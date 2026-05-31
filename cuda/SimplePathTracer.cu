#include <optix.h>
#include <optix_device.h>

#include "optix_params.h"
#include "float3_math.h"

extern "C" {
__constant__ Params params;
}

// ------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------
#define M_PI       3.14159265358979323846f
#define M_1_PI_F   0.31830988618379067154f   // 1/pi
#define EPS        3e-3f                     // ray offset epsilon to avoid self-intersection

// ==========================================================================
// RANDOM NUMBER GENERATOR
// LCG (Linear Congruential Generator). Fast on GPU.
// Good enough for path tracing when seed is varied per pixel, sample, bounce.
// ==========================================================================

__device__ unsigned int lcg_next(unsigned int& seed) {
    seed = 1664525u * seed + 22695477u;
    return seed;
}

// Returns float in [0, 1)
__device__ float rnd(unsigned int& seed) {
    return (lcg_next(seed) & 0xFFFFFF) / 16777216.0f;
}

// ==========================================================================
// NORMAL HELPERS
// Geometric normal: flat, computed from edge cross product. Used for
//   enter/exit decisions on glass (interpolated normals unreliable near silhouettes).
// Interpolated normal: smooth, from per-vertex normals. Used for shading
//   to produce smooth appearance regardless of polygon count.
// ==========================================================================

__device__ float3 getGeometricNormal(const HitGroupDataCommon* sbt, const float3& ray_dir)
{
    const int   prim_idx = optixGetPrimitiveIndex();
    const uint3 tri      = sbt->indices[prim_idx];
    float3 e1 = sbt->vertices[tri.y].position - sbt->vertices[tri.x].position;
    float3 e2 = sbt->vertices[tri.z].position - sbt->vertices[tri.x].position;
    // optixTransformNormalFromObjectToWorldSpace uses inverse-transpose of instance
    // transform, which is correct for normals (regular transform distorts them under scaling)
    float3 n = normalize(optixTransformNormalFromObjectToWorldSpace(cross(e1, e2)));
    if (dot(n, ray_dir) > 0.f) n = -n;  // flip so normal always faces incoming ray
    return n;
}

__device__ float3 getInterpolatedNormal(const HitGroupDataCommon* sbt, const float3& ray_dir)
{
    const int   prim_idx = optixGetPrimitiveIndex();
    const uint3 tri      = sbt->indices[prim_idx];
    const float2 bary    = optixGetTriangleBarycentrics();
    const float  b0      = 1.f - bary.x - bary.y;
    // Barycentric interpolation of per-vertex normals gives smooth Phong shading
    float3 n_obj = b0     * sbt->vertices[tri.x].normal
                 + bary.x * sbt->vertices[tri.y].normal
                 + bary.y * sbt->vertices[tri.z].normal;
    float3 n = normalize(optixTransformNormalFromObjectToWorldSpace(n_obj));
    if (dot(n, ray_dir) > 0.f) n = -n;
    return n;
}

// ==========================================================================
// ENVIRONMENT MAP — SAMPLING AND PDF
//
// Equirectangular projection maps a sphere onto a rectangle:
//   phi   = azimuth [-pi, pi]  -> u [0, 1]
//   theta = polar   [0,  pi]   -> v [0, 1]
// ==========================================================================

__device__ float2 dirToEnvmapUV(float3 dir) {
    float phi   = atan2f(dir.z, dir.x);
    float theta = acosf(clamp(dir.y, -1.f, 1.f));
    return make_float2((phi / (2.f * M_PI)) + 0.5f, theta / M_PI);
}

__device__ float3 uvToEnvmapDir(float2 uv) {
    float phi   = (uv.x - 0.5f) * 2.f * M_PI;
    float theta = uv.y * M_PI;
    return make_float3(sinf(theta) * cosf(phi), cosf(theta), sinf(theta) * sinf(phi));
}

__device__ float3 sampleEnvmap(cudaTextureObject_t tex, float3 dir) {
    float2 uv  = dirToEnvmapUV(dir);
    float4 val = tex2D<float4>(tex, uv.x, uv.y);
    return make_float3(val.x, val.y, val.z);
}

// ------------------------------------------------------------------
// ENVMAP IMPORTANCE SAMPLING via precomputed 2D CDF
//
// Problem: uniform random directions mostly land on dark parts of the envmap.
// Solution: precompute a 2D CDF weighted by luminance * sin(theta).
//   - sin(theta) corrects for equirectangular distortion: pixels near the
//     poles represent less solid angle and should be sampled proportionally less.
//   - CDF lookup concentrates samples toward bright directions.
//
// The CDF is built on the CPU and uploaded as two textures:
//   envmap_cdf_marginal_v:      1D, length H — CDF over rows (which row is bright?)
//   envmap_cdf_conditional_u:   2D, size W x H — per-row CDF over columns
//
// Sampling: binary search in marginal CDF -> get row v, then binary search in
//   conditional CDF for that row -> get column u.
// ------------------------------------------------------------------

__device__ float2 sampleEnvmapCDF(float r1, float r2, float& pdf_out)
{
    int W = params.envmap.width;
    int H = params.envmap.height;

    // --- Step 1: sample row v from marginal CDF ---
    float v_flt = 0.f;
    {
        int lo = 0, hi = H - 1;
        while (lo < hi) {
            int   mid     = (lo + hi) / 2;
            float cdf_mid = tex1Dfetch<float>(params.envmap.cdf_marginal_v, mid);
            if (cdf_mid < r1) lo = mid + 1; else hi = mid;
        }
        // Linearly interpolate within the bin for sub-texel accuracy
        float cdf_lo = (lo > 0) ? tex1Dfetch<float>(params.envmap.cdf_marginal_v, lo-1) : 0.f;
        float cdf_hi = tex1Dfetch<float>(params.envmap.cdf_marginal_v, lo);
        float t      = (cdf_hi > cdf_lo) ? (r1 - cdf_lo) / (cdf_hi - cdf_lo) : 0.f;
        v_flt        = (float(lo) + t) / float(H);
    }

    // --- Step 2: sample column u from conditional CDF for this row ---
    int row = clamp(int(v_flt * H), 0, H - 1);
    float u_flt = 0.f;
    {
        int lo = 0, hi = W - 1;
        while (lo < hi) {
            int   mid     = (lo + hi) / 2;
            float cdf_mid = tex2D<float>(params.envmap.cdf_conditional_u, mid, row);
            if (cdf_mid < r2) lo = mid + 1; else hi = mid;
        }
        float cdf_lo = (lo > 0) ? tex2D<float>(params.envmap.cdf_conditional_u, lo-1, row) : 0.f;
        float cdf_hi = tex2D<float>(params.envmap.cdf_conditional_u, lo, row);
        float t      = (cdf_hi > cdf_lo) ? (r2 - cdf_lo) / (cdf_hi - cdf_lo) : 0.f;
        u_flt        = (float(lo) + t) / float(W);
    }

    // --- PDF in solid-angle measure ---
    // PDF(uv) = luminance(uv) / total_luminance  [in UV space]
    // Converting UV -> solid angle introduces Jacobian: pdf_solid = pdf_uv * W*H / (2*pi^2 * sin_theta)
    float theta  = v_flt * M_PI;
    float sin_t  = sinf(theta);
    if (sin_t < 1e-5f) { pdf_out = 1.f; return make_float2(u_flt, v_flt); }

    float4 env_val = tex2D<float4>(params.envmap.texture, u_flt, v_flt);
    float  lum     = 0.2126f * env_val.x + 0.7152f * env_val.y + 0.0722f * env_val.z;
    pdf_out = fmaxf(lum * float(W * H) / (2.f * M_PI * M_PI * sin_t), 1e-6f);

    return make_float2(u_flt, v_flt);
}

// Evaluate the envmap PDF for an arbitrary direction.
// Needed by MIS when a BRDF-sampled ray escapes to the envmap.
__device__ float envmapPdf(float3 dir)
{
    float2 uv    = dirToEnvmapUV(dir);
    float  theta = uv.y * M_PI;
    float  sin_t = sinf(theta);
    if (sin_t < 1e-5f) return 0.f;
    float4 env_val = tex2D<float4>(params.envmap.texture, uv.x, uv.y);
    float  lum     = 0.2126f * env_val.x + 0.7152f * env_val.y + 0.0722f * env_val.z;
    return fmaxf(lum * float(params.envmap.width * params.envmap.height)
                 / (2.f * M_PI * M_PI * sin_t), 1e-6f);
}

// ==========================================================================
// ALBEDO HELPER
//
// Reads surface color. Priority:
//   1. Albedo texture (if albedo_texture != 0) — sampled at interpolated UV
//   2. Flat material color (sbt->albedo) — the Kd value from the .mtl file
//
// IMPORTANT: We use sbt->albedo for the flat color, NOT vertex colors.
// Vertex colors in typical OBJ/MTL are per-material constants; sbt->albedo
// is the Kd value loaded from the material and stored in the SBT record by
// the host. Make sure your SBT setup copies mat.color -> rec.data.albedo.
// ==========================================================================

__device__ float3 getAlbedo(const HitGroupDataLambert* sbt)
{
    const int   prim_idx = optixGetPrimitiveIndex();
    const uint3 tri      = sbt->indices[prim_idx];
    const float2 bary    = optixGetTriangleBarycentrics();
    const float  b0      = 1.f - bary.x - bary.y;

    if (sbt->albedo_texture != 0) {
        // Interpolate UV coordinates across the triangle
        const float2 uv0 = sbt->vertices[tri.x].uv;
        const float2 uv1 = sbt->vertices[tri.y].uv;
        const float2 uv2 = sbt->vertices[tri.z].uv;
        float u = uv0.x * b0 + uv1.x * bary.x + uv2.x * bary.y;
        float v = uv0.y * b0 + uv1.y * bary.x + uv2.y * bary.y;
        v = 1.f - v;  // flip V: OBJ bottom-left origin, CUDA top-left
        float4 t = tex2D<float4>(sbt->albedo_texture, u, v);
        return make_float3(t.x, t.y, t.z);
    }

    // Use the flat material color (Kd from .mtl) stored in the SBT record
    return sbt->albedo;
}

// ==========================================================================
// BRDF SAMPLING HELPERS
//
// IMPORTANCE SAMPLING: sample directions proportional to the integrand.
// This reduces variance because samples are concentrated where contribution is large.
// The Monte Carlo estimator divides by the PDF to remain unbiased:
//   estimate = f(wi) * Li(wi) * NdotL / pdf(wi)
// ==========================================================================

// Cosine-weighted hemisphere sample. PDF = NdotL / pi.
// Pairs perfectly with Lambert BRDF = albedo/pi:
//   weight = BRDF * NdotL / pdf = (albedo/pi) * NdotL / (NdotL/pi) = albedo
// All pi and NdotL terms cancel — very efficient.
__device__ float3 sampleCosineHemisphere(const float3& n, float r1, float r2)
{
    float r   = sqrtf(r1);          // disk radius
    float phi = 2.f * M_PI * r2;    // azimuth angle
    float x   = r * cosf(phi);
    float y   = r * sinf(phi);
    float z   = sqrtf(fmaxf(0.f, 1.f - r1));  // cos(theta), projected up

    // Orthonormal basis around normal n
    float3 up    = fabsf(n.y) < 0.999f ? make_float3(0.f, 1.f, 0.f)
                                        : make_float3(1.f, 0.f, 0.f);
    float3 right = normalize(cross(n, up));
    float3 fwd   = cross(right, n);
    return normalize(x * right + y * fwd + z * n);
}

// Blinn-Phong specular importance sample.
// Samples half-vector h from the NDF, reflects wo around h to get wi.
__device__ float3 sampleBlinnPhong(const float3& wo, const float3& n,
                                    float shininess, float r1, float r2, float& pdf_out)
{
    float cos_th = powf(r1, 1.f / (shininess + 1.f));  // NDF CDF inversion
    float sin_th = sqrtf(fmaxf(0.f, 1.f - cos_th * cos_th));
    float phi_h  = 2.f * M_PI * r2;

    float3 h_local = make_float3(sin_th * cosf(phi_h), sin_th * sinf(phi_h), cos_th);

    float3 up    = fabsf(n.y) < 0.999f ? make_float3(0.f, 1.f, 0.f)
                                        : make_float3(1.f, 0.f, 0.f);
    float3 right = normalize(cross(n, up));
    float3 fwd   = cross(right, n);
    float3 h     = normalize(h_local.x * right + h_local.y * fwd + h_local.z * n);

    float3 wi    = normalize(2.f * dot(wo, h) * h - wo);  // reflect wo around h

    // PDF over wi: p(h) / Jacobian, where Jacobian = 4 * HdotWo
    float NdotH  = fmaxf(dot(n, h), 0.f);
    float HdotWo = fmaxf(dot(h, wo), 1e-4f);
    float pdf_h  = (shininess + 1.f) / (2.f * M_PI) * powf(NdotH, shininess);
    pdf_out      = pdf_h / (4.f * HdotWo);
    return wi;
}

// Evaluate PDF of Blinn-Phong sampling for an existing direction wi.
// Used for MIS — need to know how likely BRDF sampling would have picked wi.
__device__ float pdfBlinnPhong(const float3& wi, const float3& wo,
                                const float3& n, float shininess)
{
    float3 h     = normalize(wi + wo);
    float NdotH  = fmaxf(dot(n, h), 0.f);
    float HdotWo = fmaxf(dot(h, wo), 1e-4f);
    float pdf_h  = (shininess + 1.f) / (2.f * M_PI) * powf(NdotH, shininess);
    return pdf_h / (4.f * HdotWo);
}

// PDF of cosine-weighted hemisphere sampling for direction wi.
__device__ float pdfCosine(const float3& wi, const float3& n) {
    return fmaxf(dot(n, wi), 0.f) * M_1_PI_F;  // NdotL / pi
}

// ==========================================================================
// NORMALIZED BLINN-PHONG BRDF EVALUATION
//
// f(wi, wo) = kd * albedo/pi  +  ks * F(HdotV) * (n+2)/(2pi) * pow(NdotH, n)
//   Diffuse (Lambertian):  scatters uniformly. 1/pi normalizes to integrate to kd.
//   Specular (Blinn-Phong): concentrated near mirror direction. (n+2)/(2pi) is the
//     normalization factor that ensures specular lobe integrates to ks regardless
//     of shininess. Without this, high shininess = more total energy = nonphysical.
//   Fresnel (Schlick): makes specular stronger at grazing angles, weaker at normal.
//
// Energy conservation: kd + ks <= 1
// The caller is responsible for multiplying by NdotL = dot(n, wi).
// This BRDF does NOT include NdotL — that belongs in the rendering equation, not BRDF.
// ==========================================================================

__device__ float3 evalBlinnPhong(
    const float3& wi,         // direction toward light (incoming)
    const float3& wo,         // direction toward camera (outgoing)
    const float3& n,          // shading normal
    const float3& albedo,     // diffuse color
    const float3& spec_color, // specular F0 color
    float shininess,          // specular exponent (higher = sharper)
    float kd,                 // diffuse weight
    float ks)                 // specular weight (kd + ks <= 1)
{
    float NdotL = fmaxf(dot(n, wi), 0.f);
    float NdotV = fmaxf(dot(n, wo), 0.f);
    if (NdotL <= 0.f || NdotV <= 0.f) return make_float3(0.f);

    float3 h    = normalize(wi + wo);
    float NdotH = fmaxf(dot(n, h), 0.f);
    float HdotV = fmaxf(dot(h, wo), 0.f);

    // Diffuse: Lambert = albedo / pi
    float3 f_diff = kd * albedo * M_1_PI_F;

    // Specular: normalized Blinn-Phong
    float norm    = (shininess + 2.f) / (2.f * M_PI);  // energy conservation normalization
    float D       = powf(NdotH, shininess);              // NDF value

    // Schlick Fresnel: F = F0 + (1-F0)*(1-HdotV)^5
    float  t      = powf(1.f - HdotV, 5.f);
    float3 F      = spec_color + (make_float3(1.f) - spec_color) * t;

    float3 f_spec = ks * F * norm * D;

    return f_diff + f_spec;
    // Note: do NOT multiply by NdotL here — caller does that
}

// ==========================================================================
// EMISSIVE TRIANGLE SAMPLING (NEE for area lights)
//
// Physical area lights = mesh triangles with an emission field.
// Works for any shape: quads (2 triangles), spheres (many triangles), etc.
//
// A sphere light is just a sphere mesh with emission set — no special math.
// The NEE samples points on the triangulated surface, so a sphere light
// automatically works correctly once its triangles are in emissive_triangles[].
// ==========================================================================
/*
__device__ float3 sampleEmissiveTriangle(int idx, float r1, float r2,
                                          float3& pos_out, float& pdf_area_out)
{
    const EmissiveTriangle& tri = params.emissive_triangles[idx];

    // Uniform sampling over triangle area using the sqrt trick
    float su = sqrtf(r1);
    float u  = 1.f - su, v = r2 * su, w = 1.f - u - v;
    pos_out  = u * tri.v0 + v * tri.v1 + w * tri.v2;

    float area    = 0.5f * length(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
    pdf_area_out  = 1.f / fmaxf(area, 1e-6f);  // uniform over area

    return tri.emission;
}
*/
// Convert area-measure PDF to solid-angle-measure PDF.
// Needed for MIS — all PDFs must be in the same measure.
// pdf_solid = pdf_area * dist^2 / |cos(theta_light)|
__device__ float areaPdfToSolidAngle(float pdf_area, float dist, float cos_light) {
    return pdf_area * dist * dist / fmaxf(fabsf(cos_light), 1e-4f);
}

// ==========================================================================
// SHADOW RAY
// Returns true if the path from origin to origin+dir*max_dist is blocked.
// Uses TERMINATE_ON_FIRST_HIT: we only care about occlusion, not which object.
// ==========================================================================

__device__ bool isOccluded(float3 origin, float3 dir, float max_dist)
{
    optixTraverse(params.traversable, origin, dir, EPS, max_dist - EPS, 0.f,
                  OptixVisibilityMask(255),
                  OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                  0, 1, 0);
    return optixHitObjectIsHit();
}

// ==========================================================================
// MIS — MULTIPLE IMPORTANCE SAMPLING (Balance Heuristic)
//
// When combining two sampling strategies with PDFs p_a and p_b:
//   w_a = p_a / (p_a + p_b)
//
// The weight is high when strategy A is "more responsible" for this sample.
// This reduces variance compared to using only one strategy:
//   - Light sampling: good for small bright lights, bad for mirror surfaces
//   - BRDF sampling: good for mirror surfaces, bad for small bright lights
//   - MIS: good for both simultaneously
//
// MIS requires importance sampling (IS) to function — the PDFs from IS are
// the weights' input. So yes: if you use MIS, you are also using IS.
// ==========================================================================

__device__ float misWeight(float pdf_a, float pdf_b) {
    return pdf_a / fmaxf(pdf_a + pdf_b, 1e-8f);
}

// ==========================================================================
// RAY GENERATION PROGRAM
// Entry point — one thread per pixel. Generates camera rays and accumulates.
// ==========================================================================

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    if (idx.x >= params.width || idx.y >= params.height) return;

    const int pixel = idx.y * params.width + idx.x;
    float3 frame_color = make_float3(0.f);

    for (int s = 0; s < params.samples_per_pixel; ++s) {
        // Unique seed: XOR of pixel coords, frame seed, sample index
        unsigned int seed = params.random_seed
            ^ (idx.x * 73856093u) ^ (idx.y * 19349663u) ^ ((unsigned int)s * 83492791u);

        // Jitter pixel sample for anti-aliasing (without jitter: all samples hit same center)
        float u = ((float)idx.x + rnd(seed)) / (float)params.width;
        float v = ((float)idx.y + rnd(seed)) / (float)params.height;

        float3 dir = normalize(
            params.camera.lower_left_corner
            + u * params.camera.horizontal
            + v * params.camera.vertical
            - params.camera.origin);

        // Payload: [p0,p1,p2]=color, [p3]=depth, [p4,p5,p6]=throughput
        unsigned int p0=0,p1=0,p2=0, p3=0;
        unsigned int p4=__float_as_uint(1.f), p5=__float_as_uint(1.f), p6=__float_as_uint(1.f);

        // SER (Shader Execution Reordering): split trace into traverse + reorder + invoke
        // traverse: fires ray, runs BVH + anyhit, stores hit info (no shading yet)
        // reorder:  groups threads by hit material -> coherent warp execution
        // invoke:   runs closesthit or miss with better warp coherence
        optixTraverse(params.traversable, params.camera.origin, dir,
                      EPS, 1e16f, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                      0, 1, 0, p0, p1, p2, p3, p4, p5, p6);
        optixReorder();
        optixInvoke(p0, p1, p2, p3, p4, p5, p6);

        frame_color = frame_color + make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
    }
    frame_color = frame_color / (float)params.samples_per_pixel;

    // Progressive accumulation: running average over frames
    // current_sample is the 1-indexed frame counter (incremented by host before launch)
    if (params.current_sample == 0) {
        params.accum_buffer[pixel] = frame_color;
    } else {
        float N = (float)params.current_sample;
        params.accum_buffer[pixel] = (params.accum_buffer[pixel] * (N-1.f) + frame_color) / N;
    }

    // float3 fc = clamp(params.accum_buffer[pixel], 0.f, 1.f);


    // params.image[pixel] = make_uchar4(
    //     (unsigned char)(fc.x * 255.99f),
    //     (unsigned char)(fc.y * 255.99f),
    //     (unsigned char)(fc.z * 255.99f), 255);

    float3 final_color = params.accum_buffer[pixel] / (1.f + params.accum_buffer[pixel]);
    final_color = final_color * 255.99f;

    params.image[pixel] = make_uchar4(
        (unsigned char)(final_color.x),
        (unsigned char)(final_color.y),
        (unsigned char)(final_color.z), 
        255);
}

// ==========================================================================
// MISS PROGRAM
// Called when ray escapes scene. Returns envmap radiance (IBL background).
// ==========================================================================

extern "C" __global__ void __miss__ms()
{
    float3 ray_dir = normalize(optixGetWorldRayDirection());
    float3 Le = make_float3(1.f);
    if (params.envmap.has_envmap) {
        Le = sampleEnvmap(params.envmap.texture, ray_dir)
           * params.envmap.scale * powf(2.f, params.envmap.exposure);
        Le = clamp(Le, 0.f, 10.f);  // prevent extreme fireflies
    }
    optixSetPayload_0(__float_as_uint(Le.x));
    optixSetPayload_1(__float_as_uint(Le.y));
    optixSetPayload_2(__float_as_uint(Le.z));
}

// ==========================================================================
// CLOSEST HIT — OPAQUE SURFACE (Blinn-Phong BRDF + NEE + MIS)
//
// At each surface hit, light reaches this point via two paths:
//   DIRECT (NEE): we explicitly sample light sources and test visibility.
//     - Area lights: sample random point on emissive triangle, shadow ray
//     - Envmap: sample bright direction from CDF, shadow ray to infinity
//   INDIRECT: fire a new ray in BRDF-sampled direction, recurse.
//     - The miss shader handles envmap contribution for escaped rays
//
// MIS combines direct and indirect contributions:
//   - For direct (light sample): weight = p_light / (p_light + p_brdf)
//   - For indirect (BRDF sample hitting envmap): weight = p_brdf / (p_brdf + p_env)
// This gives low variance for both small sharp lights and smooth glossy surfaces.
// ==========================================================================

extern "C" __global__ void __closesthit__ch()
{
    const float  t_hit   = optixGetRayTmax();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 hit_pos = optixGetWorldRayOrigin() + t_hit * ray_dir;

    const HitGroupDataLambert* sbt = (HitGroupDataLambert*)optixGetSbtDataPointer();

    float3 normal = getInterpolatedNormal(sbt, ray_dir);    // smooth shading normal
    float3 n_geom = getGeometricNormal(sbt, ray_dir);       // geometric normal
    float3 albedo = getAlbedo(sbt);                         // material color
    float3 wo     = -normalize(ray_dir);                    // toward camera


    unsigned int depth     = optixGetPayload_3();
    float3       throughput = make_float3(
        __uint_as_float(optixGetPayload_4()),
        __uint_as_float(optixGetPayload_5()),
        __uint_as_float(optixGetPayload_6()));

    float3 added_light = make_float3(0.f);
    if (nonZero(sbt->emission)) {
        added_light = throughput * sbt->emission * params.light_intensity;
    }

    // Per-bounce seed: unique per pixel + frame + depth + hit position
    unsigned int seed = params.random_seed
        ^ (optixGetLaunchIndex().x * 73856093u)
        ^ (optixGetLaunchIndex().y * 19349663u)
        ^ (depth * 83492791u)
        ^ __float_as_uint(hit_pos.x + hit_pos.y);

    // Blinn-Phong parameters (could be per-material via SBT in a full implementation)
    const float  kd        = 0.7f;
    const float  ks        = 0.3f;
    const float  shininess = 64.f;
    const float3 spec_col  = make_float3(1.f);  // white specular

    float3 direct = make_float3(0.f);

    // ------------------------------------------------------------------
    // NEE: Area lights (emissive triangle meshes)
    // Emissive spheres work here too — they are just many emissive triangles.
    // ------------------------------------------------------------------
    /*
    for (int li = 0; li < params.num_emissive_triangles; li++) {
        float  pdf_area;
        float3 light_pos;
        float3 Le = sampleEmissiveTriangle(li, rnd(seed), rnd(seed), light_pos, pdf_area);

        float3 to_light  = light_pos - hit_pos;
        float  dist      = length(to_light);
        float3 wi_light  = normalize(to_light);
        float  NdotL     = dot(n_geom, wi_light);
        if (NdotL <= 0.f) continue;  // light behind surface

        // Convert area PDF to solid-angle for MIS
        const EmissiveTriangle& et = params.emissive_triangles[li];
        float3 light_n  = normalize(cross(et.v1 - et.v0, et.v2 - et.v0));
        float  cos_l    = fabsf(dot(light_n, -wi_light));
        float  pdf_sa   = areaPdfToSolidAngle(pdf_area, dist, cos_l);

        if (!isOccluded(hit_pos + n_geom * EPS, wi_light, dist)) {
            float3 f       = evalBlinnPhong(wi_light, wo, normal, albedo, spec_col, shininess, kd, ks);
            // BRDF PDF for this direction (needed for MIS weight)
            float pdf_brdf = kd * pdfCosine(wi_light, normal)
                           + ks * pdfBlinnPhong(wi_light, wo, normal, shininess);
            float w_light  = misWeight(pdf_sa, pdf_brdf);  // favor light sample when pdf_sa >> pdf_brdf
            direct = direct + w_light * f * Le * NdotL / fmaxf(pdf_sa, 1e-6f);
        }
    }
    */
    // ------------------------------------------------------------------
    // NEE: Envmap (IBL direct lighting via CDF importance sampling)
    // Fires a shadow ray toward a bright envmap direction.
    // Without this, envmap lighting only arrives via random bounces that
    // happen to escape — very slow convergence, especially indoors.
    // ------------------------------------------------------------------
    if (params.envmap.has_envmap) {
        float  pdf_env;
        float2 env_uv  = sampleEnvmapCDF(rnd(seed), rnd(seed), pdf_env);
        float3 wi_env  = uvToEnvmapDir(env_uv);
        float  NdotL_e = dot(n_geom, wi_env);
        if (NdotL_e > 0.f && !isOccluded(hit_pos + n_geom * EPS, wi_env, 1e16f)) {
            float3 Le_env  = sampleEnvmap(params.envmap.texture, wi_env)
                           * params.envmap.scale * powf(2.f, params.envmap.exposure);
            float3 f_env   = evalBlinnPhong(wi_env, wo, normal, albedo, spec_col, shininess, kd, ks);
            float pdf_brdf = kd * pdfCosine(wi_env, normal)
                           + ks * pdfBlinnPhong(wi_env, wo, normal, shininess);
            // MIS: favor envmap sample when pdf_env >> pdf_brdf (sharp bright spots)
            float w_env    = misWeight(pdf_env, pdf_brdf);
            direct = direct + w_env * f_env * Le_env * NdotL_e / fmaxf(pdf_env, 1e-6f);
        }
    }

    // ------------------------------------------------------------------
    // RUSSIAN ROULETTE (throughput-based, unbiased)
    // The survival probability = luminance of throughput.
    // Black albedo -> low throughput -> high termination chance.
    // White/mirror albedo -> full throughput -> path continues.
    // Surviving paths are boosted by 1/prob to keep the estimator unbiased.
    // ------------------------------------------------------------------
    float3 new_throughput = throughput * albedo;
    bool   cont           = (depth < params.max_bounce_depth);
    float  rr_prob        = 1.f;

    if (cont && depth >= 2) {
        float lum = 0.2126f * new_throughput.x + 0.7152f * new_throughput.y + 0.0722f * new_throughput.z;
        rr_prob   = clamp(lum, 0.05f, 0.95f);
        if (rnd(seed) > rr_prob) cont = false;
        else                     new_throughput = new_throughput / rr_prob;  // unbiased boost
    }

    // ------------------------------------------------------------------
    // BRDF Importance Sampling — indirect lighting (next bounce)
    // Sample a direction from the BRDF distribution. The direction chosen
    // here determines where the next ray goes. If it escapes to the envmap,
    // the miss shader returns the envmap color, weighted by the MIS w_brdf.
    // ------------------------------------------------------------------
    float3 indirect = make_float3(0.f);

    if (cont) {
        float  prob_diff  = kd / (kd + ks);
        float3 wi_bounce;
        float  pdf_bounce;

        if (rnd(seed) < prob_diff) {
            // Sample diffuse lobe: cosine-weighted hemisphere
            wi_bounce       = sampleCosineHemisphere(normal, rnd(seed), rnd(seed));
            float pd        = pdfCosine(wi_bounce, normal);
            float ps        = pdfBlinnPhong(wi_bounce, wo, normal, shininess);
            pdf_bounce      = prob_diff * pd + (1.f - prob_diff) * ps;  // mixture PDF
        } else {
            // Sample specular lobe: Blinn-Phong NDF
            float ps;
            wi_bounce       = sampleBlinnPhong(wo, normal, shininess, rnd(seed), rnd(seed), ps);
            float pd        = pdfCosine(wi_bounce, normal);
            pdf_bounce      = prob_diff * pd + (1.f - prob_diff) * ps;  // mixture PDF
        }

        float NdotL_geom_b = dot(n_geom, wi_bounce);
        float NdotL_b = fmaxf(dot(normal, wi_bounce), 0.f);
        if (NdotL_geom_b > 0.f && NdotL_b > 0.f && pdf_bounce > 1e-6f) {
            float3 f = evalBlinnPhong(wi_bounce, wo, normal, albedo, spec_col, shininess, kd, ks);

            // MIS weight for BRDF sample vs envmap sampling.
            // If this bounce escapes to the envmap, the miss shader returns Le.
            // We pre-weight the child throughput by w_brdf so the miss result
            // is automatically MIS-weighted without any extra code in miss shader.
            float w_brdf = 1.f;
            if (params.envmap.has_envmap) {
                float pdf_env = envmapPdf(wi_bounce);
                w_brdf = misWeight(pdf_bounce, pdf_env);
            }

            // Child throughput = accumulated throughput * BRDF weight * MIS weight
            float3 child_tp = new_throughput * f * NdotL_b / pdf_bounce;

            unsigned int p0=0,p1=0,p2=0, p3=depth+1;
            unsigned int p4=__float_as_uint(child_tp.x * w_brdf);
            unsigned int p5=__float_as_uint(child_tp.y * w_brdf);
            unsigned int p6=__float_as_uint(child_tp.z * w_brdf);

            optixTrace(params.traversable, hit_pos + n_geom * EPS, wi_bounce,
                       EPS, 1e16f, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                       0, 1, 0, p0, p1, p2, p3, p4, p5, p6);

            indirect = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
        }
    }

    float3 final_color = direct + indirect * albedo + added_light;  // modulate indirect by albedo for energy conservation
    optixSetPayload_0(__float_as_uint(final_color.x));
    optixSetPayload_1(__float_as_uint(final_color.y));
    optixSetPayload_2(__float_as_uint(final_color.z));
}

// ==========================================================================
// CLOSEST HIT — GLASS (Delta BSDF: Fresnel reflect + refract)
//
// Glass is a DELTA BSDF: at each point, only one exact direction is valid.
// Consequence: NEE does NOT work (BRDF = 0 for any arbitrary direction).
// Instead: stochastically pick reflect or refract with Fresnel probability.
//
// For an emissive sphere light you would want to sample points on its surface
// from inside the glass — this is advanced and can be skipped for now.
//
// The Blinn-Phong surface highlights are an ADDITION to the delta BSDF,
// modeling slight surface roughness of real glass (dust, micro-scratches).
// They are evaluated against directional lights only.
//
// EMISSIVE SPHERE LIGHT FAQ:
//   Q: How does a sphere emit light?
//   A: Tessellate the sphere into triangles. Set emission on those triangles.
//      Upload them to params.emissive_triangles[]. The opaque closesthit
//      uses NEE to sample points on those triangles exactly like any other
//      area light. No special code needed for sphere vs quad vs any shape.
// ==========================================================================

extern "C" __global__ void __closesthit__glass()
{
    const float  t_hit   = optixGetRayTmax();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 hit_pos = optixGetWorldRayOrigin() + t_hit * ray_dir;

    const HitGroupDataGlass* sbt = (HitGroupDataGlass*)optixGetSbtDataPointer();

    const int   prim_idx = optixGetPrimitiveIndex();
    const uint3 tri      = sbt->indices[prim_idx];

    // Geometric normal for enter/exit decision — reliable near silhouettes
    float3 geom_n = normalize(optixTransformNormalFromObjectToWorldSpace(
        cross(sbt->vertices[tri.y].position - sbt->vertices[tri.x].position,
              sbt->vertices[tri.z].position - sbt->vertices[tri.x].position)));
    bool entering = dot(geom_n, ray_dir) < 0.f;
    if (!entering) geom_n = -geom_n;  // always points against ray

    // Smooth normal for highlight shading only
    const float2 bary = optixGetTriangleBarycentrics();
    const float  b0   = 1.f - bary.x - bary.y;
    float3 shade_n = normalize(optixTransformNormalFromObjectToWorldSpace(
        b0     * sbt->vertices[tri.x].normal
        + bary.x * sbt->vertices[tri.y].normal
        + bary.y * sbt->vertices[tri.z].normal));
    if (dot(shade_n, ray_dir) > 0.f) shade_n = -shade_n;

    // Glass tint from vertex color (fallback to white if unset)
    float3 glass_tint = sbt->albedo * b0
                      + sbt->albedo * bary.x
                      + sbt->albedo * bary.y;
    float tint_lum = 0.2126f*glass_tint.x + 0.7152f*glass_tint.y + 0.0722f*glass_tint.z;
    if (tint_lum < 0.01f) glass_tint = make_float3(1.f);  // uncolored glass = white

    float ior = sbt->refraction_index;

    unsigned int depth     = optixGetPayload_3();
    float3       throughput = make_float3(
        __uint_as_float(optixGetPayload_4()),
        __uint_as_float(optixGetPayload_5()),
        __uint_as_float(optixGetPayload_6()));

    float3 added_light = make_float3(0.f);
    if (nonZero(sbt->emission)) {
        added_light = throughput * sbt->emission * params.light_intensity;
    }

    unsigned int seed = params.random_seed
        ^ (optixGetLaunchIndex().x * 73856093u)
        ^ (optixGetLaunchIndex().y * 19349663u)
        ^ (depth * 83492791u)
        ^ __float_as_uint(hit_pos.x + hit_pos.y);

    float3 wo = -normalize(ray_dir);

    // ------------------------------------------------------------------
    // Surface highlights (Blinn-Phong sheen from directional lights)
    // Models the rough surface component of real glass.
    // ------------------------------------------------------------------
    float3 surface_color = make_float3(0.f);
    /*
    for (int li = 0; li < params.num_lights; li++) {
        if (params.lights[li].type != 1) continue;  // directional lights only
        // position_or_direction stores the direction the light is POINTING (toward scene)
        // Negate to get the direction from surface toward light (wi)
        float3 light_dir = normalize(-params.lights[li].position_or_direction);
        float  NdotL     = fmaxf(dot(shade_n, light_dir), 0.f);
        if (NdotL <= 0.f) continue;

        if (!isOccluded(hit_pos + geom_n * EPS, light_dir, 1e16f)) {
            float3 h   = normalize(wo + light_dir);
            float NdotH = fmaxf(dot(shade_n, h), 0.f);
            float HdotV = fmaxf(dot(h, wo), 0.f);
            float norm  = (128.f + 2.f) / (2.f * M_PI);  // shininess = 128
            float D     = powf(NdotH, 128.f);
            // Schlick Fresnel: glass reflects more at grazing angles
            float r0    = (1.f - ior) / (1.f + ior); r0 = r0 * r0;
            float F_hl  = r0 + (1.f - r0) * powf(1.f - HdotV, 5.f);
            surface_color = surface_color + params.lights[li].color * F_hl * norm * D * NdotL;
        }
    }
    */
    // ------------------------------------------------------------------
    // Stochastic Fresnel: reflect or refract
    //
    // eta = n_incoming / n_outgoing (Snell's law ratio)
    // cos_i = angle of incidence at surface
    // sin2_t = sin^2(refracted angle) via Snell: n1*sin1 = n2*sin2
    //
    // Total Internal Reflection (TIR): when inside glass at steep angle,
    //   sin2_t > 1 -> refraction impossible -> force reflection.
    // Otherwise: reflect with probability F (Fresnel), refract with (1-F).
    //
    // Monte Carlo weight: f/pdf = tint (F cancels with probability F).
    // ------------------------------------------------------------------
    float3 indirect = make_float3(0.f);
    if (depth < params.max_bounce_depth) {
        float eta    = entering ? (1.f / ior) : ior;
        float cos_i  = fmaxf(dot(geom_n, wo), 0.f);
        float sin2_t = eta * eta * (1.f - cos_i * cos_i);

        float r0 = (1.f - ior) / (1.f + ior); r0 = r0 * r0;
        float F  = r0 + (1.f - r0) * powf(1.f - cos_i, 5.f);

        bool reflect = (sin2_t > 1.f) || (rnd(seed) < F);  // TIR or Fresnel coin flip

        float3 new_dir, new_origin, weight;
        if (reflect) {
            // Mirror reflection: r = d - 2*(d·n)*n
            new_dir    = normalize(ray_dir - 2.f * dot(ray_dir, geom_n) * geom_n);
            new_origin = hit_pos + geom_n * EPS;   // offset away from surface
            weight     = glass_tint;
        } else {
            // Snell refraction: t = eta*d + (eta*cos_i - cos_t)*n
            float cos_t = sqrtf(1.f - sin2_t);
            new_dir     = normalize(eta * ray_dir + (eta * cos_i - cos_t) * geom_n);
            new_origin  = hit_pos + new_dir * EPS;  // offset into surface
            // eta^2: radiance scales by n^2 across a refractive boundary
            weight      = glass_tint * (eta * eta);
        }

        float3 child_tp = throughput * weight;
        unsigned int p0=0,p1=0,p2=0, p3=depth+1;
        unsigned int p4=__float_as_uint(child_tp.x);
        unsigned int p5=__float_as_uint(child_tp.y);
        unsigned int p6=__float_as_uint(child_tp.z);

        optixTrace(params.traversable, new_origin, new_dir,
                   EPS, 1e16f, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                   0, 1, 0, p0, p1, p2, p3, p4, p5, p6);

        indirect = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
    }

    float3 final_color = added_light + surface_color + indirect * glass_tint ;

    optixSetPayload_0(__float_as_uint(final_color.x));
    optixSetPayload_1(__float_as_uint(final_color.y));
    optixSetPayload_2(__float_as_uint(final_color.z));
    optixSetPayload_3(depth);
}
