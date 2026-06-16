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
#define M_1_PI_F   0.31830988618379067154f
#define EPS        3e-3f    // ray offset to avoid self-intersection


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





// ------------------------------------------------------------------
// Register layout RadiancePRD
//   p0      : ray_type       (uint32,   R by CH/MS)
//   p1..p3  : throughput     (float3,   RW)
//   p4      : done           (uint32,   W by CH/MS, R by caller)
//   p5..p7  : emitted        (float3,   W by CH/MS)
//   p8..p10 : radiance       (float3,   W by CH/MS — reserved for NEE)
//   p11..p13: next_origin    (float3,   W by CH)
//   p14..p16: next_direction (float3,   W by CH)
//   p17     : is_specular    (uint32,   W by CH)
// ------------------------------------------------------------------

static __forceinline__ __device__ void storeClosesthitRadiancePRD(const RadiancePRD& prd)
{
    optixSetPayload_1(__float_as_uint(prd.throughput.x));
    optixSetPayload_2(__float_as_uint(prd.throughput.y));
    optixSetPayload_3(__float_as_uint(prd.throughput.z));
    optixSetPayload_4(prd.done);
    optixSetPayload_5(__float_as_uint(prd.emitted.x));
    optixSetPayload_6(__float_as_uint(prd.emitted.y));
    optixSetPayload_7(__float_as_uint(prd.emitted.z));
    optixSetPayload_8(__float_as_uint(prd.radiance.x));
    optixSetPayload_9(__float_as_uint(prd.radiance.y));
    optixSetPayload_10(__float_as_uint(prd.radiance.z));
    optixSetPayload_11(__float_as_uint(prd.next_origin.x));
    optixSetPayload_12(__float_as_uint(prd.next_origin.y));
    optixSetPayload_13(__float_as_uint(prd.next_origin.z));
    optixSetPayload_14(__float_as_uint(prd.next_direction.x));
    optixSetPayload_15(__float_as_uint(prd.next_direction.y));
    optixSetPayload_16(__float_as_uint(prd.next_direction.z));
    optixSetPayload_17(prd.is_specular);
}

static __forceinline__ __device__ RadiancePRD loadClosesthitRadiancePRD()
{
    RadiancePRD prd = {};
    prd.throughput.x = __uint_as_float(optixGetPayload_1());
    prd.throughput.y = __uint_as_float(optixGetPayload_2());
    prd.throughput.z = __uint_as_float(optixGetPayload_3());
    return prd;
}

static __forceinline__ __device__ void storeMissRadiancePRD(const RadiancePRD& prd)
{
    // Miss only writes: done, emitted, radiance
    optixSetPayload_4(prd.done);
    optixSetPayload_5(__float_as_uint(prd.emitted.x));
    optixSetPayload_6(__float_as_uint(prd.emitted.y));
    optixSetPayload_7(__float_as_uint(prd.emitted.z));
    optixSetPayload_8(__float_as_uint(prd.radiance.x));
    optixSetPayload_9(__float_as_uint(prd.radiance.y));
    optixSetPayload_10(__float_as_uint(prd.radiance.z));
}

static __forceinline__ __device__ RadiancePRD loadMissRadiancePRD()
{
    // Miss only reads what the caller passed
    RadiancePRD prd = {};
    return prd;
}


// ------------------------------------------------------------------
// traceRadiance — fires a ray and returns filled RadiancePRD
// ------------------------------------------------------------------
static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    RadiancePRD& prd)
{
	unsigned int u0 = prd.ray_type;
    unsigned int u1 = __float_as_uint(prd.throughput.x);
    unsigned int u2 = __float_as_uint(prd.throughput.y);
    unsigned int u3 = __float_as_uint(prd.throughput.z);
    unsigned int u4 = 0u;                               // done
    unsigned int u5 = 0u, u6 = 0u, u7 = 0u;          // emitted
    unsigned int u8 = 0u, u9 = 0u, u10 = 0u;          // radiance
    unsigned int u11 = 0u, u12 = 0u, u13 = 0u;          // next_origin
    unsigned int u14 = 0u, u15 = 0u, u16 = 0u;          // next_direction
    unsigned int u17 = 0u;                               // is_specular

    optixTraverse(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.f,                    // ray time
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,                      // SBT offset  (ray type 0)
        1,                      // SBT stride
        0,                      // miss SBT index
        u0, u1, u2, u3, u4,
        u5, u6, u7, u8, u9,
        u10, u11, u12, u13, u14,
        u15, u16, u17
    );
    optixReorder();
    optixInvoke(
        u0, u1, u2, u3, u4,
        u5, u6, u7, u8, u9,
        u10, u11, u12, u13, u14,
        u15, u16, u17
    );

    // Unpack outputs back into prd
	prd.ray_type = u0;
    prd.throughput.x = __uint_as_float(u1);
    prd.throughput.y = __uint_as_float(u2);
    prd.throughput.z = __uint_as_float(u3);
    prd.done = u4;
    prd.emitted.x = __uint_as_float(u5);
    prd.emitted.y = __uint_as_float(u6);
    prd.emitted.z = __uint_as_float(u7);
    prd.radiance.x = __uint_as_float(u8);
    prd.radiance.y = __uint_as_float(u9);
    prd.radiance.z = __uint_as_float(u10);
    prd.next_origin.x = __uint_as_float(u11);
    prd.next_origin.y = __uint_as_float(u12);
    prd.next_origin.z = __uint_as_float(u13);
    prd.next_direction.x = __uint_as_float(u14);
    prd.next_direction.y = __uint_as_float(u15);
    prd.next_direction.z = __uint_as_float(u16);
    prd.is_specular = u17;
}


// ------------------------------------------------------------------
// Register layout DenoiserGuidePRD
//   p0      : ray_type       (uint32,   R by CH/MS)
//   p1..p3  : albedo         (float3,   RW)
//   p4..p6  : normal         (float3,   RW)
// ------------------------------------------------------------------

static __forceinline__ __device__ void storeClosesthitDenoiserGuidePRD(const DenoiserGuidePRD& prd)
{
    optixSetPayload_1(__float_as_uint(prd.albedo.x));
    optixSetPayload_2(__float_as_uint(prd.albedo.y));
    optixSetPayload_3(__float_as_uint(prd.albedo.z));
    optixSetPayload_4(__float_as_uint(prd.normal.x));
    optixSetPayload_5(__float_as_uint(prd.normal.y));
    optixSetPayload_6(__float_as_uint(prd.normal.z));
}

static __forceinline__ __device__ DenoiserGuidePRD loadClosesthitDenoiserGuidePRD()
{
    DenoiserGuidePRD prd = {};
    return prd;
}

static __forceinline__ __device__ DenoiserGuidePRD loadMissDenoiserGuidePRD()
{
    // Miss only reads what the caller passed in
    DenoiserGuidePRD prd = {};
    return prd;
}
// ------------------------------------------------------------------
// traceDenoiserGuide — fires a ray and returns filled DenoiserGuidePRD
// ------------------------------------------------------------------
static __forceinline__ __device__ void traceDenoiserGuide(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    DenoiserGuidePRD& prd)
{
    unsigned int u0 = prd.ray_type;
    unsigned int u1 = __float_as_uint(prd.albedo.x);
    unsigned int u2 = __float_as_uint(prd.albedo.y);
    unsigned int u3 = __float_as_uint(prd.albedo.z);
    unsigned int u4 = __float_as_uint(prd.normal.x);
    unsigned int u5 = __float_as_uint(prd.normal.y);
    unsigned int u6 = __float_as_uint(prd.normal.z);

    optixTraverse(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.f,                    // ray time
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,                      // SBT offset  (ray type 0)
        1,                      // SBT stride
        0,                      // miss SBT index
        u0, 
        u1, u2, u3, 
        u4, u5, u6
    );
    optixReorder();
    optixInvoke(
        u0, 
        u1, u2, u3, 
        u4, u5, u6
    );

    // Unpack outputs back into prd
	prd.ray_type = u0;
	prd.albedo.x = __uint_as_float(u1);
	prd.albedo.y = __uint_as_float(u2);
	prd.albedo.z = __uint_as_float(u3);
	prd.normal.x = __uint_as_float(u4);
	prd.normal.y = __uint_as_float(u5);
	prd.normal.z = __uint_as_float(u6);
}


// ------------------------------------------------------------------
// Geometry helpers — shared by both hit programs
// ------------------------------------------------------------------

// Interpolate shading normal from vertex buffer using barycentrics.
// Flips toward the incoming ray (two-sided shading).
static __forceinline__ __device__ float3 getInterpolatedNormal(
    const HitGroupDataCommon* sbt,
    const float3& ray_dir)
{
    const int    prim_idx = optixGetPrimitiveIndex();
    const uint3  tri = sbt->indices[prim_idx];
    const float2 bary = optixGetTriangleBarycentrics();
    const float  b0 = 1.f - bary.x - bary.y;

    float3 n = normalize(
        b0 * sbt->vertices[tri.x].normal +
        bary.x * sbt->vertices[tri.y].normal +
        bary.y * sbt->vertices[tri.z].normal
    );
    n = normalize(optixTransformNormalFromObjectToWorldSpace(n));

    // Flip if ray hits back face (two-sided)
    if (dot(n, ray_dir) > 0.f)
        n = -n;

    return n;
}

// Interpolate UV coordinates across the triangle.
static __forceinline__ __device__ float2 getInterpolatedUV(
    const HitGroupDataCommon* sbt)
{
    const int    prim_idx = optixGetPrimitiveIndex();
    const uint3  tri = sbt->indices[prim_idx];
    const float2 bary = optixGetTriangleBarycentrics();
    const float  b0 = 1.f - bary.x - bary.y;

    return make_float2(
        b0 * sbt->vertices[tri.x].uv.x + bary.x * sbt->vertices[tri.y].uv.x + bary.y * sbt->vertices[tri.z].uv.x,
        b0 * sbt->vertices[tri.x].uv.y + bary.x * sbt->vertices[tri.y].uv.y + bary.y * sbt->vertices[tri.z].uv.y
    );
}

// Sample emissive — texture takes priority over constant.
static __forceinline__ __device__ float3 getEmissive(
    const HitGroupDataCommon* sbt, float2 uv)
{
    if (sbt->emission_texture != 0) {
        float4 t = tex2D<float4>(sbt->emission_texture, uv.x, uv.y);
        return make_float3(t.x, t.y, t.z);
    }
    return sbt->emission * params.light_intensity;
}

// Sample alpha.
static __forceinline__ __device__ float3 getAlpha(
    const HitGroupDataCookTorrance* sbt, float2 uv)
{
    if (sbt->alpha_texture != 0) {
        float4 t = tex2D<float4>(sbt->alpha_texture, uv.x, uv.y);
		return make_float3(t.x, t.y, t.z);  // Assuming alpha is in RGB channels, returns (alpha, alpha, alpha)
    }
	return make_float3(1.f);  //returns white (opaque) if no alpha texture
}

// ------------------------------------------------------------------
// Cook-Torrance material helpers
// ------------------------------------------------------------------

// Returns base color sampled from texture
// Falls back to sbt->base_color if no texture.
static __forceinline__ __device__ float3 getBaseColor(
    const HitGroupDataCookTorrance* sbt, float2 uv)
{
    if (sbt->albedo_texture != 0) {
        float4 t = tex2D<float4>(sbt->albedo_texture, uv.x, uv.y);
        return make_float3(t.x, t.y, t.z);
    }
    return sbt->base_color;
}

static __forceinline__ __device__ float3 getTint(
    const HitGroupDataGlass* sbt, float2 uv)
{
    if (sbt->tint_texture != 0) {
        float4 t = tex2D<float4>(sbt->tint_texture, uv.x, uv.y);
        return make_float3(t.x, t.y, t.z);
    }
    return sbt->tint;
}

// Returns scalar roughness. Texture (R channel) takes priority.
static __forceinline__ __device__ float getRoughness(
    const HitGroupDataCookTorrance* sbt, float2 uv)
{
    if (sbt->roughness_texture != 0) {
        float4 t = tex2D<float4>(sbt->roughness_texture, uv.x, uv.y);
        return t.x;  // R channel = roughness (map_Pr is grayscale)
    }
    return sbt->roughness;
}

// Returns scalar metallic. Texture (R channel) takes priority.
static __forceinline__ __device__ float getMetallic(
    const HitGroupDataCookTorrance* sbt, float2 uv)
{
    if (sbt->metallic_texture != 0) {
        float4 t = tex2D<float4>(sbt->metallic_texture, uv.x, uv.y);
        return t.x;  // R channel = metallic (map_Pm is grayscale)
    }
    return sbt->metallic;
}

// Specular color for per-texel F0 override (map_Ks).
// Used when computing F0 for materials with a specular texture.
static __forceinline__ __device__ float3 getSpecularColor(
    const HitGroupDataCookTorrance* sbt, float2 uv)
{
    if (sbt->specular_texture != 0) {
        float4 t = tex2D<float4>(sbt->specular_texture, uv.x, uv.y);
        return make_float3(t.x, t.y, t.z);
    }
    return sbt->specular_color;
}


// ------------------------------------------------------------------
// Math utilities
// ------------------------------------------------------------------

static __forceinline__ __device__ float luminance(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

static __forceinline__ __device__ float3 lerp3(float3 a, float3 b, float t)
{
    return a + t * (b - a);
}

// Build orthonormal tangent frame around a shading normal.
static __forceinline__ __device__ void buildONB(
    const float3& n,
    float3& tangent,
    float3& bitangent)
{
    float sign = copysignf(1.0f, n.z);
    float a = -1.0f / (sign + n.z);
    float b = n.x * n.y * a;
    tangent = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    bitangent = make_float3(b, sign + n.y * n.y * a, -n.y);
}

// Transform a direction from tangent space to world space.
static __forceinline__ __device__ float3 tangentToWorld(
    const float3& v,
    const float3& n,
    const float3& tangent,
    const float3& bitangent)
{
    return v.x * tangent + v.y * bitangent + v.z * n;
}

// Transform a direction from world space to tangent space.
static __forceinline__ __device__ float3 worldToTangent(
    const float3& v,
    const float3& n,
    const float3& tangent,
    const float3& bitangent)
{
    return make_float3(dot(v, tangent), dot(v, bitangent), dot(v, n));
}

// Reflect a direction around a surface normal.
// Formula: r = d - 2(d.n)n
// d:      incoming direction (does NOT need to be normalized)
// n:      surface normal (MUST be normalized)
// returns: reflected direction (NOT normalized; same magnitude as d)
static __forceinline__ __device__ float3 reflect(const float3& d, const float3& n)
{
    return d - 2.0f * dot(d, n) * n;
}


// ------------------------------------------------------------------
// BRDF functions
// ------------------------------------------------------------------

// Schlick Fresnel approximation.
// cosTheta: dot(V, H) or dot(L, H)
// F0:       reflectance at normal incidence
static __forceinline__ __device__ float3 fresnelSchlick(float cosTheta, float3 F0)
{
    float fc = powf(1.0f - cosTheta, 5.0f);
    return F0 + (1.0f - F0) * fc;
}

// GGX Normal Distribution Function (Trowbridge-Reitz).
// alpha2: GGX alpha squared = (roughness^2)^2
// NdotH:  dot(N, H), clamped to [0,1]
static __forceinline__ __device__ float D_GGX(float NdotH, float alpha2)
{
    float d = NdotH * NdotH * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (M_PI * d * d + 1e-7f);
}

// Height-correlated Smith G2 masking-shadowing (Heitz 2014).
// Returned value already incorporates the 1/(4*NdotL*NdotV) Cook-Torrance denominator,
// so the full specular weight is simply: F * G2_combined.
// Formula: G2 / (4 * NdotL * NdotV)
//        = 0.5 / ( NdotV*sqrt(alpha2 + (1-alpha2)*NdotL^2) + NdotL*sqrt(alpha2 + (1-alpha2)*NdotV^2) )
static __forceinline__ __device__ float G2_SmithCombined(
    float NdotL, float NdotV, float alpha2)
{
    float gl = NdotV * sqrtf(alpha2 + (1.0f - alpha2) * NdotL * NdotL);
    float gv = NdotL * sqrtf(alpha2 + (1.0f - alpha2) * NdotV * NdotV);
    return 0.5f / (gl + gv + 1e-7f);
}

// ------------------------------------------------------------------
// VNDF Sampling (Dupuy & Benyoub 2023 — Spherical Cap method)
//
// Samples the GGX distribution of visible normals.
// All vectors in tangent space (Z = up = shading normal).
//
// wi:       incident (view) direction in tangent space — must have wi.z > 0
// alpha:    isotropic GGX roughness (perceptual roughness^2)
// u1, u2:   uniform random numbers in [0, 1)
// returns:  half vector wm in tangent space (normalized)
// ------------------------------------------------------------------
static __device__ __forceinline__ float3 SampleVNDF_Hemisphere(
    float3 wi, float u1, float u2)
{
    // Sample a spherical cap in (-wi.z, 1]
    float phi = 2.0f * M_PI * u1;
    float z = fmaf(1.0f - u2, 1.0f + wi.z, -wi.z);
    float sinTheta = sqrtf(clamp(1.0f - z * z, 0.0f, 1.0f));
    float3 c = make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), z);
    // Half vector between incident direction and cap sample (returned unnormalized)
    return c + wi;
}

static __device__ __forceinline__ float3 SampleVNDF_GGX(
    float3 wi, float alpha, float u1, float u2)
{
    // Step 1: warp wi to the unit hemisphere configuration
    float3 wiStd = normalize(make_float3(wi.x * alpha, wi.y * alpha, wi.z));
    // Step 2: sample the hemisphere via spherical cap (Dupuy & Benyoub 2023)
    float3 wmStd = SampleVNDF_Hemisphere(wiStd, u1, u2);
    // Step 3: warp back to ellipsoid (GGX) configuration
    return normalize(make_float3(wmStd.x * alpha, wmStd.y * alpha, wmStd.z));
}


// ------------------------------------------------------------------
// Tone mapping and display
// ------------------------------------------------------------------

static __device__ __forceinline__ float3 acesTMO(const float3 color)
{
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return make_float3(
        clamp((color.x * (a * color.x + b)) / (color.x * (c * color.x + d) + e), 0.f, 1.f),
        clamp((color.y * (a * color.y + b)) / (color.y * (c * color.y + d) + e), 0.f, 1.f),
        clamp((color.z * (a * color.z + b)) / (color.z * (c * color.z + d) + e), 0.f, 1.f)
    );
}

static __device__ __forceinline__ float linearToSRGB(float x)
{
    return (x <= 0.0031308f) ? 12.92f * x : 1.055f * powf(x, 1.f / 2.4f) - 0.055f;
}

static __device__ __forceinline__ uchar4 convertToFinalSRGB(float3 color)
{
    float3 tm = acesTMO(color);
    return make_uchar4(
        (unsigned char)(clamp(linearToSRGB(tm.x), 0.f, 1.f) * 255.99f),
        (unsigned char)(clamp(linearToSRGB(tm.y), 0.f, 1.f) * 255.99f),
        (unsigned char)(clamp(linearToSRGB(tm.z), 0.f, 1.f) * 255.99f),
        255u
    );
}


// ------------------------------------------------------------------
// Environment map helpers
// ------------------------------------------------------------------
__device__ float2 dirToEnvmapUV(float3 dir)
{
    float phi = atan2f(dir.z, dir.x);
    float theta = acosf(clamp(dir.y, -1.f, 1.f));
    return make_float2((phi / (2.f * M_PI)) + 0.5f, theta / M_PI);
}

__device__ float3 sampleEnvmap(cudaTextureObject_t tex, float3 dir)
{
    float2 uv = dirToEnvmapUV(dir);
    float4 val = tex2D<float4>(tex, uv.x, uv.y);
    return make_float3(val.x, val.y, val.z);
}


// ==================================================================================
// RAYGEN
// Accumulation: running average across frames using params.current_sample.
// ==================================================================================
extern "C" __global__ void __raygen__pathTracer()
{
    
    const uint3 idx = optixGetLaunchIndex();
    if (idx.x >= params.width || idx.y >= params.height) return;

    const unsigned int pixel_index = idx.y * params.width + idx.x;

    // Per-pixel sample loop
    // Accumulation across frames is handled below via the running average.
    float3 frame_color = make_float3(0.f, 0.f, 0.f);

    for (int s = 0; s < params.samples_per_pixel; ++s){        
        // Unique seed: XOR of pixel coords, frame seed, sample index
        unsigned int seed = params.random_seed ^ (idx.x * 73856093u) ^ (idx.y * 19349663u) ^ ((unsigned int)s * 83492791u);

        // Primary ray (pinhole camera, subpixel jitter for AA)
        float u = ((float)idx.x + rnd(seed)) / (float)params.width;
        float v = ((float)idx.y + rnd(seed)) / (float)params.height;

        float3 ray_origin = params.camera.origin;
        float3 ray_dir = normalize(
            params.camera.lower_left_corner
            + u * params.camera.horizontal
            + v * params.camera.vertical
            - params.camera.origin
        );

        // Path state
        float3 throughput = make_float3(1.f, 1.f, 1.f);
        float3 radiance = make_float3(0.f, 0.f, 0.f);

        // Iterative bounce loop
        for (int bounce = 0; bounce <= params.max_bounce_depth; ++bounce){
            // Initialize payload for this trace call.
            // throughput and seed are passed IN to the CH shader.
            RadiancePRD prd = {};
            prd.throughput = throughput;
            prd.ray_type = 0;  // Set the ray type for this path
            prd.done = 0u;

            traceRadiance(
                params.traversable,
                ray_origin,
                ray_dir,
                EPS,
                1e16f,
                prd
            );

            // Accumulate emissive / environment light 
            // prd.emitted is Le at this surface (or sky radiance from miss).
            radiance = radiance + (throughput * prd.emitted);

            // prd.radiance is reserved for NEE direct light (added here when implemented)
            //radiance = radiance + prd.radiance;  // uncomment when NEE is in place

            //  Terminate if miss shader or surface flagged path as done 
            if (prd.done)
                break;

            // Update throughput with BRDF weight written by CH shader 
            // CH writes the new (attenuated) throughput back into prd.throughput.
            throughput = prd.throughput;

            // Russian Roulette path termination 
            // Skip RR for the first rr_start_depth bounces to avoid bias
            // on direct and first-indirect lighting.
            if (bounce >= params.rr_start_depth){
                float q = fmaxf(0.05f, 1.0f - luminance(throughput));
                if (rnd(seed) < q)
                    break;
                throughput = throughput * (1.0f / (1.0f - q));
            }

            // Safety: terminate paths with negligible throughput (numerical stability)
            if (luminance(throughput) < 1e-6f)
                break;

            // Advance ray to next bounce
            ray_origin = prd.next_origin;
            ray_dir = prd.next_direction;
        }

        frame_color = frame_color + radiance;
    }

    // Average over samples_per_pixel within this launch
    frame_color = frame_color * (1.0f / (float)params.samples_per_pixel);

    // Progressive accumulation (running average across frames)
    // Formula: accum = accum + (new - accum) / (n + 1)
    // This is equivalent to a weighted average of all samples so far.
    float3 accumulated;
    if (params.current_sample == 0){
        accumulated = frame_color;
    }
    else{
        float3 prev = make_float3(params.accum_buffer[pixel_index].x, params.accum_buffer[pixel_index].y, params.accum_buffer[pixel_index].z);
        float  w = 1.0f / (float)(params.current_sample + 1);
        accumulated = make_float3(
            prev.x + (frame_color.x - prev.x) * w,
            prev.y + (frame_color.y - prev.y) * w,
            prev.z + (frame_color.z - prev.z) * w
        );
    }
    
    // Trace for denoiser guide buffers (albedo, normal) on the first sample only
    if (params.current_sample == 0){
        float u = (float)idx.x / (float)params.width;
        float v = (float)idx.y / (float)params.height;

        float3 ray_origin = params.camera.origin;
        float3 ray_dir = normalize(
            params.camera.lower_left_corner
            + u * params.camera.horizontal
            + v * params.camera.vertical
            - params.camera.origin
        );

        // Path state


        DenoiserGuidePRD prd = {};
		prd.ray_type = 1;  // Set the ray type for denoiser guide
        prd.albedo = make_float3(0.f, 0.f, 0.f);
        prd.normal = make_float3(0.f, 0.f, 0.f);

        traceDenoiserGuide(
            params.traversable,
            ray_origin,
            ray_dir,
            EPS,
            1e16f,
            prd
        );

		float3 albedo = prd.albedo;
		float3 normal = prd.normal;

        params.albedo_buffer[pixel_index] = make_float4(albedo.x, albedo.y, albedo.z, 1.0f);
        params.normal_buffer[pixel_index] = make_float4(normal.x, normal.y, normal.z, 1.0f);
    }

    params.accum_buffer[pixel_index] = make_float4(accumulated.x, accumulated.y, accumulated.z, 1.0f);

    // Tonemap and write to display buffer 
    //params.image[pixel_index] = convertToFinalSRGB(accumulated);
    //params.image[pixel_index] = convertToFinalSRGB(make_float3(params.albedo_buffer[pixel_index].x,params.albedo_buffer[pixel_index].y,params.albedo_buffer[pixel_index].z));
    //params.image[pixel_index] = convertToFinalSRGB(make_float3(params.normal_buffer[pixel_index].x, params.normal_buffer[pixel_index].y,params.normal_buffer[pixel_index].z));
}


// ==================================================================================
// MISS — environment map or constant sky
// ==================================================================================
extern "C" __global__ void __miss__envMap()
{
    unsigned int ray_type = optixGetPayload_0();
    if (ray_type == 1) {
        return;
    }

    RadiancePRD prd = loadMissRadiancePRD();

    float3 ray_dir = normalize(optixGetWorldRayDirection());
    float3 Le = make_float3(1.f);

    if (params.envmap.has_envmap) {
        Le = sampleEnvmap(params.envmap.texture, ray_dir) * params.envmap.scale * powf(2.f, params.envmap.exposure);
        Le = clamp(Le, 0.f, 10.f);  // prevent extreme fireflies
    }

    prd.emitted = Le;
    prd.radiance = make_float3(0.f);
    prd.done = 1u;

    storeMissRadiancePRD(prd);
}


// ==================================================================================
// CLOSESTHIT — Cook-Torrance microfacet BRDF
//
// Implements:
//   - GGX NDF + height-correlated Smith G2 + Schlick Fresnel
//   - VNDF importance sampling (Dupuy & Benyoub 2023)
//   - Metallic-roughness workflow:
//       F0      = lerp(0.04, base_color, metallic)
//       diffuse = (1 - F) * (1 - metallic) * base_color / pi
//       specular= F * G2_combined        [D and PDF cancel with VNDF sampling]
//   - Stochastic lobe selection (diffuse vs specular) based on luminance(F)
//   - Emissive support via emission constant or texture
//   - RNG state carried through payload for per-bounce uncorrelated samples
//
// Material sources (from MTL via host-side SBT filling):
//   base_color      <- map_Kd texture or Kd constant
//   roughness       <- map_Pr texture or sqrt(2/(Ns+2)) from Ns
//   metallic        <- map_Pm texture or Kd/Ks luminance heuristic
//   specular_color  <- map_Ks texture or Ks constant (for per-texel F0)
// ==================================================================================
extern "C" __global__ void __closesthit__cookTorrance()
{
    const HitGroupDataCookTorrance* sbt =
        (const HitGroupDataCookTorrance*)optixGetSbtDataPointer();

    unsigned int ray_type = optixGetPayload_0();
    if (ray_type == 1) {
		DenoiserGuidePRD prd = loadClosesthitDenoiserGuidePRD();
        const float3 ray_dir = normalize(optixGetWorldRayDirection());

        const float2 uv = getInterpolatedUV(sbt);
        prd.albedo = getBaseColor(sbt, uv);
        prd.normal = getInterpolatedNormal(sbt, ray_dir);  // world space, flipped
        storeClosesthitDenoiserGuidePRD(prd);
        return;
    }

    RadiancePRD prd = loadClosesthitRadiancePRD();

    // Reconstruct hit geometry 
    const float3 ray_dir = normalize(optixGetWorldRayDirection());
    const float3 hit_pos = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    const float2 uv = getInterpolatedUV(sbt);
    const float3 normal = getInterpolatedNormal(sbt, ray_dir);  // world space, flipped

    // View direction (pointing away from surface toward camera)
    const float3 V = -ray_dir;

    // Sample material parameters 
    float3 base_color = getBaseColor(sbt, uv);
    float  roughness = getRoughness(sbt, uv);
    //float  roughness = 0.8f;
    float  metallic = getMetallic(sbt, uv);

    // Clamp roughness to avoid degenerate GGX (perfectly smooth is handled but
    // produces a near-delta lobe; 0.02 minimum avoids numerical issues)
    roughness = clamp(roughness, 0.02f, 1.0f);

    // GGX alpha: perceptual roughness -> linear roughness -> alpha
    // alpha = roughness^2 (Disney remapping — perceptually linear slider)
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    // Compute F0 (reflectance at normal incidence) 
    // For dielectrics: F0 = 0.04 (covers most non-metals)
    // For metals:      F0 = base_color (colored metallic reflectance)
    // Per-texel specular override: if map_Ks is present, its luminance can
    // scale F0 from 0 to 0.08 (Allegorithmic PBR guide specular level trick).
    float3 dielectric_F0 = make_float3(0.04f, 0.04f, 0.04f);

    // If specular texture exists, use it to override per-texel F0 for dielectrics
    // (the specular workflow: luminance(Ks) remapped to [0, 0.08])
    if (sbt->specular_texture != 0) {
        float3 ks = getSpecularColor(sbt, uv);
        // Remap: luminance(Ks) in [0,1] -> F0 in [0, 0.08]
        dielectric_F0 = make_float3(luminance(ks) * 0.08f);
    }

    // Final F0: lerp between dielectric value and base_color by metallic
    float3 F0 = lerp3(dielectric_F0, base_color, metallic);

    // Emissive 
    prd.emitted = getEmissive(sbt, uv);

    const uint3 launch_idx = optixGetLaunchIndex();
    const unsigned int pixel = launch_idx.y * optixGetLaunchDimensions().x + launch_idx.x;

    // Per-bounce seed: unique per pixel + frame + depth + hit position
    unsigned int seed1 = params.random_seed
        ^ (optixGetLaunchIndex().x * 73856093u)
        ^ (optixGetLaunchIndex().y * 19349663u)
        ^ (params.current_sample * 83492791u)
        ^ __float_as_uint(hit_pos.x + hit_pos.y);

    unsigned int seed2 = params.random_seed
        ^ (optixGetLaunchIndex().x * 73856093u)
        ^ (optixGetLaunchIndex().y * 19349663u)
        ^ (params.current_sample * 83492791u)
        ^ __float_as_uint(hit_pos.x / hit_pos.y);

    unsigned int seed3 = params.random_seed
        ^ (optixGetLaunchIndex().x * 73856093u)
        ^ (optixGetLaunchIndex().y * 19349663u)
        ^ (params.current_sample * 83492791u)
        ^ __float_as_uint(hit_pos.x * hit_pos.y);

    float u1 = rnd(seed1);
    float u2 = rnd(seed2);
    float u3 = rnd(seed3);  // for lobe selection

    // Build tangent frame 
    float3 tangent, bitangent;
    buildONB(normal, tangent, bitangent);

    // View direction in tangent space
    float3 V_local = worldToTangent(V, normal, tangent, bitangent);

    // Safety: if V is below the surface (can happen with normal mapping or
    // backfacing geometry), flip it to avoid NaN in VNDF sampling
    if (V_local.z < 1e-4f)
        V_local.z = 1e-4f;

    // Stochastic lobe selection: specular or diffuse 
    // Estimate Fresnel at the view angle (using NdotV as approximation for VdotH)
    float NdotV = clamp(dot(normal, V), 0.f, 1.f);
    float3 F_approx = fresnelSchlick(NdotV, F0);
    float  p_specular = clamp(luminance(F_approx), 0.1f, 0.9f);
    // Ensure some minimum diffuse contribution for dielectrics
    p_specular = metallic > 0.9f ? 1.0f : p_specular;

    float3 scatter_dir_world;
    float3 brdf_weight;

    if (u3 < p_specular)
    {
        // Specular lobe: VNDF-sampled GGX reflection 

        // Sample half vector in tangent space via VNDF
        float3 Wm_local = SampleVNDF_GGX(V_local, alpha, u1, u2);

        // Reflect view direction around half vector to get scattered direction
        float3 L_local = reflect(-V_local, Wm_local);

        // Discard below-surface samples (can happen at high roughness)
        if (L_local.z <= 0.f) {
            prd.done = 1u;
            storeClosesthitRadiancePRD(prd);
            return;
        }

        // Evaluate BRDF terms
        float NdotL = clamp(L_local.z, 0.f, 1.f);  // in tangent space: L.z = dot(N,L)
        float NdotH = clamp(Wm_local.z, 0.f, 1.f);
        float VdotH = clamp(dot(V_local, Wm_local), 0.f, 1.f);

        float3 F = fresnelSchlick(VdotH, F0);
        float  G2 = G2_SmithCombined(NdotL, V_local.z, alpha2);
        // D_GGX(NdotH, alpha2) * NdotL appears in the numerator,
        // but with VNDF sampling the PDF = D_GGX * G1 * VdotH / NdotV,
        // and the weight simplifies to: F * G2_combined * VdotH * NdotL / (G1 * NdotV * ...)
        // Using the height-correlated G2_combined (which folds in the 4*NdotL*NdotV denom):
        //   weight = F * G2 * VdotH / NdotH  (D and most G terms cancel)
        // Reference: Heitz 2014, eq. 15 + VNDF importance sampling
        brdf_weight = F * (G2 * VdotH * NdotL / (NdotH + 1e-7f));

        // Divide by lobe selection probability
        brdf_weight = brdf_weight / p_specular;

        scatter_dir_world = tangentToWorld(L_local, normal, tangent, bitangent);
    }
    else
    {
        // Diffuse lobe: cosine-weighted hemisphere sampling 
        // Lambertian BRDF: f = base_color / pi
        // PDF of cosine sampling: pdf = NdotL / pi
        // Weight: f * NdotL / pdf = base_color   (pi and NdotL cancel)

        // Sample cosine-weighted direction (Malley's method)
        float r = sqrtf(u1);
        float phi = 2.0f * M_PI * u2;
        float3 L_local = make_float3(r * cosf(phi), r * sinf(phi),
            sqrtf(fmaxf(0.f, 1.f - r * r)));

        // Energy conservation: diffuse only carries the (1-F) complement
        float3 F_at_normal = fresnelSchlick(clamp(L_local.z, 0.f, 1.f), F0);
        float3 kD = (1.0f - F_at_normal) * (1.0f - metallic);
        brdf_weight = kD * base_color;

        // Divide by lobe selection probability
        brdf_weight = brdf_weight / (1.0f - p_specular);

        scatter_dir_world = tangentToWorld(L_local, normal, tangent, bitangent);
    }

    // Fill payload 
    // Multiply current throughput by the BRDF weight for this bounce.
    prd.throughput = prd.throughput * brdf_weight;
    prd.next_origin = hit_pos + EPS * normal;
    prd.next_direction = normalize(scatter_dir_world);
    prd.radiance = make_float3(0.f);  // no NEE yet
    prd.is_specular = 0u;               // false — diffuse/glossy, MIS applies
    prd.done = 0u;

    storeClosesthitRadiancePRD(prd);
}

// ==================================================================================
// ANYHIT - Opacity map alpha test 
// ==================================================================================
extern "C" __global__ void __anyhit__opacity()
{
    const HitGroupDataCookTorrance* sbt =
        (const HitGroupDataCookTorrance*)optixGetSbtDataPointer();

    const float2 uv = getInterpolatedUV(sbt);

    const float3 ray_dir = normalize(optixGetWorldRayDirection());
    const float3 hit_pos = optixGetWorldRayOrigin()
        + optixGetRayTmax() * ray_dir;
    unsigned int seed = params.random_seed
        ^ (optixGetLaunchIndex().x * 73856093u)
        ^ (optixGetLaunchIndex().y * 19349663u)
        ^ (params.current_sample * 83492791u)
        ^ __float_as_uint(hit_pos.x + hit_pos.y);
    float rand_val = rnd(seed);

    // Sample alpha
	float alpha = luminance(getAlpha(sbt, uv));  // Assuming alpha is stored in RGB channels as a grayscale value

    if(alpha < rand_val)
    {
        optixIgnoreIntersection();
    }
}

// ==================================================================================
// CLOSESTHIT — Perfect dielectric glass
//
// Implements:
//   - Schlick Fresnel to stochastically choose reflect vs. refract
//   - Snell's law refraction in vector form
//   - Total Internal Reflection (TIR) handled automatically via discriminant check
//   - Front/back face detection for correct IOR ratio (air->glass vs glass->air)
//   - Tint support (colored glass)
//   - Emissive support (glowing glass — unusual but valid)
//
// is_specular is set to 1 — raygen will skip NEE MIS for this bounce.
// ==================================================================================
extern "C" __global__ void __closesthit__glass()
{
    const HitGroupDataGlass* sbt =
        (const HitGroupDataGlass*)optixGetSbtDataPointer();

    unsigned int ray_type = optixGetPayload_0();
    if (ray_type == 1) {
        DenoiserGuidePRD prd = loadClosesthitDenoiserGuidePRD();
        const float3 ray_dir = normalize(optixGetWorldRayDirection());

        const float2 uv = getInterpolatedUV(sbt);
        prd.albedo = getTint(sbt, uv);
        prd.normal = getInterpolatedNormal(sbt, ray_dir);  // world space, flipped
        storeClosesthitDenoiserGuidePRD(prd);
        return;
    }

    RadiancePRD prd = loadClosesthitRadiancePRD();

    // Hit geometry 
    const float3 ray_dir = normalize(optixGetWorldRayDirection());
    const float3 hit_pos = optixGetWorldRayOrigin()
        + optixGetRayTmax() * ray_dir;

    const float2 uv = getInterpolatedUV(sbt);

    // Raw interpolated normal 
    const int    prim_idx = optixGetPrimitiveIndex();
    const uint3  tri = sbt->indices[prim_idx];
    const float2 bary = optixGetTriangleBarycentrics();
    const float  b0 = 1.f - bary.x - bary.y;
    float3 n_raw = normalize(
        b0 * sbt->vertices[tri.x].normal +
        bary.x * sbt->vertices[tri.y].normal +
        bary.y * sbt->vertices[tri.z].normal
    );
    n_raw = normalize(optixTransformNormalFromObjectToWorldSpace(n_raw));

    // Determine ray side (front face = hitting outside) 
    bool   front_face = dot(ray_dir, n_raw) < 0.f;
    float3 outward_normal = front_face ? n_raw : -n_raw;

    // IOR ratio: n1/n2
    // entering glass: n1=1.0 (air), n2=ior  -> ratio = 1/ior
    // leaving  glass: n1=ior, n2=1.0 (air)  -> ratio = ior
    float ior_ratio = front_face ? (1.0f / sbt->refraction_index) : sbt->refraction_index;

    // Emissive 
    //prd.emitted = getEmissive(sbt, uv);

    // Fresnel (Schlick)
    float cos_theta = fminf(dot(-ray_dir, outward_normal), 1.0f);
    float r0 = (1.0f - ior_ratio) / (1.0f + ior_ratio);
    r0 = r0 * r0;
    float reflectance = r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);

    // Total Internal Reflection check
    float sin_theta_sq = 1.0f - cos_theta * cos_theta;
    float discriminant = 1.0f - ior_ratio * ior_ratio * sin_theta_sq;
    bool  can_refract = discriminant > 0.0f;

    // Stochastic reflect / refract decision 
    // Reconstruct RNG from payload seed (same scheme as Cook-Torrance)
    const uint3 launch_idx = optixGetLaunchIndex();
    const unsigned int pixel = launch_idx.y * optixGetLaunchDimensions().x + launch_idx.x;

    float3 scattered;
    float3 next_origin;

    // Init random seed
    unsigned int seed = params.random_seed
        ^ (optixGetLaunchIndex().x * 73856093u)
        ^ (optixGetLaunchIndex().y * 19349663u)
        ^ (params.current_sample * 83492791u)
        ^ __float_as_uint(hit_pos.x + hit_pos.y);

    if (!can_refract || reflectance > rnd(seed))
    {
        // Reflection 
        scattered = reflect(ray_dir, outward_normal);     // r = d - 2(d.n)n
        next_origin = hit_pos + EPS * outward_normal;     // offset to same side
    }
    else
    {
        // Refraction (Snell's law in vector form) 
        float3 r_perp = ior_ratio * (ray_dir + cos_theta * outward_normal);
        float3 r_paral = -sqrtf(fabsf(discriminant)) * outward_normal;
        scattered = normalize(r_perp + r_paral);
        next_origin = hit_pos - EPS * outward_normal;  // offset to other side
    }

    // Fill payload 
    prd.throughput = prd.throughput * getTint(sbt, uv);
    prd.next_origin = next_origin;
    prd.next_direction = normalize(scattered);
    prd.radiance = make_float3(0.f);
    prd.is_specular = 1u;  // skip NEE MIS for this delta event
    prd.done = 0u;

    storeClosesthitRadiancePRD(prd);
}
