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
#define EPS        1e-4f    // ray offset to avoid self-intersection


// ---------------------------------------------------------
// TEA (Tiny Encryption Algorithm)
// Used exclusively to generate a strong, uncorrelated seed 
// for each pixel based on its index and the current frame.
// ---------------------------------------------------------
__device__ __forceinline__ void tea_seed(uint32_t val0, uint32_t val1, uint32_t& v0, uint32_t& v1) {
    v0 = val0;
    v1 = val1;
    uint32_t sum = 0;

    // 4 iterations are perfectly sufficient for generating a random seed
    for (int i = 0; i < 4; i++) {
        sum += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0 >> 5) + 0x7e95761e);
    }
}

// ---------------------------------------------------------
// PCG32 (Permuted Congruential Generator)
// The primary, fast RNG used during the ray tracing loop.
// ---------------------------------------------------------
__device__ __forceinline__ uint32_t pcg32_random(PCG32& rng) {
    uint64_t oldstate = rng.state;
    // Advance internal state
    rng.state = oldstate * 6364136223846793005ULL + (rng.inc | 1);

    // Calculate output function (XSH RR), uses old state for max instruction-level parallelism
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);

    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Generates a random float in the range [0.0, 1.0)
__device__ __forceinline__ float rnd(PCG32& rng) {
    uint32_t res = pcg32_random(rng);
    return (float)res * (1.0f / 4294967296.0f);
}

// ---------------------------------------------------------
// Pointer Packing
// ---------------------------------------------------------

// Splits a 64-bit pointer into two 32-bit payload registers
__device__ __forceinline__ void pack_pointer(void* ptr, uint32_t& p0, uint32_t& p1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    p0 = static_cast<uint32_t>(uptr >> 32);
    p1 = static_cast<uint32_t>(uptr & 0xFFFFFFFF);
}

// Reconstructs the pointer directly from OptiX payload registers
template <typename T>
__device__ __forceinline__ T* get_prd() {
    const uint32_t p0 = optixGetPayload_0();
    const uint32_t p1 = optixGetPayload_1();
    const uint64_t uptr = (static_cast<uint64_t>(p0) << 32) | p1;
    return reinterpret_cast<T*>(uptr);
}

// ------------------------------------------------------------------
// traceRadiance - fires a ray and returns filled RadiancePRD
// ------------------------------------------------------------------
static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    RadiancePRD& prd)
{
    // 1. Pack the 64-bit memory address of the prd reference into two 32-bit registers
    const uint64_t uptr = reinterpret_cast<uint64_t>(&prd);
    unsigned int u0 = static_cast<unsigned int>(uptr >> 32);
    unsigned int u1 = static_cast<unsigned int>(uptr & 0xFFFFFFFF);

    // 2. Perform the hardware BVH traversal
    optixTraverse(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.f,                     // ray time
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        RayType::RADIANCE,       // SBT offset (RAY_TYPE_RADIANCE)
        RayType::COUNT,          // SBT stride (Will be 2 for Radiance + Shadow types, when added)
        RayType::RADIANCE,       // miss SBT index (RAY_TYPE_RADIANCE)
        u0, u1                   // Only pass the 2 pointer registers
    );

    // Reorder execution for warp coherence
    optixReorder();

    // Invoke the Hit or Miss programs
    optixInvoke(u0, u1);
}

static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    float                  alpha_threshold ) {
    unsigned int is_visible = 0; // Assume occluded
    unsigned int alpha_payload = __float_as_uint(alpha_threshold);
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.f,                     // ray time
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, // Disable ClosestHit entirely for maximum speed
        RayType::OCCLUSION,   // sbtOffset (1)  // SBT offset 
        RayType::COUNT,     // sbtStride (2)  // SBT stride (Will be 2 for Radiance + Shadow types, when added)
        RayType::OCCLUSION,    // missSBTIndex (1)// miss SBT index 
        is_visible, alpha_payload
    );
    return (is_visible != 0);
}

// ------------------------------------------------------------------
// Geometry helpers - shared by both hit programs
// ------------------------------------------------------------------

static __forceinline__ __device__ float3 getGeometricNormal(const HitGroupDataCommon* sbt) {
    const int prim_idx = optixGetPrimitiveIndex();
    const uint3 tri = sbt->indices[prim_idx];
    const float3 v0 = sbt->vertices[tri.x].position;
    const float3 v1 = sbt->vertices[tri.y].position;
    const float3 v2 = sbt->vertices[tri.z].position;
    const float3 edge1 = v1 - v0;
    const float3 edge2 = v2 - v0;

    // The cross product of the edges gives the raw perpendicular object-space normal.
    // The order (edge1 x edge2) matters! It defines the "front" face 
    // based on clockwise vs counter-clockwise vertex winding order.
    float3 n_object = cross(edge1, edge2);
    return normalize(optixTransformNormalFromObjectToWorldSpace(n_object));
}

// Interpolate shading normal from vertex buffer using barycentrics.
// Flips toward the incoming ray (two-sided shading).
static __forceinline__ __device__ float3 getInterpolatedNormal( const HitGroupDataCommon* sbt){
    const int    prim_idx = optixGetPrimitiveIndex();
    const uint3  tri = sbt->indices[prim_idx];
    const float2 bary = optixGetTriangleBarycentrics();
    const float  b0 = 1.f - bary.x - bary.y;

    // Linearly interpolate raw object-space normals
    float3 n = b0     * sbt->vertices[tri.x].normal +
               bary.x * sbt->vertices[tri.y].normal +
               bary.y * sbt->vertices[tri.z].normal;

    n = normalize(optixTransformNormalFromObjectToWorldSpace(n));

    // Fallback for degenerate shading normals
    if (dot(n, n) < EPS) {
        n = getGeometricNormal(sbt);
    }
    return n;
}

// Interpolate UV coordinates across the triangle.
static __forceinline__ __device__ float2 getInterpolatedUV( const HitGroupDataCommon* sbt){
    const int    prim_idx = optixGetPrimitiveIndex();
    const uint3  tri = sbt->indices[prim_idx];
    const float2 bary = optixGetTriangleBarycentrics();
    const float  b0 = 1.f - bary.x - bary.y;

    return make_float2(
        b0 * sbt->vertices[tri.x].uv.x + bary.x * sbt->vertices[tri.y].uv.x + bary.y * sbt->vertices[tri.z].uv.x,
        b0 * sbt->vertices[tri.x].uv.y + bary.x * sbt->vertices[tri.y].uv.y + bary.y * sbt->vertices[tri.z].uv.y
    );
}

// Sample emissive - texture takes priority over constant.
static __forceinline__ __device__ float3 getEmissive( const HitGroupDataCommon* sbt, float2 uv){
    if (sbt->emission_texture != 0) {
        float4 t = tex2D<float4>(sbt->emission_texture, uv.x, uv.y);
        return make_float3(t.x, t.y, t.z);
    }
    return sbt->emission * params.light_intensity;
}

// Sample alpha.
static __forceinline__ __device__ float3 getAlpha( const HitGroupDataCookTorrance* sbt, float2 uv){
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
static __forceinline__ __device__ float3 getBaseColor( const HitGroupDataCookTorrance* sbt, float2 uv){
    if (sbt->albedo_texture != 0) {
        float4 t = tex2D<float4>(sbt->albedo_texture, uv.x, uv.y);
        return make_float3(t.x, t.y, t.z);
    }
    return sbt->base_color;
}

static __forceinline__ __device__ float3 getTint( const HitGroupDataGlass* sbt, float2 uv)
{
    if (sbt->tint_texture != 0) {
        float4 t = tex2D<float4>(sbt->tint_texture, uv.x, uv.y);
        return make_float3(t.x, t.y, t.z);
    }
    return sbt->tint;
}

// Returns scalar roughness. Texture (R channel) takes priority.
static __forceinline__ __device__ float getRoughness( const HitGroupDataCookTorrance* sbt, float2 uv)
{
    if (sbt->roughness_texture != 0) {
        float4 t = tex2D<float4>(sbt->roughness_texture, uv.x, uv.y);
        return t.x;  // R channel = roughness (map_Pr is grayscale)
    }
    return sbt->roughness;
}

// Returns scalar metallic. Texture (R channel) takes priority.
static __forceinline__ __device__ float getMetallic( const HitGroupDataCookTorrance* sbt, float2 uv)
{
    if (sbt->metallic_texture != 0) {
        float4 t = tex2D<float4>(sbt->metallic_texture, uv.x, uv.y);
        return t.x;  // R channel = metallic (map_Pm is grayscale)
    }
    return sbt->metallic;
}

// Specular color for per-texel F0 override (map_Ks).
// Used when computing F0 for materials with a specular texture.
static __forceinline__ __device__ float3 getSpecularColor( const HitGroupDataCookTorrance* sbt, float2 uv)
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

// Build orthonormal tangent frame around a shading normal.
static __forceinline__ __device__ void buildONB( const float3& normal, float3& tangent, float3& bitangent){
    float sign = copysignf(1.0f, normal.z);
    float a = -1.0f / (sign + normal.z);
    float b = normal.x * normal.y * a;
    tangent = make_float3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    bitangent = make_float3(b, sign + normal.y * normal.y * a, -normal.y);
}

// Transform a direction from tangent space to world space.
static __forceinline__ __device__ float3 tangentToWorld( const float3& v, const float3& n, 
    const float3& tangent, const float3& bitangent){
    return v.x * tangent + v.y * bitangent + v.z * n;
}

// Transform a direction from world space to tangent space.
static __forceinline__ __device__ float3 worldToTangent(const float3& v, const float3& n, 
    const float3& tangent, const float3& bitangent){
    return make_float3(dot(v, tangent), dot(v, bitangent), dot(v, n));
}

static __forceinline__ __device__ float power_heuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + 1e-16f); // EPS prevents division by zero
}

// ------------------------------------------------------------------
// BRDF functions 
// Equations mainly from Eric Heitz's 2018 paper
// Sampling the GGX Distribution of Visible Normals
// ------------------------------------------------------------------

// Reflect a direction around a surface normal.
// Formula: r = d - 2(d.n)n
// d:      incoming direction (does NOT need to be normalized)
// n:      surface normal (MUST be normalized)
// returns: reflected direction (NOT normalized; same magnitude as d)
static __forceinline__ __device__ float3 reflect(const float3& d, const float3& n){
    return d - 2.0f * dot(d, n) * n;
}

// Equation 2: The Smith Masking A(V), which is typically denoted as Lambda
static __forceinline__ __device__ float Lambda_GGX(float3 V, float alpha_x, float alpha_y) {
    if (V.z <= 0.0f) return 0.0f;
    float term = (alpha_x * V.x) * (alpha_x * V.x) + (alpha_y * V.y) * (alpha_y * V.y);
    term = term / (V.z * V.z);
    return 0.5f * (-1.0f + sqrtf(1.0f + term));
}

// G1 Masking function (Equation 2)
static __forceinline__ __device__ float G1_GGX(float3 V, float alpha_x, float alpha_y) {
    return 1.0f / (1.0f + Lambda_GGX(V, alpha_x, alpha_y));
}

// G2 Shadowing-Masking function (Height-correlated Smith)
static __forceinline__ __device__ float G2_GGX(float3 Ve, float3 Li, float alpha_x, float alpha_y) {
    return 1.0f / (1.0f + Lambda_GGX(Ve, alpha_x, alpha_y) + Lambda_GGX(Li, alpha_x, alpha_y));
}

// Equation 1: GGX Normal Distribution Function (NDF)
static __forceinline__ __device__ float D_GGX(float3 Ne, float alpha_x, float alpha_y) {
    float a2 = (Ne.x * Ne.x) / (alpha_x * alpha_x) +
        (Ne.y * Ne.y) / (alpha_y * alpha_y) +
        (Ne.z * Ne.z);
    return 1.0f / (M_PI * alpha_x * alpha_y * a2 * a2);
}

// Schlick Fresnel approximation.
// Input:
// cosTheta: dot(V, H) or dot(L, H)
// F0:       reflectance at normal incidence
// Equation:
// F0 + (1.0 - F0)(1.0 - VoH)^5
static __forceinline__ __device__ float3 fresnelSchlick(float cosTheta, float3 F0){
    return F0 + (1.0f - F0) * powf(1.0f - cosTheta, 5.0f);
}

// Input Ve: view direction
// Input alpha_x, alpha_y: roughness parameters
// Input U1, U2: uniform random numbers
// Output Ne: normal sampled with PDF D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
static __device__ __forceinline__ float3 sampleGGXVNDF( float3 Ve, float alpha_x, float alpha_y, float U1, float U2) {
    // Section 3.2: transforming the view direction to the hemisphere configuration
    float3 Vh = normalize(make_float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0f ?
        make_float3(-Vh.y, Vh.x, 0.0f) * rsqrtf(lensq) :
        make_float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);

    // Section 4.2: parameterization of the projected area
    float r = sqrtf(U1);
    float phi = 2.0f * M_PI * U2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;

    // Section 4.3: reprojection onto hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // Section 3.4: transforming the normal back to the ellipsoid configuration
    float3 Ne = normalize(make_float3(alpha_x * Nh.x, alpha_y * Nh.y, fmaxf(0.0f, Nh.z)));

    return Ne;
}

struct BSDFSample {
    float3 Li;     // The newly sampled light direction for the next bounce
    float3 weight; // The BRDF multiplier to apply to path throughput
    float  pdf;    // Exact PDF (in case you need it for MIS)
    bool   valid;  // True if the bounce stays above the hemisphere
};

static __device__ __forceinline__ BSDFSample evaluateCookTorrance( float3 Ve, float alpha_x, float alpha_y,
    float U1, float U2, float3 F0){
    BSDFSample result;
    result.valid = false;

    // 1. Sample the visible microfacet normal (Listing 1)
    float3 Ne = sampleGGXVNDF(Ve, alpha_x, alpha_y, U1, U2);

    // 2. Compute the sampled light direction (Equation 16)
    // Graphics reflect() expects ray pointing IN. Ve points OUT. So we pass -Ve.
    result.Li = reflect(-Ve, Ne);

    // Ensure the resulting ray isn't pointing back inside the surface
    if (result.Li.z <= 0.0f || Ve.z <= 0.0f) {
        return result;
    }

    // 3. Compute exact PDF (Equation 17)
    // PDF(Li) = (G1(Ve) * D(Ne)) / (4 * Ve.z) as derived from the cancellation
    float g1_Ve = G1_GGX(Ve, alpha_x, alpha_y);
    float d_Ne = D_GGX(Ne, alpha_x, alpha_y);
    result.pdf = (g1_Ve * d_Ne) / (4.0f * Ve.z);

    // 4. Compute Fresnel F(Ve, Ne) using Schlick's approximation
    float vDotH = fmaxf(0.0f, dot(Ve, Ne));
    float3 F = F0 + (make_float3(1.0f) - F0) * powf(1.0f - vDotH, 5.0f);

    // 5. Compute the final Throughput Weight (Equation 19)
    // Weight = F * G2 / G1
    float g2 = G2_GGX(Ve, result.Li, alpha_x, alpha_y);
    result.weight = F * (g2 / g1_Ve);

    result.valid = true;
    return result;
}

// Evaluates a Lambertian diffuse bounce
inline __device__ BSDFSample evaluateLambertian(float U1, float U2, float3 base_color, float metallic) {
    BSDFSample result;

    // 1. Cosine-weighted hemisphere sampling
    float r = sqrtf(U1);
    float theta = 2.0f * M_PI * U2;

    // Tangent space direction
    result.Li = make_float3(r * cosf(theta), r * sinf(theta), sqrtf(fmaxf(0.0f, 1.0f - U1)));

    // 2. Compute PDF ( cosine / PI )
    result.pdf = result.Li.z / M_PI;

    // 3. Compute Weight
    // The BRDF is base_color / PI. The Monte Carlo estimator is: (BRDF * cos) / PDF.
    // ((base_color / PI) * Li.z) / (Li.z / PI) = base_color

    // Metals do not have diffuse reflection (photons are absorbed instantly).
    // We scale the diffuse albedo down based on the metallic value.
    result.weight = base_color * (1.0f - metallic);

    result.valid = (result.Li.z > 0.0f);
    return result;
}

// ------------------------------------------------------------------
// NEE functions
// ------------------------------------------------------------------

// Explicitly evaluates the combined BRDF (Specular + Diffuse) for a specific light direction.
// We need this to calculate NEE weights, as we aren't generating a random ray here.
static __forceinline__ __device__ void evaluateBRDF_NEE(
    float3 Ve, float3 Ld, float alpha_x, float alpha_y,
    float3 F0, float3 base_color, float metallic,
    float3& brdf_val)
{
    brdf_val = make_float3(0.0f);
    if (Ve.z <= 0.0f || Ld.z <= 0.0f) return;

    // Specular Lobe
    float3 H = normalize(Ve + Ld);
    if (H.z > 0.0f) {
        float VdotH = fmaxf(0.0f, dot(Ve, H));
        float D = D_GGX(H, alpha_x, alpha_y);
        float G2 = G2_GGX(Ve, Ld, alpha_x, alpha_y);
        float3 F = F0 + (make_float3(1.0f) - F0) * powf(1.0f - VdotH, 5.0f);
        brdf_val = brdf_val + (D * G2 * F) / (4.0f * Ve.z * Ld.z);
    }

    // Diffuse Lobe
    float3 f_diff = base_color * (1.0f - metallic) / M_PI;
    brdf_val = brdf_val + f_diff;
}

// Computes the PDF of our stochastic lobe selection choosing a specific direction
static __forceinline__ __device__ float computeBRDFPdf(float3 Ve, float3 Ld, float alpha_x, float alpha_y, float p_specular) {
    float pdf_spec = 0.0f;
    float3 H = normalize(Ve + Ld);
    if (H.z > 0.0f && Ve.z > 0.0f && Ld.z > 0.0f) {
        float D = D_GGX(H, alpha_x, alpha_y);
        float G1 = G1_GGX(Ve, alpha_x, alpha_y);
        pdf_spec = (G1 * D) / (4.0f * Ve.z); // The exact VNDF PDF
    }
    float pdf_diff = fmaxf(0.0f, Ld.z / M_PI);

    return (pdf_spec * p_specular) + (pdf_diff * (1.0f - p_specular));
}

// Samples a random point on an emissive triangle
static __forceinline__ __device__ void sampleLight(
    const Params& params, float3 hit_pos,
    float u1, float u2, float u3,
    float3& light_dir, float& light_dist, float3& light_radiance, float& light_pdf)
{
    light_pdf = 0.0f;
    light_radiance = make_float3(0.f);

    if (params.num_emissive_triangles == 0) return;

    // A. Binary search the CDF to pick a light based on its power (luminance * area)
    int left = 0;
    int right = params.num_emissive_triangles - 1;
    int light_idx = right;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (params.emissive_triangles[mid].cdf >= u1) {
            light_idx = mid;
            right = mid - 1;
        }
        else {
            left = mid + 1;
        }
    }

    EmissiveTriangle tri = params.emissive_triangles[light_idx];

    // B. Uniformly sample a point on the chosen triangle
    float sq = sqrtf(u2);
    float u = 1.0f - sq;
    float v = u3 * sq;
    float3 p = tri.v0 * u + tri.v1 * v + tri.v2 * (1.0f - u - v);

    // Calculate normal and distance
    float3 edge1 = tri.v1 - tri.v0;
    float3 edge2 = tri.v2 - tri.v0;
    float3 n = normalize(cross(edge1, edge2));

    float3 dir = p - hit_pos;
    float dist2 = dot(dir, dir);
    light_dist = sqrtf(dist2);
    light_dir = dir / light_dist;

    float cos_theta = dot(n, -light_dir);

    if (cos_theta > 0.0f) {
        light_radiance = tri.emission;

        // C. Calculate the exact PDF of picking this point
        // PDF = P(picking triangle) * P(picking point on triangle) * Jacobian(Area to Solid Angle)
        // Because P(triangle) is proportional to (luminance * area), the Area elegantly cancels out!
        float pdf_area = luminance(tri.emission) / params.total_emissive_weight;

        // Convert area PDF to solid angle PDF using the distance and angle
        light_pdf = pdf_area * (dist2 / cos_theta);
    }
}

// ------------------------------------------------------------------
// Environment map helpers
// ------------------------------------------------------------------
__device__ float2 dirToEnvmapUV(float3 dir){
    float phi = atan2f(dir.z, dir.x);
    float theta = acosf(clamp(dir.y, -1.f, 1.f));
    return make_float2((phi / (2.f * M_PI)) + 0.5f, theta / M_PI);
}

__device__ float3 sampleEnvmap(cudaTextureObject_t tex, float3 dir){
    float2 uv = dirToEnvmapUV(dir);
    float4 val = tex2D<float4>(tex, uv.x, uv.y);
    return make_float3(val.x, val.y, val.z);
}


// ==================================================================================
// RAYGEN
// Accumulation: running average across frames using params.current_sample.
// ==================================================================================
extern "C" __global__ void __raygen__pathTracer(){
    
    const uint3 idx = optixGetLaunchIndex();
    if (idx.x >= params.width || idx.y >= params.height) return;

    const unsigned int pixel_index = idx.y * params.width + idx.x;

    uint32_t seed0, seed1;
    tea_seed(pixel_index, params.current_frame, seed0, seed1);

    PCG32 rng;  // Random number generator
    rng.state = ((uint64_t)seed0 << 32) | seed1;
    rng.inc = pixel_index;
    pcg32_random(rng);

    // Per-pixel sample loop
    // Accumulation across frames is handled below via the running average.
    float3 frame_color = make_float3(0.f, 0.f, 0.f);
    float3 frame_albedo = make_float3(0.f, 0.f, 0.f);
    float3 frame_normal = make_float3(0.f, 0.f, 0.f);

    for (int s = 0; s < params.samples_per_pixel; ++s) {

        float u = ((float)idx.x + rnd(rng)) / (float)params.width;
        float v = ((float)idx.y + rnd(rng)) / (float)params.height;

        float3 ray_origin = params.camera.origin;
        float3 ray_dir = normalize(
            params.camera.lower_left_corner
            + u * params.camera.horizontal
            + v * params.camera.vertical
            - params.camera.origin
        );

        float3 throughput = make_float3(1.f);
        float3 radiance = make_float3(0.f);

        // Variables to carry context from the previous bounce
        float prev_btdf_pdf = 0.0f;
        unsigned int prev_is_specular = 1u; // Primary ray acts like a perfect specular bounce

        for (int bounce = 0; bounce <= params.max_bounce_depth; ++bounce) {

            RadiancePRD prd = {};
            prd.throughput = throughput;
            prd.done = 0u;
            prd.rng = &rng;
            prd.alpha_threshold = rnd(rng);

            traceRadiance(
                params.traversable,
                ray_origin, ray_dir,
                EPS, 1e16f, prd
            );

            // 1. Accumulate Light Source Emission (Implicit/BRDF hitting a light)
            if (luminance(prd.emitted) > 0.0f) {
                // If it's the primary ray OR we bounced off a mirror, NEE cannot sample it. 
                // So we give it 100% weight.
                if (bounce == 0 || prev_is_specular) {
                    radiance = radiance + (throughput * prd.emitted);
                }
                // Otherwise, MIS: Balance between BRDF probability and Light sampling probability
                else {
                    float mis_weight = power_heuristic(prev_btdf_pdf, prd.hit_light_pdf);
                    //float mis_weight = 1.0f;
                    radiance = radiance + (throughput * prd.emitted * mis_weight);
                }
            }

            // 2. Accumulate Direct Lighting (Next Event Estimation)
            // prd.radiance was strictly calculated and shadow-tested in the Closest Hit program
            radiance = radiance + (throughput * prd.radiance);

            if (prd.done) break;

            // Denoiser layers
            if (bounce == 0) {
                frame_albedo = frame_albedo + prd.albedo;
                frame_normal = frame_normal + prd.normal;
            }

            // Update state for the NEXT loop iteration
            throughput = prd.throughput;
            prev_btdf_pdf = prd.btdf_pdf;     // Cache the PDF of the ray we just shot
            prev_is_specular = prd.is_specular;

            // Russian Roulette
            if (bounce >= params.rr_start_depth) {
                float q = fmaxf(0.05f, 1.0f - luminance(throughput));
                if (rnd(rng) < q) break;
                throughput = throughput * (1.0f / (1.0f - q));
            }

            if (luminance(throughput) < 1e-6f) break;

            ray_origin = prd.next_origin;
            ray_dir = prd.next_direction;
        }

        frame_color = frame_color + radiance;
    }

    float inv_spp = 1.0f / (float)params.samples_per_pixel;

    // Average over samples_per_pixel within this launch
    frame_color  = frame_color * inv_spp;
    frame_albedo = frame_albedo * inv_spp;
    if(length_squared(frame_normal) != 0.f)
        frame_normal = normalize(frame_normal);

    // Progressive accumulation (running average across frames)
    // Formula: accum = accum + (new - accum) / (n + 1)
    // This is equivalent to a weighted average of all samples so far.
    float3 accumulated    = make_float3(0.f);
    float3 current_albedo = make_float3(0.f);
    float3 current_normal = make_float3(0.f);

    if (params.current_sample == 0){
        accumulated = frame_color;
        current_albedo = frame_albedo;
        current_normal = frame_normal;
    }
    else{
        float frame_weight = 1.0f / (float)(params.current_sample + 1.f);
        // color
        float3 prev = make_float3(params.accum_buffer[pixel_index].x, params.accum_buffer[pixel_index].y, params.accum_buffer[pixel_index].z);
        accumulated = lerp3(prev, frame_color, frame_weight);

        // albedo
        float3 old_albedo = make_float3(params.albedo_buffer[pixel_index].x, params.albedo_buffer[pixel_index].y, params.albedo_buffer[pixel_index].z);
        current_albedo = lerp3(old_albedo, frame_albedo, frame_weight);

        // normal
        float3 old_normal = make_float3(params.normal_buffer[pixel_index].x, params.normal_buffer[pixel_index].y, params.normal_buffer[pixel_index].z);
        current_normal = normalize(lerp3(old_normal, frame_normal, frame_weight));
    }
    
    params.accum_buffer[pixel_index] = make_float4(accumulated.x, accumulated.y, accumulated.z, 1.0f);
    params.albedo_buffer[pixel_index] = make_float4(current_albedo.x, current_albedo.y, current_albedo.z, 1.0f);
    params.normal_buffer[pixel_index] = make_float4(current_normal.x, current_normal.y, current_normal.z, 1.0f);

    //params.accum_buffer[pixel_index] = make_float4(current_albedo.x, current_albedo.y, current_albedo.z, 1.0f);
    //params.accum_buffer[pixel_index] = make_float4(accumulated.x, accumulated.y, accumulated.z, 1.0f);
}


// ==================================================================================
// MISS - environment map or constant sky
// ==================================================================================
extern "C" __global__ void __miss__envMap()
{
    RadiancePRD* prd = get_prd<RadiancePRD>();

    float3 ray_dir = normalize(optixGetWorldRayDirection());
    float3 Le = params.background_color;

    if (params.envmap.has_envmap) {
        Le = sampleEnvmap(params.envmap.texture, ray_dir) * params.envmap.scale * powf(2.f, params.envmap.exposure);
        Le = clamp(Le, 0.f, 10.f);  // prevent extreme fireflies
    }

    prd->emitted = Le;
    prd->radiance = make_float3(0.f);
    prd->done = 1u;
}

extern "C" __global__ void __miss__occlusion()
{
    // Flip the payload register to 1 (visible)
    optixSetPayload_0(1);
}

// ==================================================================================
// CLOSESTHIT - Cook-Torrance microfacet BRDF
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
    // 1. Get Payload and Hit Info
    const HitGroupDataCookTorrance* sbt = (const HitGroupDataCookTorrance*)optixGetSbtDataPointer();
    RadiancePRD* prd = get_prd<RadiancePRD>();

    const float3 ray_dir = normalize(optixGetWorldRayDirection());
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float  hit_t = optixGetRayTmax();
    const float3 hit_pos = ray_orig + hit_t * ray_dir;

    const float2 uv = getInterpolatedUV(sbt);
    float3 normal_s = getInterpolatedNormal(sbt);
    float3 normal_g = getGeometricNormal(sbt);

    // Normal Flipping
    if (dot(normal_g, ray_dir) > 0.0f) {
        normal_g = -normal_g;
        normal_s = -normal_s;
    }

    float3 V_world = -ray_dir;
    if (dot(V_world, normal_g) <= 0.0f) {
        prd->done = true;
        return;
    }

    // Build the Tangent Frame 
    float3 tangent, bitangent;
    buildONB(normal_s, tangent, bitangent);
    float3 Ve = worldToTangent(V_world, normal_s, tangent, bitangent);

    // Fetch Material Parameters 
    float  roughness = getRoughness(sbt, uv);
    float3 base_color = getBaseColor(sbt, uv);
    float  metallic = getMetallic(sbt, uv);
    float  alpha_x = clamp(roughness * roughness, 0.001f, 1.f);
    float  alpha_y = alpha_x;

    prd->emitted = getEmissive(sbt, uv);

    // Determine PDF of hitting this exact point directly via Light Sampling (for RayGen MIS)
    if (luminance(prd->emitted) > 0.0f) {
        float pdf_area = luminance(prd->emitted) / params.total_emissive_weight;
        float cos_theta_light = fmaxf(1e-6f, dot(normal_g, V_world));
        prd->hit_light_pdf = pdf_area * (hit_t * hit_t) / cos_theta_light;
    }
    else {
        prd->hit_light_pdf = 0.0f;
    }

    float3 F0_dielectric = make_float3(0.04f);
    float3 F0 = F0_dielectric * (1.0f - metallic) + base_color * metallic;

    float VdotN = fmaxf(0.0f, Ve.z);
    float3 F_guess = F0 + (make_float3(1.0f) - F0) * powf(1.0f - VdotN, 5.0f);
    float p_specular = clamp((F_guess.x + F_guess.y + F_guess.z) / 3.0f, 0.1f, 0.9f);

    prd->radiance = make_float3(0.0f); // Reset NEE container

    // ------------------------------------------------------------------
    //  NEXT EVENT ESTIMATION (Direct Light Sampling)
    // ------------------------------------------------------------------
    float3 light_dir, light_radiance;
    float light_dist, light_pdf;

    // Pass PRNG state safely
    float l_u1 = rnd(*prd->rng), l_u2 = rnd(*prd->rng), l_u3 = rnd(*prd->rng);

    sampleLight(params, hit_pos, l_u1, l_u2, l_u3, light_dir, light_dist, light_radiance, light_pdf);

    if (light_pdf > 0.0f) {
        float3 Ld_tangent = worldToTangent(light_dir, normal_s, tangent, bitangent);

        // Only evaluate if light is physically above the horizon
        if (Ld_tangent.z > 0.0f && dot(light_dir, normal_g) > 0.0f) {

            float3 brdf_val;
            evaluateBRDF_NEE(Ve, Ld_tangent, alpha_x, alpha_y, F0, base_color, metallic, brdf_val);

            // What is the probability we would have generated this ray blindly via the BRDF?
            float brdf_pdf = computeBRDFPdf(Ve, Ld_tangent, alpha_x, alpha_y, p_specular);

            if (brdf_pdf > 0.0f) {
                float mis_weight = power_heuristic(light_pdf, brdf_pdf);
                //float mis_weight = 1.0f;

                float shadow_tmax = light_dist * 0.999f;

                bool occluded = traceOcclusion(
                    params.traversable,
                    hit_pos + normal_g * EPS, // Origin (pushed off the floor)
                    light_dir,                // Direction
                    EPS,                      // tmin
                    shadow_tmax,              // tmax (stops right before the light)
                    rnd(*prd->rng)
                );


                //// Trace the shadow ray locally
                //bool occluded = traceOcclusion(
                //    params.traversable, hit_pos + normal_g * EPS, light_dir,
                //    0.0f, light_dist - EPS, rnd(*prd->rng)
                //);

                //occluded = false;

                if (occluded) {
                    // Note: Ld_tangent.z IS the cos(theta) term in the rendering equation!
                    prd->radiance = light_radiance * brdf_val * Ld_tangent.z * mis_weight / light_pdf;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    //  BRDF SCATTERING (Indirect Bounces)
    // ------------------------------------------------------------------
    float u1 = rnd(*prd->rng), u2 = rnd(*prd->rng), u3 = rnd(*prd->rng);
    BSDFSample sample;

    if (u3 < p_specular) {
        sample = evaluateCookTorrance(Ve, alpha_x, alpha_y, u1, u2, F0);
        if (sample.valid) {
            sample.weight = sample.weight / p_specular;
            prd->is_specular = (roughness < 0.05f) ? 1u : 0u;
        }
    }
    else {
        sample = evaluateLambertian(u1, u2, base_color, metallic);
        if (sample.valid) {
            sample.weight = sample.weight / (1.0f - p_specular);
            prd->is_specular = 0u;
        }
    }

    if (!sample.valid) {
        prd->done = true;
        return;
    }

    float3 L_world = normalize(tangentToWorld(sample.Li, normal_s, tangent, bitangent));

    if (dot(L_world, normal_g) <= 0.0f) {
        prd->done = true;
        return;
    }

    // Calculate exact PDF of the ray we just shot for MIS on the NEXT hit
    prd->btdf_pdf = computeBRDFPdf(Ve, sample.Li, alpha_x, alpha_y, p_specular);

    // Finalize Payload
    prd->throughput = prd->throughput * sample.weight;
    prd->next_origin = hit_pos + normal_g * EPS;
    prd->next_direction = L_world;
    prd->done = 0u;

    prd->albedo = base_color;
    prd->normal = normal_s;
}
// ==================================================================================
// ANYHIT - Opacity map alpha test 
// ==================================================================================
extern "C" __global__ void __anyhit__opacity()
{
    const HitGroupDataCookTorrance* sbt = (const HitGroupDataCookTorrance*)optixGetSbtDataPointer();

    RadiancePRD* prd = get_prd<RadiancePRD>();

    const float2 uv = getInterpolatedUV(sbt);
    // Sample alpha
	float alpha = luminance(getAlpha(sbt, uv));  // Assuming alpha is stored in RGB channels as a grayscale value

    if(alpha < prd->alpha_threshold)
    {
        optixIgnoreIntersection();
    }
}

extern "C" __global__ void __anyhit__occlusion()
{
    const HitGroupDataCookTorrance* sbt = (const HitGroupDataCookTorrance*)optixGetSbtDataPointer();

    const float2 uv = getInterpolatedUV(sbt);
    // Sample alpha
    float alpha = luminance(getAlpha(sbt, uv));  // Assuming alpha is stored in RGB channels as a grayscale value

    if (alpha < __uint_as_float(optixGetPayload_1()))
    {
        optixIgnoreIntersection();
    }
}

// ==================================================================================
// CLOSESTHIT - Perfect dielectric glass
//
// Implements:
//   - Schlick Fresnel to stochastically choose reflect vs. refract
//   - Snell's law refraction in vector form
//   - Total Internal Reflection (TIR) handled automatically via discriminant check
//   - Front/back face detection for correct IOR ratio (air->glass vs glass->air)
//   - Tint support (colored glass)
//   - Emissive support (glowing glass - unusual but valid)
//
// is_specular is set to 1 - raygen will skip NEE MIS for this bounce.
// ==================================================================================
extern "C" __global__ void __closesthit__glass()
{
    const HitGroupDataGlass* sbt =
        (const HitGroupDataGlass*)optixGetSbtDataPointer();

    RadiancePRD* prd = get_prd<RadiancePRD>();

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
    float3 n_raw = b0 *     sbt->vertices[tri.x].normal +
                   bary.x * sbt->vertices[tri.y].normal +
                   bary.y * sbt->vertices[tri.z].normal;
    n_raw = normalize(optixTransformNormalFromObjectToWorldSpace(n_raw));

    // Determine ray side (front face = hitting outside) 
    bool   front_face = dot(ray_dir, n_raw) < 0.f;
    float3 outward_normal = front_face ? n_raw : -n_raw;

    // IOR ratio: n1/n2
    // entering glass: n1=1.0 (air), n2=ior  -> ratio = 1/ior
    // leaving  glass: n1=ior, n2=1.0 (air)  -> ratio = ior
    float ior_ratio = front_face ? (1.0f / sbt->refraction_index) : sbt->refraction_index;

    // Emissive 
    prd->emitted = getEmissive(sbt, uv);

    // Fresnel (Schlick)
    float cos_theta = fminf(dot(-ray_dir, outward_normal), 1.0f);
    float r0 = (1.0f - ior_ratio) / (1.0f + ior_ratio);
    r0 = r0 * r0;
    float reflectance = r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);

    // Total Internal Reflection check
    float sin_theta_sq = 1.0f - cos_theta * cos_theta;
    float discriminant = 1.0f - ior_ratio * ior_ratio * sin_theta_sq;
    bool  can_refract = discriminant > 0.0f;

    float3 scattered;
    float3 next_origin;
    if (!can_refract || reflectance > rnd(*prd->rng))
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
    float3 tint = getTint(sbt, uv);
    prd->throughput = prd->throughput * tint;
    prd->next_origin = next_origin;
    prd->next_direction = normalize(scattered);
    prd->radiance = make_float3(0.f);
    prd->is_specular = 1u;  // skip NEE MIS for this delta event
    prd->done = 0u;
    prd->albedo = tint;
    prd->normal = outward_normal;
}
