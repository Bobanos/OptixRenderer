#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>

// This file defines the data structures that will be shared between the host and device code.

struct Camera {
    float3 origin;
    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;
};

struct ColoredVertex {
    float3 position;
    float3 normal;
    float2 uv;
};


// Environment map parameters
struct EnvironmentMap {
    cudaTextureObject_t texture; // HDR environment map
    cudaTextureObject_t cdf_marginal_v; // 1D texture: CDF over rows
    cudaTextureObject_t cdf_conditional_u; // 2D texture: per-row CDF over columns
	int width;
	int height;
    float               scale;   // intensity multiplier (can be > 1)
    float               exposure; // additional exposure in stops
    bool                has_envmap;
};

struct Params {
    uchar4* image;
    float3* accum_buffer; 
    int     width;
    int     height;
    Camera  camera;
    OptixTraversableHandle traversable;

    float   light_intensity;

    int     max_bounce_depth;
    int     rr_start_depth;
    int     samples_per_pixel;
    int     current_sample;
    unsigned int random_seed;

    EnvironmentMap envmap;
};

//===================================================================================================

constexpr unsigned int RAY_TYPE_COUNT = 1;


// ------------------------------------------------------------------
// RadiancePRD payload layout
//
// Register map (18 registers total):
//   p0..p2   : throughput      (float3, RW by caller and CH)
//   p3       : seed            (uint,   RW by caller and CH)
//   p4       : done            (uint,   W by CH and MS, R by caller)
//   p5..p7   : emitted         (float3, W by CH and MS, R by caller)
//   p8..p10  : radiance        (float3, W by CH and MS, R by caller - unused for now, kept for NEE)
//   p11..p13 : next_origin     (float3, W by CH, R by caller)
//   p14..p16 : next_direction  (float3, W by CH, R by caller)
//   p17      : is_specular     (uint,   W by CH, R by caller - for MIS later)
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// RadiancePRD struct - holds path state communicated via payload registers
// ------------------------------------------------------------------
struct RadiancePRD
{
    // Caller writes before trace, CH reads and writes back
    float3       throughput;    // current path weight (starts at 1,1,1)
    unsigned int seed;          // PCG32 state (lower 32 bits; inc is derived from pixel)

    // CH / MS write, caller reads after trace
    unsigned int done;          // 1 = path terminates (miss or absorbed)
    float3       emitted;       // Le at this surface (for emissive geometry)
    float3       radiance;      // direct light contribution (unused until NEE)
    float3       next_origin;   // scattered ray origin
    float3       next_direction;// scattered ray direction
    unsigned int is_specular;   // 1 = delta BRDF event (glass) - skip NEE MIS later
    float3       albedo;        // base color (for denoiser)
    float3       normal;        // shading normal (for denoiser)
};

//===================================================================================================

struct RayGenData {};

struct MissData {};

struct HitGroupDataCommon
{
    ColoredVertex* vertices;
    uint3* indices; 
	//float3 albedo;  // Base color (if no texture)
	float3 emission;  // Emissive color (can be > 1 for bright materials)
	cudaTextureObject_t emission_texture; // 0 = no texture, use emission color
};

struct HitGroupDataCookTorrance : public HitGroupDataCommon
{
    // Base color / albedo
    float3              base_color;         // constant base color (Kd for dielectric, Ks for metal)
    cudaTextureObject_t albedo_texture;     // map_Kd — 0 if not present

    // Specular (used to compute F0 per-texel when map_Ks is present)
    float3              specular_color;     // constant Ks
    cudaTextureObject_t specular_texture;   // map_Ks — 0 if not present

    // Roughness — scalar fallback + optional texture (R channel)
    float               roughness;          // scalar roughness in [0,1]
    cudaTextureObject_t roughness_texture;  // map_Pr — 0 if not present

    // Metallic — scalar fallback + optional texture (R channel)
    float               metallic;           // scalar metallic in [0,1]
    cudaTextureObject_t metallic_texture;   // map_Pm — 0 if not present

    cudaTextureObject_t alpha_texture;   // map_d — 0 if not present
};

struct HitGroupDataGlass : public HitGroupDataCommon
{
    float refraction_index; // IOR (e.g. 1.5 for standard glass)
    float3 tint;            // color tint — (1,1,1) for clear glass
    cudaTextureObject_t tint_texture; // map_Kd — 0 if not present
};

// Compile-time verification template for derived hit group types
template <typename DerivedType>
struct VerifyHitGroupLayout {
    static_assert(std::is_base_of_v<HitGroupDataCommon, DerivedType>,
        "DerivedType must inherit from HitGroupDataCommon");

    static_assert(offsetof(DerivedType, vertices) == offsetof(HitGroupDataCommon, vertices),
        "vertices offset mismatch in derived type");

    static_assert(offsetof(DerivedType, indices) == offsetof(HitGroupDataCommon, indices),
        "indices offset mismatch in derived type");

    // Verify derived members come after base members
    static_assert(sizeof(HitGroupDataCommon) <= sizeof(DerivedType),
        "DerivedType must be at least as large as HitGroupDataCommon");
};

// Instantiate verification for each hit group type
template struct VerifyHitGroupLayout<HitGroupDataCookTorrance>;
template struct VerifyHitGroupLayout<HitGroupDataGlass>;