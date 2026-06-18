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


struct EmissiveTriangle {
    float3 v0, v1, v2;   // world-space vertices
    float3 emission;     // constant emissive radiance (Ke), in world/linear units
    float  area;         // triangle area in world units, precomputed at build time
    float  cdf;          // cumulative (area * luminance) fraction, normalized to [0,1]
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
    float               total_weight; // sum of luminance*sin(theta) over all texels,
    // computed on host alongside the CDF. Used to balance NEE strategy selection (envmap vs
    // light triangles) proportional to how much each source actually contributes.
};

struct Params {
    uchar4* image;
    float4* accum_buffer; 
    float4* albedo_buffer;  // For denoiser guide layer
    float4* normal_buffer;  // For denoiser guide layer
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
    float3 background_color;


    EmissiveTriangle* emissive_triangles;
    int                num_emissive_triangles;
    float              total_emissive_weight;  // sum of area*luminance across all lights,
};

//===================================================================================================

// ------------------------------------------------------------------
// RadiancePRD payload layout
//
// Register map (24 registers total):
//   p0      : ray_type         (uint32, R by CH/MS) - 0 = radiance, 1 = shadow/occlusion
//   p1..p3  : throughput       (float3, RW)
//   p4       : done            (uint,   W by CH and MS, R by caller)
//   p5..p7   : emitted         (float3, W by CH and MS, R by caller)
//   p8..p10  : radiance        (float3, W by CH and MS, R by caller - unused for now, kept for NEE)
//   p11..p13 : next_origin     (float3, W by CH, R by caller)
//   p14..p16 : next_direction  (float3, W by CH, R by caller)
//   p17      : is_specular     (uint,   W by CH, R by caller - for MIS later)
//   p18..p20 : albedo          (float3, W by CH, R by caller - for denoiser guide)
//   p21..p23 : normal          (float3, W by CH, R by caller - for denoiser guide)
//   p24      : brdf_pdf        (float,  W by CH, R by caller - solid-angle PDF of the
//                                        sampled scatter direction, used for MIS when
//                                        the NEXT bounce implicitly hits a light)
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// RadiancePRD struct - holds path state communicated via payload registers
// ------------------------------------------------------------------
struct RadiancePRD
{
    unsigned int ray_type;
    // Caller writes before trace, CH reads and writes back
    float3       throughput;    // current path weight (starts at 1,1,1)

    // CH / MS write, caller reads after trace
    unsigned int done;          // 1 = path terminates (miss or absorbed)
    float3       emitted;       // Le at this surface (for emissive geometry)
    float3       radiance;      // direct light contribution (NEE result, added by CH)
    float3       next_origin;   // scattered ray origin
    float3       next_direction;// scattered ray direction
    unsigned int is_specular;   // 1 = delta BRDF event (glass) - skip NEE MIS later
    float3       albedo;        // Base color for diffuse materials, used for denoiser guide
    float3       normal;        // Surface normal at hit point, used for denoiser guide
    float        brdf_pdf;      // solid-angle PDF of the sampled scatter direction (for MIS)
};

//===================================================================================================

struct RayGenData {};

struct MissData {};

struct HitGroupDataCommon
{
    ColoredVertex* vertices;
    uint3* indices; 
	float3 emission;  // Emissive color (can be > 1 for bright materials)
	cudaTextureObject_t emission_texture; // 0 = no texture, use emission color
};

struct HitGroupDataCookTorrance : public HitGroupDataCommon
{
    // Base color / albedo
    float3              base_color;         // constant base color (Kd for dielectric, Ks for metal)
    cudaTextureObject_t albedo_texture;     // map_Kd - 0 if not present

    // Specular (used to compute F0 per-texel when map_Ks is present)
    float3              specular_color;     // constant Ks
    cudaTextureObject_t specular_texture;   // map_Ks - 0 if not present

    // Roughness - scalar fallback + optional texture (R channel)
    float               roughness;          // scalar roughness in [0,1]
    cudaTextureObject_t roughness_texture;  // map_Pr - 0 if not present

    // Metallic - scalar fallback + optional texture (R channel)
    float               metallic;           // scalar metallic in [0,1]
    cudaTextureObject_t metallic_texture;   // map_Pm - 0 if not present

    cudaTextureObject_t alpha_texture;   // map_d - 0 if not present
};

struct HitGroupDataGlass : public HitGroupDataCommon
{
    float refraction_index; // IOR (e.g. 1.5 for standard glass)
    float3 tint;            // color tint - (1,1,1) for clear glass
    cudaTextureObject_t tint_texture; // map_Kd - 0 if not present
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