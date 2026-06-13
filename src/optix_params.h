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

    //int     max_bounce_depth;
    int     samples_per_pixel;
    int     current_sample;
    unsigned int random_seed;

    EnvironmentMap envmap;
};

//===================================================================================================

constexpr unsigned int RAY_TYPE_COUNT = 1;

constexpr OptixPayloadTypeID PAYLOAD_TYPE_RADIANCE = OPTIX_PAYLOAD_TYPE_ID_0;

const unsigned int radiancePayloadSemantics[3] = {
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
};
//const unsigned int radiancePayloadSemantics[18] =
//{
//    // RadiancePRD::attenuation
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
//    // RadiancePRD::seed
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
//    // RadiancePRD::depth
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
//
//    // RadiancePRD::emitted
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
//    // RadiancePRD::radiance
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
//    // RadiancePRD::origin
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
//    // RadiancePRD::direction
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
//    // RadiancePRD::done
//    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
//};

struct RadiancePRD {
    float3 radiance;
};

//struct RadiancePRD
//{
//    // these are produced by the caller, passed into trace, consumed/modified by CH and MS and consumed again by the caller after trace returned.
//    float3       attenuation;
//    unsigned int seed;
//    int          depth;
//
//    // these are produced by CH and MS, and consumed by the caller after trace returned.
//    float3       emitted;
//    float3       radiance;
//    float3       origin;
//    float3       direction;
//    int          done;
//};

//===================================================================================================

struct RayGenData {};

struct MissData {};

struct HitGroupDataCommon
{
    ColoredVertex* vertices;
    uint3* indices; 
	float3 albedo;  // Base color (if no texture)
	float3 emission;  // Emissive color (can be > 1 for bright materials)
	cudaTextureObject_t emission_texture; // 0 = no texture, use emission color
};

struct HitGroupDataCookTorrance : public HitGroupDataCommon
{
    cudaTextureObject_t albedo_texture; // 0 = no texture, use vertex color
    float roughness;
	float metallic;
    float3 base_color;
};

struct HitGroupDataGlass : public HitGroupDataCommon
{
    float refraction_index;
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