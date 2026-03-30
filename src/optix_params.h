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
    float3 color;
    float2 uv;
};

// Light types: 0 = point light, 1 = directional light
struct Light {
    float3 position_or_direction;  // position for point light, direction for directional light
    float3 color;
    int    type;                   // 0 = point, 1 = directional
    int    _padding;               // for 16-byte alignment
};

struct Params {
    uchar4* image;
    int     width;
    int     height;
    Camera  camera;
    OptixTraversableHandle traversable;

    // Flexible light system
    Light   lights[4];
    int     num_lights;

    int max_recursion_depth;
};

struct RayGenData {};
struct MissData {};



struct HitGroupDataCommon
{
    ColoredVertex* vertices;
    uint3* indices;
};

struct HitGroupDataLambert : public HitGroupDataCommon
{
    float3 albedo;
    cudaTextureObject_t albedo_texture; // 0 = no texture, use vertex color
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
template struct VerifyHitGroupLayout<HitGroupDataLambert>;
template struct VerifyHitGroupLayout<HitGroupDataGlass>;