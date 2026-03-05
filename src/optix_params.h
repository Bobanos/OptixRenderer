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

struct Params {
    uchar4* image;
    int     width;
    int     height;
    Camera  camera;
    OptixTraversableHandle traversable;

    // Multiple lights
    float3  light_position[2];
    float3  light_color[2];
    int     num_lights;

    int max_recursion_depth;
};

struct RayGenData{};
struct MissData{};
struct HitGroupData
{
    ColoredVertex* vertices;
    uint3* indices;
    float   refraction_index;  // For glass material (1.0 = opaque, 1.5 = glass)
    cudaTextureObject_t albedo_texture; // 0 = no texture, use vertex color
};