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
};

struct RayGenData
{
};


struct MissData
{
    //float4 bg_color;
};


struct HitGroupData
{
    //float3  emission_color;
    float3  diffuse_color;
    //float4* vertices;
};