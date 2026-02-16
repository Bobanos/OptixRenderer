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
    float3  light_position;
    float3  light_color;
};