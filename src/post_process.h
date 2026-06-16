#pragma once

#include <cuda_runtime.h>

void launchTonemapKernel(
    float4* d_denoised,
    uchar4* d_image,
    int      width,
    int      height,
    cudaStream_t stream
);