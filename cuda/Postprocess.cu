#include "post_process.h"
#include "float3_math.h"
#include <device_launch_parameters.h>

__global__ void tonemapKernel(
    const float4* __restrict__ hdr_input,
    uchar4* __restrict__ ldr_output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float4 hdr = hdr_input[idx];
    float3 color = make_float3(hdr.x, hdr.y, hdr.z);

    // ACES tonemapping
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    color.x = fmaxf(0.0f, fminf(1.0f, (color.x * (a * color.x + b)) / (color.x * (c * color.x + d) + e)));
    color.y = fmaxf(0.0f, fminf(1.0f, (color.y * (a * color.y + b)) / (color.y * (c * color.y + d) + e)));
    color.z = fmaxf(0.0f, fminf(1.0f, (color.z * (a * color.z + b)) / (color.z * (c * color.z + d) + e)));

    auto toSRGB = [](float val) {
        return (val <= 0.0031308f) ? 12.92f * val : 1.055f * powf(val, 1.f / 2.4f) - 0.055f;
        };

    ldr_output[idx] = make_uchar4(
        (unsigned char)(fmaxf(0.0f, fminf(1.0f, toSRGB(color.x))) * 255.99f),
        (unsigned char)(fmaxf(0.0f, fminf(1.0f, toSRGB(color.y))) * 255.99f),
        (unsigned char)(fmaxf(0.0f, fminf(1.0f, toSRGB(color.z))) * 255.99f),
        255u
    );
}

void launchTonemapKernel(
    float4* d_denoised,
    uchar4* d_image,
    int     width,
    int     height,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    tonemapKernel << <grid, block, 0, stream >> > (d_denoised, d_image, width, height);

    cudaError_t err = cudaGetLastError();
}