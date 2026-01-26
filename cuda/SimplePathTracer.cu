#include <optix.h>
#include <optix_device.h>

struct Params
{
    uchar4* image;
    int     width;
    int     height;
};

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();

    if (idx.x >= params.width || idx.y >= params.height)
        return;

    const int i = idx.y * params.width + idx.x;

    params.image[i] = make_uchar4(255, 0, 0, 255);
}

extern "C" __global__ void __miss__ms()
{
    /* never reached, but required for pipeline correctness */
}
