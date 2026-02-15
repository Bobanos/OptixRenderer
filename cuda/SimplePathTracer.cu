#include <optix.h>
#include <optix_device.h>

#include "optix_params.h"

extern "C" {
__constant__ Params params;
}

// Helper functions
__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(float t, const float3& v) {
    return make_float3(t * v.x, t * v.y, t * v.z);
}

__device__ float3 operator*(const float3& v, float t) {
    return make_float3(v.x * t, v.y * t, v.z * t);
}

__device__ float3 operator+(const float3& v, float t) {
    return make_float3(v.x + t, v.y + t, v.z + t);
}

__device__ float3 normalize(const float3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();

    if (idx.x >= params.width || idx.y >= params.height)
        return;

    const int i = idx.y * params.width + idx.x;

    // Calculate normalized pixel coordinates (0 to 1)
    float u = (float)idx.x / (float)(params.width - 1);
    float v = (float)idx.y / (float)(params.height - 1);

    // Generate ray from camera
    float3 ray_origin = params.camera.origin;
    float3 ray_direction = normalize(
        params.camera.lower_left_corner +
        u * params.camera.horizontal +
        v * params.camera.vertical -
        params.camera.origin
    );

    // Trace ray
    unsigned int p0 = 0, p1 = 0, p2 = 0;  // Payload for color
    optixTrace(
        params.traversable,
        ray_origin,
        ray_direction,
        0.001f,              // tmin
        1e16f,               // tmax
        0.0f,                // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset
        1,                   // SBT stride
        0,                   // missSBTIndex
        p0, p1, p2
    );

    // Unpack color from payload
    float r = __uint_as_float(p0);
    float g = __uint_as_float(p1);
    float b = __uint_as_float(p2);

    params.image[i] = make_uchar4(
        (unsigned char)(r * 255.99f),
        (unsigned char)(g * 255.99f),
        (unsigned char)(b * 255.99f),
        255
    );
}

extern "C" __global__ void __miss__ms()
{
    // Sky blue gradient
    const float3 ray_dir = optixGetWorldRayDirection();
    float t = 0.5f * (ray_dir.y + 1.0f);
    float3 color = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + 
                   t * make_float3(0.5f, 0.7f, 1.0f);

    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

extern "C" __global__ void __closesthit__ch()
{
    // Get triangle normal
    const float3 ray_dir = optixGetWorldRayDirection();
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const int prim_idx = optixGetPrimitiveIndex();
    
    // Calculate geometric normal
    const float3 normal = optixGetWorldRayDirection();  // Placeholder - will calculate properly
    
    // Simple diffuse shading based on normal
    float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(
        make_float3(0, 0, 1)  // Simplified - just use a fixed normal for now
    ));
    
    // Color based on normal (for visualization)
    float3 color = world_normal * 0.5f + 0.5f;  // Map -1..1 to 0..1
    
    // Simple lighting
    float3 light_dir = normalize(make_float3(1.0f, 1.0f, 1.0f));
    float ndotl = fmaxf(0.0f, dot(world_normal, light_dir));
    color = color * (0.3f + 0.7f * ndotl);  // Ambient + diffuse

    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}