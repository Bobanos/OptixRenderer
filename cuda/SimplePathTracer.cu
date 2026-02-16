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

__device__ float3 operator*(const float3& v, const float3& t) {
    return make_float3(v.x * t.x, v.y * t.y, v.z * t.z);
}

__device__ float3 operator+(const float3& v, float t) {
    return make_float3(v.x + t, v.y + t, v.z + t);
}

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__device__ float length_squared(const float3& v) {
    return dot(v, v);
}

__device__ float3 normalize(const float3& v) {
    float len = length(v);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ float3 clamp(const float3& v, float min_val, float max_val) {
    return make_float3(
        fminf(fmaxf(v.x, min_val), max_val),
        fminf(fmaxf(v.y, min_val), max_val),
        fminf(fmaxf(v.z, min_val), max_val)
    );
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
    // Get hit point
    const float t_hit = optixGetRayTmax();
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float3 hit_point = ray_origin + t_hit * ray_direction;
    
    const float3 object_normal = make_float3(0, 1, 0);
    
    float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
    
    // Flip normal if facing away from ray
    if (dot(world_normal, ray_direction) > 0.0f) {
        world_normal = world_normal * -1.0f;;
    }
    
    // Calculate lighting
    float3 to_light = params.light_position - hit_point;
    float distance_to_light = length(to_light);
    float3 light_dir = normalize(to_light);
    
    // Trace shadow ray
    unsigned int shadow_hit = 0;
    optixTrace(
        params.traversable,
        hit_point + world_normal * 0.001f,  // Offset to avoid self-intersection
        light_dir,
        0.001f,                              // tmin
        distance_to_light - 0.001f,         // tmax (stop at light)
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        0,
        1,
        0,
        shadow_hit  // Payload: will be set to 1 if hit
    );
    
    float shadow = shadow_hit ? 1.0f : 0.0f;  // 1 = in shadow, 0 = lit
    
    // Calculate color
    float ndotl = fmaxf(0.0f, dot(world_normal, light_dir));
    float attenuation = 1.0f / (1.0f + 0.1f * distance_to_light);
    
    float3 base_color = make_float3(0.8f, 0.8f, 0.8f);
    float3 ambient = make_float3(0.1f, 0.1f, 0.1f);
    float3 diffuse = params.light_color * base_color * ndotl * attenuation * shadow;
    float3 color = ambient + diffuse;
    
    // Clamp to valid range
    color = clamp(color, 0.0f, 1.0f);

    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}