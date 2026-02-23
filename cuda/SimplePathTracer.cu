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
    unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0;  // Payload for color
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
        p0, p1, p2, p3
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
    

    // fetch current triangle vertices
    float3 data[3];
    optixGetTriangleVertexData( optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(),
        optixGetRayTime(), data );

    // compute triangle normal
    data[1] = data[1] - data[0];
    data[2] = data[2] - data[0];
    float3 object_normal = make_float3(
        data[1].y*data[2].z - data[1].z*data[2].y,
        data[1].z*data[2].x - data[1].x*data[2].z,
        data[1].x*data[2].y - data[1].y*data[2].x );
    
    float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
    
    // Flip normal if facing away from ray
    if (dot(world_normal, ray_direction) > 0.0f) {
        world_normal = world_normal * -1.0f;;
    }
    
    // Accumulate lighting from all lights
    float3 total_diffuse = make_float3(0.0f, 0.0f, 0.0f);

    for (int light_idx = 0; light_idx < params.num_lights; light_idx++) {
        // Calculate lighting for this light
        float3 to_light = params.light_position[light_idx] - hit_point;
        float distance_to_light = length(to_light);
        float3 light_dir = normalize(to_light);
        
        // Trace shadow ray
        unsigned int shadow_hit = 0;
        optixTrace(
            params.traversable,
            hit_point + world_normal * 0.001f,
            light_dir,
            0.001f,
            distance_to_light - 0.001f,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            0,
            1,
            0,
            shadow_hit
        );
        
        float shadow = shadow_hit ? 1.0f : 0.0f;
        
        // Calculate contribution from this light
        float ndotl = fmaxf(0.0f, dot(world_normal, light_dir));
        float attenuation = 1.0f / (1.0f + 0.1f * distance_to_light);
        
        total_diffuse = total_diffuse + params.light_color[light_idx] * ndotl * attenuation * shadow;
    }
    



    const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();
    //const int    prim_idx        = optixGetPrimitiveIndex();
    //const int    vert_idx_offset = prim_idx*3;
    float3 base_color = sbt_data->diffuse_color;

    if (length_squared(base_color) < 0.001f) {
        base_color = make_float3(1.0f, 0.0f, 1.0f);  // Magenta = error indicator
    }


    float3 ambient = make_float3(0.1f, 0.1f, 0.1f);
    float3 color = ambient + base_color * total_diffuse;
    
    // Clamp to valid range
    color = clamp(color, 0.0f, 1.0f);

    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}


extern "C" __global__ void __closesthit__glass()
{
    // Get hit point
    const float t_hit = optixGetRayTmax();
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float3 hit_point = ray_origin + t_hit * ray_direction;
    
    // Fetch current triangle vertices
    float3 data[3];
    optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(),
        optixGetRayTime(), data);

    // Compute triangle normal
    data[1] = data[1] - data[0];
    data[2] = data[2] - data[0];
    float3 object_normal = make_float3(
        data[1].y*data[2].z - data[1].z*data[2].y,
        data[1].z*data[2].x - data[1].x*data[2].z,
        data[1].x*data[2].y - data[1].y*data[2].x);
    
    float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
    
    // Determine if we're entering or exiting the material
    bool entering = dot(world_normal, ray_direction) < 0.0f;
    if (!entering) {
        world_normal = world_normal * -1.0f;
    }
    
    const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();
    float3 base_color = sbt_data->diffuse_color;
    float ior = sbt_data->refraction_index;

    // Get current recursion depth
    unsigned int current_depth = optixGetPayload_3();

    float3 color;

    // If we've reached max depth, just return the tinted color (no more recursion)
    if (current_depth >= 3) {  // Max 3 bounces for glass
        color = base_color * 0.8f;  // Slightly darker fallback
    }
    else {
        // Simple refraction/reflection mix
        float eta = entering ? (1.0f / ior) : ior;
        float cos_theta = fminf(dot(ray_direction * -1.0f, world_normal), 1.0f);
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    
        // Schlick's approximation for Fresnel
        float r0 = (1.0f - ior) / (1.0f + ior);
        r0 = r0 * r0;
        float reflectance = r0 + (1.0f - r0) * powf((1.0f - cos_theta), 5.0f);
    
        // Total internal reflection or reflection
        if (eta * sin_theta > 1.0f || reflectance > 0.5f) {
            // Reflection
            float3 reflected = ray_direction - 2.0f * dot(ray_direction, world_normal) * world_normal;
        
            // Trace reflection ray
            unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = current_depth + 1;
            optixTrace(
                params.traversable,
                hit_point + world_normal * 0.001f,
                normalize(reflected),
                0.001f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0,
                1,
                0,
                p0, p1, p2, p3
            );
        
            color = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
            color = color * base_color; // Tint by glass color
        }
        else {
            // Refraction
            float3 refracted_perp = eta * (ray_direction + cos_theta * world_normal);
            float3 refracted_parallel = -sqrtf(fabsf(1.0f - dot(refracted_perp, refracted_perp))) * world_normal;
            float3 refracted = refracted_perp + refracted_parallel;
        
            // Trace refraction ray
            unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = current_depth + 1;
            optixTrace(
                params.traversable,
                hit_point - world_normal * 0.001f, // Move slightly inside
                normalize(refracted),
                0.001f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0,
                1,
                0,
                p0, p1, p2, p3
            );
        
            color = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
            color = color * base_color; // Tint by glass color
        }
    }
    
    // Clamp to valid range
    color = clamp(color, 0.0f, 1.0f);

    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
    optixSetPayload_3(current_depth);
}