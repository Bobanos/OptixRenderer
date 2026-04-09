#include <optix.h>
#include <optix_device.h>

#include "optix_params.h"
#include "float3_math.h"

extern "C" {
__constant__ Params params;
}

#define M_PI 3.14159265358979323846f

// ------------------------------------------------------------------
// Simple Linear Congruential Generator (LCG) for random numbers
// ------------------------------------------------------------------
__device__ unsigned int lcg_next(unsigned int& seed) {
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 22695477u;
    seed = LCG_A * seed + LCG_C;
    return seed;
}

__device__ float random_float(unsigned int& seed) {
    return (lcg_next(seed) & 0xFFFFFF) / 16777216.0f; // 24-bit precision
}

// Random direction in hemisphere (cosine-weighted for Lambertian)
__device__ float3 random_hemisphere_direction(const float3& normal, unsigned int& seed) {
    float u = random_float(seed);
    float v = random_float(seed);
    
    // Cosine-weighted hemisphere sampling
    float r = sqrtf(u);
    float theta = 2.0f * M_PI * v;
    
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(1.0f - u); // Cosine-weighted
    
    // Build orthonormal basis from normal
    float3 up = fabsf(normal.y) < 0.999f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 right = normalize(cross(normal, up));
    float3 forward = cross(right, normal);
    
    // Transform to world space
    return normalize(
        x * right +
        y * forward +
        z * normal
    );
}

// ------------------------------------------------------------------
// Helper: read per-vertex color from SBT data using barycentrics
// ------------------------------------------------------------------
__device__ float3 getVertexColor(const HitGroupDataCommon * sbt)
{
    const int   prim_idx = optixGetPrimitiveIndex();
    const uint3 tri      = sbt->indices[prim_idx];

    const float2 bary = optixGetTriangleBarycentrics();
    const float  b0   = 1.0f - bary.x - bary.y;

    return sbt->vertices[tri.x].color * b0
         + sbt->vertices[tri.y].color * bary.x
         + sbt->vertices[tri.z].color * bary.y;
}

// ------------------------------------------------------------------
// Helper: compute world-space normal for the hit triangle
// ------------------------------------------------------------------
__device__ float3 getGeometricNormal(const HitGroupDataCommon* sbt, const float3& ray_dir)
{
    const int   prim_idx = optixGetPrimitiveIndex();
    const uint3 tri      = sbt->indices[prim_idx];

    const float3 v0 = sbt->vertices[tri.x].position;
    const float3 v1 = sbt->vertices[tri.y].position;
    const float3 v2 = sbt->vertices[tri.z].position;

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 object_normal = make_float3(
        (e1.y*e2.z - e1.z*e2.y),
        (e1.z*e2.x - e1.x*e2.z),
        (e1.x*e2.y - e1.y*e2.x));

    float3 world_normal = normalize(
        optixTransformNormalFromObjectToWorldSpace(object_normal));

    // Flip if facing away from ray
    if (dot(world_normal, ray_dir) > 0.0f)
        world_normal = world_normal * -1.0f;

    return world_normal;
}

__device__ float3 getInterpolatedNormal(const HitGroupDataCommon* sbt, const float3& ray_dir)
{
    const int   prim_idx = optixGetPrimitiveIndex();
    const uint3 tri      = sbt->indices[prim_idx];
    const float2 bary    = optixGetTriangleBarycentrics();
    const float  b0      = 1.0f - bary.x - bary.y;
    const float  b1       = bary.x;
    const float  b2       = bary.y;

    float3 world_normal;
    //if (sbt->hasVertexNormals) {
        const float3 n0 = sbt->vertices[tri.x].normal;
        const float3 n1 = sbt->vertices[tri.y].normal;
        const float3 n2 = sbt->vertices[tri.z].normal;

        float3 object_normal = b0 * n0 + b1 * n1 + b2 * n2;

        world_normal = normalize(
            optixTransformNormalFromObjectToWorldSpace(object_normal));
    //} else {
    //     world_normal = getGeometricNormal(sbt, ray_dir);
    //}
    // Flip if facing away from ray
    if (dot(world_normal, ray_dir) > 0.0f)
        world_normal = world_normal * -1.0f;

    return world_normal;
}

// ------------------------------------------------------------------
// Helper: sample environment map using ray direction
// ------------------------------------------------------------------

// Convert a direction to equirectangular UV
__device__ float2 dirToEnvmapUV(float3 dir) {
    float phi   = atan2f(dir.z, dir.x);           // [-pi, pi]
    float theta = acosf(clamp(dir.y, -1.f, 1.f)); // [0, pi]
    float zoom = 1.0f; // Could be a parameter to control field of view
    return make_float2(
        (phi   / (2.f * M_PI)) * zoom + 0.5f,  // u: [0,1]
        theta /       M_PI                    // v: [0,1]
    );
}

__device__ float3 sampleEnvmap(cudaTextureObject_t envmap, float3 dir) {
    float2 uv = dirToEnvmapUV(dir);
    float4 val = tex2D<float4>(envmap, uv.x, uv.y);
    return make_float3(val.x, val.y, val.z);
}

// ------------------------------------------------------------------
// Helper: get albedo - samples texture if present, else vertex color
// ------------------------------------------------------------------
__device__ float3 getAlbedo(const HitGroupDataLambert* sbt)
{
    const int   prim_idx = optixGetPrimitiveIndex();
    const uint3 tri      = sbt->indices[prim_idx];
    const float2 bary    = optixGetTriangleBarycentrics();
    const float  b0      = 1.0f - bary.x - bary.y;

    if (sbt->albedo_texture != 0) {
        // Interpolate UV coordinates using barycentrics
        const float2 uv0 = sbt->vertices[tri.x].uv;
        const float2 uv1 = sbt->vertices[tri.y].uv;
        const float2 uv2 = sbt->vertices[tri.z].uv;
        float u = uv0.x * b0 + uv1.x * bary.x + uv2.x * bary.y;
        float v = uv0.y * b0 + uv1.y * bary.x + uv2.y * bary.y;

        v = 1.0f - v; // flip V: OBJ origin is bottom-left, stb_image is top-left

        // tex2D returns float4 in [0,1] because of cudaReadModeNormalizedFloat
        float4 t = tex2D<float4>(sbt->albedo_texture, u, v);
        return make_float3(t.x, t.y, t.z);
    }

    // No texture - interpolate vertex color
    return sbt->vertices[tri.x].color * b0
         + sbt->vertices[tri.y].color * bary.x
         + sbt->vertices[tri.z].color * bary.y;
}

// Russian roulette termination
__device__ float russian_roulette_probability(unsigned int depth, float threshold, float decay) {
    // Probability of continuing the path
    // Decreases with depth to naturally terminate paths
    float prob = threshold * powf(decay, (float)depth);
    prob = fmaxf(prob, 0.05f);  // Clamp minimum probability to avoid division issues
    return prob;
}

// ------------------------------------------------------------------
// Ray generation
// ------------------------------------------------------------------
extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();

    if (idx.x >= params.width || idx.y >= params.height)
        return;

    const int i = idx.y * params.width + idx.x;

    // Initialize random seed per pixel (different for each sample)
    unsigned int seed = params.random_seed + (idx.x * 73856093 ^ idx.y * 19349663 ^ params.current_sample * 83492791);

    // Calculate normalized pixel coordinates with jitter for antialiasing
    float jitter_x = random_float(seed);
    float jitter_y = random_float(seed);
    
    float u = (float)idx.x + jitter_x / (float)(params.width - 1);
    float v = (float)idx.y + jitter_y / (float)(params.height - 1);
    
    u = u / (float)(params.width);
    v = v / (float)(params.height);

    // Generate ray from camera
    float3 ray_origin = params.camera.origin;
    float3 ray_direction = normalize(
        params.camera.lower_left_corner +
        u * params.camera.horizontal +
        v * params.camera.vertical -
        params.camera.origin
    );

    // Trace ray - pass seed in payload slot 3
    unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0;
    optixTrace(
        params.traversable,
        ray_origin,
        ray_direction,
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

    // Unpack color from payload
    float r = __uint_as_float(p0);
    float g = __uint_as_float(p1);
    float b = __uint_as_float(p2);
    
    float3 sample_color = make_float3(r, g, b);

    // Accumulate into buffer
    if (params.current_sample == 0) {
        params.accum_buffer[i] = sample_color;
    } else {
        // Weighted average: older samples have less weight as we accumulate
        float weight = 1.0f / (float)(params.current_sample);
        params.accum_buffer[i] = params.accum_buffer[i] * (1.0f - weight) + sample_color * weight;
    }

    // Convert to uchar4 for display
    float3 final_color = clamp(params.accum_buffer[i], 0.0f, 1.0f);
    params.image[i] = make_uchar4(
        (unsigned char)(final_color.x * 255.99f),
        (unsigned char)(final_color.y * 255.99f),
        (unsigned char)(final_color.z * 255.99f),
        255
    );
}
// ------------------------------------------------------------------
// Miss: sky gradient
// ------------------------------------------------------------------
/*
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
*/

extern "C" __global__ void __miss__ms() {
    //MissData* miss = (MissData*)optixGetSbtDataPointer();
    const float3 ray_dir = optixGetWorldRayDirection();
    float3 Le;

    if (params.has_envmap) {
        float3 envColor = sampleEnvmap(params.envmap, ray_dir);
        // Apply exposure: exposure = log2 scale
        envColor = envColor * params.envmap_scale * powf(2.0f, params.envmap_exposure);
        Le = clamp(envColor, 0.0f, 10.0f);  // Allow HDR values
    } else {
        float t = 0.5f * (ray_dir.y + 1.0f);
        Le = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
    }
    optixSetPayload_0(__float_as_uint(Le.x));
    optixSetPayload_1(__float_as_uint(Le.y));
    optixSetPayload_2(__float_as_uint(Le.z));
}

// ------------------------------------------------------------------
// Closest hit: solid/diffuse
// ------------------------------------------------------------------
extern "C" __global__ void __closesthit__ch()
{
    const float t_hit          = optixGetRayTmax();
    const float3 ray_origin    = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float3 hit_point     = ray_origin + t_hit * ray_direction;

    const HitGroupDataLambert* sbt = (HitGroupDataLambert*)optixGetSbtDataPointer();

    float3 world_normal = getInterpolatedNormal(sbt, ray_direction);
    float3 base_color   = getAlbedo(sbt);

    if (length_squared(base_color) < 0.001f)
        base_color = make_float3(1.0f, 0.0f, 1.0f);

    unsigned int depth = optixGetPayload_3();
    
    // Initialize seed for this bounce
    unsigned int seed = params.random_seed + (depth * 12345) + 
                        __float_as_uint(hit_point.x) ^ __float_as_uint(hit_point.y);

    // Direct lighting
    float3 direct_color = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int light_idx = 0; light_idx < params.num_lights; light_idx++) {
        const Light& light = params.lights[light_idx];
        float3 light_dir;
        float distance_to_light;
        float atten = 1.0f;

        if (light.type == 0) {
            float3 to_light = light.position_or_direction - hit_point;
            distance_to_light = length(to_light);
            light_dir = normalize(to_light);
            atten = 1.0f / (1.0f + 0.1f * distance_to_light);
        } else {
            light_dir = normalize(light.position_or_direction);
            distance_to_light = 1e16f;
            atten = 1.0f;
        }

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
            0, 1, 0,
            shadow_hit
        );

        float shadow  = shadow_hit ? 1.0f : 0.0f;
        float ndotl   = fmaxf(0.0f, dot(world_normal, light_dir));
        direct_color = direct_color + light.color * ndotl * atten * shadow;
    }

    float3 ambient = make_float3(0.02f, 0.02f, 0.02f);
    direct_color = direct_color + ambient;

    // Russian roulette path termination
    float rr_prob = russian_roulette_probability(depth, params.rr_threshold, params.rr_decay);
    float rr_random = random_float(seed);
    
    if (rr_random < rr_prob && depth < params.max_bounce_depth) {
        // Continue path with weighted contribution to account for probability
        float3 bounce_dir = random_hemisphere_direction(world_normal, seed);

        unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = depth + 1;
        optixTrace(
            params.traversable,
            hit_point + world_normal * 0.001f,
            bounce_dir,
            0.001f,
            1e16f,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            p0, p1, p2, p3
        );

        float3 indirect_color = make_float3(
            __uint_as_float(p0),
            __uint_as_float(p1),
            __uint_as_float(p2)
        );

        // Divide by probability to account for Russian roulette
        float3 final_color = direct_color * base_color + base_color * indirect_color / rr_prob;
        
        optixSetPayload_0(__float_as_uint(final_color.x));
        optixSetPayload_1(__float_as_uint(final_color.y));
        optixSetPayload_2(__float_as_uint(final_color.z));
    } else {
        // Path terminated by Russian roulette
        optixSetPayload_0(__float_as_uint(direct_color.x));
        optixSetPayload_1(__float_as_uint(direct_color.y));
        optixSetPayload_2(__float_as_uint(direct_color.z));
    }
}
// ------------------------------------------------------------------
// Closest hit: glass/refractive
// ------------------------------------------------------------------
extern "C" __global__ void __closesthit__glass()
{
    const float t_hit          = optixGetRayTmax();
    const float3 ray_origin    = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float3 hit_point     = ray_origin + t_hit * ray_direction;

    const HitGroupDataGlass* sbt = (HitGroupDataGlass*)optixGetSbtDataPointer();

    // Compute normal (without flipping - we need the raw outward normal for IOR)
    const int   prim_idx = optixGetPrimitiveIndex();
    const uint3 tri      = sbt->indices[prim_idx];
    const float3 v0 = sbt->vertices[tri.x].position;
    const float3 v1 = sbt->vertices[tri.y].position;
    const float3 v2 = sbt->vertices[tri.z].position;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 object_normal = make_float3(
        e1.y*e2.z - e1.z*e2.y,
        e1.z*e2.x - e1.x*e2.z,
        e1.x*e2.y - e1.y*e2.x);
    float3 world_normal = normalize(
        optixTransformNormalFromObjectToWorldSpace(object_normal));

    // Determine enter/exit
    bool entering = dot(world_normal, ray_direction) < 0.0f;
    if (!entering)
        world_normal = world_normal * -1.0f;

    float3 base_color = make_float3(0,1,0);// = getAlbedo( sbt );
    float  ior        = sbt->refraction_index;

    unsigned int current_depth = optixGetPayload_3();
    float3 color;

    if (current_depth >= 3) {
        // Max depth fallback
        color = base_color * 0.8f;
    }
    else {
        float eta         = entering ? (1.0f / ior) : ior;
        float cos_theta   = fminf(dot(ray_direction * -1.0f, world_normal), 1.0f);
        float sin_theta   = sqrtf(1.0f - cos_theta * cos_theta);
        float r0          = (1.0f - ior) / (1.0f + ior);
        r0                = r0 * r0;
        float reflectance = r0 + (1.0f - r0) * powf((1.0f - cos_theta), 5.0f);

        if (eta * sin_theta > 1.0f || reflectance > 0.5f) {
            // Reflection
            float3 reflected = ray_direction - 2.0f * dot(ray_direction, world_normal) * world_normal;
            unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = current_depth + 1;
            optixTrace(params.traversable,
                       hit_point + world_normal * 0.001f, normalize(reflected),
                       0.001f, 1e16f, 0.0f,
                       OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0,
                       p0, p1, p2, p3);
            color = make_float3(__uint_as_float(p0),
                                __uint_as_float(p1),
                                __uint_as_float(p2)) * base_color;
        }
        else {
            // Refraction
            float3 refracted_perp     = eta * (ray_direction + cos_theta * world_normal);
            float3 refracted_parallel = -sqrtf(fabsf(1.0f - dot(refracted_perp, refracted_perp))) * world_normal;
            float3 refracted          = refracted_perp + refracted_parallel;
            unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = current_depth + 1;
            optixTrace(params.traversable,
                       hit_point - world_normal * 0.001f, normalize(refracted),
                       0.001f, 1e16f, 0.0f,
                       OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0,
                       p0, p1, p2, p3);
            color = make_float3(__uint_as_float(p0),
                                __uint_as_float(p1),
                                __uint_as_float(p2)) * base_color;
        }
    }

    color = clamp(color, 0.0f, 1.0f);

    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
    optixSetPayload_3(current_depth);
}