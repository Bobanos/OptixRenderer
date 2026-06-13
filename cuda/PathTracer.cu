#include <optix.h>
#include <optix_device.h>

#include "optix_params.h"
#include "float3_math.h"

extern "C" {
__constant__ Params params;
}

// ------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------
#define M_PI       3.14159265358979323846f
#define M_1_PI_F   0.31830988618379067154f   // 1/pi
#define EPS        3e-3f                     // ray offset epsilon to avoid self-intersection


// ------------------------------------------------------------------
// RNG (Permuted congruential generator)
// ------------------------------------------------------------------
struct PCG32
{
    unsigned long long state;
    unsigned long long inc;

    __device__ void seed(unsigned long long init_state, unsigned long long init_seq)
    {
        state = 0u;
        inc = (init_seq << 1u) | 1u;
        next_uint();
        state += init_state;
        next_uint();
    }

    __device__ unsigned int next_uint()
    {
        unsigned long long old = state;
        state = old * 6364136223846793005ULL + inc;
        unsigned int xorshifted = ((old >> 18u) ^ old) >> 27u;
        unsigned int rot = old >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    // Returns uniform float in [0, 1)
    __device__ float next_float()
    {
        return (float)next_uint() * (1.0f / 4294967296.0f);
    }
};


// ------------------------------------------------------------------
// 
// ------------------------------------------------------------------

static __forceinline__ __device__ float3 fresnelShlick(float cosTheta, float3 F0) {
    return F0 + (1.0f - F0) * powf(1.0f - cosTheta, 5.0f);
}

// Method for Sampling Visible GGX Normals with Spherical Caps by Jonathan Dupuy and Anis Benyoub (2023)
static __device__ __forceinline__ float3 SampleVNDF_Hemisphere(float u1, float u2, float3 wi)
{
    //sample a spherical cap in (-wi.z,1]
    float phi = 2.0f * M_PI * u1;
    float z = fma((1.0f - u2), (1.0f + wi.z), -wi.z);
    float sinTheta = sqrt(clamp(1.0f - z * z, 0.0f, 1.0f));
    float x = sinTheta * cos(phi);
	float y = sinTheta * sin(phi);
    float3 c = make_float3(x, y, z);
    //compute halfway direction;
    float3 h = c + wi;
    //return without normalization (as this is done later)
    return h;
}



__device__ float3 getAlbedo(const HitGroupDataCookTorrance* sbt)
{
    const int   prim_idx = optixGetPrimitiveIndex();
    const uint3 tri = sbt->indices[prim_idx];
    const float2 bary = optixGetTriangleBarycentrics();
    const float  b0 = 1.f - bary.x - bary.y;

    if (sbt->albedo_texture != 0) {
        // Interpolate UV coordinates across the triangle
        const float2 uv0 = sbt->vertices[tri.x].uv;
        const float2 uv1 = sbt->vertices[tri.y].uv;
        const float2 uv2 = sbt->vertices[tri.z].uv;
        float u = uv0.x * b0 + uv1.x * bary.x + uv2.x * bary.y;
        float v = uv0.y * b0 + uv1.y * bary.x + uv2.y * bary.y;
        float4 t = tex2D<float4>(sbt->albedo_texture, u, v);
        return make_float3(t.x, t.y, t.z);
    }

    // Use the flat material color (Kd from .mtl) stored in the SBT record
    return sbt->albedo;
}


static __device__ __forceinline__ float luminance(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

//------------------------------------------------------------------------------
//
// Utility functions
//
//------------------------------------------------------------------------------

static __forceinline__ __device__ RadiancePRD loadClosesthitRadiancePRD() {
    RadiancePRD prd = {};
    return prd;
}

static __forceinline__ __device__ RadiancePRD loadMissRadiancePRD() {
    RadiancePRD prd = {};
    return prd;
}

static __forceinline__ __device__ void storeClosesthitRadiancePRD(RadiancePRD prd) {
    optixSetPayload_0(__float_as_uint(prd.radiance.x));
    optixSetPayload_1(__float_as_uint(prd.radiance.y));
    optixSetPayload_2(__float_as_uint(prd.radiance.z));
}

static __forceinline__ __device__ void storeMissRadiancePRD(RadiancePRD prd) {
    optixSetPayload_0(__float_as_uint(prd.radiance.x));
    optixSetPayload_1(__float_as_uint(prd.radiance.y));
    optixSetPayload_2(__float_as_uint(prd.radiance.z));
}
//static __forceinline__ __device__ RadiancePRD loadClosesthitRadiancePRD()
//{
//    RadiancePRD prd = {};
//
//    prd.attenuation.x = __uint_as_float(optixGetPayload_0());
//    prd.attenuation.y = __uint_as_float(optixGetPayload_1());
//    prd.attenuation.z = __uint_as_float(optixGetPayload_2());
//    prd.seed = optixGetPayload_3();
//    prd.depth = optixGetPayload_4();
//    return prd;
//}
//
//static __forceinline__ __device__ RadiancePRD loadMissRadiancePRD()
//{
//    RadiancePRD prd = {};
//    return prd;
//}
//
//static __forceinline__ __device__ void storeClosesthitRadiancePRD(RadiancePRD prd)
//{
//    optixSetPayload_0(__float_as_uint(prd.attenuation.x));
//    optixSetPayload_1(__float_as_uint(prd.attenuation.y));
//    optixSetPayload_2(__float_as_uint(prd.attenuation.z));
//
//    optixSetPayload_3(prd.seed);
//    optixSetPayload_4(prd.depth);
//
//    optixSetPayload_5(__float_as_uint(prd.emitted.x));
//    optixSetPayload_6(__float_as_uint(prd.emitted.y));
//    optixSetPayload_7(__float_as_uint(prd.emitted.z));
//
//    optixSetPayload_8(__float_as_uint(prd.radiance.x));
//    optixSetPayload_9(__float_as_uint(prd.radiance.y));
//    optixSetPayload_10(__float_as_uint(prd.radiance.z));
//
//    optixSetPayload_11(__float_as_uint(prd.origin.x));
//    optixSetPayload_12(__float_as_uint(prd.origin.y));
//    optixSetPayload_13(__float_as_uint(prd.origin.z));
//
//    optixSetPayload_14(__float_as_uint(prd.direction.x));
//    optixSetPayload_15(__float_as_uint(prd.direction.y));
//    optixSetPayload_16(__float_as_uint(prd.direction.z));
//
//    optixSetPayload_17(prd.done);
//}
//
//
//static __forceinline__ __device__ void storeMissRadiancePRD(RadiancePRD prd)
//{
//    optixSetPayload_5(__float_as_uint(prd.emitted.x));
//    optixSetPayload_6(__float_as_uint(prd.emitted.y));
//    optixSetPayload_7(__float_as_uint(prd.emitted.z));
//
//    optixSetPayload_8(__float_as_uint(prd.radiance.x));
//    optixSetPayload_9(__float_as_uint(prd.radiance.y));
//    optixSetPayload_10(__float_as_uint(prd.radiance.z));
//
//    optixSetPayload_17(prd.done);
//}

static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    RadiancePRD& prd){

    unsigned int u0, u1, u2;

    // SER (Shader Execution Reordering): split trace into traverse + reorder + invoke
    // traverse: fires ray, runs BVH + anyhit, stores hit info (no shading yet)
    // reorder:  groups threads by hit material -> coherent warp execution
    // invoke:   runs closesthit or miss with better warp coherence
    optixTraverse(
        PAYLOAD_TYPE_RADIANCE,
        handle,
        ray_origin,
        ray_direction,
        tmin,                   // tmin
        tmax,                   // tmax    
        0.f,                    // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,                      // SBT offset  (ray type 0 = radiance)
        1,                      // SBT stride  (1 ray type)
        0,                      // miss SBT index
        u0, u1, u2
    );
    optixReorder();
    optixInvoke(PAYLOAD_TYPE_RADIANCE, 
        u0, u1, u2);

	prd.radiance = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
}

// ======================================================================================
// Scene-to-Display Transform Pipeline (Tone Mapping -> Gamma Correction -> Quantization)
// ======================================================================================

// ACES Filmic Tone Mapping Operator
static __device__ __forceinline__ float3 acesTMO(const float3 color) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return make_float3(
        clamp((color.x * (a * color.x + b)) / (color.x * (c * color.x + d) + e), 0.0f, 1.0f),
        clamp((color.y * (a * color.y + b)) / (color.y * (c * color.y + d) + e), 0.0f, 1.0f),
        clamp((color.z * (a * color.z + b)) / (color.z * (c * color.z + d) + e), 0.0f, 1.0f)
    );
}

// Gamma correction (sRGB) for display. Assumes input is in linear space [0,1].
static __device__ __forceinline__ float linearToSRGB(float value) {
    if (value <= 0.0031308f) {
        return 12.92f * value;
    }
    else {
        return 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
    }
}

// Full pipeline: applies tone mapping, then gamma corrects, then quantizes to 8 bits per channel
static __device__ __forceinline__ uchar4 convertToFinalSRGB(float3 inputColor){
	float3 tonemappedColor = acesTMO(inputColor);
    float3 sRGBColor = make_float3(
        linearToSRGB(tonemappedColor.x),
        linearToSRGB(tonemappedColor.y),
        linearToSRGB(tonemappedColor.z)
    );
    return make_uchar4(
        (unsigned char)(sRGBColor.x * 255.99f),
        (unsigned char)(sRGBColor.y * 255.99f),
        (unsigned char)(sRGBColor.z * 255.99f), 
        255u
    );
}

// ======================================================================================
// ENVIRONMENT MAP — SAMPLING AND PDF
//
// Equirectangular projection maps a sphere onto a rectangle:
//   phi   = azimuth [-pi, pi]  -> u [0, 1]
//   theta = polar   [0,  pi]   -> v [0, 1]
// ======================================================================================

__device__ float2 dirToEnvmapUV(float3 dir) {
    float phi = atan2f(dir.z, dir.x);
    float theta = acosf(clamp(dir.y, -1.f, 1.f));
    return make_float2((phi / (2.f * M_PI)) + 0.5f, theta / M_PI);
}

__device__ float3 uvToEnvmapDir(float2 uv) {
    float phi = (uv.x - 0.5f) * 2.f * M_PI;
    float theta = uv.y * M_PI;
    return make_float3(sinf(theta) * cosf(phi), cosf(theta), sinf(theta) * sinf(phi));
}

__device__ float3 sampleEnvmap(cudaTextureObject_t tex, float3 dir) {
    float2 uv = dirToEnvmapUV(dir);
    float4 val = tex2D<float4>(tex, uv.x, uv.y);
    return make_float3(val.x, val.y, val.z);
}


// ======================================================================================
// RAYGEN, MISS, and CLOSESTHIT PROGRAMS
// ======================================================================================
extern "C" __global__ void __raygen__pathTracer() {
    const uint3 idx = optixGetLaunchIndex();
    if (idx.x >= params.width || idx.y >= params.height) return;

    const unsigned int pixel_index = idx.y * params.width + idx.x;

    // RNG init
    // Unique seed per pixel per frame to avoid pattern repetition.
    PCG32 rng;
    rng.seed(
        (unsigned long long)pixel_index * 6364136223846793005ULL + 1442695040888963407ULL,
        (unsigned long long)params.current_sample * 2654435761ULL + 1
    );

    float3 frame_color = make_float3(0.0f, 0.0f, 0.0f);
    int samples_per_pixel =  params.samples_per_pixel;
    for (int samples = 0; samples < samples_per_pixel; ++samples){

        // Jitter pixel sample for anti-aliasing (without jitter: all samples hit same center)
        float u = ((float)idx.x + rng.next_float()) / (float)params.width;
        float v = ((float)idx.y + rng.next_float()) / (float)params.height;

        float3 ray_origin = params.camera.origin;
        float3 ray_dir = normalize(
            params.camera.lower_left_corner
            + u * params.camera.horizontal
            + v * params.camera.vertical
            - params.camera.origin);

        // Path state
        //float3 throughput = make_float3(1.0f, 1.0f, 1.0f); // path weight
        //float3 radiance = make_float3(0.0f, 0.0f, 0.0f); // accumulated light

        RadiancePRD prd;

        const float T_MIN = 1e-3f;
        const float T_MAX = 1e16f;

        traceRadiance(
            params.traversable,
            ray_origin,
            ray_dir,
            T_MIN,
            T_MAX,
            prd
		);
   //     // Payload: [p0,p1,p2]=color, [p3]=depth, [p4,p5,p6]=throughput
   //     unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0;
   //     unsigned int p4 = __float_as_uint(1.f), p5 = __float_as_uint(1.f), p6 = __float_as_uint(1.f);

   //     // SER (Shader Execution Reordering): split trace into traverse + reorder + invoke
   //     // traverse: fires ray, runs BVH + anyhit, stores hit info (no shading yet)
   //     // reorder:  groups threads by hit material -> coherent warp execution
   //     // invoke:   runs closesthit or miss with better warp coherence
   //     optixTraverse(params.traversable,
   //         ray_origin,
   //         ray_dir,
   //         T_MIN,                  // tmin
   //         T_MAX,                  // tmax    
			//0.f,                    // rayTime
   //         OptixVisibilityMask(255),
   //         OPTIX_RAY_FLAG_NONE,
   //         0,                      // SBT offset  (ray type 0 = radiance)
   //         1,                      // SBT stride  (1 ray type)
   //         0,                      // miss SBT index
   //         p0, p1, p2
   //     );
   //     optixReorder();
   //     optixInvoke(p0, p1, p2);

        frame_color = frame_color + prd.radiance;

    }

	params.image[pixel_index] = convertToFinalSRGB(frame_color);
}

// ======================================================================================

extern "C" __global__ void __miss__envMap() {
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    RadiancePRD prd = loadMissRadiancePRD();

    float3 ray_dir = normalize(optixGetWorldRayDirection());
	float3 Le = make_float3(1.f);  // default white if no envmap provided
    if (params.envmap.has_envmap) {
        Le = sampleEnvmap(params.envmap.texture, ray_dir) * params.envmap.scale * powf(2.f, params.envmap.exposure);
        //Le = clamp(Le, 0.f, 10.f);  // prevent extreme fireflies
    }
    
	prd.radiance = Le;
    storeMissRadiancePRD(prd);

}

// ======================================================================================

extern "C" __global__ void __closesthit__cookTorrance() {
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    const float  t_hit = optixGetRayTmax();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 hit_pos = optixGetWorldRayOrigin() + t_hit * ray_dir;

    const HitGroupDataCookTorrance* sbt = (HitGroupDataCookTorrance*)optixGetSbtDataPointer();

    RadiancePRD prd = loadClosesthitRadiancePRD();

    //float3 normal = getInterpolatedNormal(sbt, ray_dir);    // smooth shading normal
    //float3 n_geom = getGeometricNormal(sbt, ray_dir);       // geometric normal
    float3 albedo = getAlbedo(sbt);                             // material color
    float3 wo     = -normalize(ray_dir);                        // toward camera


    //prd.radiance = albedo;
	prd.radiance = sbt->base_color;
    storeClosesthitRadiancePRD(prd);
}

// ======================================================================================

extern "C" __global__ void __closesthit__glass() {
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);
    RadiancePRD prd = loadClosesthitRadiancePRD();
    storeClosesthitRadiancePRD(prd);
	// placeholder until we implement the full glass shader below
    /*
    PathPayload* payload = get_payload();
    const GlassData& mat = *reinterpret_cast<const GlassData*>(optixGetSbtDataPointer());

    // Hit geometry
    float3 ray_dir = normalize(optixGetWorldRayDirection());

    // Geometric normal (object space -> world space)
    // Replace with interpolated vertex normals in production.
    float3 normal_obj = make_float3(0.0f, 1.0f, 0.0f); // placeholder
    float3 normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal_obj));

    float3 hit_point = optixGetWorldRayOrigin()
        + optixGetRayTmax() * optixGetWorldRayDirection();

    //  Determine ray side 
    // front_face = ray hitting outside of surface
    bool   front_face = dot(ray_dir, normal) < 0.0f;
    float3 outward_normal = front_face ? normal
        : make_float3(-normal.x, -normal.y, -normal.z);

    // n1/n2 ratio: air->glass or glass->air
    float ior_ratio = front_face ? (1.0f / mat.ior) : mat.ior;

    //  Emissive (rarely used for glass, but supported)
    payload->emitted = mat.emissive;

    //  Fresnel (Schlick) 
    float cos_theta = fminf(dot(-ray_dir, outward_normal), 1.0f);
    float reflectance = fresnel_schlick(cos_theta, ior_ratio);

    //  Refraction via Snell's law 
    // Check for Total Internal Reflection
    float sin_theta_sq = 1.0f - cos_theta * cos_theta;
    float discriminant = 1.0f - ior_ratio * ior_ratio * sin_theta_sq;
    bool  can_refract = discriminant > 0.0f;

    // RNG for reflect/refract decision
    PCG32 rng;
    rng.seed(
        (unsigned long long)optixGetPrimitiveIndex() * 1442695040888963407ULL
        ^ __float_as_uint(hit_point.x * 1000.0f),
        (unsigned long long)optixGetLaunchIndex().x + optixGetLaunchIndex().y * dims_seed_helper()
    );
    float rand_val = rng.next_float();

    float3 scattered;

    if (!can_refract || reflectance > rand_val)
    {
        //  Reflect
        // r = d - 2(d·n)n
        scattered = ray_dir - 2.0f * dot(ray_dir, outward_normal) * outward_normal;
        scattered = normalize(scattered);

        // Offset origin to avoid self-intersection on same side
        payload->next_origin = hit_point + 1e-4f * outward_normal;
    }
    else
    {
        // Refract
        // Snell's law in vector form
        float3 r_perp = ior_ratio * (ray_dir + cos_theta * outward_normal);
        float3 r_paral = -sqrtf(fabsf(discriminant)) * outward_normal;
        scattered = normalize(r_perp + r_paral);

        // Offset origin to inside of surface to avoid self-intersection
        payload->next_origin = hit_point - 1e-4f * outward_normal;
    }

    // Glass is a perfect specular event:
    // throughput_scale = tint (color absorption)
    // No division by PDF needed — delta distribution cancels.
    payload->throughput_scale = mat.tint;
    payload->next_direction = scattered;
    payload->is_specular = true;  // skip NEE MIS for delta events
    payload->done = false;
    */
}