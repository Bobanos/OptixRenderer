#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>
#include <optix.h>
#include <optix_stubs.h>

#include "optix_params.h"
#include "obj_loader.h"
#include "camera.h"
#include "scene.h"


#ifdef _DEBUG
#define DEBUG_LOG(msg) std::cout << msg << std::endl
#define DEBUG_LOGF(fmt, ...) printf(fmt "\n", __VA_ARGS__)
#else
#define DEBUG_LOG(msg) (void)0
#define DEBUG_LOGF(fmt, ...) (void)0
#endif

#define CUDA_CHECK(x) r_util::cudaCheck((x), __FILE__, __LINE__)
#define OPTIX_CHECK(x) r_util::optixCheck((x), __FILE__, __LINE__)

namespace r_util {
    inline void cudaCheck(cudaError_t error, const char* file, int line) {
        if (error != cudaSuccess) {
            std::cerr << "CUDA error at " << file << ":" << line << ": "
                << cudaGetErrorString(error) << std::endl;
            std::exit(1);
        }
    }

    inline void optixCheck(OptixResult res, const char* file, int line) {
        if (res != OPTIX_SUCCESS) {
            std::cerr << "OptiX error at " << file << ":" << line << ": "
                << res << std::endl;
            std::exit(1);
        }
    }
}

class OptixRenderer {
public:
    OptixRenderer(int width, int height);
    ~OptixRenderer();

    // Initialization
    void initCUDA();
    void initOptix();
    void loadScene(SceneID scene_id);
    void loadMap(const std::string& path);
    void setupShaders();
    void setupLighting();

    // Rendering
    void render(const Camera& camera, int samples_per_pixel);
    void updateInstanceTransform(int instance_idx, const float transform[12]);
    void updateLightParametersPos(int light_idx, float3 pos);
    void updateLightParametersColor(int light_idx, float3 color);
    void updateEnvmapParameters(float scale, float exposure);
    void updateLightIntensity(float light_intensity);

    // Multi-object transforms
    void updateObjectTransform(int object_idx, float rotation_x, float rotation_y, float rotation_z);
    void getObjectRotation(int object_idx, float& rotation_x, float& rotation_y, float& rotation_z) const;

    // Scene switching
    void switchScene(SceneID scene_id);
    SceneID getCurrentSceneID() const { return current_scene_id; }
    std::string getCurrentSceneName() const { return current_scene_data.name; }
    int getObjectCount() const { return (int)object_transforms.size(); }

    // Getters
    uchar4* getPixelBuffer() const { return params.image; }
    float3* getAccumBuffer() const { return params.accum_buffer; }
    int getWidth() const { return params.width; }
    int getHeight() const { return params.height; }
    Params& getParams() { return params; }
    SceneData getCurrentSceneData() const { return current_scene_data; }

    // Parameter updates
    void setMaxBounceDepth(int depth) { params.max_bounce_depth = depth; }
    void setSamplesPerPixel(int spp) { params.samples_per_pixel = spp; }
    void setRandomSeed(unsigned int seed) { params.random_seed = seed; }

    void updateCamera(const Camera& camera);
    void resetAccumulationBuffer();

private:
    // Device pointers
    struct DeviceBuffers {
        CUdeviceptr d_pixels = 0;
        CUdeviceptr d_accum_buffer = 0;
        CUdeviceptr d_params = 0;
        CUdeviceptr d_ias_output_buffer = 0;
        CUdeviceptr d_instances = 0;
        CUdeviceptr d_hg = 0;
        CUdeviceptr d_rg = 0;
        CUdeviceptr d_ms = 0;
    } device_buffers;

    // OptiX state
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixShaderBindingTable sbt = {};

    // Program groups
    OptixProgramGroup raygen_program_group = nullptr;
    OptixProgramGroup miss_program_group = nullptr;
    OptixProgramGroup hitgroup_lambert_program_group = nullptr;
    OptixProgramGroup hitgroup_glass_program_group = nullptr;

    // Acceleration structures
    OptixTraversableHandle ias_handle = 0;
    std::vector<OptixAccelBufferSizes> gas_sizes;
    OptixAccelBufferSizes  ias_buffer_sizes;

    // One per placed object in the scene — references a SceneObject
    struct SceneObjectInstance {
        LoadedSceneObject* object = nullptr;       // which geometry
        float              transform[12]{ 1,0,0,0,
                                           0,1,0,0,
                                           0,0,1,0 }; // where/how it's placed
    };

    // Scene data
	std::vector<LoadedSceneObject>   loaded_scene_objects;
    std::vector<SceneObjectInstance> scene_instances;

    struct TextureData {
        std::string         name = "";
        cudaTextureObject_t tex = 0;
        cudaArray_t         array = nullptr;
    };

    // Global cache — key is the resolved absolute path
    std::unordered_map<std::string, TextureData> texture_cache;

    // Host params
    Params params = {};

    // Object transform state (per object)
    struct ObjectTransform {
        float rotation_x = 0.0f;
        float rotation_y = 0.0f;
        float rotation_z = 0.0f;
    };
    std::vector<ObjectTransform> object_transforms;

    // Track material offsets for each object
    std::vector<uint32_t> object_material_offsets;

    Camera last_camera = {};

    // Scene management
    SceneID current_scene_id = SceneID::ALLIED_AVENGER;
    SceneData current_scene_data;

    cudaArray_t envmap_cdf_marginal_array = nullptr;
    cudaArray_t envmap_cdf_conditional_array = nullptr;

    void rebuildIAS();

    // Helper methods
    void createModuleAndProgramGroups();
    void createPipeline();
    void uploadSceneObject(LoadedSceneObject& obj);
    void buildGAS(LoadedSceneObject& obj, int index);
    void buildIAS();
    void buildSBT();
    void cleanup();
    void FreeDeviceBuffers();
    void FreeTexturesAndEnvMaps();

    void createTransformMatrix(
        float(&transform)[12],
        const float3 translation,
        const float3 scale,
        float rot_x,
        float rot_y,
        float rot_z
    );

	uint32_t current_sbt_offset = 0;  // Used for IAS. Tracks the next available SBT offset for hit groups as we build IAS

    std::vector<char> loadFile(const std::string& path);
    cudaTextureObject_t loadTextureCached(const std::string& resolved_path);
    //cudaTextureObject_t loadTextureFromFile(const std::string& path, cudaArray_t& out_array);
};