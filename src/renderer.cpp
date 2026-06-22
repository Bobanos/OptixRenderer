#include "renderer.h"
#include <iostream>
#include <vector>
#include <cuda_gl_interop.h>
#include <optix_micromap.h>
#include <stdexcept>


void optixLogCallback(unsigned int level, const char* tag, const char* message, void*) {
    std::cerr << "[OptiX][" << level << "][" << tag << "] "
        << message << std::endl;
}

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupDataCookTorrance> HitGroupRecordCookTorrance;
typedef Record<HitGroupDataGlass> HitGroupRecordGlass;


// Constructor to initialize renderer parameters, including setting the image width and height for the output buffer.
OptixRenderer::OptixRenderer(int width, int height) {
    params.width = width;
    params.height = height;
}

// Destructor to clean up resources
OptixRenderer::~OptixRenderer() {
    cleanup();
}

// Initializes cuda
void OptixRenderer::initCUDA() {
    CUDA_CHECK(cudaFree(nullptr));
    DEBUG_LOG("[CUDA] Initialized");
}

// Initializes the OptiX context with appropriate options, including setting up a log callback for debugging and enabling validation layers in debug builds.
void OptixRenderer::initOptix() {
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions ctx_opts = {};
    ctx_opts.logCallbackFunction = optixLogCallback;
    ctx_opts.logCallbackLevel = 4;
#ifdef _DEBUG
    DEBUG_LOG("[OptiX] DEBUG ENABLED");
    ctx_opts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

    CUcontext cuCtx = 0;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &ctx_opts, &context));
    DEBUG_LOG("[OptiX] Context created");
}

// Loads the scene specified by scene_id.
void OptixRenderer::loadScene(SceneID scene_id) {
    current_scene_data = SceneManager::getSceneConfig(scene_id);  // Load scene configuration data (camera settings, object list, envmap path, etc.) for the specified scene ID and store in current_scene_data
    current_scene_id = scene_id;  // Update the current scene ID to the newly loaded scene

    loaded_scene_objects.clear();  // Clear previously loaded scene objects to free memory and prepare for new scene
    scene_instances.clear();  // Clear previously created scene instances to prepare for new scene

    params.background_color = current_scene_data.background_color;
    current_sbt_offset = 0;

    loaded_scene_objects.reserve(current_scene_data.objects.size());

    gas_sizes.resize(current_scene_data.objects.size());// Resize GAS sizes vector to match the number of objects in the scene to be used later during GAS construction

    for (size_t i = 0; i < current_scene_data.objects.size(); ++i) {
        loaded_scene_objects.push_back(loadSceneObject(
            current_scene_data.objects[i].name,
            current_scene_data.objects[i].obj_path,
            current_scene_data.objects[i].base_dir)); // Load geometry and materials for this object and store in loaded_scene_objects
        loaded_scene_objects.back().sbt_base = current_sbt_offset;  // Assign SBT base index for this instance based on the current offset in the SBT. This will be used later when building the SBT to know which hit group entries correspond to this instance's geometry and materials.
        current_sbt_offset += (uint32_t)loaded_scene_objects.back().materials.size();  // Update the SBT offset for the next instance by adding the number of materials in this object, since each material corresponds to one hit group entry in the SBT
        uploadSceneObject(loaded_scene_objects.back()); // Upload geometry and material data to GPU buffers for this object
        buildGAS(loaded_scene_objects.back(), i);  // Build a GAS for this object and store the handle in loaded_scene_objects.back().gas_handle

        DEBUG_LOGF("[Scene] - [GAS]: %s with %zu vertices, %d materials",
            loaded_scene_objects.back().name.c_str(),
            loaded_scene_objects.back().vertices.size(),
            (int)loaded_scene_objects.back().materials.size());

        // Create instance for this object
        SceneObjectInstance instance;
        instance.object = &loaded_scene_objects.back();  // Point to the loaded scene object
        createTransformMatrix(
            instance.transform,
            current_scene_data.objects[i].position,
            current_scene_data.objects[i].scale,
            current_scene_data.objects[i].rotation_x,
            current_scene_data.objects[i].rotation_y,
            current_scene_data.objects[i].rotation_z
        );  // Create the initial transform matrix for this instance based on the scene configuration
        scene_instances.push_back(instance);  // Add this instance to the list of scene instances
    }

    DEBUG_LOGF("[Scene] Loading: %s with %zu objects",
        current_scene_data.name.c_str(),
        current_scene_data.objects.size());

    buildIAS();  // Build the top-level IAS for all instances in the scene using the GAS handles stored in loaded_scene_objects
    loadMap(current_scene_data.envmap_path);  // Load the environment map specified in the scene configuration and upload it to the GPU
    buildLightList();  // Build the flat NEE light list (emissive triangles) and upload it to the GPU
}

// Builds the NEE light list from all currently loaded scene objects, computes
// its area*luminance-weighted CDF, and uploads both the triangle array and
// the total weight to GPU memory. Stores the resulting pointer + counts
// directly into params so the next render() call picks them up automatically.
void OptixRenderer::buildLightList() {
    // Free any previously uploaded light list before rebuilding
    if (params.emissive_triangles) {
        CUDA_CHECK(cudaFree((void*)params.emissive_triangles));
        params.emissive_triangles = nullptr;
    }

    // Collect one transform pointer per object, in the same order as
    // loaded_scene_objects. scene_instances[i].object points back into
    // loaded_scene_objects, and instance.transform holds the current
    // world transform for that object.
    std::vector<float(*)[12]> object_transforms;
    object_transforms.reserve(scene_instances.size());
    for (auto& instance : scene_instances)
        object_transforms.push_back(&instance.transform);

    std::vector<EmissiveTriangle> lights =
        buildEmissiveTriangleList(loaded_scene_objects, object_transforms);

    float total_weight = buildLightCDF(lights);

    params.num_emissive_triangles = (int)lights.size();
    params.total_emissive_weight = total_weight;

    if (lights.empty()) {
        params.emissive_triangles = nullptr;
        DEBUG_LOG("[NEE] No emissive triangles found, NEE will be skipped at render time");
        return;
    }

    CUDA_CHECK(cudaMalloc((void**)&params.emissive_triangles,
        lights.size() * sizeof(EmissiveTriangle)));
    CUDA_CHECK(cudaMemcpy((void*)params.emissive_triangles,
        lights.data(),
        lights.size() * sizeof(EmissiveTriangle),
        cudaMemcpyHostToDevice));

    DEBUG_LOGF("[NEE] Uploaded %d emissive triangles, total weight = %f",
        params.num_emissive_triangles, params.total_emissive_weight);
}

// Loads a texture from disk if not already loaded, and returns a CUDA texture object handle.
cudaTextureObject_t OptixRenderer::loadTextureCached(const std::string& resolved_path)
{
    // Already loaded - return existing handle
    auto it = texture_cache.find(resolved_path);
    if (it != texture_cache.end()) {
        //printf("[Texture] Cache hit: %s\n", resolved_path.c_str());
        return it->second.tex;
    }

    // Load for the first time
    int w, h, channels;
    unsigned char* data = stbi_load(resolved_path.c_str(), &w, &h, &channels, 4);
    if (!data) {
        printf("[Texture] FAILED to load: %s\n", resolved_path.c_str());
        return 0;
    }

    TextureData td;
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<uchar4>();
    CUDA_CHECK(cudaMallocArray(&td.array, &fmt, w, h));
    CUDA_CHECK(cudaMemcpy2DToArray(td.array, 0, 0,
        data, w * 4, w * 4, h, cudaMemcpyHostToDevice));
    stbi_image_free(data);

    cudaResourceDesc res = {};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = td.array;

    cudaTextureDesc tex = {};
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.readMode = cudaReadModeNormalizedFloat;
    tex.normalizedCoords = 1;
    tex.sRGB = 1;  // most albedo textures are sRGB

    CUDA_CHECK(cudaCreateTextureObject(&td.tex, &res, &tex, nullptr));

    texture_cache[resolved_path] = td;
    //printf("[Texture] Loaded: %s (%dx%d)\n", resolved_path.c_str(), w, h);
    return td.tex;
}

// Loads an HDR environment map from the specified path, uploads it to the GPU as a CUDA texture,
// and computes the CDFs. If the path is empty or loading fails, it sets has_envmap to false in the params.
void OptixRenderer::loadMap(const std::string& path) {
    if (path.empty()) {
        params.envmap.has_envmap = false;
        return;
    }

    int width, height, channels;
    float* data = stbi_loadf(path.c_str(), &width, &height, &channels, 4);  // Force RGBA
    if (!data) {
        std::cerr << "[Env] Failed to load: " << path << " - " << stbi_failure_reason() << "\n";
        params.envmap.has_envmap = false;
        return;
    }

    std::cout << "[Env] Loaded " << path << " (" << width << "x" << height << ")\n";

    // --- Upload envmap texture ---
    {
        cudaArray_t env_array;
        cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float4>();
        CUDA_CHECK(cudaMallocArray(&env_array, &fmt, width, height));
        CUDA_CHECK(cudaMemcpy2DToArray(
            env_array, 0, 0,
            data,
            width * 4 * sizeof(float),
            width * 4 * sizeof(float),
            height,
            cudaMemcpyHostToDevice));

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = env_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(&params.envmap.texture, &res_desc, &tex_desc, nullptr));
    }

    // --- Compute and upload CDF textures ---
    {
        std::vector<float> marginal_cdf, conditional_cdf;
        computeEnvmapCDF(data, width, height, marginal_cdf, conditional_cdf);
        uploadEnvmapCDFTextures(
            marginal_cdf, conditional_cdf,
            width, height,
            params.envmap.cdf_marginal_v,
            params.envmap.cdf_conditional_u,
            envmap_cdf_marginal_array,
            envmap_cdf_conditional_array);

        params.envmap.width = width;
        params.envmap.height = height;
    }

    stbi_image_free(data);

    params.envmap.has_envmap = true;
    DEBUG_LOG("[Env] Successfully loaded environment map with CDF");
}

// Uploads the vertex buffer, index buffer, and SBT index buffer for a given LoadedSceneObject to GPU memory.
// The vertex buffer contains the geometry data (positions, normals, UVs) for the object's triangles, the index
// buffer defines how vertices are connected into triangles, and the SBT index buffer maps each triangle to
// a material index for shader binding. After this function is called, the LoadedSceneObject will have its
// d_vertices, d_indices, and d_sbt_indices members populated with device pointers to the uploaded data.
void OptixRenderer::uploadSceneObject(LoadedSceneObject& obj)
{
    // 1. Full vertex buffer - read by shaders via sbt->vertices
    //    AND used by OptiX GAS build via stride
    {
        size_t bytes = obj.vertices.size() * sizeof(ColoredVertex);
        CUDA_CHECK(cudaMalloc((void**)&obj.d_vertices, bytes));
        CUDA_CHECK(cudaMemcpy((void*)obj.d_vertices,
            obj.vertices.data(), bytes, cudaMemcpyHostToDevice));
    }

    // 2. Index buffer - triangles
    {
        size_t bytes = obj.indices.size() * sizeof(uint3);
        CUDA_CHECK(cudaMalloc((void**)&obj.d_indices, bytes));
        CUDA_CHECK(cudaMemcpy((void*)obj.d_indices,
            obj.indices.data(), bytes, cudaMemcpyHostToDevice));
    }

    // 3. SBT index buffer - one uint32 per triangle, routes to material
    {
        size_t bytes = obj.sbt_index_buffer.size() * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc((void**)&obj.d_sbt_indices, bytes));
        CUDA_CHECK(cudaMemcpy((void*)obj.d_sbt_indices,
            obj.sbt_index_buffer.data(), bytes, cudaMemcpyHostToDevice));
    }

    printf("[GPU] '%s': %.1f MB\n", obj.name.c_str(),
        (float)(obj.vertices.size()
            * sizeof(ColoredVertex)
            + obj.indices.size() * sizeof(uint3)
            + obj.sbt_index_buffer.size()
            * sizeof(uint32_t)) / (1024.f * 1024.f));
}

// Creates OptiX modules and program groups by compiling the OptiX IR (PTX) code for the ray generation,
// miss, and hit group shaders. The compiled modules and program groups are stored in the OptixRenderer
// for later use when building the pipeline and SBT.
void OptixRenderer::createModuleAndProgramGroups() {
    auto optix_ir = loadFile("generated/optixir/PathTracer.optixir");

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.pipelineLaunchParamsSizeInBytes = sizeof(Params);
	pipeline_compile_options.allowOpacityMicromaps = true;

    char log[4096];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(optixModuleCreate(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        optix_ir.data(),
        optix_ir.size(),
        log,
        &logSize,
        &module));
    if (logSize > 1)
        std::cerr << "[OptiX] Module log:\n" << log << std::endl;

    DEBUG_LOGF("[OptiX] Module created, params size: %d", sizeof(Params));

    OptixProgramGroupOptions pg_opts = {};

    // Create program groups
    OptixProgramGroupDesc descriptor_raygen = {};
    descriptor_raygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descriptor_raygen.raygen.module = module;
    descriptor_raygen.raygen.entryFunctionName = "__raygen__pathTracer";
    OPTIX_CHECK(optixProgramGroupCreate(context, &descriptor_raygen, 1, &pg_opts, log, &logSize, &raygen_program_group));

    OptixProgramGroupDesc descriptor_miss = {};
    descriptor_miss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descriptor_miss.miss.module = module;
    descriptor_miss.miss.entryFunctionName = "__miss__envMap";
    OPTIX_CHECK(optixProgramGroupCreate(context, &descriptor_miss, 1, &pg_opts, log, &logSize, &miss_program_group));

    OptixProgramGroupDesc descriptor_miss_occlusion = {};
    descriptor_miss_occlusion.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descriptor_miss_occlusion.miss.module = module;
    descriptor_miss_occlusion.miss.entryFunctionName = "__miss__occlusion";
    OPTIX_CHECK(optixProgramGroupCreate(context, &descriptor_miss_occlusion, 1, &pg_opts, log, &logSize, &miss_occlusion_program_group));

    OptixProgramGroupDesc descriptor_hitgroup_occlusion = {};
    descriptor_hitgroup_occlusion.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descriptor_hitgroup_occlusion.hitgroup.moduleCH = nullptr;
    descriptor_hitgroup_occlusion.hitgroup.entryFunctionNameCH = nullptr;
    descriptor_hitgroup_occlusion.hitgroup.moduleAH = module;
    descriptor_hitgroup_occlusion.hitgroup.entryFunctionNameAH = "__anyhit__occlusion";
    OPTIX_CHECK(optixProgramGroupCreate(context, &descriptor_hitgroup_occlusion, 1, &pg_opts, log, &logSize, &hitgroup_occlusion_program_group));

    OptixProgramGroupDesc descriptor_hitgroup_cooktorrance = {};
    descriptor_hitgroup_cooktorrance.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descriptor_hitgroup_cooktorrance.hitgroup.moduleCH = module;
    descriptor_hitgroup_cooktorrance.hitgroup.entryFunctionNameCH = "__closesthit__cookTorrance";
    descriptor_hitgroup_cooktorrance.hitgroup.moduleAH = module;
    descriptor_hitgroup_cooktorrance.hitgroup.entryFunctionNameAH = "__anyhit__opacity";
    OPTIX_CHECK(optixProgramGroupCreate(context, &descriptor_hitgroup_cooktorrance, 1, &pg_opts, log, &logSize, &hitgroup_cooktorrance_program_group));

    OptixProgramGroupDesc descriptor_hitgroup_glass = {};
    descriptor_hitgroup_glass.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descriptor_hitgroup_glass.hitgroup.moduleCH = module;
    descriptor_hitgroup_glass.hitgroup.entryFunctionNameCH = "__closesthit__glass";
    OPTIX_CHECK(optixProgramGroupCreate(context, &descriptor_hitgroup_glass, 1, &pg_opts, log, &logSize, &hitgroup_glass_program_group));

    DEBUG_LOG("[OptiX] Program groups created");
}

// Creates the OptiX pipeline by linking the previously created program groups together,
// and specifying pipeline-level options such as maximum trace depth. The resulting pipeline
// is stored in the OptixRenderer for use during rendering.
void OptixRenderer::createPipeline() {
    OptixProgramGroup groups[] = {
        raygen_program_group,
        miss_program_group,
        miss_occlusion_program_group,
        hitgroup_occlusion_program_group,
        hitgroup_cooktorrance_program_group,
        hitgroup_glass_program_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;

    char log[4096];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(optixPipelineCreate(
        context,
        &pipeline_compile_options,
        &pipeline_link_options,
        groups,
        6,
        log,
        &logSize,
        &pipeline
    ));
    DEBUG_LOG("[OptiX] Pipeline created");
}

// ============================================================
// OMM helpers
// ============================================================

// Sample a single texel from a CPU-side RGBA uint8 image.
// Returns the opacity value in [0,1].
// The Bistro's map_d textures are greyscale stored in the R channel.
// If your specific texture has it in the alpha channel, change [0] to [3].
static float sampleAlphaCPU(const unsigned char* pixels, int w, int h, float2 uv){
    // Wrap UVs into [0,1)
    float fu = uv.x - floorf(uv.x);
    float fv = uv.y - floorf(uv.y);
    int x = (int)(fu * w) % w;
    int y = (int)(fv * h) % h;
    if (x < 0) x += w;
    if (y < 0) y += h;
    // RGBA layout from stbi_load(..., 4)
    // R channel = opacity for standard greyscale map_d
    return pixels[(y * w + x) * 4 + 0] / 255.f;
}

// Classify 3 corner opacity samples into a 2-bit OMM state.
static uint8_t classifyOpacity(float a0, float a1, float a2, float threshold){
    bool any_opaque = (a0 >= threshold) || (a1 >= threshold) || (a2 >= threshold);
    bool any_transparent = (a0 < threshold) || (a1 < threshold) || (a2 < threshold);

    if (any_opaque && any_transparent)
        return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;   // boundary - fire any-hit
    else if (any_opaque)
        return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;           // always hit - skip any-hit
    else
        return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;      // always miss - skip any-hit
}

// Pack a 2-bit state into a flat byte array at absolute micro-triangle index m.
// byte_offset: start of this triangle's OMM data in the flat array (in bytes).
static void packOMMState(std::vector<uint8_t>& data, uint32_t byte_offset, int micro_idx, uint8_t state){
    int abs_bit = (int)(byte_offset * 8) + micro_idx * 2;
    int byte_idx = abs_bit / 8;
    int shift = abs_bit % 8;
    data[byte_idx] |= (state & 0x3) << shift;
}


bool OptixRenderer::buildOMMData(
    const LoadedSceneObject& object, int subdivision_level, std::vector<uint8_t>& packed_states,
    std::vector<OptixOpacityMicromapDesc>& descs, std::vector<uint32_t>& index_buffer){
    
    struct CPUAlphaTex {
        unsigned char* pixels = nullptr;
        int texture_width = 0;
        int texture_height = 0;
    };
    std::vector<CPUAlphaTex> per_material_alpha(object.materials.size());

    bool any_alpha_material = false;
    for (size_t material_index = 0; material_index < object.materials.size(); ++material_index){
        const std::string& path = object.materials[material_index].texture_paths.alpha_path;
        if (path.empty()) continue;

        int w, h, ch;
        unsigned char* px = stbi_load(path.c_str(), &w, &h, &ch, 4);
        if (!px) {
            printf("[OMM] Failed to load alpha texture: %s\n", path.c_str());
            continue;
        }
        per_material_alpha[material_index] = { px, w, h };
        any_alpha_material = true;
    }

    if (!any_alpha_material) {
        return false;
    }

    const int num_divs = 1 << subdivision_level;              // e.g. 8 for level 3
    const int num_micro_tris = num_divs * num_divs;           // 4^level micro-tris per source tri
    const int bytes_per_omm = (num_micro_tris * 2 + 7) / 8;   // 2 bits per micro-tri, packed
    const float threshold = 0.5f;

    index_buffer.resize(object.indices.size());

    for (int triangle_index = 0; triangle_index < (int)object.indices.size(); ++triangle_index)
    {
        // Which material does THIS triangle actually use?
        uint32_t mat_idx = object.sbt_index_buffer[triangle_index];
        const CPUAlphaTex& atex = per_material_alpha[mat_idx];

        if (atex.pixels == nullptr)
        {
            // This triangle's material has no alpha texture, then it is fully
            // opaque and needs no classification work or descriptor at all.
            // OptiX predefined special index: skips OMM lookup entirely,
            // any-hit is never invoked for it (same effect as DISABLE_ANYHIT,
            // but expressed via the OMM index so the array build stays valid
            // for the whole geometry in INDEXED mode).
            index_buffer[triangle_index] = OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_OPAQUE;
            continue;
        }

        const uint3& tri = object.indices[triangle_index];
        float2 uv0 = object.vertices[tri.x].uv;
        float2 uv1 = object.vertices[tri.y].uv;
        float2 uv2 = object.vertices[tri.z].uv;

        // Record where this OMM's packed data begins
        uint32_t byte_offset = (uint32_t)packed_states.size();
        packed_states.resize(byte_offset + bytes_per_omm, 0u);

        for (int m = 0; m < num_micro_tris; ++m)
        {
            // Get the 3 barycentric coordinates of this micro-triangle's vertices
            // in the space of the source (base) triangle.
            // Each coordinate is a float2 (u,v) with w = 1-u-v implied.
            float2 bary0, bary1, bary2;
            optixMicromapIndexToBaseBarycentrics(m, subdivision_level,
                bary0, bary1, bary2);

            // Interpolate source triangle UVs at each micro-triangle corner
            auto baryToTexUV = [&](float2 b) -> float2 {
                float w = 1.f - b.x - b.y;
                return make_float2(
                    w * uv0.x + b.x * uv1.x + b.y * uv2.x,
                    w * uv0.y + b.x * uv1.y + b.y * uv2.y
                );
                };

            float2 tex0 = baryToTexUV(bary0);
            float2 tex1 = baryToTexUV(bary1);
            float2 tex2 = baryToTexUV(bary2);

            // Sample and classify
            uint8_t state = classifyOpacity(
                sampleAlphaCPU(atex.pixels, atex.texture_width, atex.texture_height, tex0),
                sampleAlphaCPU(atex.pixels, atex.texture_width, atex.texture_height, tex1),
                sampleAlphaCPU(atex.pixels, atex.texture_width, atex.texture_height, tex2),
                threshold);

            packOMMState(packed_states, byte_offset, m, state);
        }

        // One descriptor per source triangle
        OptixOpacityMicromapDesc desc{};
        desc.byteOffset = byte_offset;
        desc.subdivisionLevel = (uint16_t)subdivision_level;
        desc.format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;

        // 1:1 mapping: triangle triangle_index uses descriptor triangle_index
        index_buffer[triangle_index] = (int32_t)descs.size();  // index of the desc we're about to push
        descs.push_back(desc);
    }

    // Free CPU alpha texture memory
    for (auto& atex : per_material_alpha)
        if (atex.pixels) stbi_image_free(atex.pixels);

    int classified = (int)descs.size();
    int skipped = (int)object.indices.size() - classified;
    printf("[OMM] '%s': %d triangles classified at level %d (%d micro-tris each), "
        "%d triangles skipped (no alpha material)\n",
        object.name.c_str(), classified, subdivision_level, num_micro_tris, skipped);

    return classified > 0;
}


// Builds a GAS for the given scene object using its vertex and index buffers, and stores the resulting traversable handle in the object.
void OptixRenderer::buildGAS(LoadedSceneObject& object, int index) {
    std::vector<uint32_t> geometryFlags(object.materials.size());

    for (size_t i = 0; i < object.materials.size(); ++i) {
        if (object.materials[i].texture_paths.alpha_path.empty()) {
            geometryFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        }
        else {
            geometryFlags[i] = OPTIX_GEOMETRY_FLAG_NONE;
        }
    }

    // OMM: check if ANY material in this object has an alpha texture.
    // We no longer pick a single alpha texture for the whole object --
    // buildOMMData now loads every alpha texture used by this object's
    // materials and classifies each triangle against ITS OWN material.
    // This flag just decides whether it's worth attempting OMM at all.
    bool object_has_alpha_material = false;
    for (const auto& mat : object.materials) {
        if (!mat.texture_paths.alpha_path.empty()) {
            object_has_alpha_material = true;
            break;
        }
    }

    bool has_omm = false;

    OptixBuildInputOpacityMicromap omm_build_input_attachment{};  // zero-init

    if (object_has_alpha_material)
    {
        const int subdivision_level = 4;  // 4^3 = 64 micro-tris per triangle

        std::vector<uint8_t>                   packed_states;
        std::vector<OptixOpacityMicromapDesc>  descs;
        std::vector<uint32_t>                  tri_index_buffer;

        bool ok = buildOMMData(object, subdivision_level, packed_states, descs, tri_index_buffer);
        if (ok)
        {
            // Upload packed state data
            CUdeviceptr d_omm_data = 0;
            CUdeviceptr d_omm_descs = 0;

            CUDA_CHECK(cudaMalloc((void**)&d_omm_data,
                packed_states.size()));
            CUDA_CHECK(cudaMemcpy((void*)d_omm_data,
                packed_states.data(), packed_states.size(),
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMalloc((void**)&d_omm_descs,
                descs.size() * sizeof(OptixOpacityMicromapDesc)));
            CUDA_CHECK(cudaMemcpy((void*)d_omm_descs,
                descs.data(),
                descs.size() * sizeof(OptixOpacityMicromapDesc),
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMalloc((void**)&object.d_omm_index_buffer,
                tri_index_buffer.size() * sizeof(int32_t)));
            CUDA_CHECK(cudaMemcpy((void*)object.d_omm_index_buffer,
                tri_index_buffer.data(),
                tri_index_buffer.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice));

            // Build OptixOpacityMicromapArray
            OptixOpacityMicromapHistogramEntry histogram{};
            histogram.count = (unsigned int)descs.size();
            histogram.subdivisionLevel = (uint16_t)subdivision_level;
            histogram.format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;

            OptixOpacityMicromapArrayBuildInput omm_array_build_input{};
            omm_array_build_input.flags = OPTIX_OPACITY_MICROMAP_FLAG_NONE;
            omm_array_build_input.inputBuffer = d_omm_data;
            omm_array_build_input.numMicromapHistogramEntries = 1;
            omm_array_build_input.micromapHistogramEntries = &histogram;
            omm_array_build_input.perMicromapDescBuffer = d_omm_descs;
            omm_array_build_input.perMicromapDescStrideInBytes = sizeof(OptixOpacityMicromapDesc);

            OptixMicromapBufferSizes omm_sizes{};
            OPTIX_CHECK(optixOpacityMicromapArrayComputeMemoryUsage(
                context, &omm_array_build_input, &omm_sizes));

            CUdeviceptr d_omm_temp = 0;
            CUDA_CHECK(cudaMalloc((void**)&d_omm_temp, omm_sizes.tempSizeInBytes));
            CUDA_CHECK(cudaMalloc((void**)&object.d_omm_array_output, omm_sizes.outputSizeInBytes));

            OptixMicromapBuffers omm_buffers{};
            omm_buffers.output = object.d_omm_array_output;
            omm_buffers.outputSizeInBytes = omm_sizes.outputSizeInBytes;
            omm_buffers.temp = d_omm_temp;
            omm_buffers.tempSizeInBytes = omm_sizes.tempSizeInBytes;

            OPTIX_CHECK(optixOpacityMicromapArrayBuild(
                context, 0, &omm_array_build_input, &omm_buffers));
            CUDA_CHECK(cudaStreamSynchronize(0));

            // Temp buffers no longer needed after build
            CUDA_CHECK(cudaFree((void*)d_omm_temp));
            CUDA_CHECK(cudaFree((void*)d_omm_data));
            CUDA_CHECK(cudaFree((void*)d_omm_descs));

            //  Describe OMM attachment for GAS build
            object.omm_usage_count = {};
            object.omm_usage_count.count = (unsigned int)descs.size();
            object.omm_usage_count.subdivisionLevel = subdivision_level;
            object.omm_usage_count.format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;

            omm_build_input_attachment.indexingMode =
                OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;
            omm_build_input_attachment.opacityMicromapArray = object.d_omm_array_output;
            omm_build_input_attachment.indexBuffer = object.d_omm_index_buffer;
            omm_build_input_attachment.indexSizeInBytes = sizeof(uint32_t);
            omm_build_input_attachment.numMicromapUsageCounts = 1;
            omm_build_input_attachment.micromapUsageCounts = &object.omm_usage_count;

            has_omm = true;
            DEBUG_LOGF("[OMM] Built array for '%s': %zu triangles",
                object.name.c_str(), descs.size());
        }
    }

    //  GAS build input
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // Vertex Data
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(ColoredVertex);
    build_input.triangleArray.numVertices = (unsigned int)object.vertices.size();
    build_input.triangleArray.vertexBuffers = &object.d_vertices;

    // Index Data
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    build_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(object.indices.size());
    build_input.triangleArray.indexBuffer = object.d_indices;

    // SBT Mapping Data
    build_input.triangleArray.flags = geometryFlags.data();
    build_input.triangleArray.numSbtRecords = static_cast<uint32_t>(geometryFlags.size());
    build_input.triangleArray.sbtIndexOffsetBuffer = object.d_sbt_indices;
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    // Attach OMM if we built one for this object
    if (has_omm) {
        build_input.triangleArray.opacityMicromap = omm_build_input_attachment;
    }

    OptixAccelBuildOptions buildOptions = {};
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
        | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context, &buildOptions, &build_input, 1, &gas_sizes[index]));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer, gas_sizes[index].tempSizeInBytes));
    CUDA_CHECK(cudaMalloc((void**)&object.d_gas_output, gas_sizes[index].outputSizeInBytes));

    // Emit compacted size so we can shrink the GAS after build
    CUdeviceptr d_compact_size;
    CUDA_CHECK(cudaMalloc((void**)&d_compact_size, sizeof(size_t)));
    OptixAccelEmitDesc emit{};
    emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit.result = d_compact_size;

    OPTIX_CHECK(optixAccelBuild(
        context, 0,
        &buildOptions,
        &build_input, 1,
        d_temp_buffer, gas_sizes[index].tempSizeInBytes,
        object.d_gas_output, gas_sizes[index].outputSizeInBytes,
        &object.gas_handle,
        &emit, 1
    ));
    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaFree((void*)d_temp_buffer));

    // Compact the GAS 
    size_t compact_size = 0;
    CUDA_CHECK(cudaMemcpy(&compact_size, (void*)d_compact_size,
        sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree((void*)d_compact_size));

    CUdeviceptr d_compact_output = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_compact_output, compact_size));
    OPTIX_CHECK(optixAccelCompact(context, 0,
        object.gas_handle,
        d_compact_output, compact_size,
        &object.gas_handle));
    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaFree((void*)object.d_gas_output));
    object.d_gas_output = d_compact_output;

    DEBUG_LOGF("[GAS] Built '%s' with %zu materials%s",
        object.name.c_str(), object.materials.size(),
        has_omm ? " + OMM" : "");
}

// Builds instance for each scene object and then builds the IAS referencing those instances
// The IAS acts as TLAS (top-level acceleration structure) that allows ray traversal to reference
// the GAS of each instance and apply the appropriate transformations for each instance
void OptixRenderer::buildIAS() {
    // Create OptixInstance for each scene instance, which describes how to transform the geometry and which GAS to reference for that instance. 
    // The instance data is uploaded to a GPU buffer and used as input for building the IAS.
    std::vector<OptixInstance> instances(scene_instances.size());
    for (size_t i = 0; i < scene_instances.size(); ++i) {
        OptixInstance& instance = instances[i];
        memset(&instance, 0, sizeof(OptixInstance));
        memcpy(instance.transform, scene_instances[i].transform, sizeof(float) * 12); // Copy the 3x4 transform matrix for this instance from corresponding SceneObjectInstance

        instance.instanceId = (unsigned int)i; // Set instance ID to the index of this instance, which can be used in shaders to identify which instance was hit
        instance.visibilityMask = 255; // Set visibility mask to 255 (all bits on) so that this instance is visible to all ray types
        instance.sbtOffset = scene_instances[i].object->sbt_base * RayType::COUNT;; // Set SBT offset for this instance based on the SBT base index assigned during scene loading. This tells OptiX which hit group entries in the SBT correspond to this instance's geometry and materials.
        instance.flags = OPTIX_INSTANCE_FLAG_NONE; // No special flags for this instance
        instance.traversableHandle = scene_instances[i].object->gas_handle; // Set the traversable handle for this instance to point to the GAS built for the corresponding scene object
    }

    // Upload instance data to GPU buffer for IAS build
    size_t instances_size = instances.size() * sizeof(OptixInstance);
    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_instances, instances_size)); // Allocate GPU buffer for instance data
    CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_instances, instances.data(), instances_size, cudaMemcpyHostToDevice)); // Upload instance data to GPU

    // Configure the OptixBuildInput 
    OptixBuildInput ias_build_input = {};
    ias_build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES; // Build input type is instances for IAS
    ias_build_input.instanceArray.instances = device_buffers.d_instances; // Set pointer to instance data on GPU
    ias_build_input.instanceArray.numInstances = static_cast<uint32_t>(instances.size()); // Set number of instances for IAS build

    // Build IAS with the instance data, which creates the top-level acceleration structure that references the GAS for each instance. The resulting IAS handle is stored in ias_handle for use during rendering.
    OptixAccelBuildOptions ias_build_options = {};
    ias_build_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    ias_build_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &ias_build_options, &ias_build_input, 1, &ias_buffer_sizes)); // Compute memory usage for IAS build
    CUdeviceptr d_ias_temp_buffer; // Temporary buffer for IAS build
    CUDA_CHECK(cudaMalloc((void**)&d_ias_temp_buffer, ias_buffer_sizes.tempSizeInBytes)); // Allocate temporary buffer for IAS build
    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_ias_output_buffer, ias_buffer_sizes.outputSizeInBytes)); // Allocate output buffer for IAS build
    OPTIX_CHECK(optixAccelBuild(
        context,
        0,
        &ias_build_options,
        &ias_build_input,
        1,
        d_ias_temp_buffer,
        ias_buffer_sizes.tempSizeInBytes,
        device_buffers.d_ias_output_buffer,
        ias_buffer_sizes.outputSizeInBytes,
        &ias_handle,
        nullptr,
        0)
    ); // Build IAS and store handle in ias_handle

    CUDA_CHECK(cudaStreamSynchronize(0)); // Ensure IAS build is complete before freeing temp buffer
    CUDA_CHECK(cudaFree((void*)d_ias_temp_buffer)); // Free temporary buffer after IAS build
    DEBUG_LOGF("[IAS] Built with %zu instances", instances.size());
}


// Builds SBT constaining one hit record per material
// Each record points to either HitGroupRecordCookTorrance or HitGroupDataGlass
void OptixRenderer::buildSBT() {
    size_t max_stride = (sizeof(HitGroupRecordCookTorrance) > sizeof(HitGroupRecordGlass))
        ? sizeof(HitGroupRecordCookTorrance) : sizeof(HitGroupRecordGlass);

    max_stride = ((max_stride + OPTIX_SBT_RECORD_ALIGNMENT - 1) / OPTIX_SBT_RECORD_ALIGNMENT) * OPTIX_SBT_RECORD_ALIGNMENT;

    // Count total materials first
    size_t total_materials = 0;
    for (const auto& obj : loaded_scene_objects) {
        total_materials += obj.materials.size();
    }

    // Size = Total Materials * 2 (Radiance + Occlusion) * Size of largest record
    std::vector<char> hit_records(total_materials * RayType::COUNT * max_stride, 0);
    int hit_record_count = 0;

    for (const auto& scene_object : loaded_scene_objects) {
        for (size_t i = 0; i < scene_object.materials.size(); ++i) {
            const auto& material = scene_object.materials[i];

            // Base SBT index multiplied by RayType::COUNT to leave gaps for the second ray type
            uint32_t base_sbt_index = (scene_object.sbt_base + (uint32_t)i) * RayType::COUNT;

            // -------------------------------------------------------------
            // SLOT A: Pack the RADIANCE Record (RayType::RADIANCE = 0)
            // -------------------------------------------------------------
            uint32_t radiance_sbt_index = base_sbt_index + RayType::RADIANCE;

            if (material.is_glass) {
                HitGroupRecordGlass record = {};
                OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_glass_program_group, &record));
                record.data.vertices = (ColoredVertex*)scene_object.d_vertices;
                record.data.indices = (uint3*)scene_object.d_indices;
                record.data.tint = material.albedo;
                record.data.emission = material.emission;
                record.data.emission_texture = material.texture_paths.emissive_path.empty() ? 0 : loadTextureCached(material.texture_paths.emissive_path);
                record.data.tint_texture = material.texture_paths.diffuse_path.empty() ? 0 : loadTextureCached(material.texture_paths.diffuse_path);
                record.data.refraction_index = material.ior;

                std::memcpy(hit_records.data() + radiance_sbt_index * max_stride, &record, sizeof(HitGroupRecordGlass));
            }
            else {
                HitGroupRecordCookTorrance record = {};
                OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_cooktorrance_program_group, &record));
                record.data.vertices = (ColoredVertex*)scene_object.d_vertices;
                record.data.indices = (uint3*)scene_object.d_indices;
                record.data.base_color = material.base_color;
                record.data.emission = material.emission;
                record.data.albedo_texture = material.texture_paths.diffuse_path.empty() ? 0 : loadTextureCached(material.texture_paths.diffuse_path);
                record.data.emission_texture = material.texture_paths.emissive_path.empty() ? 0 : loadTextureCached(material.texture_paths.emissive_path);
                record.data.alpha_texture = material.texture_paths.alpha_path.empty() ? 0 : loadTextureCached(material.texture_paths.alpha_path);

                record.data.roughness = material.roughness;
                record.data.metallic = material.metallic;
                record.data.specular_color = material.specular;
                record.data.metallic_texture = 0;
                record.data.roughness_texture = 0;
                record.data.specular_texture = 0;

                std::memcpy(hit_records.data() + radiance_sbt_index * max_stride, &record, sizeof(HitGroupRecordCookTorrance));
            }
            hit_record_count++;

            // -------------------------------------------------------------
            // SLOT B: Pack the OCCLUSION Record (RayType::OCCLUSION = 1)
            // -------------------------------------------------------------
            uint32_t occlusion_sbt_index = base_sbt_index + RayType::OCCLUSION;

            HitGroupRecordCookTorrance occ_record = {};

            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_occlusion_program_group, &occ_record));

            // Pass geometry data so the Any-Hit shader can calculate UVs for the alpha mask
            occ_record.data.vertices = (ColoredVertex*)scene_object.d_vertices;
            occ_record.data.indices = (uint3*)scene_object.d_indices;
            occ_record.data.alpha_texture = material.texture_paths.alpha_path.empty() ? 0 : loadTextureCached(material.texture_paths.alpha_path);

            std::memcpy(hit_records.data() + occlusion_sbt_index * max_stride, &occ_record, sizeof(HitGroupRecordCookTorrance));
            hit_record_count++;
        }
    }

    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_hg, hit_records.size()));
    CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_hg, hit_records.data(), hit_records.size(), cudaMemcpyHostToDevice));

    // --- Raygen Record ---
    RayGenRecord raygen_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_program_group, &raygen_record));
    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_rg, sizeof(RayGenRecord)));
    CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_rg, &raygen_record, sizeof(raygen_record), cudaMemcpyHostToDevice));

    // --- Miss Records ---
    std::vector<MissRecord> miss_records(RayType::COUNT);

    // Radiance Miss (Slot 0)
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_program_group, &miss_records[RayType::RADIANCE]));

    // Occlusion Miss (Slot 1)
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_occlusion_program_group, &miss_records[RayType::OCCLUSION]));

    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_ms, sizeof(MissRecord) * RayType::COUNT));
    CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_ms, miss_records.data(), sizeof(MissRecord) * RayType::COUNT, cudaMemcpyHostToDevice));

    // Setup SBT
    sbt.raygenRecord = device_buffers.d_rg;

    // Miss configuration
    sbt.missRecordBase = device_buffers.d_ms;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = RayType::COUNT;

    // Hitgroup configuration
    sbt.hitgroupRecordBase = device_buffers.d_hg;
    sbt.hitgroupRecordStrideInBytes = max_stride;
    sbt.hitgroupRecordCount = hit_record_count;

    DEBUG_LOG("[SBT] Built with Radiance and Occlusion ray types.");
}

// Setup programs, module, pipeline, and SBT for the renderer.
// Runs once per scene load
void OptixRenderer::setupShaders() {
    createModuleAndProgramGroups();
    createPipeline();
    buildSBT();
}

// Mainly setups parameters for the params struct
// Runs once per scene load
void OptixRenderer::setupLighting() {

    // Path tracer settings
    params.light_intensity = 1.f;
    params.max_bounce_depth = 8;   // Start with x bounces
    params.rr_start_depth = 3;      // Start Russian Roulette after x bounces
    params.samples_per_pixel = 1;  // Progressive sampling
    params.current_frame = 0;
    params.current_sample = 0;

    params.envmap.scale = current_scene_data.envmap_scale;
    params.envmap.exposure = current_scene_data.envmap_exposure;
}

// Render the scene using the provided camera and number of samples per pixel. 
// This function sets up the necessary parameters, allocates buffers if needed, and launches the OptiX pipeline to perform ray tracing.
// Runs every frame
void OptixRenderer::render(const Camera& camera, int samples_per_pixel) {
    // Allocate buffers if not already done
    if (!device_buffers.d_pixels) {
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_pixels, params.width * params.height * sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_accum_buffer, params.width * params.height * sizeof(float4)));
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_albedo_buffer, params.width * params.height * sizeof(float4)));
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_normal_buffer, params.width * params.height * sizeof(float4)));
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_params, sizeof(Params)));

        params.image = (uchar4*)device_buffers.d_pixels;
        params.accum_buffer = (float4*)device_buffers.d_accum_buffer;
        params.albedo_buffer = (float4*)device_buffers.d_albedo_buffer;
        params.normal_buffer = (float4*)device_buffers.d_normal_buffer;
    }

    params.camera = camera;
    params.traversable = ias_handle;
    params.samples_per_pixel = samples_per_pixel;

    CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_params, &params, sizeof(Params), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(
        pipeline,
        0,
        device_buffers.d_params,
        sizeof(Params),
        &sbt,
        params.width,
        params.height,
        1)
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    params.current_sample++;
    params.current_frame++;
}

// Update the environment map scale and exposure parameters in the renderer's Params structure,
// which are used during rendering to adjust the appearance of the environment map.
void OptixRenderer::updateEnvmapParameters(float scale, float exposure) {
    params.envmap.scale = scale;
    params.envmap.exposure = exposure;
}

// Update the light intensity parameter in the renderer's Params structure, 
// which is used during rendering to scale the contribution of light sources in the scene.
void OptixRenderer::updateLightIntensity(float light_intensity) {
    params.light_intensity = light_intensity;
}

// Reset the accumulation buffer to zero and reset the current sample count to 0, 
// effectively restarting the progressive rendering process.
void OptixRenderer::resetBuffersOnCameraUpdate() {
    if (device_buffers.d_accum_buffer) {
        CUDA_CHECK(cudaMemset((void*)device_buffers.d_accum_buffer, 0, params.width * params.height * sizeof(float4)));
    }
    if (device_buffers.d_albedo_buffer) {
        CUDA_CHECK(cudaMemset((void*)device_buffers.d_albedo_buffer, 0, params.width * params.height * sizeof(float4)));
    }
    if (device_buffers.d_normal_buffer) {
        CUDA_CHECK(cudaMemset((void*)device_buffers.d_normal_buffer, 0, params.width * params.height * sizeof(float4)));
    }
    params.current_sample = 0;
}

// Cleanup all OptiX resources, including pipeline, context, denoiser, textures, environment maps, and device buffers.
void OptixRenderer::cleanup() {
    if (denoiser != nullptr) OPTIX_CHECK(optixDenoiserDestroy(denoiser));
    if (pipeline != nullptr) OPTIX_CHECK(optixPipelineDestroy(pipeline));
    if (context != nullptr)  OPTIX_CHECK(optixDeviceContextDestroy(context));
    //if (stream != nullptr)   CUDA_CHECK(cudaStreamDestroy(stream));

    FreeTexturesAndEnvMaps();
    FreeDeviceBuffers();
}

// Free all textures and environment map resources, including CUDA texture objects and arrays.
void OptixRenderer::FreeTexturesAndEnvMaps() {
    // Clean up textures
    for (auto& mat_tex : texture_cache) {
        if (mat_tex.second.tex) {
            CUDA_CHECK(cudaDestroyTextureObject(mat_tex.second.tex));
        }
        if (mat_tex.second.array) {
            CUDA_CHECK(cudaFreeArray(mat_tex.second.array));
        }
    }
    texture_cache.clear();

    // Clean up envmap CDF arrays
    if (envmap_cdf_marginal_array != nullptr) {
        CUDA_CHECK(cudaFreeArray(envmap_cdf_marginal_array));
        envmap_cdf_marginal_array = nullptr;
    }
    if (envmap_cdf_conditional_array != nullptr) {
        CUDA_CHECK(cudaFreeArray(envmap_cdf_conditional_array));
        envmap_cdf_conditional_array = nullptr;
    }
}

// Free all device buffers associated with loaded scene objects and renderer resources, including vertex/index buffers,
// SBT indices, GAS outputs, pixel buffers, accumulation buffers, and parameter buffers.
void OptixRenderer::FreeDeviceBuffers() {
    for (LoadedSceneObject object : loaded_scene_objects) {
        CUDA_CHECK(cudaFree((void*)object.d_vertices));          object.d_vertices = 0;
        CUDA_CHECK(cudaFree((void*)object.d_indices));           object.d_indices = 0;
        CUDA_CHECK(cudaFree((void*)object.d_sbt_indices));       object.d_sbt_indices = 0;
        CUDA_CHECK(cudaFree((void*)object.d_gas_output));        object.d_gas_output = 0;
        // Free OMM buffers if this object had OMM built for it.
        // These must outlive the GAS - free them only here during full cleanup.
        if (object.d_omm_array_output) {
            CUDA_CHECK(cudaFree((void*)object.d_omm_array_output));
            object.d_omm_array_output = 0;
        }
        if (object.d_omm_index_buffer) {
            CUDA_CHECK(cudaFree((void*)object.d_omm_index_buffer));
            object.d_omm_index_buffer = 0;
        }
    }

    auto freeAndNull = [](CUdeviceptr& ptr) {
        CUDA_CHECK(cudaFree((void*)ptr));
        ptr = 0;
        };

    freeAndNull(device_buffers.d_pixels);
    freeAndNull(device_buffers.d_accum_buffer);
    freeAndNull(device_buffers.d_albedo_buffer);
    freeAndNull(device_buffers.d_normal_buffer);
    freeAndNull(device_buffers.d_params);
    freeAndNull(device_buffers.d_ias_output_buffer);
    freeAndNull(device_buffers.d_instances);
    freeAndNull(device_buffers.d_hg);
    freeAndNull(device_buffers.d_rg);
    freeAndNull(device_buffers.d_ms);

    if (params.emissive_triangles) {
        CUDA_CHECK(cudaFree((void*)params.emissive_triangles));
        params.emissive_triangles = nullptr;
    }
    params.num_emissive_triangles = 0;
    params.total_emissive_weight = 0.f;
}

// Helper function to create a 3x4 transform matrix
void OptixRenderer::createTransformMatrix(float(&transform)[12], float3 translation, float3 scale,
    float rotation_x = 0.0f, float rotation_y = 0.0f, float rotation_z = 0.0f)
{
    // Precompute sin/cos for each axis
    float cx = cosf(rotation_x), sx = sinf(rotation_x);
    float cy = cosf(rotation_y), sy = sinf(rotation_y);
    float cz = cosf(rotation_z), sz = sinf(rotation_z);

    // Combined rotation matrix R = Ry * Rx * Rz
    // Each element is the dot product of the combined basis vectors
    float r00 = cy * cz + sy * sx * sz;   float r01 = -cy * sz + sy * sx * cz;  float r02 = sy * cx;
    float r10 = cx * sz;                  float r11 = cx * cz;                  float r12 = -sx;
    float r20 = -sy * cz + cy * sx * sz;  float r21 = sy * sz + cy * sx * cz;   float r22 = cy * cx;

    // Row-major 3x4 [R*Scale | T]
    transform[0] = r00 * scale.x;  transform[1] = r01 * scale.y;  transform[2] = r02 * scale.z;  transform[3] = translation.x;
    transform[4] = r10 * scale.x;  transform[5] = r11 * scale.y;  transform[6] = r12 * scale.z;  transform[7] = translation.y;
    transform[8] = r20 * scale.x;  transform[9] = r21 * scale.y;  transform[10] = r22 * scale.z;  transform[11] = translation.z;
}

// Load a binary file into a vector of chars. Throws an exception if the file cannot be opened.
std::vector<char> OptixRenderer::loadFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Failed to open file: " + path);

    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    f.read(data.data(), size);
    return data;
}

// Update camera parameters and reset accumulation buffer if camera moved since the last frame.
void OptixRenderer::updateCamera(const Camera& camera) {
    // Check if camera has moved significantly
    float3 pos_diff = camera.origin - last_camera.origin;
    float pos_distance = sqrtf(pos_diff.x * pos_diff.x + pos_diff.y * pos_diff.y + pos_diff.z * pos_diff.z);

    // Check if camera direction changed significantly
    float3 dir_diff = camera.lower_left_corner - last_camera.lower_left_corner;
    float dir_distance = sqrtf(dir_diff.x * dir_diff.x + dir_diff.y * dir_diff.y + dir_diff.z * dir_diff.z);

    if (pos_distance > 0.001f || dir_distance > 0.001f) {
        // Camera moved - reset accumulation
        resetBuffersOnCameraUpdate();
    }
    last_camera = camera;
}

// Switch to a different scene by loading its configuration, cleaning up old resources, and rebuilding the scene's geometry and SBT.
void OptixRenderer::switchScene(SceneID scene_id) {
    if (scene_id == current_scene_id) {
        return;  // Already on this scene
    }

    // Get new scene configuration
    current_scene_data = SceneManager::getSceneConfig(scene_id);
    current_scene_id = scene_id;

    // Clean up old geometry and textures
    FreeTexturesAndEnvMaps();
    FreeDeviceBuffers();
    ias_handle = 0;

    loadScene(current_scene_id);
    buildSBT();

    DEBUG_LOGF("[Scene] Switched to: %s with %zu objects.",
        current_scene_data.name.c_str(),
        current_scene_data.objects.size());
}


// Denoiser setup function to initialize OptiX denoiser with specified options and allocate necessary GPU buffers for denoising operations. 
// This function creates an OptiX denoiser, computes memory requirements, allocates GPU memory for the denoiser state and scratch buffers, and sets up the denoiser state for use in rendering.
// Runs once during renderer initialization to prepare for denoising operations on rendered images.
void OptixRenderer::setupDenoiser() {
    const OptixDenoiserOptions denoiser_options = {
        .guideAlbedo = 1,  // Will expect use guide albedo for better results (added later in OptixDenoiserGuideLayer)
        .guideNormal = 1,  // Will expect use guide normal for better results (added later in OptixDenoiserGuideLayer)
        .denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY  // Ignores aplha channel
    };

    OPTIX_CHECK(optixDenoiserCreate(   // Create an OptiX denoiser with the specified options and store handle in denoiser variable
        context,                       // Optix device context
        OPTIX_DENOISER_MODEL_KIND_HDR, // Which denoise model to use
        &denoiser_options,             // Denoiser options
        &denoiser));                   // Output denoiser handle


    OPTIX_CHECK(optixDenoiserComputeMemoryResources(  // Compute the memory requirements for the denoiser state and scratch buffers based on the image dimensions and store in denoiser_sizes
        denoiser,          // Denoiser handle
        params.width,      // Width of the images to be denoised
        params.height,     // Height of the images to be denoised
        &denoiser_sizes)); // Store the computed memory requirements for the denoiser state and scratch buffers in denoiser_sizes

    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_denoiser_state, denoiser_sizes.stateSizeInBytes));                   // Allocate GPU memory for the denoiser state buffer based on the computed size in denoiser_sizes
    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_denoiser_scratch, denoiser_sizes.withoutOverlapScratchSizeInBytes)); // Allocate GPU memory for the denoiser scratch buffer based on the computed size in denoiser_sizes

    OPTIX_CHECK(optixDenoiserSetup( // Setup the denoiser state with the specified image dimensions and allocated buffers
        denoiser,                   // Denoiser handle
        0,                          // Cuda stream
        params.width,               // Width of the images to be denoised
        params.height,              // Height of the images to be denoised
        device_buffers.d_denoiser_state,                 // Pointer to the allocated GPU memory for the denoiser state buffer
        denoiser_sizes.stateSizeInBytes,                 // Size of the denoiser state buffer in bytes
        device_buffers.d_denoiser_scratch,               // Pointer to the allocated GPU memory for the denoiser scratch buffer
        denoiser_sizes.withoutOverlapScratchSizeInBytes) // Size of the denoiser scratch buffer in bytes
    );

    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_denoised_buffer, params.width * params.height * sizeof(float4)));
}

void OptixRenderer::runDenoiser(float blendFactor) {
    //// Describe input layers
    OptixDenoiserGuideLayer guide_layer{};
    guide_layer.albedo.data = (CUdeviceptr)device_buffers.d_albedo_buffer;
    guide_layer.albedo.width = getWidth();
    guide_layer.albedo.height = getHeight();
    guide_layer.albedo.rowStrideInBytes = getWidth() * sizeof(float4);
    guide_layer.albedo.pixelStrideInBytes = sizeof(float4);
    guide_layer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    guide_layer.normal.data = (CUdeviceptr)device_buffers.d_normal_buffer;
    guide_layer.normal.width = getWidth();
    guide_layer.normal.height = getHeight();
    guide_layer.normal.rowStrideInBytes = getWidth() * sizeof(float4);
    guide_layer.normal.pixelStrideInBytes = sizeof(float4);
    guide_layer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // Describe the noisy beauty input and denoised output
    OptixDenoiserLayer color_layer{};
    color_layer.input.data = (CUdeviceptr)device_buffers.d_accum_buffer; // your HDR accum
    color_layer.input.width = getWidth();
    color_layer.input.height = getHeight();
    color_layer.input.rowStrideInBytes = getWidth() * sizeof(float4);
    color_layer.input.pixelStrideInBytes = sizeof(float4);
    color_layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    color_layer.output.data = (CUdeviceptr)device_buffers.d_denoised_buffer; // separate output
    color_layer.output.width = getWidth();
    color_layer.output.height = getHeight();
    color_layer.output.rowStrideInBytes = getWidth() * sizeof(float4);
    color_layer.output.pixelStrideInBytes = sizeof(float4);
    color_layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // Run denoiser
    OptixDenoiserParams denoiser_params{};
    denoiser_params.blendFactor = blendFactor; // 0 = full denoiser output, 1 = full noisy input

    OPTIX_CHECK(optixDenoiserInvoke(
        denoiser,
        0,
        &denoiser_params,
        device_buffers.d_denoiser_state,
        denoiser_sizes.stateSizeInBytes,
        &guide_layer,
        &color_layer,
        1,                  // one color layer
        0, 0,               // offset x, y (0 for full image)
        device_buffers.d_denoiser_scratch,
        denoiser_sizes.withoutOverlapScratchSizeInBytes)
    );
}

void OptixRenderer::postprocessAccum() {
    launchTonemapKernel(
        reinterpret_cast<float4*>(device_buffers.d_accum_buffer),
        reinterpret_cast<uchar4*>(device_buffers.d_pixels),
        params.width,
        params.height,
        0
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

void OptixRenderer::postprocessDenoised() {
    launchTonemapKernel(
        reinterpret_cast<float4*>(device_buffers.d_denoised_buffer),
        reinterpret_cast<uchar4*>(device_buffers.d_pixels),
        params.width,
        params.height,
        0
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}