#include "renderer.h"
#include <iostream>
#include <vector>
#include <cuda_gl_interop.h>


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
    //CUDA_CHECK(cudaStreamCreate(&stream));
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
}

// Loads a texture from disk if not already loaded, and returns a CUDA texture object handle.
cudaTextureObject_t OptixRenderer::loadTextureCached(const std::string& resolved_path)
{
    // Already loaded — return existing handle
    auto it = texture_cache.find(resolved_path);
    if (it != texture_cache.end()) {
        printf("[Texture] Cache hit: %s\n", resolved_path.c_str());
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
    printf("[Texture] Loaded: %s (%dx%d)\n", resolved_path.c_str(), w, h);
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
        std::cerr << "[Env] Failed to load: " << path << " — " << stbi_failure_reason() << "\n";
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
    // 1. Full vertex buffer — read by shaders via sbt->vertices
    //    AND used by OptiX GAS build via stride
    {
        size_t bytes = obj.vertices.size() * sizeof(ColoredVertex);
        CUDA_CHECK(cudaMalloc((void**)&obj.d_vertices, bytes));
        CUDA_CHECK(cudaMemcpy((void*)obj.d_vertices,
            obj.vertices.data(), bytes, cudaMemcpyHostToDevice));
    }

    // 2. Index buffer — triangles
    {
        size_t bytes = obj.indices.size() * sizeof(uint3);
        CUDA_CHECK(cudaMalloc((void**)&obj.d_indices, bytes));
        CUDA_CHECK(cudaMemcpy((void*)obj.d_indices,
            obj.indices.data(), bytes, cudaMemcpyHostToDevice));
    }

    // 3. SBT index buffer — one uint32 per triangle, routes to material
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
    pipeline_compile_options.numPayloadValues = 23;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.pipelineLaunchParamsSizeInBytes = sizeof(Params);

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
        hitgroup_cooktorrance_program_group, 
        hitgroup_glass_program_group 
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    //pipeline_link_options.maxTraceDepth = 7;
    pipeline_link_options.maxTraceDepth = 1;

    char log[4096];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(optixPipelineCreate(
        context, 
        &pipeline_compile_options, 
        &pipeline_link_options, 
        groups,
        4, 
        log, 
        &logSize, 
        &pipeline
    ));
    DEBUG_LOG("[OptiX] Pipeline created");
}

// Builds a GAS for the given scene object using its vertex and index buffers, and stores the resulting traversable handle in the object.
void OptixRenderer::buildGAS(LoadedSceneObject& object, int index) {
    std::vector<uint32_t> geometryFlags(object.materials.size());

	for (size_t i = 0; i < object.materials.size(); ++i) {  // Set flag to disable any-hit shader for materials without alpha texture, no flags for the rest 
        if (object.materials[i].texture_paths.alpha_path.empty()) {
            geometryFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            continue;
        }
        geometryFlags[i] = OPTIX_GEOMETRY_FLAG_NONE;
    }

	OptixBuildInput build_input = {};
	build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	// Vertex Data: using the full ColoredVertex buffer for both GAS build and shader access, relying on stride to skip unused data during build
	build_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
	build_input.triangleArray.vertexStrideInBytes = sizeof(ColoredVertex);
	build_input.triangleArray.numVertices         = (unsigned int)object.vertices.size();
	build_input.triangleArray.vertexBuffers       = &object.d_vertices;

	// Index Data
	build_input.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
	build_input.triangleArray.numIndexTriplets   = static_cast<uint32_t>(object.indices.size());
	build_input.triangleArray.indexBuffer        = object.d_indices;

	// SBT Mapping Data: one uint32 per triangle, used to route to material in shader
	build_input.triangleArray.flags         = geometryFlags.data(); 
	build_input.triangleArray.numSbtRecords = static_cast<uint32_t>(geometryFlags.size());

	// SBT index buffer describes which SBT record (material) each triangle uses
	build_input.triangleArray.sbtIndexOffsetBuffer        = object.d_sbt_indices;
	build_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof(uint32_t);
	build_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

	OptixAccelBuildOptions buildOptions = {};
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
	buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context, 
        &buildOptions, 
        &build_input, 
        1, 
        &gas_sizes[index]
        )
    );

	CUdeviceptr d_temp_buffer;  // Temporary buffer for GAS build
	CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer, gas_sizes[index].tempSizeInBytes));  // Allocate temporary buffer for GAS build
	CUDA_CHECK(cudaMalloc((void**)&object.d_gas_output, gas_sizes[index].outputSizeInBytes));  // Allocate output buffer for GAS
	OPTIX_CHECK(optixAccelBuild(  // Build GAS for this object
        context,
        0, 
        &buildOptions, 
        &build_input, 
        1,
        d_temp_buffer, 
        gas_sizes[index].tempSizeInBytes,
        object.d_gas_output, 
        gas_sizes[index].outputSizeInBytes,
        &object.gas_handle, 
        nullptr, 
        0
    ));
    CUDA_CHECK(cudaStreamSynchronize(0)); // Ensure build is complete before freeing temp buffer
	CUDA_CHECK(cudaFree((void*)d_temp_buffer));  // Free temporary buffer after build
	DEBUG_LOGF("[GAS] Built '%s' with %zu materials", object.name.c_str(), object.materials.size());
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
        instance.sbtOffset = scene_instances[i].object->sbt_base; // Set SBT offset for this instance based on the SBT base index assigned during scene loading. This tells OptiX which hit group entries in the SBT correspond to this instance's geometry and materials.
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
    const size_t max_stride = (sizeof(HitGroupRecordCookTorrance) > sizeof(HitGroupRecordGlass))
        ? sizeof(HitGroupRecordCookTorrance) : sizeof(HitGroupRecordGlass);

    // Count total materials first
    size_t total_materials = 0;
    for (const auto& obj : loaded_scene_objects)
        total_materials += obj.materials.size();

    std::vector<char> hit_records(total_materials * max_stride, 0);
	int hit_record_count = 0;

    for (const auto& scene_object : loaded_scene_objects) {
        for (size_t i = 0; i < scene_object.materials.size(); ++i) {
            const auto& material = scene_object.materials[i];

			uint32_t sbt_index = scene_object.sbt_base + (uint32_t)i; // Calculate the SBT index for this material based on the object's SBT base and material index

            if (material.is_glass) {
                HitGroupRecordGlass record;
                OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_glass_program_group, &record));
                record.data.vertices = (ColoredVertex*)scene_object.d_vertices; // Set vertex buffer pointer for this record to the vertex buffer of the corresponding scene object
                record.data.indices = (uint3*)scene_object.d_indices; // Set index buffer pointer for this record to the index buffer of the corresponding scene object
                record.data.tint = material.albedo; // Set albedo color for lambert material in this record
                record.data.emission = material.emission; // Set emission color for this material
                if (!material.texture_paths.emissive_path.empty()) {
                    record.data.emission_texture = loadTextureCached(material.texture_paths.emissive_path); // Load the emission texture for this material and set the texture handle in the record
                }
                else {
                    record.data.emission_texture = 0; // If no emission texture, set texture handle to 0
                }
                if (!material.texture_paths.diffuse_path.empty()) {
                    record.data.tint_texture = loadTextureCached(material.texture_paths.diffuse_path); 
                }
                else {
                    record.data.tint_texture = 0; // If no tint texture, set texture handle to 0
                }
                record.data.refraction_index = material.ior; // Set refraction index for glass material

				hit_records.resize(hit_records.size() + max_stride);
				std::memcpy(hit_records.data() + sbt_index * max_stride, &record, sizeof(HitGroupRecordGlass)); // Copy the hit group record data into the correct position in the hit_records vector based on the calculated SBT index
                ++hit_record_count;
            }
            else {
                HitGroupRecordCookTorrance record;
                OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_cooktorrance_program_group, &record));
                record.data.vertices = (ColoredVertex*)scene_object.d_vertices; // Set vertex buffer pointer for this record to the vertex buffer of the corresponding scene object
                record.data.indices = (uint3*)scene_object.d_indices; // Set index buffer pointer for this record to the index buffer of the corresponding scene object
                record.data.base_color = material.base_color; // Set base color for Cook-Torrance material in this record
                record.data.emission = material.emission; // Set emission color for this material

                if (!material.texture_paths.diffuse_path.empty()) {
					record.data.albedo_texture = loadTextureCached(material.texture_paths.diffuse_path); // Load the diffuse texture for this material and set the texture handle in the record
                }
                else {
					record.data.albedo_texture = 0; // If no diffuse texture, set texture handle to 0
                }

                if (!material.texture_paths.emissive_path.empty()) {
					record.data.emission_texture = loadTextureCached(material.texture_paths.emissive_path); // Load the emission texture for this material and set the texture handle in the record
                }
                else {
					record.data.emission_texture = 0; // If no emission texture, set texture handle to 0
                }

                if (!material.texture_paths.alpha_path.empty()) { // Adjust to your actual path string name
                    record.data.alpha_texture = loadTextureCached(material.texture_paths.alpha_path);
                }
                else {
                    record.data.alpha_texture = 0; // 0 means not present / fully opaque material
                }

                record.data.roughness = material.roughness;
                record.data.metallic  = material.metallic;
                record.data.specular_color = material.specular;

				record.data.metallic_texture = 0;
                record.data.roughness_texture = 0;
                record.data.specular_texture = 0;
                //record.data.albedo_texture = 0; // TODO REMOVE 


                hit_records.resize(hit_records.size() + max_stride);
                std::memcpy(hit_records.data() + sbt_index * max_stride, &record, sizeof(HitGroupRecordCookTorrance));
                ++hit_record_count;
            }
        }
    }

    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_hg, hit_records.size()));
    CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_hg, hit_records.data(), hit_records.size(), cudaMemcpyHostToDevice));

    RayGenRecord raygen_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_program_group, &raygen_record));
    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_rg, sizeof(RayGenRecord)));
    CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_rg, &raygen_record, sizeof(raygen_record), cudaMemcpyHostToDevice));

    MissRecord miss_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_program_group, &miss_record));
    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_ms, sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_ms, &miss_record, sizeof(miss_record), cudaMemcpyHostToDevice));

    // Setup SBT
    sbt.raygenRecord = device_buffers.d_rg;
    sbt.missRecordBase = device_buffers.d_ms;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = device_buffers.d_hg;
    sbt.hitgroupRecordStrideInBytes = max_stride;
    sbt.hitgroupRecordCount = hit_record_count;

    DEBUG_LOG("[SBT] Built");

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
    params.current_sample = 0;
    params.random_seed = 1415;

    params.envmap.scale = 1.0f;
    params.envmap.exposure = 0.0f;

    //DEBUG_LOGF("[Lighting] Setup complete %d lights", params.num_lights);
}

// Render the scene using the provided camera and number of samples per pixel. 
// This function sets up the necessary parameters, allocates buffers if needed, and launches the OptiX pipeline to perform ray tracing.
// Runs every frame
void OptixRenderer::render(const Camera& camera, int samples_per_pixel) {
    // Allocate buffers if not already done
    if (!device_buffers.d_pixels) {
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_pixels,          params.width * params.height * sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_accum_buffer,    params.width * params.height * sizeof(float4)));
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_albedo_buffer,   params.width * params.height * sizeof(float4)));
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_normal_buffer,   params.width * params.height * sizeof(float4)));
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_params, sizeof(Params)));

        params.image = (uchar4*)device_buffers.d_pixels;
        params.accum_buffer = (float4*)device_buffers.d_accum_buffer;
        params.albedo_buffer = (float4*)device_buffers.d_albedo_buffer;
        params.normal_buffer = (float4*)device_buffers.d_normal_buffer;
    }

    params.camera = camera;
    params.traversable = ias_handle;
    params.samples_per_pixel = samples_per_pixel;
    params.random_seed = params.random_seed * 1103515245 + 12345;

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
    if (pipeline != nullptr) OPTIX_CHECK(optixPipelineDestroy(pipeline));
    if (context != nullptr)  OPTIX_CHECK(optixDeviceContextDestroy(context));
    //if (stream != nullptr)   CUDA_CHECK(cudaStreamDestroy(stream));

    FreeTexturesAndEnvMaps();
    FreeDeviceBuffers();

	if (denoiser != nullptr) OPTIX_CHECK(optixDenoiserDestroy(denoiser));
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
        CUDA_CHECK(cudaFree((void*)object.d_vertices));    object.d_vertices = 0;
        CUDA_CHECK(cudaFree((void*)object.d_indices));     object.d_indices = 0;
        CUDA_CHECK(cudaFree((void*)object.d_sbt_indices)); object.d_sbt_indices = 0;
        CUDA_CHECK(cudaFree((void*)object.d_gas_output));  object.d_gas_output = 0;
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

    optixDenoiserInvoke(
        denoiser, 
        0,
        &denoiser_params,
        device_buffers.d_denoiser_state,
        denoiser_sizes.stateSizeInBytes,
        &guide_layer,
        &color_layer, 
        1,    // one color layer
        0, 0,               // offset x, y (0 for full image)
        device_buffers.d_denoiser_scratch,
        denoiser_sizes.withoutOverlapScratchSizeInBytes);

    // //Then tonemap d_denoised_buffer -> your display output
    // //(run your ACES + sRGB kernel on the denoised HDR output)
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