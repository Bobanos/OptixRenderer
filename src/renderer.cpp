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
typedef Record<HitGroupDataLambert> HitGroupRecordLambert;
typedef Record<HitGroupDataGlass> HitGroupRecordGlass;



OptixRenderer::OptixRenderer(int width, int height) {
    params.width = width;
    params.height = height;
}

OptixRenderer::~OptixRenderer() {
    cleanup();
}

void OptixRenderer::initCUDA() {
    CUDA_CHECK(cudaFree(nullptr));
    DEBUG_LOG("[CUDA] Initialized");
}

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

void OptixRenderer::loadScene(SceneID scene_id) {
    current_scene_data = SceneManager::getSceneConfig(scene_id);
    current_scene_id = scene_id;

    // Load all objects in the scene and merge them
    MergedObjMesh combined_mesh;
    object_material_bases.clear();

    for (size_t i = 0; i < current_scene_data.objects.size(); ++i) {
        const auto& obj_config = current_scene_data.objects[i];

        // Record material base BEFORE merging
        object_material_bases.push_back((uint32_t)combined_mesh.materials.size());

        auto meshes = loadObj(obj_config.obj_path, obj_config.mtl_path);
        if (meshes.empty()) {
            throw std::runtime_error("Failed to load object: " + obj_config.name);
        }

        auto obj_mesh = mergeObjMeshes(meshes);

        // Merge this object into combined mesh
        uint32_t vertex_base = (uint32_t)combined_mesh.vertices.size();
        uint32_t material_base = (uint32_t)combined_mesh.materials.size();

        // Copy vertices
        for (const auto& v : obj_mesh.vertices) {
            combined_mesh.vertices.push_back(v);
        }

        // Copy indices and remap material indices
        for (const auto& tri : obj_mesh.indices) {
            combined_mesh.indices.push_back(make_uint3(
                tri.x + vertex_base,
                tri.y + vertex_base,
                tri.z + vertex_base
            ));
        }

        // Copy SBT index buffer with material offset
        for (uint32_t sbt_idx : obj_mesh.sbt_index_buffer) {
            combined_mesh.sbt_index_buffer.push_back(sbt_idx + material_base);
        }

        // Copy materials
        for (const auto& mat : obj_mesh.materials) {
            combined_mesh.materials.push_back(mat);
        }
    }

    merged_mesh = combined_mesh;

    // Initialize transform state for each object
    object_transforms.clear();
    object_transforms.resize(current_scene_data.objects.size());
    for (size_t i = 0; i < current_scene_data.objects.size(); ++i) {
        object_transforms[i].rotation_x = current_scene_data.objects[i].rotation_x;
        object_transforms[i].rotation_y = current_scene_data.objects[i].rotation_y;
        object_transforms[i].rotation_z = current_scene_data.objects[i].rotation_z;
    }

    DEBUG_LOGF("[Scene] Loaded: %s with %zu objects, %d materials",
        current_scene_data.name.c_str(),
        current_scene_data.objects.size(),
        (int)merged_mesh.materials.size());

    uploadGeometryData();
    loadMap(current_scene_data.envmap_path);
}

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

void OptixRenderer::uploadGeometryData() {
    // Vertices (full buffer)
    {
        size_t cv = merged_mesh.vertices.size() * sizeof(ColoredVertex);
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_vertices, cv));
        CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_vertices, merged_mesh.vertices.data(), cv, cudaMemcpyHostToDevice));
    }

    // Positions (extracted from vertices for GAS)
    {
        std::vector<float3> pos(merged_mesh.vertices.size());
        for (size_t i = 0; i < pos.size(); i++) pos[i] = merged_mesh.vertices[i].position;
        size_t pb = pos.size() * sizeof(float3);
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_positions, pb));
        CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_positions, pos.data(), pb, cudaMemcpyHostToDevice));
    }

    // Indices
    {
        size_t ib = merged_mesh.indices.size() * sizeof(uint3);
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_indices, ib));
        CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_indices, merged_mesh.indices.data(), ib, cudaMemcpyHostToDevice));
    }

    // SBT index buffer (one per primitive)
    {
        size_t sib = merged_mesh.sbt_index_buffer.size() * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_sbt_indices, sib));
        CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_sbt_indices, merged_mesh.sbt_index_buffer.data(), sib, cudaMemcpyHostToDevice));
    }

    DEBUG_LOG("[Memory] Geometry uploaded");
}

void OptixRenderer::createModuleAndProgramGroups() {
    auto optix_ir = loadFile("generated/optixir/SimplePathTracer.optixir");

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.numPayloadValues = 7;
    pipeline_compile_options.numAttributeValues = 2;
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

    // Create program groups
    OptixProgramGroupDesc descriptor_raygen = {};
    descriptor_raygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descriptor_raygen.raygen.module = module;
    descriptor_raygen.raygen.entryFunctionName = "__raygen__rg";

    OptixProgramGroupDesc descriptor_miss = {};
    descriptor_miss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descriptor_miss.miss.module = module;
    descriptor_miss.miss.entryFunctionName = "__miss__ms";

    OptixProgramGroupDesc descriptor_hitgroup_lambert = {};
    descriptor_hitgroup_lambert.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descriptor_hitgroup_lambert.hitgroup.moduleCH = module;
    descriptor_hitgroup_lambert.hitgroup.entryFunctionNameCH = "__closesthit__ch";

    OptixProgramGroupDesc descriptor_hitgroup_glass = {};
    descriptor_hitgroup_glass.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descriptor_hitgroup_glass.hitgroup.moduleCH = module;
    descriptor_hitgroup_glass.hitgroup.entryFunctionNameCH = "__closesthit__glass";

    OptixProgramGroupOptions pg_opts = {};

    OPTIX_CHECK(optixProgramGroupCreate(context, &descriptor_raygen, 1, &pg_opts, log, &logSize, &raygen_program_group));
    OPTIX_CHECK(optixProgramGroupCreate(context, &descriptor_miss, 1, &pg_opts, log, &logSize, &miss_program_group));
    OPTIX_CHECK(optixProgramGroupCreate(context, &descriptor_hitgroup_lambert, 1, &pg_opts, log, &logSize, &hitgroup_lambert_program_group));
    OPTIX_CHECK(optixProgramGroupCreate(context, &descriptor_hitgroup_glass, 1, &pg_opts, log, &logSize, &hitgroup_glass_program_group));

    DEBUG_LOG("[OptiX] Program groups created");
}

void OptixRenderer::createPipeline() {
    OptixProgramGroup groups[] = { 
        raygen_program_group, 
        miss_program_group, 
        hitgroup_lambert_program_group, 
        hitgroup_glass_program_group 
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 8;

    char log[4096];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(optixPipelineCreate(
        context, 
        &pipeline_compile_options, 
        &pipeline_link_options, groups,
        4, 
        log, 
        &logSize, 
        &pipeline
    ));
    DEBUG_LOG("[OptiX] Pipeline created");
}

void OptixRenderer::buildAccelerationStructures() {
    buildGAS();
    buildIAS();
}

// ------------------------------------------------------------------
// Build single GAS with 1 build input and SbtIndexOffsetBuffer
// ------------------------------------------------------------------
void OptixRenderer::buildGAS() {
    std::vector<unsigned int> geom_flags(merged_mesh.materials.size(), OPTIX_GEOMETRY_FLAG_NONE);

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    build_input.triangleArray.numVertices = (unsigned int)merged_mesh.vertices.size();
    build_input.triangleArray.vertexBuffers = &device_buffers.d_positions;
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    build_input.triangleArray.numIndexTriplets = (unsigned int)merged_mesh.indices.size();
    build_input.triangleArray.indexBuffer = device_buffers.d_indices;
    build_input.triangleArray.flags = geom_flags.data();
    build_input.triangleArray.numSbtRecords = (unsigned int)merged_mesh.materials.size();
    build_input.triangleArray.sbtIndexOffsetBuffer = device_buffers.d_sbt_indices;
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    OptixAccelBuildOptions accel_opts = {};
    accel_opts.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_opts.operation = OPTIX_BUILD_OPERATION_BUILD;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_opts, &build_input, 1, &gas_sizes));

    CUdeviceptr d_gas_temp;
    CUDA_CHECK(cudaMalloc((void**)&d_gas_temp, gas_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_gas_output, gas_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(context, 
        0, 
        &accel_opts, 
        &build_input, 
        1,
        d_gas_temp, 
        gas_sizes.tempSizeInBytes,
        device_buffers.d_gas_output, 
        gas_sizes.outputSizeInBytes,
        &gas_handle, 
        nullptr, 
        0
    ));

    CUDA_CHECK(cudaFree((void*)d_gas_temp));
    DEBUG_LOGF("[GAS] Built with %d materials", merged_mesh.materials.size());
}

// ----------------------------------------------------------
// Build IAS — per-instance SBT offset for correct material lookup
// ----------------------------------------------------------
void OptixRenderer::buildIAS() {
    // Create instances for each object
    instances.clear();
    for (size_t i = 0; i < current_scene_data.objects.size(); ++i) {
        const auto& obj_config = current_scene_data.objects[i];

        OptixInstance instance = {};
        createTransformMatrix(
            obj_config.position,
            obj_config.scale,
            instance.transform,
            object_transforms[i].rotation_x,
            object_transforms[i].rotation_y,
            object_transforms[i].rotation_z
        );
        instance.instanceId = (unsigned int)i;
        instance.sbtOffset = 0;//object_material_bases[i];  // Per-object material offset
        instance.visibilityMask = 255;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = gas_handle;

        DEBUG_LOGF("[IAS] Max offset: %d", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET);

        instances.push_back(instance);
    }

    // Upload instances to GPU
    size_t instances_size = instances.size() * sizeof(OptixInstance);
    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_instances, instances_size));
    CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_instances, instances.data(), instances_size, cudaMemcpyHostToDevice));

    OptixBuildInput inst_input = {};
    inst_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    inst_input.instanceArray.instances = device_buffers.d_instances;
    inst_input.instanceArray.numInstances = (unsigned int)instances.size();

    OptixAccelBuildOptions ias_opts = {};
    ias_opts.buildFlags = OPTIX_BUILD_FLAG_NONE;
    ias_opts.operation = OPTIX_BUILD_OPERATION_BUILD;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &ias_opts, &inst_input, 1, &ias_sizes));

    CUdeviceptr d_ias_temp;
    CUDA_CHECK(cudaMalloc((void**)&d_ias_temp, ias_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_ias_output, ias_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(context, 0, &ias_opts, &inst_input, 1,
        d_ias_temp, ias_sizes.tempSizeInBytes,
        device_buffers.d_ias_output, ias_sizes.outputSizeInBytes,
        &ias_handle, nullptr, 0));

    CUDA_CHECK(cudaFree((void*)d_ias_temp));
    DEBUG_LOGF("[IAS] Built with %zu instances", instances.size());
}

// ----------------------------------------------------------
// SBT — one record per material
// Records are in the same order as merged.materials
// Each record points to either HitGroupDataLambert or HitGroupDataGlass
// ----------------------------------------------------------
void OptixRenderer::buildSBT() {
    const size_t max_stride = (sizeof(HitGroupRecordLambert) > sizeof(HitGroupRecordGlass))
        ? sizeof(HitGroupRecordLambert) : sizeof(HitGroupRecordGlass);

    std::vector<char> hit_records;

    material_textures.clear();
    material_textures.resize(merged_mesh.materials.size());

    for (size_t mat_idx = 0; mat_idx < merged_mesh.materials.size(); ++mat_idx) {
        const auto& mat = merged_mesh.materials[mat_idx];

        if (mat.is_glass) {
            HitGroupRecordGlass rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_glass_program_group, &rec));
            rec.data.vertices = (ColoredVertex*)device_buffers.d_vertices;
            rec.data.indices = (uint3*)device_buffers.d_indices;
            rec.data.refraction_index = mat.ior;
            
            //for (auto& v : merged_mesh.vertices) {
            //    v.color = mat.color;  // Set vertex color to material color
            //}

            hit_records.resize(hit_records.size() + max_stride);
            std::memcpy(hit_records.data() + mat_idx * max_stride, &rec, sizeof(HitGroupRecordGlass));
        }
        else {
            HitGroupRecordLambert rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_lambert_program_group, &rec));
            rec.data.vertices = (ColoredVertex*)device_buffers.d_vertices;
            rec.data.indices = (uint3*)device_buffers.d_indices;
            rec.data.albedo = mat.color;
            rec.data.shininess = 32.f;
            rec.data.specular_color = { 0.04, 0.04, 0.04 };

            // Load texture if available
            if (!mat.texture_path.empty()) {
                material_textures[mat_idx].albedo_tex = loadTextureFromFile(
                    mat.texture_path,
                    material_textures[mat_idx].albedo_array
                );
                rec.data.albedo_texture = material_textures[mat_idx].albedo_tex;
                DEBUG_LOGF("[Texture] Loaded for material %zu: %s", mat_idx, mat.texture_path.c_str());
            }
            else {
                rec.data.albedo_texture = 0;
            }

            hit_records.resize(hit_records.size() + max_stride);
            std::memcpy(hit_records.data() + mat_idx * max_stride, &rec, sizeof(HitGroupRecordLambert));
        }
    }

    CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_hg, hit_records.size()));
    CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_hg, hit_records.data(), hit_records.size(), cudaMemcpyHostToDevice));

    //// Re-upload vertices with corrected colors
    //size_t cv = merged_mesh.vertices.size() * sizeof(ColoredVertex);
    //CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_vertices, merged_mesh.vertices.data(), cv, cudaMemcpyHostToDevice));


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
    sbt.hitgroupRecordCount = (unsigned int)merged_mesh.materials.size();

    DEBUG_LOG("[SBT] Built");
}

void OptixRenderer::setupShaders() {
    createModuleAndProgramGroups();
    createPipeline();
    buildSBT();
}

void OptixRenderer::setupLighting() {
    params.num_lights = 3;

    params.lights[0].type = 0;
    params.lights[0].position_or_direction = make_float3(2.0f, 3.0f, 2.0f);
    params.lights[0].color = make_float3(1.0f, 0.4f, 0.4f);

    params.lights[1].type = 0;
    params.lights[1].position_or_direction = make_float3(2.0f, 3.0f, -2.0f);
    params.lights[1].color = make_float3(0.4f, 0.4f, 1.0f);

    params.lights[2].type = 1;
    params.lights[2].position_or_direction = normalize(make_float3(0.0f, -1.0f, -0.2f));
    params.lights[2].color = make_float3(0.5f, 0.5f, 0.5f);

    // Path tracer settings
    params.max_bounce_depth = 4;   // Start with x bounces
    params.samples_per_pixel = 2;  // Progressive sampling
    params.current_sample = 0;
    params.random_seed = 1415;

    params.envmap.scale = 1.0f;
    params.envmap.exposure = 0.0f;

    DEBUG_LOGF("[Lighting] Setup complete %d lights", params.num_lights);
}

void OptixRenderer::render(const Camera& camera, int samples_per_pixel) {
    // Allocate buffers if not already done
    if (!device_buffers.d_pixels) {
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_pixels, params.width * params.height * sizeof(uchar4)));
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_accum_buffer, params.width * params.height * sizeof(float3)));
        CUDA_CHECK(cudaMalloc((void**)&device_buffers.d_params, sizeof(Params)));

        params.image = (uchar4*)device_buffers.d_pixels;
        params.accum_buffer = (float3*)device_buffers.d_accum_buffer;
    }

    params.camera = camera;
    params.traversable = ias_handle;
    params.samples_per_pixel = samples_per_pixel;
    params.random_seed = params.random_seed * 1103515245 + 12345;

    CUDA_CHECK(cudaMemcpy((void*)device_buffers.d_params, &params, sizeof(Params), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipeline, 0, device_buffers.d_params, sizeof(Params), &sbt,
        params.width, params.height, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

	params.current_sample++;
}

void OptixRenderer::updateLightParametersPos(int light_idx, float3 position_or_direction) {
    if (light_idx >= 0 && light_idx < params.num_lights) {
        params.lights[light_idx].position_or_direction = position_or_direction;
    }
}

void OptixRenderer::updateLightParametersColor(int light_idx, float3 color) {
    if (light_idx >= 0 && light_idx < params.num_lights) {
        params.lights[light_idx].color = color;
    }
}

void OptixRenderer::updateEnvmapParameters(float scale, float exposure) {
	params.envmap.scale = scale;
	params.envmap.exposure = exposure;
}

void OptixRenderer::resetAccumulationBuffer() {
    if (device_buffers.d_accum_buffer) {
        CUDA_CHECK(cudaMemset((void*)device_buffers.d_accum_buffer, 0, params.width * params.height * sizeof(float3)));
		//DEBUG_LOG("[Render] Accumulation buffer reset");
    }
	params.current_sample = 0;
}

void OptixRenderer::cleanup() {
    if (pipeline) optixPipelineDestroy(pipeline);
    if (context) optixDeviceContextDestroy(context);

    // Clean up textures
    for (auto& mat_tex : material_textures) {
        if (mat_tex.albedo_tex) {
            CUDA_CHECK(cudaDestroyTextureObject(mat_tex.albedo_tex));
        }
        if (mat_tex.albedo_array) {
            CUDA_CHECK(cudaFreeArray(mat_tex.albedo_array));
        }
    }
    material_textures.clear();

    // Clean up envmap CDF arrays
    if (envmap_cdf_marginal_array) {
        CUDA_CHECK(cudaFreeArray(envmap_cdf_marginal_array));
    }
    if (envmap_cdf_conditional_array) {
        CUDA_CHECK(cudaFreeArray(envmap_cdf_conditional_array));
    }

    CUDA_CHECK(cudaFree((void*)device_buffers.d_pixels));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_accum_buffer));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_params));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_vertices));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_positions));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_indices));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_sbt_indices));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_gas_output));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_ias_temp_rt));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_ias_output));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_instances));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_hg));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_rg));
    CUDA_CHECK(cudaFree((void*)device_buffers.d_ms));
}

// Helper function to create a 3x4 transform matrix
void OptixRenderer::createTransformMatrix(float3 translation, float3 scale, float transform[12],
    float rotation_x = 0.0f, float rotation_y = 0.0f, float rotation_z = 0.0f)
{
    // Precompute sin/cos for each axis
    float cx = cosf(rotation_x), sx = sinf(rotation_x);
    float cy = cosf(rotation_y), sy = sinf(rotation_y);
    float cz = cosf(rotation_z), sz = sinf(rotation_z);

    // Combined rotation matrix R = Ry * Rx * Rz
    // Each element is the dot product of the combined basis vectors
    float r00 = cy * cz + sy * sx * sz;   float r01 = -cy * sz + sy * sx * cz;  float r02 = sy * cx;
    float r10 = cx * sz;              float r11 = cx * cz;               float r12 = -sx;
    float r20 = -sy * cz + cy * sx * sz;  float r21 = sy * sz + cy * sx * cz;   float r22 = cy * cx;

    // Row-major 3x4 [R*Scale | T]
    transform[0] = r00 * scale.x;  transform[1] = r01 * scale.y;  transform[2] = r02 * scale.z;  transform[3] = translation.x;
    transform[4] = r10 * scale.x;  transform[5] = r11 * scale.y;  transform[6] = r12 * scale.z;  transform[7] = translation.y;
    transform[8] = r20 * scale.x;  transform[9] = r21 * scale.y;  transform[10] = r22 * scale.z;  transform[11] = translation.z;
}

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

cudaTextureObject_t OptixRenderer::loadTextureFromFile(const std::string& path, cudaArray_t& out_array)
{
    if (path.empty()) return 0;

    int w, h, ch;
    // Force 4 channels (RGBA) for consistent upload
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 4);
    if (!data) {
        std::cerr << "[TEX] Failed to load: " << path
            << " — " << stbi_failure_reason() << "\n";
        return 0;
    }

    std::cout << "[TEX] Loaded " << path << " (" << w << "x" << h << ")\n";

    // Allocate CUDA array (uchar4)
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<uchar4>();
    CUDA_CHECK(cudaMallocArray(&out_array, &fmt, w, h));
    CUDA_CHECK(cudaMemcpy2DToArray(
        out_array, 0, 0,
        data,
        w * 4 * sizeof(unsigned char),
        w * 4 * sizeof(unsigned char),
        h,
        cudaMemcpyHostToDevice
    ));
    stbi_image_free(data);

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = out_array;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    // uchar4 needs normalized read to get [0,1] floats in the shader
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;

    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr));
    return tex;
}

void OptixRenderer::updateObjectTransform(int object_idx, float rotation_x, float rotation_y, float rotation_z) {
    if (object_idx < 0 || object_idx >= (int)object_transforms.size()) return;

    object_transforms[object_idx].rotation_x = rotation_x;
    object_transforms[object_idx].rotation_y = rotation_y;
    object_transforms[object_idx].rotation_z = rotation_z;

    const auto& obj_config = current_scene_data.objects[object_idx];
    createTransformMatrix(
        obj_config.position,
        obj_config.scale,
        instances[object_idx].transform,
        rotation_x,
        rotation_y,
        rotation_z
    );

    // Update this instance on GPU
    size_t offset = object_idx * sizeof(OptixInstance);
    CUDA_CHECK(cudaMemcpy(
        (void*)((CUdeviceptr)device_buffers.d_instances + offset),
        &instances[object_idx],
        sizeof(OptixInstance),
        cudaMemcpyHostToDevice
    ));

    rebuildIAS();
}

void OptixRenderer::getObjectRotation(int object_idx, float& rotation_x, float& rotation_y, float& rotation_z) const {
    if (object_idx < 0 || object_idx >= (int)object_transforms.size()) return;
    rotation_x = object_transforms[object_idx].rotation_x;
    rotation_y = object_transforms[object_idx].rotation_y;
    rotation_z = object_transforms[object_idx].rotation_z;
}

void OptixRenderer::rebuildIAS() {
    // Rebuild the IAS with the updated instance transform
    OptixBuildInput inst_input = {};
    inst_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    inst_input.instanceArray.instances = device_buffers.d_instances;
    inst_input.instanceArray.numInstances = (unsigned int)instances.size();;

    OptixAccelBuildOptions ias_opts = {};
    ias_opts.buildFlags = OPTIX_BUILD_FLAG_NONE;
    ias_opts.operation = OPTIX_BUILD_OPERATION_BUILD;  // Rebuild, not update

    CUdeviceptr d_ias_temp;
    CUDA_CHECK(cudaMalloc((void**)&d_ias_temp, ias_sizes.tempSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(context, 0, &ias_opts, &inst_input, 1,
        d_ias_temp, ias_sizes.tempSizeInBytes,
        device_buffers.d_ias_output, ias_sizes.outputSizeInBytes,
        &ias_handle, nullptr, 0));

    CUDA_CHECK(cudaFree((void*)d_ias_temp));
    DEBUG_LOG("[IAS] Rebuilt");
}

void OptixRenderer::updateCamera(const Camera& camera) {
    // Check if camera has moved significantly
    float3 pos_diff = camera.origin - last_camera.origin;
    float pos_distance = sqrtf(pos_diff.x * pos_diff.x + pos_diff.y * pos_diff.y + pos_diff.z * pos_diff.z);

    // Check if camera direction changed significantly
    float3 dir_diff = camera.lower_left_corner - last_camera.lower_left_corner;
    float dir_distance = sqrtf(dir_diff.x * dir_diff.x + dir_diff.y * dir_diff.y + dir_diff.z * dir_diff.z);

    if (pos_distance > 0.001f || dir_distance > 0.001f) {
        // Camera moved - reset accumulation
        resetAccumulationBuffer();
        //DEBUG_LOG("[Camera] Moved - accumulation reset");
    }


    last_camera = camera;
}

void OptixRenderer::switchScene(SceneID scene_id) {
    if (scene_id == current_scene_id) {
        return;  // Already on this scene
    }

    // Get new scene configuration
    current_scene_data = SceneManager::getSceneConfig(scene_id);
    current_scene_id = scene_id;

    // Clean up old geometry and textures
    for (auto& mat_tex : material_textures) {
        if (mat_tex.albedo_tex) {
            CUDA_CHECK(cudaDestroyTextureObject(mat_tex.albedo_tex));
        }
        if (mat_tex.albedo_array) {
            CUDA_CHECK(cudaFreeArray(mat_tex.albedo_array));
        }
    }
    material_textures.clear();

    if (device_buffers.d_vertices) CUDA_CHECK(cudaFree((void*)device_buffers.d_vertices));
    if (device_buffers.d_positions) CUDA_CHECK(cudaFree((void*)device_buffers.d_positions));
    if (device_buffers.d_indices) CUDA_CHECK(cudaFree((void*)device_buffers.d_indices));
    if (device_buffers.d_sbt_indices) CUDA_CHECK(cudaFree((void*)device_buffers.d_sbt_indices));
    if (device_buffers.d_gas_output) CUDA_CHECK(cudaFree((void*)device_buffers.d_gas_output));
    if (device_buffers.d_hg) CUDA_CHECK(cudaFree((void*)device_buffers.d_hg));
    if (device_buffers.d_instances) CUDA_CHECK(cudaFree((void*)device_buffers.d_instances));

    object_material_bases.clear();
    uint32_t mat_offset = 0;

    // Load all objects in the scene and merge them
    MergedObjMesh combined_mesh;

    for (size_t i = 0; i < current_scene_data.objects.size(); ++i) {
        object_material_bases.push_back(mat_offset);
        const auto& obj_config = current_scene_data.objects[i];

        auto meshes = loadObj(obj_config.obj_path, obj_config.mtl_path);
        if (meshes.empty()) {
            throw std::runtime_error("Failed to load object: " + obj_config.name);
        }

        auto obj_mesh = mergeObjMeshes(meshes);

        // Merge this object into combined mesh
        uint32_t vertex_base = (uint32_t)combined_mesh.vertices.size();
        uint32_t material_base = (uint32_t)combined_mesh.materials.size();

        // Copy vertices
        for (const auto& v : obj_mesh.vertices) {
            combined_mesh.vertices.push_back(v);
        }

        // Copy indices and remap material indices
        for (const auto& tri : obj_mesh.indices) {
            combined_mesh.indices.push_back(make_uint3(
                tri.x + vertex_base,
                tri.y + vertex_base,
                tri.z + vertex_base
            ));
        }

        // Copy SBT index buffer with material offset
        for (uint32_t sbt_idx : obj_mesh.sbt_index_buffer) {
            combined_mesh.sbt_index_buffer.push_back(sbt_idx + material_base);
        }

        // Copy materials
        for (const auto& mat : obj_mesh.materials) {
            combined_mesh.materials.push_back(mat);
        }
    }

    merged_mesh = combined_mesh;

    // Initialize transform state for each object
    object_transforms.clear();
    object_transforms.resize(current_scene_data.objects.size());
    for (size_t i = 0; i < current_scene_data.objects.size(); ++i) {
        object_transforms[i].rotation_x = current_scene_data.objects[i].rotation_x;
        object_transforms[i].rotation_y = current_scene_data.objects[i].rotation_y;
        object_transforms[i].rotation_z = current_scene_data.objects[i].rotation_z;
    }

    DEBUG_LOGF("[Scene] Loading: %s with %zu objects",
        current_scene_data.name.c_str(),
        current_scene_data.objects.size());

    uploadGeometryData();
    loadMap(current_scene_data.envmap_path);

    // Rebuild acceleration structures
    buildGAS();
    buildIAS();

    // Rebuild SBT
    buildSBT();

    // Update lighting
    params.num_lights = current_scene_data.num_lights;
    for (int i = 0; i < current_scene_data.num_lights; ++i) {
        params.lights[i] = current_scene_data.lights[i];
    }

    // Update envmap
    if (!current_scene_data.envmap_path.empty()) {
        loadMap(current_scene_data.envmap_path);
    }
    else {
        params.envmap.has_envmap = false;
    }

    params.envmap.scale = current_scene_data.envmap_scale;
    params.envmap.exposure = current_scene_data.envmap_exposure;

    updateCamera({
         current_scene_data.camera_position,
         current_scene_data.camera_lookat,
         current_scene_data.camera_up,
         current_scene_data.camera_vfov
        });

    // Reset accumulation
    resetAccumulationBuffer();

    DEBUG_LOGF("[Scene] Switched to: %s with %zu objects, %d total materials",
        current_scene_data.name.c_str(),
        current_scene_data.objects.size(),
        (int)merged_mesh.materials.size());
}