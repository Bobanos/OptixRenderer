#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>

#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>

// ------------------------------------------------------------------
// Error helpers
// ------------------------------------------------------------------

#define CUDA_CHECK(x) do {                                  \
    cudaError_t rc = x;                                     \
    if (rc != cudaSuccess) {                                \
        std::cerr << "CUDA error: "                          \
                  << cudaGetErrorString(rc) << std::endl;  \
        std::exit(1);                                       \
    }                                                       \
} while(0)

#define OPTIX_CHECK(x) do {                                 \
    OptixResult rc = x;                                     \
    if (rc != OPTIX_SUCCESS) {                              \
        std::cerr << "OptiX error: " << rc << std::endl;    \
        std::exit(1);                                       \
    }                                                       \
} while(0)

// ------------------------------------------------------------------
// OptiX log callback
// ------------------------------------------------------------------

static void optixLogCallback(
    unsigned int level,
    const char* tag,
    const char* message,
    void*)
{
    std::cerr << "[OptiX][" << level << "][" << tag << "] "
        << message << std::endl;
}

// ------------------------------------------------------------------
// Launch params
// ------------------------------------------------------------------

struct Params
{
    uchar4* image;
    int     width;
    int     height;
};

// ------------------------------------------------------------------
// Utility: load file
// ------------------------------------------------------------------

static std::vector<char> loadFile(const std::string& path)
{
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





// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------

int main()
{
    try
    {
        // ----------------------------------------------------------
        // CUDA + OptiX init
        // ----------------------------------------------------------
        CUDA_CHECK(cudaFree(nullptr));
        OPTIX_CHECK(optixInit());

        OptixDeviceContext context = nullptr;
        CUcontext cuCtx = 0;

        OptixDeviceContextOptions ctx_opts = {};
        ctx_opts.logCallbackFunction = optixLogCallback;
        ctx_opts.logCallbackLevel = 4;

        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &ctx_opts, &context));

        // ----------------------------------------------------------
        // Module
        // ----------------------------------------------------------
        auto ir = loadFile("generated/optixir/SimplePathTracer.optixir");

        OptixModuleCompileOptions module_opts = {};
        module_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

        OptixPipelineCompileOptions pipeline_opts = {};
        pipeline_opts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_opts.usesMotionBlur = false;
        pipeline_opts.numPayloadValues = 0;
        pipeline_opts.numAttributeValues = 0;
        pipeline_opts.exceptionFlags = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
        pipeline_opts.pipelineLaunchParamsVariableName = "params";

        char log[4096];
        size_t logSize = sizeof(log);

        OptixModule module = nullptr;
        OPTIX_CHECK(optixModuleCreate(
            context,
            &module_opts,
            &pipeline_opts,
            ir.data(),
            ir.size(),
            log,
            &logSize,
            &module
        ));

        if (logSize > 1)
            std::cerr << "Module log:\n" << log << std::endl;

        // ----------------------------------------------------------
        // Program groups
        // ----------------------------------------------------------
        OptixProgramGroup raygen_pg = nullptr;
        OptixProgramGroup miss_pg = nullptr;

        OptixProgramGroupDesc rg_desc = {};
        rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        rg_desc.raygen.module = module;
        rg_desc.raygen.entryFunctionName = "__raygen__rg";

        OptixProgramGroupDesc ms_desc = {};
        ms_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        ms_desc.miss.module = module;
        ms_desc.miss.entryFunctionName = "__miss__ms";

        OptixProgramGroupOptions pg_opts = {};

        OPTIX_CHECK(optixProgramGroupCreate(
            context, &rg_desc, 1, &pg_opts, log, &logSize, &raygen_pg));

        OPTIX_CHECK(optixProgramGroupCreate(
            context, &ms_desc, 1, &pg_opts, log, &logSize, &miss_pg));

        // ----------------------------------------------------------
        // Pipeline
        // ----------------------------------------------------------
        OptixProgramGroup groups[] = { raygen_pg, miss_pg };

        OptixPipelineLinkOptions link_opts = {};
        link_opts.maxTraceDepth = 1;

        OptixPipeline pipeline = nullptr;
        OPTIX_CHECK(optixPipelineCreate(
            context,
            &pipeline_opts,
            &link_opts,
            groups,
            2,
            log,
            &logSize,
            &pipeline
        ));

        // ----------------------------------------------------------
        // SBT
        // ----------------------------------------------------------
        struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
        {
            char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        };

        struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
        {
            char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        };

        RaygenRecord rg = {};
        MissRecord   ms = {};

        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg, &rg));
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &ms));

        CUdeviceptr d_rg, d_ms;
        CUDA_CHECK(cudaMalloc((void**)&d_rg, sizeof(rg)));
        CUDA_CHECK(cudaMalloc((void**)&d_ms, sizeof(ms)));

        CUDA_CHECK(cudaMemcpy((void*)d_rg, &rg, sizeof(rg), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)d_ms, &ms, sizeof(ms), cudaMemcpyHostToDevice));

        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = d_rg;
        sbt.missRecordBase = d_ms;
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = 1;

        // ----------------------------------------------------------
        // Output buffer
        // ----------------------------------------------------------
        const int width = 256;
        const int height = 256;

        CUdeviceptr d_pixels;
        CUDA_CHECK(cudaMalloc((void**)&d_pixels, width * height * sizeof(uchar4)));

        Params params = {};
        params.image = (uchar4*)d_pixels;
        params.width = width;
        params.height = height;

        CUdeviceptr d_params;
        CUDA_CHECK(cudaMalloc((void**)&d_params, sizeof(Params)));
        CUDA_CHECK(cudaMemcpy((void*)d_params, &params, sizeof(Params), cudaMemcpyHostToDevice));

        // ----------------------------------------------------------
        // Launch
        // ----------------------------------------------------------
        OPTIX_CHECK(optixLaunch(
            pipeline,
            0,
            d_params,
            sizeof(Params),
            &sbt,
            width,
            height,
            1
        ));

        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << "SUCCESS: OptiX launch completed." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}