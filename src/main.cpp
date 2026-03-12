#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>

#include "renderer.h"
#include "optix_params.h"
#include "camera.h"
#include "obj_loader.h"
#include "stb_image.h"

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

const int width = 800;
const int height = 600;

const int window_width = 1200;
const int window_height = 1000;

const float PI = 3.14159265f;
const float DEG2RAD = PI / 180.0f;
const float RAD2DEG = 180.0f / PI;

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;

// Camera instance (global for mouse callback)
CameraController* g_camera = nullptr;
bool g_mouse_captured = false;

// ------------------------------------------------------------------
// Utility: load 
// ------------------------------------------------------------------

static std::vector<char> loadFile(const std::string& path){
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

static cudaTextureObject_t loadTextureFromFile(const std::string& path,
    cudaArray_t& out_array)
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

// ------------------------------------------------------------------
// Geometry data and Transforms
// ------------------------------------------------------------------


// Helper function to create a 3x4 transform matrix
void createTransformMatrix(float3 translation, float3 scale, float transform[12],
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

// ------------------------------------------------------------------
// Callbacks and input processing
// ------------------------------------------------------------------
static void optixLogCallback(unsigned int level, const char* tag, const char* message, void*) {
    std::cerr << "[OptiX][" << level << "][" << tag << "] "
        << message << std::endl;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (g_camera && g_mouse_captured) {
        g_camera->processMouseMovement((float)xpos, (float)ypos);
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (g_camera) {
        g_camera->processMouseScroll((float)yoffset);
    }
}

void processInput(GLFWwindow* window, float deltaTime)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Toggle mouse capture with TAB
    static bool tab_pressed = false;
    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS && !tab_pressed) {
        tab_pressed = true;
        g_mouse_captured = !g_mouse_captured;

        if (g_mouse_captured) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_RELEASE) {
        tab_pressed = false;
    }

    if (g_camera && g_mouse_captured) {
        g_camera->processKeyboard(window, deltaTime);
    }
}

// ------------------------------------------------------------------
// Display Buffer for ImGui
// ------------------------------------------------------------------

class ImGuiDisplayBuffer {
public:
    ImGuiDisplayBuffer(int w, int h) : width(w), height(h) {
        // Create OpenGL texture
        glGenTextures(1, &gl_texture);
        glBindTexture(GL_TEXTURE_2D, gl_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Register with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterImage(
            &cuda_resource,
            gl_texture,
            GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsWriteDiscard
        ));
    }

    ~ImGuiDisplayBuffer() {
        if (cuda_resource) {
            cudaGraphicsUnregisterResource(cuda_resource);
        }
        if (gl_texture) {
            glDeleteTextures(1, &gl_texture);
        }
    }

    void copyFromDevice(CUdeviceptr d_pixels) {
        // Map OpenGL texture to CUDA
        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_resource, 0));

        cudaArray_t array;
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0));

        // Copy from CUDA buffer to OpenGL texture
        CUDA_CHECK(cudaMemcpy2DToArray(
            array,
            0, 0,
            (void*)d_pixels,
            width * sizeof(uchar4),
            width * sizeof(uchar4),
            height,
            cudaMemcpyDeviceToDevice
        ));

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_resource, 0));
    }

    GLuint getTexture() const { return gl_texture; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }

private:
    int width, height;
    GLuint gl_texture = 0;
    cudaGraphicsResource_t cuda_resource = nullptr;
};


// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------

int main(){

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "Optix 9.1.0 PathTracer", NULL, NULL);
    
    if (window == NULL)
    {
        std::cout << "[GLFW] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "[GLAD] Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glViewport(0, 0, width, height);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // ----------------------------------------------------------
    // Setup Dear ImGui
    // ----------------------------------------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsClassic();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    try{
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

#ifdef _DEBUG
        // This may incur significant performance cost and should only be done during development.
        std::cout << "DEBUG ENABLED" << std::endl;
        ctx_opts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &ctx_opts, &context));


        // ----------------------------------------------------------
        // Module
        // ----------------------------------------------------------
        auto ir = loadFile("generated/optixir/SimplePathTracer.optixir");

        OptixModuleCompileOptions module_opts = {};
        module_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

        OptixPipelineCompileOptions pipeline_opts = {};
        pipeline_opts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_opts.usesMotionBlur = false;
        pipeline_opts.numPayloadValues = 4; //3 + 1(RGB + depth)
        pipeline_opts.numAttributeValues = 2; //2 for barycentrics
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
        OptixProgramGroupDesc rg_desc = {};
        rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        rg_desc.raygen.module = module;
        rg_desc.raygen.entryFunctionName = "__raygen__rg";

        OptixProgramGroup miss_pg = nullptr;
        OptixProgramGroupDesc ms_desc = {};
        ms_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        ms_desc.miss.module = module;
        ms_desc.miss.entryFunctionName = "__miss__ms";

        OptixProgramGroup hitgroup_pg = nullptr;
        OptixProgramGroupDesc hg_desc = {};
        hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hg_desc.hitgroup.moduleCH = module;
        hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

        OptixProgramGroup hitgroup_glass_pg = nullptr;
        OptixProgramGroupDesc hg_glass_desc = {};
        hg_glass_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hg_glass_desc.hitgroup.moduleCH = module;
        hg_glass_desc.hitgroup.entryFunctionNameCH = "__closesthit__glass";

        OptixProgramGroupOptions pg_opts = {};

        OPTIX_CHECK(optixProgramGroupCreate(context, &rg_desc, 1, &pg_opts, log, &logSize, &raygen_pg));
        OPTIX_CHECK(optixProgramGroupCreate(context, &ms_desc, 1, &pg_opts, log, &logSize, &miss_pg));
        OPTIX_CHECK(optixProgramGroupCreate(context, &hg_desc, 1, &pg_opts, log, &logSize, &hitgroup_pg));
        OPTIX_CHECK(optixProgramGroupCreate(context, &hg_glass_desc, 1, &pg_opts, log, &logSize, &hitgroup_glass_pg));

        // ----------------------------------------------------------
        // Pipeline
        // ----------------------------------------------------------
        OptixProgramGroup groups[] = { raygen_pg, miss_pg , hitgroup_pg, hitgroup_glass_pg };

        OptixPipelineLinkOptions link_opts = {};
        link_opts.maxTraceDepth = 8; //TODO fix shadow ray -> refraction exceding max trace depth bug

        OptixPipeline pipeline = nullptr;
        OPTIX_CHECK(optixPipelineCreate(
            context,
            &pipeline_opts,
            &link_opts,
            groups,
            4,
            log,
            &logSize,
            &pipeline
        ));


        // ----------------------------------------------------------
        // Build Geometry Acceleration Structure
        // ----------------------------------------------------------

        auto ship_meshes = loadObj(
            "assets/6887_allied_avenger.obj",
            "assets/6887_allied_avenger.mtl"
        );

        if (ship_meshes.empty())
            throw std::runtime_error("No meshes loaded from OBJ");

        MergedObjMesh merged = mergeObjMeshes(ship_meshes);

        // ------------------------------------------------------------------
        // Upload merged vertex/index buffers for both material types
        // ------------------------------------------------------------------
        CUdeviceptr d_solid_verts = 0, d_solid_positions = 0, d_solid_indices = 0;
        CUdeviceptr d_glass_verts = 0, d_glass_positions = 0, d_glass_indices = 0;

        // Solid
        {
            size_t cv = merged.solid_vertices.size() * sizeof(ColoredVertex);
            CUDA_CHECK(cudaMalloc((void**)&d_solid_verts, cv));
            CUDA_CHECK(cudaMemcpy((void*)d_solid_verts, merged.solid_vertices.data(), cv, cudaMemcpyHostToDevice));

            std::vector<float3> pos(merged.solid_vertices.size());
            for (size_t i = 0; i < pos.size(); i++) pos[i] = merged.solid_vertices[i].position;
            size_t pb = pos.size() * sizeof(float3);
            CUDA_CHECK(cudaMalloc((void**)&d_solid_positions, pb));
            CUDA_CHECK(cudaMemcpy((void*)d_solid_positions, pos.data(), pb, cudaMemcpyHostToDevice));

            size_t ib = merged.solid_indices.size() * sizeof(uint3);
            CUDA_CHECK(cudaMalloc((void**)&d_solid_indices, ib));
            CUDA_CHECK(cudaMemcpy((void*)d_solid_indices, merged.solid_indices.data(), ib, cudaMemcpyHostToDevice));
        }

        // Glass (may be empty — skip build input if so)
        bool has_glass = !merged.glass_vertices.empty();
        if (has_glass) {
            size_t cv = merged.glass_vertices.size() * sizeof(ColoredVertex);
            CUDA_CHECK(cudaMalloc((void**)&d_glass_verts, cv));
            CUDA_CHECK(cudaMemcpy((void*)d_glass_verts, merged.glass_vertices.data(), cv, cudaMemcpyHostToDevice));

            std::vector<float3> pos(merged.glass_vertices.size());
            for (size_t i = 0; i < pos.size(); i++) pos[i] = merged.glass_vertices[i].position;
            size_t pb = pos.size() * sizeof(float3);
            CUDA_CHECK(cudaMalloc((void**)&d_glass_positions, pb));
            CUDA_CHECK(cudaMemcpy((void*)d_glass_positions, pos.data(), pb, cudaMemcpyHostToDevice));

            size_t ib = merged.glass_indices.size() * sizeof(uint3);
            CUDA_CHECK(cudaMalloc((void**)&d_glass_indices, ib));
            CUDA_CHECK(cudaMemcpy((void*)d_glass_indices, merged.glass_indices.data(), ib, cudaMemcpyHostToDevice));
        }

        // ------------------------------------------------------------------
        // Build single GAS with 1 or 2 build inputs:
        //   build input 0 (geometrySbtIndex=0) -> solid -> SBT record 0
        //   build input 1 (geometrySbtIndex=1) -> glass -> SBT record 1
        //
        // numSbtRecords per build input = 1, so SBT indices are assigned
        // sequentially: solid=0, glass=1.
        // ------------------------------------------------------------------
        unsigned int solid_geom_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        unsigned int glass_geom_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

        OptixBuildInput build_inputs[2] = {};

        // Build input 0: solid (geometrySbtIndex = 0)
        build_inputs[0].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        build_inputs[0].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        build_inputs[0].triangleArray.vertexStrideInBytes = sizeof(float3);
        build_inputs[0].triangleArray.numVertices = (unsigned int)merged.solid_vertices.size();
        build_inputs[0].triangleArray.vertexBuffers = &d_solid_positions;
        build_inputs[0].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        build_inputs[0].triangleArray.indexStrideInBytes = sizeof(uint3);
        build_inputs[0].triangleArray.numIndexTriplets = (unsigned int)merged.solid_indices.size();
        build_inputs[0].triangleArray.indexBuffer = d_solid_indices;
        build_inputs[0].triangleArray.flags = solid_geom_flags;
        build_inputs[0].triangleArray.numSbtRecords = 1; // -> SBT record 0

        unsigned int num_build_inputs = has_glass ? 2u : 1u;

        if (has_glass) {
            // Build input 1: glass (geometrySbtIndex = 1)
            build_inputs[1].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            build_inputs[1].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            build_inputs[1].triangleArray.vertexStrideInBytes = sizeof(float3);
            build_inputs[1].triangleArray.numVertices = (unsigned int)merged.glass_vertices.size();
            build_inputs[1].triangleArray.vertexBuffers = &d_glass_positions;
            build_inputs[1].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            build_inputs[1].triangleArray.indexStrideInBytes = sizeof(uint3);
            build_inputs[1].triangleArray.numIndexTriplets = (unsigned int)merged.glass_indices.size();
            build_inputs[1].triangleArray.indexBuffer = d_glass_indices;
            build_inputs[1].triangleArray.flags = glass_geom_flags;
            build_inputs[1].triangleArray.numSbtRecords = 1; // -> SBT record 1
        }

        OptixAccelBuildOptions accel_opts = {};
        accel_opts.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accel_opts.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_opts,
            build_inputs, num_build_inputs, &gas_sizes));

        CUdeviceptr d_gas_temp, d_gas_output;
        CUDA_CHECK(cudaMalloc((void**)&d_gas_temp, gas_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_gas_output, gas_sizes.outputSizeInBytes));

        OptixTraversableHandle gas_handle;
        OPTIX_CHECK(optixAccelBuild(context, 0, &accel_opts,
            build_inputs, num_build_inputs,
            d_gas_temp, gas_sizes.tempSizeInBytes,
            d_gas_output, gas_sizes.outputSizeInBytes,
            &gas_handle, nullptr, 0));
        CUDA_CHECK(cudaFree((void*)d_gas_temp));

        std::cout << "Built 1 GAS (" << num_build_inputs << " build inputs)\n";


        // ----------------------------------------------------------
        // Build IAS — one instance pointing at the single GAS
        // sbtOffset = 0: the instance's SBT base
        // The GAS has numSbtRecords = 1+1 = 2, so OptiX uses:
        //   solid hit -> sbtOffset(0) + geometrySbtIndex(0) = record 0
        //   glass hit -> sbtOffset(0) + geometrySbtIndex(1) = record 1
        // ----------------------------------------------------------
        OptixInstance instance = {};
        createTransformMatrix(
            make_float3(0.0f, 1.0f, 0.0f),
            make_float3(0.05f, 0.05f, 0.05f),
            instance.transform,
            0.0f, 0.0f, 0.0f
        );
        instance.instanceId = 0;
        instance.sbtOffset = 0; // base for SBT record lookup
        instance.visibilityMask = 255;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = gas_handle;

        CUdeviceptr d_instances;
        CUDA_CHECK(cudaMalloc((void**)&d_instances, sizeof(OptixInstance)));
        CUDA_CHECK(cudaMemcpy((void*)d_instances, &instance, sizeof(OptixInstance), cudaMemcpyHostToDevice));

        OptixBuildInput inst_input = {};
        inst_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        inst_input.instanceArray.instances = d_instances;
        inst_input.instanceArray.numInstances = 1;

        OptixAccelBuildOptions ias_opts = {};
        ias_opts.buildFlags = OPTIX_BUILD_FLAG_NONE;
        ias_opts.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes ias_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &ias_opts,
            &inst_input, 1, &ias_sizes));

        CUdeviceptr d_ias_temp, d_ias_output;
        CUDA_CHECK(cudaMalloc((void**)&d_ias_temp, ias_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_ias_output, ias_sizes.outputSizeInBytes));

        OptixTraversableHandle ias_handle;
        OPTIX_CHECK(optixAccelBuild(context, 0, &ias_opts, &inst_input, 1,
            d_ias_temp, ias_sizes.tempSizeInBytes,
            d_ias_output, ias_sizes.outputSizeInBytes,
            &ias_handle, nullptr, 0));
        CUDA_CHECK(cudaFree((void*)d_ias_temp));

        // ----------------------------------------------------------
        // SBT — exactly 2 hit records
        //   record 0 -> solid/__closesthit__ch  (vertices=solid buffer)
        //   record 1 -> glass/__closesthit__glass (vertices=glass buffer)
        // ----------------------------------------------------------
        HitGroupRecord hg_records[2] = {};

        // Record 0: solid plastic
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg_records[0]));
        hg_records[0].data.vertices = (ColoredVertex*)d_solid_verts;
        hg_records[0].data.indices = (uint3*)d_solid_indices;
        hg_records[0].data.refraction_index = 1.0f;
        hg_records[0].data.albedo_texture = 0; // vertex color carries per-material Kd

        // Record 1: glass
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_glass_pg, &hg_records[1]));
        hg_records[1].data.vertices = (ColoredVertex*)d_glass_verts;
        hg_records[1].data.indices = (uint3*)d_glass_indices;
        hg_records[1].data.refraction_index = merged.glass_ior;
        hg_records[1].data.albedo_texture = 0;

        CUdeviceptr d_rg, d_ms, d_hg;
        RayGenRecord rg = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg, &rg));
        CUDA_CHECK(cudaMalloc((void**)&d_rg, sizeof(RayGenRecord)));
        CUDA_CHECK(cudaMemcpy((void*)d_rg, &rg, sizeof(rg), cudaMemcpyHostToDevice));

        MissRecord ms = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &ms));
        CUDA_CHECK(cudaMalloc((void**)&d_ms, sizeof(MissRecord)));
        CUDA_CHECK(cudaMemcpy((void*)d_ms, &ms, sizeof(ms), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void**)&d_hg, sizeof(HitGroupRecord) * 2));
        CUDA_CHECK(cudaMemcpy((void*)d_hg, hg_records, sizeof(HitGroupRecord) * 2, cudaMemcpyHostToDevice));

        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = d_rg;
        sbt.missRecordBase = d_ms;
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = d_hg;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
        sbt.hitgroupRecordCount = 2;


        // ----------------------------------------------------------
        // Output buffer + Display
        // ----------------------------------------------------------
        CUdeviceptr d_pixels;
        CUDA_CHECK(cudaMalloc((void**)&d_pixels, width * height * sizeof(uchar4)));

        Params params = {};
        params.image = (uchar4*)d_pixels;
        params.width = width;
        params.height = height;
        params.traversable = ias_handle;
        params.light_position[0] = make_float3(2.0f, 3.0f, 2.0f);
        params.light_color[0] = make_float3(1.0f, 0.4f, 0.4f);

        params.light_position[1] = make_float3(2.0f, 3.0f, -2.0f);
        params.light_color[1] = make_float3(0.4f, 0.4f, 1.0f);
        params.num_lights = 2;

        CUdeviceptr d_params;
        CUDA_CHECK(cudaMalloc((void**)&d_params, sizeof(Params)));
        //CUDA_CHECK(cudaMemcpy((void*)d_params, &params, sizeof(Params), cudaMemcpyHostToDevice));

        // Create display buffer
        ImGuiDisplayBuffer display(width, height);

        // Create camera
        CameraController camera_controller(
            make_float3(0.0f, 5.0f, 8.0f),  // position
            make_float3(0.0f, -4.0f, 0.0f),  // look at
            make_float3(0.0f, 1.0f, 0.0f),  // up
            60.0f,                           // vfov
            (float)width / (float)height    // aspect ratio
        );
        g_camera = &camera_controller;

        // ----------------------------------------------------------
        // Ship transform state — lives outside the render loop
        // ----------------------------------------------------------
        float ship_rotation_x = 0.0f;
        float ship_rotation_y = 0.0f;
        float ship_rotation_z = 0.0f;
        bool  auto_rotate = false;
        float rotation_speed = 1.0f; // radians per second

        // Keep IAS rebuild resources alive across frames
        OptixAccelBuildOptions ias_opts_rt = {};
        ias_opts_rt.buildFlags = OPTIX_BUILD_FLAG_NONE;
        ias_opts_rt.operation = OPTIX_BUILD_OPERATION_BUILD;

        CUdeviceptr d_ias_temp_rt;
        CUDA_CHECK(cudaMalloc((void**)&d_ias_temp_rt, ias_sizes.tempSizeInBytes));

        // Timing
        float deltaTime = 0.0f;
        float lastFrame = 0.0f;

        // ----------------------------------------------------------
        // Render Loop
        // ----------------------------------------------------------
        while (!glfwWindowShouldClose(window))
        {
            // Calculate delta time - ADDED
            float currentFrame = (float)glfwGetTime();
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;

            glfwPollEvents();

            // Start ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // Input processing
            processInput(window, deltaTime);


            // Update ship rotation and rebuild IAS
            if (auto_rotate)
                ship_rotation_y += rotation_speed * DEG2RAD * deltaTime;

            float sc = 0.05f;
            createTransformMatrix(
                make_float3(0.0f, 1.0f, 0.0f),
                make_float3(sc, sc, sc),
                instance.transform,
                ship_rotation_x,
                ship_rotation_y,
                ship_rotation_z
            );
            CUDA_CHECK(cudaMemcpy((void*)d_instances, &instance,
                sizeof(OptixInstance), cudaMemcpyHostToDevice));

            params.traversable = ias_handle;

            // Update camera parameters
            params.camera = camera_controller.getCameraData();
            CUDA_CHECK(cudaMemcpy((void*)d_params, &params, sizeof(Params), cudaMemcpyHostToDevice));

            // OptiX render
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

            // Copy OptiX output to OpenGL texture
            display.copyFromDevice(d_pixels);

            // ImGui viewport window
            ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar);

            // Display the texture
            ImGui::Image(
                (void*)(intptr_t)display.getTexture(),
                ImVec2(display.getWidth(), display.getHeight()),
                ImVec2(0, 1),  // UV coordinates (flip Y)
                ImVec2(1, 0)
            );
            ImGui::End();

            // Camera Controls window
            ImGui::Begin("Camera Controls");
            ImGui::Text("FPS: %.1f", 1.0f / deltaTime);
            ImGui::Text("Press TAB to toggle camera control");
            ImGui::Text("Camera Captured: %s", g_mouse_captured ? "Yes" : "No");

            float3 pos = camera_controller.getPosition();
            ImGui::Text("Position: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
            ImGui::Text("Speed: %.2f", camera_controller.getSpeed());

            ImGui::Separator();
            ImGui::Text("Controls:");
            ImGui::BulletText("WASD: Move");
            ImGui::BulletText("Q/E: Down/Up");
            ImGui::BulletText("Mouse: Look around");
            ImGui::BulletText("Scroll: Adjust speed");
            ImGui::End();

            // Light control
            ImGui::Begin("Light Controls");

            static float light1_pos[3] = { 5.0f, 5.0f, 5.0f };
            static float light1_col[3] = { 1.0f, 0.4f, 0.4f };
            static float light2_pos[3] = { -5.0f, -5.0f, -5.0f };
            static float light2_col[3] = { 0.4f, 0.4f, 1.0f };

            // Disable interaction when camera is captured
            if (g_mouse_captured) {
                ImGui::BeginDisabled();
            }

            ImGui::Text("Light 1");
            if (ImGui::DragFloat3("Light Position 1", light1_pos, 0.1f, -20.0f, 20.0f), ImGuiSliderFlags_NoInput) {
                params.light_position[0] = make_float3(light1_pos[0], light1_pos[1], light1_pos[2]);
            }
            if (ImGui::ColorEdit3("Light Color 1", light1_col), ImGuiSliderFlags_NoInput) {
                params.light_color[0] = make_float3(light1_col[0], light1_col[1], light1_col[2]);
            }
            ImGui::Separator();
            ImGui::Text("Light 2");
            if (ImGui::DragFloat3("Light Position 2", light2_pos, 0.1f, -20.0f, 20.0f), ImGuiSliderFlags_NoInput) {
                params.light_position[1] = make_float3(light2_pos[0], light2_pos[1], light2_pos[2]);
            }
            if (ImGui::ColorEdit3("Light Color 2", light2_col), ImGuiSliderFlags_NoInput) {
                params.light_color[1] = make_float3(light2_col[0], light2_col[1], light2_col[2]);
            }

            if (g_mouse_captured) {
                ImGui::EndDisabled();
            }

            ImGui::End();

            // Ship Controls window
            ImGui::Begin("Ship Controls");
            ImGui::Text("Ship Transform");
            ImGui::Checkbox("Auto Rotate Y", &auto_rotate);
            ImGui::SliderFloat("Speed (deg/s)", &rotation_speed, 1.0f, 360.0f);

            if (auto_rotate)
                ship_rotation_y += rotation_speed * DEG2RAD * deltaTime;

            float rx_deg = ship_rotation_x * RAD2DEG;
            float ry_deg = ship_rotation_y * RAD2DEG;
            float rz_deg = ship_rotation_z * RAD2DEG;

            if (ImGui::SliderFloat("Rotation X (deg)", &rx_deg, -180.0f, 180.0f))
                ship_rotation_x = rx_deg * DEG2RAD;
            if (ImGui::SliderFloat("Rotation Y (deg)", &ry_deg, -180.0f, 180.0f))
                ship_rotation_y = ry_deg * DEG2RAD;
            if (ImGui::SliderFloat("Rotation Z (deg)", &rz_deg, -180.0f, 180.0f))
                ship_rotation_z = rz_deg * DEG2RAD;

            if (ImGui::Button("Reset Rotation"))
                ship_rotation_x = ship_rotation_y = ship_rotation_z = 0.0f;
            ImGui::End();

            // Render ImGui
            ImGui::Render();

            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
        }

        std::cout << "SUCCESS: OptiX launch completed." << std::endl;

        // Cleanup
        CUDA_CHECK(cudaFree((void*)d_pixels));
        CUDA_CHECK(cudaFree((void*)d_params));
        CUDA_CHECK(cudaFree((void*)d_rg));
        CUDA_CHECK(cudaFree((void*)d_ms));
        CUDA_CHECK(cudaFree((void*)d_hg));
        CUDA_CHECK(cudaFree((void*)d_solid_verts));
        CUDA_CHECK(cudaFree((void*)d_solid_positions));
        CUDA_CHECK(cudaFree((void*)d_solid_indices));
        if (has_glass) {
            CUDA_CHECK(cudaFree((void*)d_glass_verts));
            CUDA_CHECK(cudaFree((void*)d_glass_positions));
            CUDA_CHECK(cudaFree((void*)d_glass_indices));
        }
        CUDA_CHECK(cudaFree((void*)d_gas_output));
        CUDA_CHECK(cudaFree((void*)d_ias_temp_rt));
        CUDA_CHECK(cudaFree((void*)d_ias_temp_rt));
    }
    catch (const std::exception& e){
        std::cerr << "Exception: " << e.what() << std::endl;

        // Cleanup ImGui
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwTerminate();
        return 1;
    }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();

    return 0;
}