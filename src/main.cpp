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

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

const int width = 800;
const int height = 600;

const int window_width = 1200;
const int window_height = 1000;

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
// Utility: load file
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

// ------------------------------------------------------------------
// Geometry data and Transforms
// ------------------------------------------------------------------

float3 vertices[8] = {
    {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f},
    {0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
    {-0.5f, -0.5f,  0.5f}, {0.5f, -0.5f,  0.5f},
    {0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f}
};

uint3 indices[12] = {
    {0,1,2}, {0,2,3},  // Front
    {4,6,5}, {4,7,6},  // Back
    {0,4,5}, {0,5,1},  // Bottom
    {2,6,7}, {2,7,3},  // Top
    {0,3,7}, {0,7,4},  // Left
    {1,5,6}, {1,6,2}   // Right
};

float3 ground_vertices[4] = {
    {-5.0f, 0.0f, -5.0f}, {5.0f, 0.0f, -5.0f},
    {5.0f, 0.0f,  5.0f}, {-5.0f, 0.0f,  5.0f}
};

uint3 ground_indices[2] = {
    {0, 1, 2}, {0, 2, 3}  // Two triangles for the quad
};

// Helper function to create a 3x4 transform matrix
void createTransformMatrix(float3 translation, float3 scale, float transform[12]) {
    // Create a simple transform: scale and translate
    // Row-major 3x4 matrix [R|T] where R is 3x3 rotation/scale, T is translation
    transform[0] = scale.x;  transform[1] = 0.0f;     transform[2] = 0.0f;     transform[3] = translation.x;
    transform[4] = 0.0f;     transform[5] = scale.y;  transform[6] = 0.0f;     transform[7] = translation.y;
    transform[8] = 0.0f;     transform[9] = 0.0f;     transform[10] = scale.z;  transform[11] = translation.z;
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

        // Upload cube vertices to device
        CUdeviceptr d_vertices;
        CUDA_CHECK(cudaMalloc((void**)&d_vertices, sizeof(vertices)));
        CUDA_CHECK(cudaMemcpy((void*)d_vertices, vertices, sizeof(vertices), cudaMemcpyHostToDevice));

        // Upload cube indices to device
        CUdeviceptr d_indices;
        CUDA_CHECK(cudaMalloc((void**)&d_indices, sizeof(indices)));
        CUDA_CHECK(cudaMemcpy((void*)d_indices, indices, sizeof(indices), cudaMemcpyHostToDevice));

        // Upload ground vertices  to device
        CUdeviceptr d_ground_vertices;
        CUDA_CHECK(cudaMalloc((void**)&d_ground_vertices, sizeof(ground_vertices)));
        CUDA_CHECK(cudaMemcpy((void*)d_ground_vertices, ground_vertices, sizeof(ground_vertices), cudaMemcpyHostToDevice));

        // Upload ground indices  to device
        CUdeviceptr d_ground_indices;
        CUDA_CHECK(cudaMalloc((void**)&d_ground_indices, sizeof(ground_indices)));
        CUDA_CHECK(cudaMemcpy((void*)d_ground_indices, ground_indices, sizeof(ground_indices), cudaMemcpyHostToDevice));

        // ----------------------------------------------------------
        // Build GAS for Cube
        // ----------------------------------------------------------
        OptixBuildInput cube_input = {};
        cube_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        cube_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        cube_input.triangleArray.vertexStrideInBytes = sizeof(float3);
        cube_input.triangleArray.numVertices = 8;
        cube_input.triangleArray.vertexBuffers = &d_vertices;
        cube_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        cube_input.triangleArray.indexStrideInBytes = sizeof(uint3);
        cube_input.triangleArray.numIndexTriplets = 12;
        cube_input.triangleArray.indexBuffer = d_indices;
        unsigned int cube_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        cube_input.triangleArray.flags = cube_flags;
        cube_input.triangleArray.numSbtRecords = 1;


        // Setup acceleration structure build options
        OptixAccelBuildOptions cube_accel_options = {};
        //cube_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        cube_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        cube_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        

        // Query memory requirements
        OptixAccelBufferSizes cube_gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            &cube_accel_options,
            &cube_input,
            1,  //num of geometries
            &cube_gas_buffer_sizes
        ));

        // Allocate temporary buffers
        CUdeviceptr d_cube_temp_buffer;
        CUDA_CHECK(cudaMalloc((void**)&d_cube_temp_buffer, cube_gas_buffer_sizes.tempSizeInBytes));

        CUdeviceptr d_cube_gas_output_buffer;
        CUDA_CHECK(cudaMalloc((void**)&d_cube_gas_output_buffer, cube_gas_buffer_sizes.outputSizeInBytes));

        // Build acceleration structure
        OptixTraversableHandle cube_gas_handle;
        OPTIX_CHECK(optixAccelBuild(
            context,
            0,
            &cube_accel_options,
            &cube_input,
            1,
            d_cube_temp_buffer,
            cube_gas_buffer_sizes.tempSizeInBytes,
            d_cube_gas_output_buffer,
            cube_gas_buffer_sizes.outputSizeInBytes,
            &cube_gas_handle,
            nullptr,
            0
        ));

        CUDA_CHECK(cudaFree((void*)d_cube_temp_buffer));


        // ----------------------------------------------------------
        // Build GAS for Ground
        // ----------------------------------------------------------
        OptixBuildInput ground_input = {};
        ground_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        ground_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        ground_input.triangleArray.vertexStrideInBytes = sizeof(float3);
        ground_input.triangleArray.numVertices = 4;
        ground_input.triangleArray.vertexBuffers = &d_ground_vertices;
        ground_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        ground_input.triangleArray.indexStrideInBytes = sizeof(uint3);
        ground_input.triangleArray.numIndexTriplets = 2;
        ground_input.triangleArray.indexBuffer = d_ground_indices;
        unsigned int ground_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        ground_input.triangleArray.flags = ground_flags;
        ground_input.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions ground_accel_options = {};
        ground_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        ground_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes ground_gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            &ground_accel_options,
            &ground_input,
            1,
            &ground_gas_buffer_sizes
        ));

        CUdeviceptr d_ground_temp_buffer;
        CUDA_CHECK(cudaMalloc((void**)&d_ground_temp_buffer, ground_gas_buffer_sizes.tempSizeInBytes));

        CUdeviceptr d_ground_gas_output_buffer;
        CUDA_CHECK(cudaMalloc((void**)&d_ground_gas_output_buffer, ground_gas_buffer_sizes.outputSizeInBytes));

        OptixTraversableHandle ground_gas_handle;
        OPTIX_CHECK(optixAccelBuild(
            context,
            0,
            &ground_accel_options,
            &ground_input,
            1,
            d_ground_temp_buffer,
            ground_gas_buffer_sizes.tempSizeInBytes,
            d_ground_gas_output_buffer,
            ground_gas_buffer_sizes.outputSizeInBytes,
            &ground_gas_handle,
            nullptr,
            0
        ));

        CUDA_CHECK(cudaFree((void*)d_ground_temp_buffer));

        std::cout << "Acceleration structure built successfully" << std::endl;

        // ----------------------------------------------------------
        // Create Instances with Transforms
        // ----------------------------------------------------------
        const int NUM_INSTANCES = 12;
        OptixInstance instances[NUM_INSTANCES] = {};

        // Instance 0: First cube at (0, 2, 0)
        float transform0[12];
        createTransformMatrix(make_float3(0.0f, 2.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), transform0);
        memcpy(instances[0].transform, transform0, sizeof(float) * 12);
        instances[0].instanceId = 0;
        instances[0].sbtOffset = 0;
        instances[0].visibilityMask = 255;
        instances[0].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[0].traversableHandle = cube_gas_handle;

        // Instance 1: Second cube at (2, 2, 0)
        float transform1[12];
        createTransformMatrix(make_float3(2.0f, 2.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), transform1);
        memcpy(instances[1].transform, transform1, sizeof(float) * 12);
        instances[1].instanceId = 1;
        instances[1].sbtOffset = 1;
        instances[1].visibilityMask = 255;
        instances[1].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[1].traversableHandle = cube_gas_handle;

        // Instance 2: Ground plane at (0, 0, 0)
        float transform2[12];
        createTransformMatrix(make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), transform2);
        memcpy(instances[2].transform, transform2, sizeof(float) * 12);
        instances[2].instanceId = 2;
        instances[2].sbtOffset = 2;
        instances[2].visibilityMask = 255;
        instances[2].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[2].traversableHandle = ground_gas_handle;

        // Instance 3: First glass cube at (-2, 2, 0)
        float transform3[12];
        createTransformMatrix(make_float3(-2.0f, 2.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), transform3);
        memcpy(instances[3].transform, transform3, sizeof(float) * 12);
        instances[3].instanceId = 3;
        instances[3].sbtOffset = 3; // Uses glass shader (index 3)
        instances[3].visibilityMask = 255;
        instances[3].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[3].traversableHandle = cube_gas_handle;

        // Instance 4: Second glass cube at (4, 2, 0)
        float transform4[12];
        createTransformMatrix(make_float3(4.0f, 2.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), transform4);
        memcpy(instances[4].transform, transform4, sizeof(float) * 12);
        instances[4].instanceId = 4;
        instances[4].sbtOffset = 4; // Uses glass shader (index 4)
        instances[4].visibilityMask = 255;
        instances[4].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[4].traversableHandle = cube_gas_handle;

        // Instance 5: Third glass cube at (1, 3, 0) - elevated
        float transform5[12];
        createTransformMatrix(make_float3(1.0f, 3.0f, 0.0f), make_float3(0.8f, 0.8f, 0.8f), transform5);
        memcpy(instances[5].transform, transform5, sizeof(float) * 12);
        instances[5].instanceId = 5;
        instances[5].sbtOffset = 5; // Uses glass shader (index 5)
        instances[5].visibilityMask = 255;
        instances[5].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[5].traversableHandle = cube_gas_handle;

        // Instance 6: First blue cube at (-4, 2, 0) - shares SBT record 6
        float transform6[12];
        createTransformMatrix(make_float3(-4.0f, 2.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), transform6);
        memcpy(instances[6].transform, transform6, sizeof(float) * 12);
        instances[6].instanceId = 6;
        instances[6].sbtOffset = 6;
        instances[6].visibilityMask = 255;
        instances[6].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[6].traversableHandle = cube_gas_handle;

        // Instance 7: Second blue cube at (-4, 3, 0) - shares SBT record 6
        float transform7[12];
        createTransformMatrix(make_float3(-4.0f, 3.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), transform7);
        memcpy(instances[7].transform, transform7, sizeof(float) * 12);
        instances[7].instanceId = 7;
        instances[7].sbtOffset = 6;
        instances[7].visibilityMask = 255;
        instances[7].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[7].traversableHandle = cube_gas_handle;

        // Instance 8: Third blue cube at (-4, 4, 0) - shares SBT record 6
        float transform8[12];
        createTransformMatrix(make_float3(-4.0f, 4.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), transform8);
        memcpy(instances[8].transform, transform8, sizeof(float) * 12);
        instances[8].instanceId = 8;
        instances[8].sbtOffset = 6;
        instances[8].visibilityMask = 255;
        instances[8].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[8].traversableHandle = cube_gas_handle;

        float transform9[12];
        createTransformMatrix(make_float3(5.0f, 2.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), transform9);
        memcpy(instances[9].transform, transform9, sizeof(float) * 12);
        instances[9].instanceId = 9;
        instances[9].sbtOffset = 7;
        instances[9].visibilityMask = 255;
        instances[9].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[9].traversableHandle = cube_gas_handle;

        float transform10[12];
        createTransformMatrix(make_float3(5.0f, 3.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), transform10);
        memcpy(instances[10].transform, transform10, sizeof(float) * 12);
        instances[10].instanceId = 10;
        instances[10].sbtOffset = 7;
        instances[10].visibilityMask = 255;
        instances[10].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[10].traversableHandle = cube_gas_handle;

        float transform11[12];
        createTransformMatrix(make_float3(5.0f, 4.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f), transform11);
        memcpy(instances[11].transform, transform11, sizeof(float) * 12);
        instances[11].instanceId = 11;
        instances[11].sbtOffset = 7;
        instances[11].visibilityMask = 255;
        instances[11].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[11].traversableHandle = cube_gas_handle;

        // Upload instances to device
        CUdeviceptr d_instances;
        CUDA_CHECK(cudaMalloc((void**)&d_instances, sizeof(OptixInstance)* NUM_INSTANCES));
        CUDA_CHECK(cudaMemcpy((void*)d_instances, instances, sizeof(OptixInstance)* NUM_INSTANCES, cudaMemcpyHostToDevice));

        // ----------------------------------------------------------
        // Build IAS (Instance Acceleration Structure)
        // ----------------------------------------------------------
        OptixBuildInput instance_input = {};
        instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.instances = d_instances;
        instance_input.instanceArray.numInstances = NUM_INSTANCES;

        OptixAccelBuildOptions ias_accel_options = {};
        ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes ias_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            &ias_accel_options,
            &instance_input,
            1,
            &ias_buffer_sizes
        ));

        CUdeviceptr d_ias_temp_buffer;
        CUDA_CHECK(cudaMalloc((void**)&d_ias_temp_buffer, ias_buffer_sizes.tempSizeInBytes));

        CUdeviceptr d_ias_output_buffer;
        CUDA_CHECK(cudaMalloc((void**)&d_ias_output_buffer, ias_buffer_sizes.outputSizeInBytes));

        OptixTraversableHandle ias_handle;
        OPTIX_CHECK(optixAccelBuild(
            context,
            0,
            &ias_accel_options,
            &instance_input,
            1,
            d_ias_temp_buffer,
            ias_buffer_sizes.tempSizeInBytes,
            d_ias_output_buffer,
            ias_buffer_sizes.outputSizeInBytes,
            &ias_handle,
            nullptr,
            0
        ));

        CUDA_CHECK(cudaFree((void*)d_ias_temp_buffer));


        // ----------------------------------------------------------
        // SBT
        // ----------------------------------------------------------
        CUdeviceptr d_rg;
        const size_t raygen_record_size = sizeof(RayGenRecord);
        CUDA_CHECK(cudaMalloc((void**)&d_rg, raygen_record_size));
        RayGenRecord rg = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg, &rg));
        CUDA_CHECK(cudaMemcpy((void*)d_rg, &rg, sizeof(rg), cudaMemcpyHostToDevice));

        CUdeviceptr d_ms;
        const size_t miss_record_size = sizeof(MissRecord);
        CUDA_CHECK(cudaMalloc((void**)&d_ms, miss_record_size));
        MissRecord   ms = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &ms));
        CUDA_CHECK(cudaMemcpy((void*)d_ms, &ms, sizeof(ms), cudaMemcpyHostToDevice));

        const int NUM_HIT_RECORDS = 8;
        HitGroupRecord hg[NUM_HIT_RECORDS];

        // Diffuse materials
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg[0]));
        hg[0].data.diffuse_color = make_float3(0.8f, 0.2f, 0.2f);
        hg[0].data.refraction_index = 1.0f; // Opaque

        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg[1]));
        hg[1].data.diffuse_color = make_float3(0.2f, 0.8f, 0.2f);
        hg[1].data.refraction_index = 1.0f; // Opaque

        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg[2]));
        hg[2].data.diffuse_color = make_float3(0.6f, 0.6f, 0.6f);
        hg[2].data.refraction_index = 1.0f; // Opaque (ground)

        // Glass materials
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_glass_pg, &hg[3]));
        hg[3].data.diffuse_color = make_float3(0.9f, 0.9f, 1.0f); // Slight blue tint
        hg[3].data.refraction_index = 1.5f; // Glass IOR

        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_glass_pg, &hg[4]));
        hg[4].data.diffuse_color = make_float3(1.0f, 0.9f, 0.9f); // Slight red tint
        hg[4].data.refraction_index = 1.5f; // Glass IOR

        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_glass_pg, &hg[5]));
        hg[5].data.diffuse_color = make_float3(0.9f, 1.0f, 0.9f); // Slight green tint
        hg[5].data.refraction_index = 1.5f; // Glass IOR

        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg[6]));
        hg[6].data.diffuse_color = make_float3(0.2f, 0.2f, 0.8f);  // Blue
        hg[6].data.refraction_index = 1.0f;

        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_glass_pg, &hg[7]));
        hg[7].data.diffuse_color = make_float3(0.2f, 0.2f, 0.8f);  // Blue
        hg[7].data.refraction_index = 1.5f;

        CUdeviceptr d_hg;
        const size_t hit_record_size = sizeof(HitGroupRecord);
        CUDA_CHECK(cudaMalloc((void**)&d_hg, hit_record_size * NUM_HIT_RECORDS));
        CUDA_CHECK(cudaMemcpy((void*)d_hg, &hg, sizeof(HitGroupRecord) * NUM_HIT_RECORDS, cudaMemcpyHostToDevice));

        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = d_rg;
        sbt.missRecordBase = d_ms;
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = d_hg;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
        sbt.hitgroupRecordCount = NUM_HIT_RECORDS;

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

            static float light1_pos[3] = { 2.0f, 3.0f, 2.0f };
            static float light1_col[3] = { 1.0f, 0.4f, 0.4f };
            static float light2_pos[3] = { 2.0f, 3.0f, -2.0f };
            static float light2_col[3] = { 0.4f, 0.4f, 1.0f };

            // Disable interaction when camera is captured
            if (g_mouse_captured) {
                ImGui::BeginDisabled();
            }

            ImGui::Text("Light 1");
            if (ImGui::DragFloat3("Light Position 1", light1_pos, 0.1f, -10.0f, 10.0f), ImGuiSliderFlags_NoInput) {
                params.light_position[0] = make_float3(light1_pos[0], light1_pos[1], light1_pos[2]);
            }
            if (ImGui::ColorEdit3("Light Color 1", light1_col), ImGuiSliderFlags_NoInput) {
                params.light_color[0] = make_float3(light1_col[0], light1_col[1], light1_col[2]);
            }
            ImGui::Separator();
            ImGui::Text("Light 2");
            if (ImGui::DragFloat3("Light Position 2", light2_pos, 0.1f, -10.0f, 10.0f), ImGuiSliderFlags_NoInput) {
                params.light_position[1] = make_float3(light2_pos[0], light2_pos[1], light2_pos[2]);
            }
            if (ImGui::ColorEdit3("Light Color 2", light2_col), ImGuiSliderFlags_NoInput) {
                params.light_color[1] = make_float3(light2_col[0], light2_col[1], light2_col[2]);
            }

            if (g_mouse_captured) {
                ImGui::EndDisabled();
            }

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
        CUDA_CHECK(cudaFree((void*)d_vertices));
        CUDA_CHECK(cudaFree((void*)d_indices));
        CUDA_CHECK(cudaFree((void*)d_ground_vertices));
        CUDA_CHECK(cudaFree((void*)d_ground_indices));
        CUDA_CHECK(cudaFree((void*)d_cube_gas_output_buffer));
        CUDA_CHECK(cudaFree((void*)d_ground_gas_output_buffer));
        CUDA_CHECK(cudaFree((void*)d_ias_output_buffer));
        CUDA_CHECK(cudaFree((void*)d_instances));
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