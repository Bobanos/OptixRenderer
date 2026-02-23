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
// Geometry data for a cube
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
        pipeline_opts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_opts.usesMotionBlur = false;
        pipeline_opts.numPayloadValues = 3; //3 for RGB color
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
        OptixProgramGroup miss_pg = nullptr;
        OptixProgramGroup hitgroup_pg = nullptr;

        OptixProgramGroupDesc rg_desc = {};
        rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        rg_desc.raygen.module = module;
        rg_desc.raygen.entryFunctionName = "__raygen__rg";

        OptixProgramGroupDesc ms_desc = {};
        ms_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        ms_desc.miss.module = module;
        ms_desc.miss.entryFunctionName = "__miss__ms";

        OptixProgramGroupDesc hg_desc = {};
        hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hg_desc.hitgroup.moduleCH = module;
        hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

        OptixProgramGroupOptions pg_opts = {};

        OPTIX_CHECK(optixProgramGroupCreate(context, &rg_desc, 1, &pg_opts, log, &logSize, &raygen_pg));
        OPTIX_CHECK(optixProgramGroupCreate(context, &ms_desc, 1, &pg_opts, log, &logSize, &miss_pg));
        OPTIX_CHECK(optixProgramGroupCreate(context, &hg_desc, 1, &pg_opts, log, &logSize, &hitgroup_pg));

        // ----------------------------------------------------------
        // Pipeline
        // ----------------------------------------------------------
        OptixProgramGroup groups[] = { raygen_pg, miss_pg , hitgroup_pg };

        OptixPipelineLinkOptions link_opts = {};
        link_opts.maxTraceDepth = 2;

        OptixPipeline pipeline = nullptr;
        OPTIX_CHECK(optixPipelineCreate(
            context,
            &pipeline_opts,
            &link_opts,
            groups,
            3,
            log,
            &logSize,
            &pipeline
        ));


        // ----------------------------------------------------------
        // Build Geometry Acceleration Structure (Cube)
        // ----------------------------------------------------------

        float3 vertices2[8];

        for (int i = 0; i<(sizeof(vertices)/sizeof(float3));i++) {
            vertices[i].y = vertices[i].y + 2;
            vertices2[i] = vertices[i];
            vertices2[i].x = vertices2[i].x + 2;
        }

        // Upload vertices of cube1 to device
        CUdeviceptr d_vertices;
        CUDA_CHECK(cudaMalloc((void**)&d_vertices, sizeof(vertices)));
        CUDA_CHECK(cudaMemcpy((void*)d_vertices, vertices, sizeof(vertices), cudaMemcpyHostToDevice));

        // Upload indices to device
        CUdeviceptr d_indices;
        CUDA_CHECK(cudaMalloc((void**)&d_indices, sizeof(indices)));
        CUDA_CHECK(cudaMemcpy((void*)d_indices, indices, sizeof(indices), cudaMemcpyHostToDevice));

        // Upload vertices of cube2 to device
        CUdeviceptr d_vertices2;
        CUDA_CHECK(cudaMalloc((void**)&d_vertices2, sizeof(vertices2)));
        CUDA_CHECK(cudaMemcpy((void*)d_vertices2, vertices2, sizeof(vertices2), cudaMemcpyHostToDevice));

        // Upload vertices of plane to device
        CUdeviceptr d_ground_vertices;
        CUDA_CHECK(cudaMalloc((void**)&d_ground_vertices, sizeof(ground_vertices)));
        CUDA_CHECK(cudaMemcpy((void*)d_ground_vertices, ground_vertices, sizeof(ground_vertices), cudaMemcpyHostToDevice));

        // Upload indices of plane to device
        CUdeviceptr d_ground_indices;
        CUDA_CHECK(cudaMalloc((void**)&d_ground_indices, sizeof(ground_indices)));
        CUDA_CHECK(cudaMemcpy((void*)d_ground_indices, ground_indices, sizeof(ground_indices), cudaMemcpyHostToDevice));

        // Setup triangle input
        OptixBuildInput triangle_input[3] = {};

        triangle_input[0].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input[0].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input[0].triangleArray.vertexStrideInBytes = sizeof(float3);
        triangle_input[0].triangleArray.numVertices = 8;
        triangle_input[0].triangleArray.vertexBuffers = &d_vertices;
        triangle_input[0].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input[0].triangleArray.indexStrideInBytes = sizeof(uint3);
        triangle_input[0].triangleArray.numIndexTriplets = 12;
        triangle_input[0].triangleArray.indexBuffer = d_indices;
        unsigned int cube_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        triangle_input[0].triangleArray.flags = cube_flags;
        triangle_input[0].triangleArray.numSbtRecords = 1;

        triangle_input[1].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input[1].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input[1].triangleArray.vertexStrideInBytes = sizeof(float3);
        triangle_input[1].triangleArray.numVertices = 8;
        triangle_input[1].triangleArray.vertexBuffers = &d_vertices2;
        triangle_input[1].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input[1].triangleArray.indexStrideInBytes = sizeof(uint3);
        triangle_input[1].triangleArray.numIndexTriplets = 12;
        triangle_input[1].triangleArray.indexBuffer = d_indices;
        unsigned int cube2_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        triangle_input[1].triangleArray.flags = cube2_flags;
        triangle_input[1].triangleArray.numSbtRecords = 1;

        triangle_input[2].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input[2].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input[2].triangleArray.vertexStrideInBytes = sizeof(float3);
        triangle_input[2].triangleArray.numVertices = 4;
        triangle_input[2].triangleArray.vertexBuffers = &d_ground_vertices;
        triangle_input[2].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input[2].triangleArray.indexStrideInBytes = sizeof(uint3);
        triangle_input[2].triangleArray.numIndexTriplets = 2;
        triangle_input[2].triangleArray.indexBuffer = d_ground_indices;
        unsigned int ground_flags[2] = { OPTIX_GEOMETRY_FLAG_NONE };
        triangle_input[2].triangleArray.flags = ground_flags;
        triangle_input[2].triangleArray.numSbtRecords = 1;



        // Setup acceleration structure build options
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        

        // Query memory requirements
        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            &accel_options,
            triangle_input,
            3,  //num of geometries
            &gas_buffer_sizes
        ));

        // Allocate temporary buffers
        CUdeviceptr d_temp_buffer;
        CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer, gas_buffer_sizes.tempSizeInBytes));

        CUdeviceptr d_gas_output_buffer;
        CUDA_CHECK(cudaMalloc((void**)&d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes));

        // Build acceleration structure
        OptixTraversableHandle gas_handle;
        OPTIX_CHECK(optixAccelBuild(
            context,
            0,  // CUDA stream
            &accel_options,
            triangle_input,
            3,  //num of geometries
            d_temp_buffer,
            gas_buffer_sizes.tempSizeInBytes,
            d_gas_output_buffer,
            gas_buffer_sizes.outputSizeInBytes,
            &gas_handle,
            nullptr,
            0
        ));

        // Free temporary buffer (we don't need it anymore)
        CUDA_CHECK(cudaFree((void*)d_temp_buffer));

        std::cout << "Acceleration structure built successfully" << std::endl;

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

        CUdeviceptr d_hg;
        const size_t hit_record_size = sizeof(HitGroupRecord);
        CUDA_CHECK(cudaMalloc((void**)&d_hg, hit_record_size * 3));

        HitGroupRecord hg[3];
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg[0]));
        hg[0].data.diffuse_color = make_float3(0.8f, 0.2f, 0.2f);
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg[1]));
        hg[1].data.diffuse_color = make_float3(0.2f, 0.8f, 0.2f);
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg[2]));
        hg[2].data.diffuse_color = make_float3(0.6f, 0.6f, 0.6f);

        CUDA_CHECK(cudaMemcpy((void*)d_hg, &hg, sizeof(HitGroupRecord) * 3, cudaMemcpyHostToDevice));

        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = d_rg;
        sbt.missRecordBase = d_ms;
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = d_hg;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
        sbt.hitgroupRecordCount = 3;

        // ----------------------------------------------------------
        // Output buffer + Display
        // ----------------------------------------------------------
        CUdeviceptr d_pixels;
        CUDA_CHECK(cudaMalloc((void**)&d_pixels, width * height * sizeof(uchar4)));

        Params params = {};
        params.image = (uchar4*)d_pixels;
        params.width = width;
        params.height = height;
        params.traversable = gas_handle;
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
        CUDA_CHECK(cudaFree((void*)d_vertices2));
        CUDA_CHECK(cudaFree((void*)d_ground_vertices));
        CUDA_CHECK(cudaFree((void*)d_ground_indices));
        CUDA_CHECK(cudaFree((void*)d_gas_output_buffer)); 
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