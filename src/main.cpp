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
typedef Record<HitGroupDataLambert> HitGroupRecordLambert;
typedef Record<HitGroupDataGlass> HitGroupRecordGlass;

// Global renderer instance
OptixRenderer* renderer = nullptr;

// Camera instance (global for mouse callback)
CameraController* g_camera = nullptr;
bool g_mouse_captured = false;

// ------------------------------------------------------------------
// Callbacks and input processing
// ------------------------------------------------------------------
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
int main() {
    // ----------------------------------------------------------
    // Setup OpenGL
    // ----------------------------------------------------------
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

    try {
        // Initialize renderer
        renderer = new OptixRenderer(width, height);
        renderer->initCUDA();
        renderer->initOptix();
        renderer->loadScene("assets/6887_allied_avenger.obj", "assets/6887_allied_avenger.mtl");
		renderer->loadMap("assets/golden_gate_hills_2k.hdr");
        renderer->setupShaders();
        renderer->buildAccelerationStructures();
        renderer->setupLighting();

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
        // Ship transform state
        // ----------------------------------------------------------
        float ship_rotation_x = 0.0f;
        float ship_rotation_y = 0.0f;
        float ship_rotation_z = 0.0f;
        bool  auto_rotate = false;
        float rotation_speed = 1.0f;

        // Update initial transform
        renderer->updateShipTransform(ship_rotation_x, ship_rotation_y, ship_rotation_z);

        // Timing
        float deltaTime = 0.0f;
        float lastFrame = 0.0f;

        int spp_input = renderer->getParams().samples_per_pixel;
        float rr_threshold = renderer->getParams().rr_threshold;
        float rr_decay = renderer->getParams().rr_decay;


        // ----------------------------------------------------------
        // Render Loop
        // ----------------------------------------------------------
        while (!glfwWindowShouldClose(window))
        {
            // Calculate delta time
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

            // Update ship rotation
            if (auto_rotate)
                ship_rotation_y += rotation_speed * DEG2RAD * deltaTime;

            // Only update transform if rotation changed
            static float last_rotation_x = 0.0f;
            static float last_rotation_y = 0.0f;
            static float last_rotation_z = 0.0f;

            bool rotation_changed = (ship_rotation_x != last_rotation_x) ||
                (ship_rotation_y != last_rotation_y) ||
                (ship_rotation_z != last_rotation_z);

			bool light_changed = false; 

            if (rotation_changed) {
                renderer->updateShipTransform(ship_rotation_x, ship_rotation_y, ship_rotation_z);
                last_rotation_x = ship_rotation_x;
                last_rotation_y = ship_rotation_y;
                last_rotation_z = ship_rotation_z;
            }

            {
				renderer->updateCamera(camera_controller.getCameraData());
                // Render
                renderer->render(camera_controller.getCameraData(), renderer->getParams().samples_per_pixel);

                // Copy to display
                display.copyFromDevice((CUdeviceptr)renderer->getPixelBuffer());
            }

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

            static float light0_pos[3] = { 2.0f, 3.0f, 2.0f };
            static float light0_col[3] = { 1.0f, 0.4f, 0.4f };
            static float light1_pos[3] = { 2.0f, 3.0f, -2.0f };
            static float light1_col[3] = { 0.4f, 0.4f, 1.0f };
            static float light2_dir[3] = { 0.0f, -1.0f, -0.2f };
            static float light2_col[3] = { 0.5f, 0.5f, 0.5f };

			static float env_map_scale = renderer->getParams().envmap_scale;
			static float env_map_expo = renderer->getParams().envmap_exposure;
            

            if (g_mouse_captured) {
                ImGui::BeginDisabled();
            }

            // Point Light 1
            ImGui::Text("Point Light 1 (Red)");
            if (ImGui::DragFloat3("Light 1 Position", light0_pos, 0.1f, -20.0f, 20.0f)) {
				renderer->updateLightParametersPos(0, make_float3(light0_pos[0], light0_pos[1], light0_pos[2]));
				light_changed = true;
            }
            if (ImGui::ColorEdit3("Light 1 Color", light0_col)) {
                renderer->updateLightParametersColor(0, make_float3(light0_col[0], light0_col[1], light0_col[2]));
                light_changed = true;
            }
            ImGui::Separator();

            // Point Light 2
            ImGui::Text("Point Light 2 (Blue)");
            if (ImGui::DragFloat3("Light 2 Position", light1_pos, 0.1f, -20.0f, 20.0f)) {
				renderer->updateLightParametersPos(1, make_float3(light1_pos[0], light1_pos[1], light1_pos[2]));
                light_changed = true;
            }
            if (ImGui::ColorEdit3("Light 2 Color", light1_col)) {
				renderer->updateLightParametersColor(1, make_float3(light1_col[0], light1_col[1], light1_col[2]));
                light_changed = true;
            }
            ImGui::Separator();

            // Directional Light
            ImGui::Text("Directional Light");
            if (ImGui::DragFloat3("Light 3 Direction", light2_dir, 0.05f, 100.0f, 1.0f)) {
				renderer->updateLightParametersPos(2, normalize(make_float3(light2_dir[0], light2_dir[1], light2_dir[2])));
                light_changed = true;
            }
            if (ImGui::ColorEdit3("Light 3 Color", light2_col)) {
				renderer->updateLightParametersColor(2, make_float3(light2_col[0], light2_col[1], light2_col[2]));
                light_changed = true;
            }

            ImGui::Text("Enviromental Map Settings");
            if (ImGui::DragFloat("Scale", &env_map_scale, 0.05f, 0.f, 100.0f)) {
                renderer->updateEnvmapParameters(env_map_scale, env_map_expo);
                light_changed = true;
            }

            if (ImGui::DragFloat("Exposure", &env_map_expo, 0.05f, 100.0f, 1.0f)) {
                renderer->updateEnvmapParameters(env_map_scale, env_map_expo);
                light_changed = true;
            }

            if (light_changed) {
                renderer->resetAccumulationBuffer(); // Clear accumulation when manually adjusting rotation
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

            float rx_deg = ship_rotation_x * RAD2DEG;
            float ry_deg = ship_rotation_y * RAD2DEG;
            float rz_deg = ship_rotation_z * RAD2DEG;

            if (ImGui::SliderFloat("Rotation X (deg)", &rx_deg, -180.0f, 180.0f)) {
                ship_rotation_x = rx_deg * DEG2RAD;
                rotation_changed = true;
            }
            if (ImGui::SliderFloat("Rotation Y (deg)", &ry_deg, -180.0f, 180.0f)) {
                ship_rotation_y = ry_deg * DEG2RAD;
                rotation_changed = true;
            }
            if (ImGui::SliderFloat("Rotation Z (deg)", &rz_deg, -180.0f, 180.0f)) {
                ship_rotation_z = rz_deg * DEG2RAD;
                rotation_changed = true;
            }

            if (rotation_changed) {
                renderer->updateShipTransform(ship_rotation_x, ship_rotation_y, ship_rotation_z);
				renderer->resetAccumulationBuffer(); // Clear accumulation when manually adjusting rotation
            }

            if (ImGui::Button("Reset Rotation")) {
                ship_rotation_x = 0.0f;
                ship_rotation_y = 0.0f;
                ship_rotation_z = 0.0f;
                renderer->updateShipTransform(ship_rotation_x, ship_rotation_y, ship_rotation_z);
            }
            ImGui::End();

            ImGui::Begin("Path Tracer Settings");
            // Samples per 
            if (ImGui::InputInt("Samples Per Pixel", &spp_input, 1, 10)) {
                spp_input = fmaxf(1, spp_input);
                renderer->setSamplesPerPixel(spp_input);
                renderer->resetAccumulationBuffer();
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::SetTooltip("Number of samples to accumulate per frame.\nLower = faster but noisier.\nHigher = slower but cleaner.");
            }

            ImGui::Separator();
            ImGui::Text("Russian Roulette");

            // RR Threshold slider
            if (ImGui::SliderFloat("RR Threshold##threshold", &rr_threshold, 0.5f, 1.0f, "%.3f")) {
                renderer->setRussianRouletteThreshold(rr_threshold);
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::SetTooltip("Initial probability of continuing path (higher = more bounces)");
                }
            }

            // RR Decay slider
            if (ImGui::SliderFloat("RR Decay##decay", &rr_decay, 0.85f, 0.99f, "%.3f")) {
                renderer->setRussianRouletteDecay(rr_decay);
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::SetTooltip("Rate of probability decrease per bounce (higher = longer paths)");
                }
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
        delete renderer;
    }
    catch (const std::exception& e) {
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