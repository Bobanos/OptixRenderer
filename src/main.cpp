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

#include <chrono>
#include <filesystem>

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

// Global renderer instance
OptixRenderer* renderer = nullptr;

// Camera instance (global for mouse callback)
CameraController* g_camera = nullptr;
bool g_mouse_captured = false;

// ------------------------------------------------------------------
// Screenshots and output management
// ------------------------------------------------------------------

// Helper function to save screenshot as PPM (simple, no external deps)
void saveScreenshot(const uchar4* pixel_buffer, int width, int height,
    const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[Screenshot] Failed to open: " << filename << std::endl;
        return;
    }

    // PPM header
    file << "P6\n";
    file << width << " " << height << "\n";
    file << "255\n";

    // Write RGB data (skip alpha channel)
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            file.put(pixel_buffer[idx].x);
            file.put(pixel_buffer[idx].y);
            file.put(pixel_buffer[idx].z);
        }
    }

    file.close();
    std::cout << "[Screenshot] Saved to: " << filename << std::endl;
}

// Generate timestamped filename with scene name
std::string generateScreenshotFilename(const std::string& scene_name)
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm* timeinfo = std::localtime(&time);
    std::stringstream ss;
    ss << "screenshots/"
        << scene_name << "_"
        << std::put_time(timeinfo, "%Y%m%d_%H%M%S")
        << "_" << std::setfill('0') << std::setw(3) << ms.count()
        << ".ppm";

    return ss.str();
}

void screenshot(const std::string& scene_name,std::string append, const uchar4* pixel_buffer, int width, int height) {
    // Ensure screenshots directory exists
    std::filesystem::create_directories("screenshots");

    std::string current_scene_name = scene_name;
    // Replace spaces with underscores
    std::replace(current_scene_name.begin(), current_scene_name.end(), ' ', '_');
	current_scene_name.append(append);

    // Generate timestamped filename
    std::string filename = generateScreenshotFilename(current_scene_name);
    std::vector<uchar4> pixel_data(width * height);
    CUDA_CHECK(cudaMemcpy(pixel_data.data(), pixel_buffer,
        width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
    saveScreenshot(pixel_data.data(), width, height, filename);
}

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

    // Setup Dear ImGui
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
		renderer->loadScene(SceneID::ALLIED_AVENGER);
        renderer->setupShaders();
        renderer->setupLighting();

        renderer->setupDenoiser();

        // Create display buffer
        ImGuiDisplayBuffer display(width, height);

        // Get initial scene data
        SceneData initial_scene = SceneManager::getSceneConfig(renderer->getCurrentSceneID());

        // Camera setup
        CameraController camera_controller(
            initial_scene.camera_position,
            initial_scene.camera_front,
            initial_scene.camera_up,
            initial_scene.camera_vfov,
            (float)width / (float)height
        );
        g_camera = &camera_controller;

        g_camera->setFront(initial_scene.camera_front);

        // Timing
        float deltaTime = 0.0f;
        float lastFrame = 0.0f;

        int spp_input = renderer->getParams().samples_per_pixel;
		int max_bounces_input = renderer->getParams().max_bounce_depth;
        float blendFactor = 0.0f;

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

			bool light_changed = false; 
            {
				renderer->updateCamera(camera_controller.getCameraData());
                // Render
                renderer->render(camera_controller.getCameraData(), renderer->getParams().samples_per_pixel);

				//renderer->postprocessAccum();

                //screenshot(renderer->getCurrentSceneName(), "_Accum", renderer->getPixelBuffer(), renderer->getWidth(), renderer->getHeight());

				renderer->runDenoiser(blendFactor);

                renderer->postprocessDenoised();

                //screenshot(renderer->getCurrentSceneName(), "_Denoised", renderer->getPixelBuffer(), renderer->getWidth(), renderer->getHeight());
                // Copy to display
                display.copyFromDevice((CUdeviceptr)renderer->getPixelBuffer());

                //glfwSetWindowShouldClose(window, true);
            }

            //================================================================================================================
            // ImGui viewport window
            ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar);

            // Display the texture
            ImGui::Image(
                (void*)(intptr_t)display.getTexture(),
                ImVec2(display.getWidth(), display.getHeight()),
                ImVec2(0, 1),  // UV coordinates (flip Y)
                ImVec2(1, 0)
            );
            ImVec2 window_pos1 = ImGui::GetWindowPos();
            ImVec2 window_size1 = ImGui::GetWindowSize();
            ImGui::End();

            //================================================================================================================
            // Camera Controls window
			//ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
            //ImGui::SetNextWindowSize(ImVec2(10, 10), ImGuiCond_Once);
            ImGui::Begin("Camera Controls");
            ImGui::Text("FPS: %.1f", 1.0f / deltaTime);
            ImGui::Text("Press TAB to toggle camera control");
            ImGui::Text("Camera Captured: %s", g_mouse_captured ? "Yes" : "No");

            float3 pos = camera_controller.getPosition();
            ImGui::Text("Position: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);

            float3 look = camera_controller.getFront();
            ImGui::Text("Front: (%.2f, %.2f, %.2f)", look.x, look.y, look.z);
            ImGui::Text("Samples Accumulated: (%.2d)", renderer->getCurrentSample());
            ImGui::Text("Speed: %.2f", camera_controller.getSpeed());

            ImGui::Separator();
            ImGui::Text("Controls:");
            ImGui::BulletText("WASD: Move");
            ImGui::BulletText("Q/E: Down/Up");
            ImGui::BulletText("Mouse: Look around");
            ImGui::BulletText("Scroll: Adjust speed");
            ImGui::Separator();

            ImVec2 window_pos2 = ImGui::GetWindowPos();
            ImVec2 window_size2 = ImGui::GetWindowSize();

            // Screenshot button
            if (ImGui::Button("Save Screenshot", ImVec2(-1, 0))) {
                // Ensure screenshots directory exists
                std::filesystem::create_directories("screenshots");

                // Get current scene name for filename
                std::string scene_name = renderer->getCurrentSceneName();
                // Replace spaces with underscores
                std::replace(scene_name.begin(), scene_name.end(), ' ', '_');

                // Generate timestamped filename
                std::string filename = generateScreenshotFilename(scene_name);

                // Copy pixel data from GPU to CPU
                std::vector<uchar4> pixel_data(width * height);
                CUDA_CHECK(cudaMemcpy(pixel_data.data(), renderer->getPixelBuffer(),
                    width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));

                // Save screenshot
                saveScreenshot(pixel_data.data(), width, height, filename);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::SetTooltip("Saves current viewport to screenshots/ folder as PPM");
            }
            ImGui::Separator();

            // Save Camera Values Button
            if (ImGui::Button("Save Camera Values", ImVec2(-1, 0))) {
                // Ensure logs directory exists
                std::filesystem::create_directories("logs");

                // Create log file with timestamp
                auto now = std::chrono::system_clock::now();
                auto time = std::chrono::system_clock::to_time_t(now);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) % 1000;

                std::stringstream clipboard_text;
                clipboard_text << "scene.camera_position = make_float3("
                    << std::fixed << std::setprecision(2)
                    << pos.x << "f, " << pos.y << "f, " << pos.z << "f);\n";
                clipboard_text << "scene.camera_front = make_float3(" << std::fixed << std::setprecision(2)
                    << look.x << "f, " << look.y << "f, " << look.z << "f);\n";

                // Copy to clipboard using ImGui
                ImGui::SetClipboardText(clipboard_text.str().c_str());

                std::tm* timeinfo = std::localtime(&time);
                // Single log file for all camera saves (append mode)
                std::string log_filename = "logs/camera_values.txt";

                // Write camera values to log file
                std::ofstream log_file(log_filename, std::ios::app);
                if (log_file) {
                    log_file << "=====================================\n";
                    log_file << "Scene: " << renderer->getCurrentSceneName() << "\n";
                    log_file << "Timestamp: " << std::put_time(timeinfo, "%Y-%m-%d %H:%M:%S") << "\n\n";
                    log_file << clipboard_text.str();
                    log_file.close();
                    std::cout << "[Camera] Saved to: " << log_filename << std::endl;
                }
                else {
                    std::cerr << "[Camera] Failed to open log file: " << log_filename << std::endl;
                }
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::SetTooltip("Saves current camera position and lookAt to logs/ folder");
            }
            ImGui::Separator();

            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::InputInt("Samples Per Pixel", &spp_input, 1, 10)) {
                spp_input = fmaxf(1, spp_input);
                renderer->setSamplesPerPixel(spp_input);
                renderer->resetBuffersOnCameraUpdate();
            }
            ImGui::Separator();
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::InputInt("Max bounces", &max_bounces_input, 1, 10)) {
                max_bounces_input = fmaxf(1, max_bounces_input);
                renderer->setMaxBounceDepth(max_bounces_input);
                renderer->resetBuffersOnCameraUpdate();
            }
            ImGui::End();

            //================================================================================================================
            // Light control
            ImGui::Begin("Lighting Controls");

			static float env_map_scale = renderer->getParams().envmap.scale;
			static float env_map_expo = renderer->getParams().envmap.exposure;

            static float light_intensity = renderer->getParams().light_intensity;
            
            if (g_mouse_captured) {
                ImGui::BeginDisabled();
            }

            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::DragFloat("Scale", &env_map_scale, 0.05f, 0.f, 100.0f)) {
                renderer->updateEnvmapParameters(env_map_scale, env_map_expo);
                light_changed = true;
            }

            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::DragFloat("Exposure", &env_map_expo, 0.05f, 100.0f, 1.0f)) {
                renderer->updateEnvmapParameters(env_map_scale, env_map_expo);
                light_changed = true;
            }

            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::DragFloat("Light Intensity", &light_intensity, 0.05f, 1.0f, 100.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                renderer->updateLightIntensity(light_intensity);
                light_changed = true;
            }
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::SliderFloat("Denoiser Blend factor", &blendFactor, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {}

            if (light_changed) {
                renderer->resetBuffersOnCameraUpdate(); // Clear accumulation when manually adjusting rotation
            }

            if (g_mouse_captured) {
                ImGui::EndDisabled();
            }
            ImVec2 window_pos3 = ImGui::GetWindowPos();
            ImVec2 window_size3 = ImGui::GetWindowSize();
            ImGui::End();

            //================================================================================================================
            ImGui::Begin("Scene Selection");
            static SceneID selected_scene = renderer->getCurrentSceneID();
            for (int i = 0; i < static_cast<int>(SceneID::COUNT); ++i) {
				SceneData iter_scene = SceneManager::getSceneConfig(static_cast<SceneID>(i));
                if (ImGui::RadioButton(iter_scene.name.c_str(), (int*)&selected_scene, (int)static_cast<SceneID>(i))) {
                    renderer->switchScene(static_cast<SceneID>(i));
					camera_controller.setPosition(iter_scene.camera_position);
                    camera_controller.setFront(iter_scene.camera_front);
                    renderer->resetBuffersOnCameraUpdate();
                }
            }
            ImVec2 window_pos4 = ImGui::GetWindowPos();
            ImVec2 window_size4 = ImGui::GetWindowSize();
            ImGui::Text("Current: %s", renderer->getCurrentSceneName().c_str());
            ImGui::End();
            //================================================================================================================

            ImGui::Begin("Window sizes");
            ImGui::Text("Current Position window 1: X = %.1f, Y = %.1f", window_pos1.x, window_pos1.y);
            ImGui::Text("Current Size window 1:     W = %.1f, H = %.1f", window_size1.x, window_size1.y);
            ImGui::Text("Current Position window 2: X = %.1f, Y = %.1f", window_pos2.x, window_pos2.y);
            ImGui::Text("Current Size window 2:     W = %.1f, H = %.1f", window_size2.x, window_size2.y);
            ImGui::Text("Current Position window 3: X = %.1f, Y = %.1f", window_pos3.x, window_pos3.y);
            ImGui::Text("Current Size window 3:     W = %.1f, H = %.1f", window_size3.x, window_size3.y);
            ImGui::Text("Current Position window 4: X = %.1f, Y = %.1f", window_pos4.x, window_pos4.y);
            ImGui::Text("Current Size window 4:     W = %.1f, H = %.1f", window_size4.x, window_size4.y);
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


            /* here i can benchmark
            // Ensure screenshots directory exists
            std::filesystem::create_directories("screenshots");

            // Get current scene name for filename
            std::string scene_name = renderer->getCurrentSceneName();
            // Replace spaces with underscores
            std::replace(scene_name.begin(), scene_name.end(), ' ', '_');

            // Generate timestamped filename
            std::string filename = generateScreenshotFilename(scene_name);
            std::vector<uchar4> pixel_data(width * height);
            CUDA_CHECK(cudaMemcpy(pixel_data.data(), renderer->getPixelBuffer(),
                width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
            saveScreenshot(pixel_data.data(), width, height, filename);
            glfwSetWindowShouldClose(window, true);
            */
        }

        std::cout << "SUCCESS: OptiX succesfuly completed." << std::endl;

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