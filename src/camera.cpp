#include "camera.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


CameraController::CameraController(float3 position, float3 look_at, float3 up, float vfov, float aspect_ratio)
        : position(position), look_at(look_at), world_up(up),vfov(vfov), aspect_ratio(aspect_ratio),
        yaw(-90.0f), pitch(0.0f), movement_speed(2.5f), mouse_sensitivity(0.1f),
        first_mouse(true), last_x(0), last_y(0)
    {
        updateCameraVectors();
    }

void CameraController::processKeyboard(GLFWwindow* window, float deltaTime) {
    float velocity = movement_speed * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        position = position + velocity * front;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        position = position - velocity * front;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        position = position - velocity * right;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        position = position + velocity * right;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        position = position - velocity * world_up;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        position = position + velocity * world_up;
}

void CameraController::processMouseMovement(float xpos, float ypos) {
    if (first_mouse) {
        last_x = xpos;
        last_y = ypos;
        first_mouse = false;
    }

    float xoffset = xpos - last_x;
    float yoffset = last_y - ypos; // Reversed: y ranges bottom to top
    last_x = xpos;
    last_y = ypos;

    xoffset *= mouse_sensitivity;
    yoffset *= mouse_sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // Constrain pitch
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    updateCameraVectors();
}

void CameraController::processMouseScroll(float yoffset) {
    movement_speed += yoffset * 0.5f;
    if (movement_speed < 0.1f)
        movement_speed = 0.1f;
    if (movement_speed > 10.0f)
        movement_speed = 10.0f;
}

// Returns Camera struct (from optix_params.h) for OptiX
Camera CameraController::getCameraData() const {
    Camera cam_data;

    float theta = vfov * M_PI / 180.0f;
    float viewport_height = 2.0f * std::tan(theta / 2.0f);
    float viewport_width = aspect_ratio * viewport_height;

    cam_data.origin = position;
    cam_data.horizontal = viewport_width * right;
    cam_data.vertical = viewport_height * up;
    cam_data.lower_left_corner = position - cam_data.horizontal * 0.5f - cam_data.vertical * 0.5f - front;

    return cam_data;
}

void CameraController::updateCameraVectors() {
    // Calculate new front vector
    float3 new_front;
    new_front.x = std::cos(yaw * M_PI / 180.0f) * std::cos(pitch * M_PI / 180.0f);
    new_front.y = std::sin(pitch * M_PI / 180.0f);
    new_front.z = std::sin(yaw * M_PI / 180.0f) * std::cos(pitch * M_PI / 180.0f);
    front = normalize(new_front);

    // Recalculate right and up vectors
    right = normalize(cross(front, world_up));
    up = normalize(cross(right, front));
}

