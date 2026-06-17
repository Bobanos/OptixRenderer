#pragma once

#include <cuda_runtime.h>
#include <GLFW/glfw3.h>
#include "optix_params.h"
#include "float3_math.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class CameraController {
public:
    CameraController(float3 position, float3 look_at, float3 up, float vfov, float aspect_ratio);

    void processKeyboard(GLFWwindow* window, float deltaTime);
    void processMouseMovement(float xpos, float ypos);
    void processMouseScroll(float yoffset);

    Camera getCameraData() const;// Returns Camera struct (from optix_params.h) for OptiX

    float3 getPosition() const { return position; }
    float3 getLookAt() const { return look_at; }
    float3 getFront() const { return front; }
    float3 getUp() const { return up; }
    float getSpeed() const { return movement_speed; }

	void setPosition(const float3& new_position) { position = new_position; updateCameraVectors(); }
    void setLookAt(const float3& new_look_at) {
        look_at = new_look_at;
        front = normalize(look_at - position);
        updateCameraVectors();
    }
    void setFront(const float3& new_front) {
        front = normalize(new_front);
        look_at = position + front;
        // Update yaw and pitch from front vector
        yaw = std::atan2(front.z, front.x) * 180.0f / M_PI;
        pitch = std::asin(front.y) * 180.0f / M_PI;
        updateCameraVectors();
    }

private:
    void updateCameraVectors();

    // Camera attributes
    float3 position;
    float3 look_at;
    float3 world_up;
    float3 front;
    float3 right;
    float3 up;

    // Euler angles
    float yaw;
    float pitch;

    // Camera options
    float vfov;
    float aspect_ratio;
    float movement_speed;
    float mouse_sensitivity;

    // Mouse state
    bool first_mouse;
    float last_x;
    float last_y;
};