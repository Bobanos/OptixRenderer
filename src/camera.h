#pragma once

#include <cuda_runtime.h>
#include <GLFW/glfw3.h>
#include "optix_params.h"
#include <cmath>

class CameraController {
public:
    CameraController(float3 position, float3 look_at, float3 up, float vfov, float aspect_ratio);

    void processKeyboard(GLFWwindow* window, float deltaTime);
    void processMouseMovement(float xpos, float ypos);
    void processMouseScroll(float yoffset);

    Camera getCameraData() const;// Returns Camera struct (from optix_params.h) for OptiX

    float3 getPosition() const { return position; }
    float getSpeed() const { return movement_speed; }

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


// Helper functions for float3 operations
inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline float3 operator*(float t, const float3& v) {
    return make_float3(t * v.x, t * v.y, t * v.z);
}

inline float3 operator*(const float3& v, float t) {
    return t * v;
}

inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline float3 normalize(const float3& v) {
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x / len, v.y / len, v.z / len);
}
