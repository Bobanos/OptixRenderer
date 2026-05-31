#pragma once

#include <cuda_runtime.h>
#include <cmath>

#ifdef __CUDA_ARCH__
    #define FLOAT3_DEVICE __device__
#else
    #define FLOAT3_DEVICE
#endif

FLOAT3_DEVICE inline float3 make_float3(const float a) {
    return make_float3(a, a, a);
}

// ------------------------------------------------------------------
// Arithmetic Operators
// ------------------------------------------------------------------

FLOAT3_DEVICE inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

FLOAT3_DEVICE inline float3 operator+(const float a, const float3& b) {
    return make_float3(a + b.x, a + b.y, a + b.z);
}

FLOAT3_DEVICE inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

FLOAT3_DEVICE inline float3 operator*(float t, const float3& v) {
    return make_float3(t * v.x, t * v.y, t * v.z);
}

FLOAT3_DEVICE inline float3 operator*(const float3& v, float t) {
    return make_float3(v.x * t, v.y * t, v.z * t);
}

FLOAT3_DEVICE inline float3 operator*(const float3& v, const float3& t) {
    return make_float3(v.x * t.x, v.y * t.y, v.z * t.z);
}

FLOAT3_DEVICE inline float3 operator/(const float3& v, const float3& t) {
    return make_float3(v.x / t.x, v.y / t.y, v.z / t.z);
}

FLOAT3_DEVICE inline float3 operator/(const float3& v, float t) {
    return make_float3(v.x / t, v.y / t, v.z / t);
}

FLOAT3_DEVICE inline float3 operator+(const float3& v, float t) {
    return make_float3(v.x + t, v.y + t, v.z + t);
}

FLOAT3_DEVICE inline float3 operator-(const float3& v, float t) {
    return make_float3(v.x - t, v.y - t, v.z - t);
}

FLOAT3_DEVICE inline float3 operator-(const float3& v) {
    return make_float3(-v.x, -v.y, -v.z);
}

// ------------------------------------------------------------------
// Vector Operations
// ------------------------------------------------------------------

FLOAT3_DEVICE inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

FLOAT3_DEVICE inline float length(const float3& v) {
    return sqrt(dot(v, v));
}

FLOAT3_DEVICE inline float length_squared(const float3& v) {
    return dot(v, v);
}

FLOAT3_DEVICE inline float3 normalize(const float3& v) {
    float len = length(v);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

FLOAT3_DEVICE inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

FLOAT3_DEVICE inline float distance(const float3& a, const float3& b) {
    return length(a - b);
}

FLOAT3_DEVICE inline float distance_squared(const float3& a, const float3& b) {
    float3 diff = a - b;
    return dot(diff, diff);
}

FLOAT3_DEVICE inline float3 clamp(const float3& v, float min_val, float max_val) {
    return make_float3(
    fmin(fmax(v.x, min_val), max_val),
    fmin(fmax(v.y, min_val), max_val),
    fmin(fmax(v.z, min_val), max_val)
    );
}

FLOAT3_DEVICE inline float clamp(const float v, float min_val, float max_val) {
    return fmin(fmax(v, min_val), max_val);
}

FLOAT3_DEVICE inline float3 abs(const float3& v) {
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}

FLOAT3_DEVICE inline float3 refract(const float3& v, const float3& n, float eta) {
    float cos_theta = dot(v, n);
    float k = 1.0f - eta * eta * (1.0f - cos_theta * cos_theta);
    if (k < 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return eta * v + (eta * cos_theta - sqrt(k)) * n;
}

FLOAT3_DEVICE inline float3 pow(const float3& v, float exp) {
    return make_float3(::pow(v.x, exp), ::pow(v.y, exp), ::pow(v.z, exp));
}

FLOAT3_DEVICE inline float3 sqrt(const float3& v) {
    return make_float3(::sqrt(v.x), ::sqrt(v.y), ::sqrt(v.z));
}

FLOAT3_DEVICE inline float3 exp(const float3& v) {
    return make_float3(::exp(v.x), ::exp(v.y), ::exp(v.z));
}

FLOAT3_DEVICE inline float3 log(const float3& v) {
    return make_float3(::log(v.x), ::log(v.y), ::log(v.z));
}

FLOAT3_DEVICE inline bool nonZero(const float3& v) {
    return (v.x != 0.0f) || (v.y != 0.0f) || (v.z != 0.0f);
}