#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>
#include <optix.h>

#include <iostream>


// ------------------------------------------------------------------
// Error helpers
// ------------------------------------------------------------------

#define CUDA_CHECK(x) r_util::cudaCheck((x), __FILE__, __LINE__)

#define OPTIX_CHECK(x) r_util::optixCheck((x), __FILE__, __LINE__)

namespace r_util {
    inline void cudaCheck(cudaError_t error, const char* file, int line) {
        if (error != cudaSuccess) {
            std::cerr << "CUDA error at " << file << ":" << line << ": "
                << cudaGetErrorString(error) << std::endl;
            std::exit(1);
        }
    }

    inline void optixCheck(OptixResult res, const char* file, int line) {
        if (res != OPTIX_SUCCESS) {
            std::cerr << "OptiX error at " << file << ":" << line << ": "
                << res << std::endl;
            std::exit(1);
        }
    }
}
