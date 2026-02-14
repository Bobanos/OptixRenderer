cmake_minimum_required(VERSION 3.25)
project(OptiXRenderer LANGUAGES C CXX CUDA)

# ------------------------------------------------------------------------------
# Language standards
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Set this according to your GPU (RTX 30xx = 86, RTX 40xx = 89)
set(CMAKE_CUDA_ARCHITECTURES 86)

# ------------------------------------------------------------------------------
# CUDA
# ------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------------------
# OptiX
# ------------------------------------------------------------------------------
set(OPTIX_ROOT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0")
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT}/include")

# ------------------------------------------------------------------------------
# Third-party libraries
# ------------------------------------------------------------------------------


# GLFW
add_subdirectory(external/glfw)

# GLAD
add_library(glad external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)

# ImGui
add_library(imgui
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    external/imgui
    external/imgui/backends
)

target_link_libraries(imgui PUBLIC glfw)

# ------------------------------------------------------------------------------
# Host executable (CPU-side code ONLY)
# ------------------------------------------------------------------------------
add_executable(optix_renderer
    src/main.cpp
    src/renderer.cpp
)

target_include_directories(optix_renderer PRIVATE
    src
    ${OPTIX_INCLUDE_DIR}
)

target_link_libraries(optix_renderer PRIVATE
    CUDA::cudart
    glfw
    glad
    imgui
)

set_target_properties(optix_renderer PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# ------------------------------------------------------------------------------
# OptiX device programs (.cu -> .optixir)
# ------------------------------------------------------------------------------

function(compile_optix_ir target_name)
    set(oneValueArgs OUTPUT_DIR)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(OPTIX "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    file(MAKE_DIRECTORY ${OPTIX_OUTPUT_DIR})

    set(optix_outputs)

    foreach(src ${OPTIX_SOURCES})
        get_filename_component(fname ${src} NAME_WE)
        set(output ${OPTIX_OUTPUT_DIR}/${fname}.optixir)

        add_custom_command(
            OUTPUT ${output}
            COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
                --optix-ir
                --std=c++17
                --use_fast_math
                -lineinfo
                -I${OPTIX_INCLUDE_DIR}
                -I${CMAKE_SOURCE_DIR}
                ${src}
                -o ${output}
            DEPENDS ${src}
            COMMENT "Compiling OptiX IR: ${src}"
            VERBATIM
        )

        list(APPEND optix_outputs ${output})
    endforeach()

    add_custom_target(${target_name} DEPENDS ${optix_outputs})
endfunction()

# ------------------------------------------------------------------------------
# OptiX shader compilation target
# ------------------------------------------------------------------------------
compile_optix_ir(optix_shaders
    OUTPUT_DIR ${CMAKE_BINARY_DIR}/generated/optixir
    SOURCES
        ${CMAKE_SOURCE_DIR}/cuda/SimplePathTracer.cu
)

# Ensure shaders are built before the executable runs
add_dependencies(optix_renderer optix_shaders)

