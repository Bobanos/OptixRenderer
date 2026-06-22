#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <cuda_runtime.h>

#include "optix_params.h"
#include "stb_image.h"
#include "float3_math.h"
#include "tiny_obj_loader.h"

struct ObjMaterial {
    std::string name;
    float3      albedo = { 0.8f, 0.8f, 0.8f };
    float3      emission = { 0.f,  0.f,  0.f };
    float       ior = 1.5f;
    float       shininess = 32.f;
	float3      specular = { 1.0f, 1.0f, 1.0f };
    bool        is_glass = false;
    bool        is_emissive = false;


    float roughness  = 0.0f;
    float metallic   = 0.0f;
    float3 base_color = { 0.8f, 0.8f, 0.8f };

    struct TexturePaths {
        std::string diffuse_path = "";
        std::string specular_path = ""; //Red channel: Occlusion, Green channel : Roughness, Blue channel : Metalness
        std::string bump_path = "";
        std::string alpha_path = "";
		std::string emissive_path = "";
	} texture_paths;


    // Construct from tinyobj material
    static ObjMaterial from(const tinyobj::material_t& material, const std::string& base_dir);
};

// One of these per OBJ file
struct LoadedSceneObject {
    std::string name;

    std::vector<ColoredVertex> vertices;
    std::vector<uint3>         indices;
    std::vector<uint32_t>      sbt_index_buffer; // one per triangle -> material index

    std::vector<ObjMaterial>   materials;

    // GPU buffers
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices = 0;
    CUdeviceptr d_sbt_indices = 0;
    CUdeviceptr d_gas_output = 0;

    // OMM GPU buffers
    CUdeviceptr d_omm_array_output = 0;
    CUdeviceptr d_omm_index_buffer = 0;

    OptixOpacityMicromapUsageCount omm_usage_count = {};

    OptixTraversableHandle gas_handle = 0;
    uint32_t               sbt_base = 0;
    float                  transform[12] = { 1,0,0,0, 
                                             0,1,0,0, 
                                             0,0,1,0 };
};

inline std::string resolveTexturePath(
    const std::string& raw_name,    
    const std::string& base_dir)     
{
    // Normalize base_dir to ensure it ends with a separator
    std::string search_root = base_dir;
    if (!search_root.empty() && search_root.back() != '/' && search_root.back() != '\\') {
        search_root += '/';
    }

    // Extract just the filename from the raw path
    std::filesystem::path raw_path(raw_name);
    std::string filename = raw_path.filename().string();  // e.g., "wood.png"

    // First, try the exact path as specified (relative to base_dir)
    std::string exact_attempt = search_root + raw_name;
    std::error_code ec;
    auto canonical_exact = std::filesystem::weakly_canonical(exact_attempt, ec);
    if (!ec && std::filesystem::exists(canonical_exact)) {
        //printf("[Texture] Found at exact path: %s\n", canonical_exact.string().c_str());
        return canonical_exact.string();
    }

    // Recursive search: walk through all subdirectories of base_dir looking for the filename
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(search_root)) {
            if (entry.is_regular_file() && entry.path().filename().string() == filename) {
                //printf("[Texture] Found '%s' at: %s\n", filename.c_str(), entry.path().string().c_str());
                return entry.path().string();
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "[Texture] Error searching directory: " << e.what() << "\n";
        return "";
    }

    // Nothing found
    printf("[Texture] Cannot resolve: '%s' (searched in '%s' and subdirectories)\n",
        raw_name.c_str(), search_root.c_str());
    return "";
}

// Construct from tinyobj material
inline ObjMaterial ObjMaterial::from(const tinyobj::material_t& material, const std::string& base_dir)
{
    ObjMaterial out;
    out.name = material.name;
    out.albedo = make_float3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
    out.emission = make_float3(material.emission[0], material.emission[1], material.emission[2]);
	out.specular = make_float3(material.specular[0], material.specular[1], material.specular[2]);
    out.shininess = material.shininess > 0.f ? material.shininess : 32.f;
    out.ior = material.ior > 1.f ? material.ior : 1.5f;
    out.is_glass = (material.illum == 3) || (material.illum == 4) || (material.illum == 6) || (material.illum == 7) || (material.illum == 9) || (material.name.find("Glass") != std::string::npos);// || (material.dissolve < 0.99f) || (material.ior > 1.01f);
    float emit_lum = out.emission.x + out.emission.y + out.emission.z;
    out.is_emissive = emit_lum > 0.001f || !out.texture_paths.emissive_path.empty();

    if (!material.diffuse_texname.empty()) {
        std::string resolved = resolveTexturePath(material.diffuse_texname, base_dir);
        if (!resolved.empty())
            out.texture_paths.diffuse_path = resolved;
    }
    if (!material.specular_texname.empty()) {
        std::string resolved = resolveTexturePath(material.specular_texname, base_dir);
        if (!resolved.empty())
            out.texture_paths.specular_path = resolved;
    }
    if (!material.bump_texname.empty()) {
        std::string resolved = resolveTexturePath(material.bump_texname, base_dir);
        if (!resolved.empty())
            out.texture_paths.bump_path = resolved;
    }
    if (!material.alpha_texname.empty()) {
        std::string resolved = resolveTexturePath(material.alpha_texname, base_dir);
        if (!resolved.empty())
            out.texture_paths.alpha_path = resolved;
    }
    if (!material.emissive_texname.empty()) {
        std::string resolved = resolveTexturePath(material.emissive_texname, base_dir);
        if (!resolved.empty())
            out.texture_paths.emissive_path = resolved;
    }

    if (material.roughness > 0.0f)          //if Pr present
        out.roughness = material.roughness;
    else
		out.roughness = sqrtf(2.f / (material.shininess + 2.f)); // Convert from Phong shininess to roughness (not exact but a common approximation)




    out.metallic = material.metallic;      // Pm if present
    if (out.metallic <= 0.0f) {            // If no Pm, guess metallic based on specular color intensity
        float kd_lum = 0.2126f * material.diffuse[0] + 0.7152f * material.diffuse[1] + 0.0722f * material.diffuse[2];
        float ks_lum = 0.2126f * material.specular[0] + 0.7152f * material.specular[1] + 0.0722f * material.specular[2];
        out.metallic = (kd_lum < 0.04f && ks_lum > 0.5f) ? 1.0f : 0.0f;
	}

    out.base_color = out.metallic > 0.5f
        ? make_float3(material.specular[0], material.specular[1], material.specular[2])
        : make_float3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);

    //out.base_color = out.metallic > 0.5f
    //    ? make_float3(1.f, 0.f, 0.f)
    //    : make_float3(0.f, 1.f, 0.f);

    return out;
}

inline void loadTextureFromFile(const std::string& path)
{
    if (path.empty()) return;

    int w, h, ch;
    // Force 4 channels (RGBA) for consistent upload
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 4);
    if (!data) {
        std::cerr << "[TEX] Failed to load: " << path
            << " - " << stbi_failure_reason() << "\n";
        return;
    }
    //std::cout << "[TEX] Loaded " << path << " (" << w << "x" << h << ")\n";
}


inline LoadedSceneObject loadSceneObject(const std::string& name,
                                         const std::string& obj_path,
                                         const std::string& base_dir)
{
    tinyobj::ObjReaderConfig reader_config;
	reader_config.triangulate = true;   // Ensure all faces are triangles for OptiX
	reader_config.vertex_color = false; // Disable vertex color parsing to save memory if not needed
	reader_config.mtl_search_path = base_dir; // Set the search path for MTL files to the same directory as the OBJ file

    tinyobj::ObjReader reader; 
	reader.ParseFromFile(obj_path, reader_config);   // Load OBJ file with the specified configuration
	if (!reader.Valid()) { // Check if loading was successful
        throw std::runtime_error("[OBJ] Failed to load '" + obj_path + "': " + reader.Error());
    }
	if (!reader.Warning().empty()) // Print any warnings that occurred during loading
        std::cerr << "[OBJ] Warning: " << reader.Warning() << "\n";

	const auto& attrib = reader.GetAttrib();  // Get vertex attributes (positions, normals, texcoords)
	const auto& shapes = reader.GetShapes();  // Get shapes (geometry groups) from the OBJ file. Each shape contains a mesh with indices and material IDs.
	const auto& materials = reader.GetMaterials();  // Get materials from the MTL file. Each material contains properties like diffuse color, specular color, texture paths, etc.

	LoadedSceneObject scene_object;  //Create a new LoadedSceneObject that holds the geometry and material data for data loaded from OBJ file
	scene_object.name = name;  //Set the name of the scene object

	scene_object.materials.reserve(materials.size());  //Build material list, one entry per loaded material from MTL file
    for (const auto& mat : materials)
		scene_object.materials.push_back(ObjMaterial::from(mat, base_dir));  // Convert tinyobj material to our own format and store in scene_object.materials

    // Walk every shape, every face - route directly into flat buffers
    for (const auto& shape : shapes) {
        for (size_t face_index = 0; face_index < shape.mesh.num_face_vertices.size(); face_index++) {
            int mat_id = shape.mesh.material_ids[face_index];
            if (mat_id < 0 || mat_id >= (int)materials.size()) mat_id = 0;

            uint32_t base = (uint32_t)scene_object.vertices.size();

            for (int v = 0; v < 3; v++) {
                tinyobj::index_t idx = shape.mesh.indices[face_index * 3 + v];
                ColoredVertex vert = {};

                int pi = idx.vertex_index * 3;
                vert.position = make_float3( attrib.vertices[pi + 0], attrib.vertices[pi + 1], attrib.vertices[pi + 2]);

                if (idx.normal_index >= 0) {
                    int ni = idx.normal_index * 3;
                    vert.normal = make_float3( attrib.normals[ni + 0], attrib.normals[ni + 1], attrib.normals[ni + 2]);
                }

                if (idx.texcoord_index >= 0) {
                    int ti = idx.texcoord_index * 2;
                    vert.uv = make_float2( attrib.texcoords[ti + 0], 1.f - attrib.texcoords[ti + 1]);
                }

                //vert.color = scene_object.materials[mat_id].albedo;
                scene_object.vertices.push_back(vert);
            }

            scene_object.indices.push_back(make_uint3(base, base + 1, base + 2));
            // This triangle maps to material mat_id's SBT record
            scene_object.sbt_index_buffer.push_back((uint32_t)mat_id);
        }
    }

    printf("[Scene] '%s': %zu verts, %zu tris, %zu materials\n", name.c_str(), scene_object.vertices.size(), scene_object.indices.size(), scene_object.materials.size());

    return scene_object;
}

// ===================================================================
// NEE LIGHT LIST CONSTRUCTION
// Builds the flat EmissiveTriangle array used by Next Event Estimation.
// ===================================================================

// Applies the object's 3x4 row-major transform (3 rows of 4 floats:
// rotation/scale 3x3 followed by a translation column) to a local-space
// point, producing a world-space point. This is the same layout written
// by OptixRenderer::createTransformMatrix.
inline float3 transformPoint(const float(&t)[12], float3 p)
{
    return make_float3(
        t[0] * p.x + t[1] * p.y + t[2] * p.z + t[3],
        t[4] * p.x + t[5] * p.y + t[6] * p.z + t[7],
        t[8] * p.x + t[9] * p.y + t[10] * p.z + t[11]
    );
}

// Walks every loaded object's triangles, and for every triangle whose
// material is emissive, transforms its vertices to world space (using the
// object's current transform) and appends one EmissiveTriangle to the list.
inline std::vector<EmissiveTriangle> buildEmissiveTriangleList(
    const std::vector<LoadedSceneObject>& objects,
    const std::vector<float(*)[12]>& object_transforms) // one transform pointer per object, same order as `objects`
{
    std::vector<EmissiveTriangle> lights;

    for (size_t obj_idx = 0; obj_idx < objects.size(); ++obj_idx)
    {
        const LoadedSceneObject& obj = objects[obj_idx];
        const float(&transform)[12] = *object_transforms[obj_idx];

        for (size_t t = 0; t < obj.indices.size(); ++t)
        {
            uint32_t mat_idx = obj.sbt_index_buffer[t];
            const ObjMaterial& mat = obj.materials[mat_idx];

            // Simple version: only constant Ke. Textured emissive (map_Ke)
            // is intentionally skipped here - extend later by subdividing
            // and sampling the texture per sub-triangle.
            if (!mat.is_emissive) continue;
            if (luminance(mat.emission) < 1e-4f) continue;

            const uint3& tri = obj.indices[t];
            float3 local_v0 = obj.vertices[tri.x].position;
            float3 local_v1 = obj.vertices[tri.y].position;
            float3 local_v2 = obj.vertices[tri.z].position;

            float3 v0 = transformPoint(transform, local_v0);
            float3 v1 = transformPoint(transform, local_v1);
            float3 v2 = transformPoint(transform, local_v2);

            float3 e1 = v1 - v0;
            float3 e2 = v2 - v0;
            float3 cr = cross(e1, e2);
            float  area = 0.5f * length(cr);

            if (area < 1e-10f) continue; // degenerate triangle, skip

            EmissiveTriangle light{};
            light.v0 = v0;
            light.v1 = v1;
            light.v2 = v2;
            light.emission = mat.emission;
            light.area = area;
            light.cdf = 0.f; // filled in by buildLightCDF

            lights.push_back(light);
        }
    }

    printf("[NEE] Built light list: %zu emissive triangles\n", lights.size());
    return lights;
}

// Computes the area*luminance-weighted CDF over the light list, in place,
// and returns the total weight (sum of area*luminance across all lights).
// This total is needed in the NEE estimator to convert the per-triangle
// selection probability (weight / total) combined with a uniform point
// pick (1 / area) into a proper PDF - see Params::total_emissive_weight.
//
// Mirrors the same cumulative-sum-then-normalize structure as
// computeEnvmapCDF above, just over a 1D list of triangles instead of a
// 2D grid of pixels.
inline float buildLightCDF(std::vector<EmissiveTriangle>& lights)
{
    if (lights.empty()) return 0.f;

    float total = 0.f;
    for (const auto& l : lights)
        total += l.area * luminance(l.emission);

    if (total < 1e-10f) {
        // All lights ended up with negligible weight, avoids divide by zero.
        for (auto& l : lights) l.cdf = 1.f;
        return 0.f;
    }

    float running = 0.f;
    for (auto& l : lights) {
        running += l.area * luminance(l.emission);
        l.cdf = running / total;
    }
    lights.back().cdf = 1.0f; // guarantee exact 1.0 at the end (float safety)

    return total;
}


// Compute 2D CDF weighted by luminance * sin(theta)
inline void computeEnvmapCDF(const float* hdr_data, int width, int height,
    std::vector<float>& out_marginal_cdf,
    std::vector<float>& out_conditional_cdf)
{
    out_marginal_cdf.resize(height, 0.0f);
    out_conditional_cdf.resize(width * height, 0.0f);

    // --- Step 1: Compute conditional CDFs (per row) ---
    for (int v = 0; v < height; ++v) {
        float row_integral = 0.0f;

        // Compute row integral and build conditional CDF
        for (int u = 0; u < width; ++u) {
            int idx = (v * width + u) * 4;  // RGBA format
            float r = hdr_data[idx + 0];
            float g = hdr_data[idx + 1];
            float b = hdr_data[idx + 2];

            // Luminance with sin(theta) weighting for equirectangular distortion
            float theta = ((float)v + 0.5f) / (float)height * 3.14159265f;
            float sin_theta = sinf(theta);
            float luminance = (0.2126f * r + 0.7152f * g + 0.0722f * b) * sin_theta;

            row_integral += luminance;
            out_conditional_cdf[v * width + u] = row_integral;
        }

        // Normalize conditional CDF for this row to [0,1]
        if (row_integral > 1e-6f) {
            for (int u = 0; u < width; ++u) {
                out_conditional_cdf[v * width + u] /= row_integral;
            }
        }

        // Store row integral for marginal CDF
        out_marginal_cdf[v] = row_integral;
    }

    // --- Step 2: Compute marginal CDF (over rows) ---
    float total_integral = 0.0f;
    for (int v = 0; v < height; ++v) {
        total_integral += out_marginal_cdf[v];
        out_marginal_cdf[v] = total_integral;
    }

    // Normalize marginal CDF to [0,1]
    if (total_integral > 1e-6f) {
        for (int v = 0; v < height; ++v) {
            out_marginal_cdf[v] /= total_integral;
        }
    }
}

// Upload CDF textures to GPU and return texture objects
inline void uploadEnvmapCDFTextures(
    const std::vector<float>& marginal_cdf,
    const std::vector<float>& conditional_cdf,
    int width, int height,
    cudaTextureObject_t& out_marginal,
    cudaTextureObject_t& out_conditional,
    cudaArray_t& out_marginal_array,
    cudaArray_t& out_conditional_array)
{
    // --- Upload marginal CDF (1D) ---
    {
        cudaChannelFormatDesc ch_desc = cudaCreateChannelDesc<float>();

        cudaMallocArray(&out_marginal_array, &ch_desc, height, 1);
        cudaMemcpyToArray(
            out_marginal_array, 0, 0,
            marginal_cdf.data(),
            height * sizeof(float),
            cudaMemcpyHostToDevice);

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = out_marginal_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;

        cudaCreateTextureObject(&out_marginal, &res_desc, &tex_desc, nullptr);
    }

    // --- Upload conditional CDF (2D) ---
    {
        cudaChannelFormatDesc ch_desc = cudaCreateChannelDesc<float>();

        cudaMallocArray(&out_conditional_array, &ch_desc, width, height);
        cudaMemcpy2DToArray(
            out_conditional_array, 0, 0,
            conditional_cdf.data(),
            width * sizeof(float),
            width * sizeof(float),
            height,
            cudaMemcpyHostToDevice);

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = out_conditional_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;

        cudaCreateTextureObject(&out_conditional, &res_desc, &tex_desc, nullptr);
    }
}