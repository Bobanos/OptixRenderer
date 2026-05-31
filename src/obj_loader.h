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

// Your own material — only the fields you use
struct ObjMaterial {
    std::string name;
    float3      albedo = { 0.8f, 0.8f, 0.8f };
    float3      emission = { 0.f,  0.f,  0.f };
    float       ior = 1.5f;
    float       shininess = 32.f;
    bool        is_glass = false;
    bool        is_emissive = false;

    struct TexturePaths {
        std::string ambient_path = "";
        std::string diffuse_path = "";
        std::string specular_path = ""; //Red channel: Occlusion, Green channel : Roughness, Blue channel : Metalness
        std::string bump_path = "";
        std::string alpha_path = "";
		std::string emissive_path = "";
	} texture_paths;
    //std::string texture_path;

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

    OptixTraversableHandle gas_handle = 0;
    uint32_t               sbt_base = 0;
    float                  transform[12] = { 1,0,0,0, 
                                             0,1,0,0, 
                                             0,0,1,0 };
};

inline std::string resolveTexturePath(
    const std::string& raw_name,     // what the MTL says, e.g. "wood.png" or "textures/wood.png"
    const std::string& base_dir)     // base directory to search (e.g., "C:/Users/lukas/OneDrive/Desktop/lumberyard/")
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
        printf("[Texture] Found at exact path: %s\n", canonical_exact.string().c_str());
        return canonical_exact.string();
    }

    // Recursive search: walk through all subdirectories of base_dir looking for the filename
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(search_root)) {
            if (entry.is_regular_file() && entry.path().filename().string() == filename) {
                printf("[Texture] Found '%s' at: %s\n", filename.c_str(), entry.path().string().c_str());
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
    out.shininess = material.shininess > 0.f ? material.shininess : 32.f;
    out.ior = material.ior > 1.f ? material.ior : 1.5f;
    out.is_glass = (material.illum == 7) || (material.illum == 9);// || (material.dissolve < 0.99f) || (material.ior > 1.01f);
    float emit_lum = out.emission.x + out.emission.y + out.emission.z;
    out.is_emissive = emit_lum > 0.001f || !out.texture_paths.emissive_path.empty();

    if (!material.ambient_texname.empty()) {
        std::string resolved = resolveTexturePath(material.ambient_texname, base_dir);
        if (!resolved.empty())
            out.texture_paths.ambient_path = resolved;
    }
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
    //// In ObjMaterial::from():
    //if (!material.diffuse_texname.empty()) {
    //    std::string resolved = resolveTexturePath(material.diffuse_texname, base_dir);
    //    if (!resolved.empty())
    //        out.texture_path = resolved;  // store the resolved path
    //}

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
            << " — " << stbi_failure_reason() << "\n";
        return;
    }
    //std::cout << "[TEX] Loaded " << path << " (" << w << "x" << h << ")\n";
}


inline void computeSmoothNormals(LoadedSceneObject& scene_object)
{
    // Group vertex indices by position using a spatial hash.
    // The hash maps position bits to a list of vertex indices at that position.
    // Vertices at the same world position get the same hash bucket and share normals.
    auto hashPos = [](float3 p) -> size_t {
        uint32_t hx, hy, hz;
        memcpy(&hx, &p.x, 4);
        memcpy(&hy, &p.y, 4);
        memcpy(&hz, &p.z, 4);
        return (size_t)(hx * 2654435761u ^ hy * 805459861u ^ hz * 3674653429u);
        };

    // Map from position hash -> list of vertex indices sharing that position
    std::unordered_map<size_t, std::vector<uint32_t>> pos_groups;

    for (uint32_t i = 0; i < (uint32_t)scene_object.vertices.size(); i++)
        pos_groups[hashPos(scene_object.vertices[i].position)].push_back(i);

    // For each group, sum the face normals (area-weighted via unnormalized cross products)
    // then normalize the sum and write back to all vertices in the group.
    for (auto& [hash, group] : pos_groups) {
        float3 sum = { 0.f, 0.f, 0.f };
        for (uint32_t vi : group) {
            const float3& n = scene_object.vertices[vi].normal;
            sum.x += n.x; sum.y += n.y; sum.z += n.z;
        }
        float len = sqrtf(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
        float3 smooth = (len > 1e-6f)
            ? make_float3(sum.x / len, sum.y / len, sum.z / len)
            : make_float3(0.f, 1.f, 0.f);  // fallback: point up

        for (uint32_t vi : group)
            scene_object.vertices[vi].normal = smooth;
    }
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

    // Walk every shape, every face — route directly into flat buffers
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

    for (int i = 0; i < 100; i++) {
        //printf("[Normal] %d: x %f , y %f , z %f \n", i, scene_object.vertices[i].normal.x, scene_object.vertices[i].normal.y, scene_object.vertices[i].normal.z);
    }
    
    //computeSmoothNormals(scene_object);  // TODO Compute smooth normals where absent

    for (int i = 0; i < 100; i++) {
        //printf("[Normal] %d: x %f , y %f , z %f \n", i, scene_object.vertices[i].normal.x, scene_object.vertices[i].normal.y, scene_object.vertices[i].normal.z);
    }

    printf("[Scene] '%s': %zu verts, %zu tris, %zu materials\n", name.c_str(), scene_object.vertices.size(), scene_object.indices.size(), scene_object.materials.size());

 //   struct TextureUsage {
 //       int ambient_texname = 0;             // map_Ka. For ambient or ambient occlusion.
 //       int diffuse_texname = 0;             // map_Kd
 //       int specular_texname = 0;            // map_Ks
 //       int bump_texname = 0;                // map_bump, map_Bump, bump
 //       int alpha_texname = 0;               // map_d
 //       int roughness_texname = 0;               // map_Pr
 //       int metallic_texname = 0;                // map_Pm
 //       int sheen_texname = 0;                   // map_Ps
 //       int emissive_texname = 0;                // map_Ke
 //       int normal_texname = 0;                 // norm. For normal mapping.
	//} texture_usage;

 //   for (const auto& material: materials) {
 //       //printf("[Material]'%s', Texture: ambient '%s', diffuse '%s', specular '%s', bump '%s', alpha '%s'\n",
 //       //    material.name.c_str(),
 //       //    material.ambient_texname.c_str(),
 //       //    material.diffuse_texname.c_str(),
 //       //    material.specular_texname.c_str(),
 //       //    material.bump_texname.c_str(),
 //       //    material.alpha_texname.c_str());
	//	if (!material.ambient_texname.empty()) texture_usage.ambient_texname++;
	//	if (!material.diffuse_texname.empty()) texture_usage.diffuse_texname++;
	//	if (!material.specular_texname.empty()) texture_usage.specular_texname++;
	//	if (!material.bump_texname.empty()) texture_usage.bump_texname++;
	//	if (!material.alpha_texname.empty()) texture_usage.alpha_texname++;
 //       if (!material.roughness_texname.empty()) texture_usage.roughness_texname++;
 //       if (!material.metallic_texname.empty()) texture_usage.metallic_texname++;
 //       if (!material.sheen_texname.empty()) texture_usage.sheen_texname++;
 //       if (!material.emissive_texname.empty()) texture_usage.emissive_texname++;
 //       if (!material.normal_texname.empty()) texture_usage.normal_texname++;
	//}

 //   for (const auto& material : materials) {
 //  //     if (!material.ambient_texname.empty())
	//		////loadTextureFromFile(base_dir + material.ambient_texname);
	//	 //   loadTextureFromFile(resolveTexturePath(material.ambient_texname, base_dir));
 //  //     if (!material.diffuse_texname.empty())
 //  //         //loadTextureFromFile(base_dir + material.diffuse_texname);
	//		//loadTextureFromFile(resolveTexturePath(material.diffuse_texname, base_dir));
 //  //     if (!material.specular_texname.empty())
 //  //         //loadTextureFromFile(base_dir + material.specular_texname);
	//		//loadTextureFromFile(resolveTexturePath(material.specular_texname, base_dir));
 //  //     if (!material.bump_texname.empty())
 //  //         loadTextureFromFile(resolveTexturePath(material.bump_texname, base_dir));
 //  //     if (!material.alpha_texname.empty())
 //  //         loadTextureFromFile(resolveTexturePath(material.alpha_texname, base_dir));
 //       if (!material.emissive_texname.empty())
 //           loadTextureFromFile(resolveTexturePath(material.emissive_texname, base_dir));
	//}

	//printf("[Scene] Texture usage: ambient %d, diffuse %d, specular %d, bump %d, alpha %d, roughness %d, metallic %d, sheen %d, emissive %d, normal %d\n",
	//	texture_usage.ambient_texname,
	//	texture_usage.diffuse_texname,
	//	texture_usage.specular_texname,
	//	texture_usage.bump_texname,
	//	texture_usage.alpha_texname,
 //       texture_usage.roughness_texname,
 //       texture_usage.metallic_texname,
 //       texture_usage.sheen_texname,
 //       texture_usage.emissive_texname,
 //       texture_usage.normal_texname

	//	);

    return scene_object;
}

// ===================================================================
// ENVMAP CDF COMPUTATION for importance sampling
// Compute 2D CDF weighted by luminance * sin(theta)
// ===================================================================
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