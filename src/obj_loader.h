#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <cuda_runtime.h>

#include "optix_params.h"

struct ObjMesh {
    std::string                material_name;
    std::vector<ColoredVertex> vertices;
    std::vector<uint3>         indices;
    bool                       is_glass = false;
    float                      ior = 1.0f;
    std::string                texture_path; // empty = no texture
};

// Merged result: all geometry in one buffer with per-material SBT indexing
struct MergedObjMesh {
    std::vector<ColoredVertex> vertices;
    std::vector<uint3>         indices;
    std::vector<uint32_t>      sbt_index_buffer;  // sbt_index_buffer[prim_idx] = SBT record index

    struct MaterialInfo {
        std::string name;
        float3      color;
        float       ior;
        bool        is_glass;
        std::string texture_path;
    };
    std::vector<MaterialInfo> materials;  // materials[i] = info for SBT record i
};

struct MtlMaterial {
    float3      kd = { 0.8f, 0.8f, 0.8f };
    float       ni = 1.0f;
    int         illum = 2;
    std::string map_kd = ""; // texture filename, empty if none
};

inline std::unordered_map<std::string, MtlMaterial> loadMtl(const std::string& path)
{
    std::unordered_map<std::string, MtlMaterial> mats;
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[MTL] Could not open: " << path << std::endl;
        return mats;
    }

    std::string line, current;
    while (std::getline(f, line)) {
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos || line[start] == '#') continue;
        line = line.substr(start);

        std::istringstream ss(line);
        std::string tok;
        ss >> tok;

        if (tok == "newmtl") { ss >> current; mats[current] = {}; }
        else if (tok == "Kd") { ss >> mats[current].kd.x >> mats[current].kd.y >> mats[current].kd.z; }
        else if (tok == "Ni") { ss >> mats[current].ni; }
        else if (tok == "illum") { ss >> mats[current].illum; }
        else if (tok == "map_Kd") { ss >> mats[current].map_kd; }
    }
    return mats;
}

inline std::vector<ObjMesh> loadObj(const std::string& obj_path,
    const std::string& mtl_path)
{
    auto materials = loadMtl(mtl_path);

    std::ifstream f(obj_path);
    if (!f) throw std::runtime_error("Failed to open OBJ: " + obj_path);

    std::vector<float3> raw_positions;
    std::vector<float2> raw_uvs;

    struct Corner { float3 pos; float2 uv; };
    std::unordered_map<std::string, std::vector<Corner>> mat_corners;
    std::unordered_map<std::string, float3>              mat_color;
    std::unordered_map<std::string, float>               mat_ior;
    std::unordered_map<std::string, bool>                mat_glass;
    std::unordered_map<std::string, std::string>         mat_texture;

    std::string current_mat = "default";
    std::string line;

    while (std::getline(f, line)) {
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos || line[start] == '#') continue;
        line = line.substr(start);

        std::istringstream ss(line);
        std::string tok;
        ss >> tok;

        if (tok == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            raw_positions.push_back(make_float3(x, y, z));
        }
        else if (tok == "vt") {
            float u, v;
            ss >> u >> v;
            raw_uvs.push_back(make_float2(u, v));
        }
        else if (tok == "usemtl") {
            ss >> current_mat;
            if (mat_corners.find(current_mat) == mat_corners.end()) {
                if (materials.count(current_mat)) {
                    auto& m = materials[current_mat];
                    mat_color[current_mat] = m.kd;
                    mat_ior[current_mat] = m.ni;
                    mat_glass[current_mat] = (m.illum == 3 || m.ni > 1.01f);
                    mat_texture[current_mat] = m.map_kd;
                }
                else {
                    mat_color[current_mat] = make_float3(0.8f, 0.8f, 0.8f);
                    mat_ior[current_mat] = 1.0f;
                    mat_glass[current_mat] = false;
                    mat_texture[current_mat] = "";
                }
            }
        }
        else if (tok == "f") {
            std::vector<Corner> face_corners;
            std::string vtok;
            while (ss >> vtok) {
                Corner c;
                c.uv = make_float2(0.0f, 0.0f);

                size_t s1 = vtok.find('/');
                int vi = std::stoi(vtok.substr(0, s1));
                if (vi < 0) vi = (int)raw_positions.size() + vi + 1;
                c.pos = raw_positions[vi - 1];

                if (s1 != std::string::npos) {
                    size_t s2 = vtok.find('/', s1 + 1);
                    std::string vt_str = vtok.substr(s1 + 1,
                        s2 == std::string::npos ? std::string::npos : s2 - s1 - 1);
                    if (!vt_str.empty() && !raw_uvs.empty()) {
                        int vti = std::stoi(vt_str);
                        if (vti < 0) vti = (int)raw_uvs.size() + vti + 1;
                        c.uv = raw_uvs[vti - 1];
                    }
                }
                face_corners.push_back(c);
            }

            for (int i = 1; i + 1 < (int)face_corners.size(); i++) {
                mat_corners[current_mat].push_back(face_corners[0]);
                mat_corners[current_mat].push_back(face_corners[i]);
                mat_corners[current_mat].push_back(face_corners[i + 1]);
            }
        }
    }

    std::string asset_dir = obj_path.substr(0, obj_path.find_last_of("/\\") + 1);

    std::vector<ObjMesh> meshes;
    for (auto& [mat, corners] : mat_corners) {
        if (corners.empty()) continue;

        ObjMesh mesh;
        mesh.material_name = mat;
        mesh.is_glass = mat_glass.count(mat) ? mat_glass[mat] : false;
        mesh.ior = mat_ior.count(mat) ? mat_ior[mat] : 1.0f;
        mesh.texture_path = mat_texture.count(mat) ? mat_texture[mat] : "";

        if (!mesh.texture_path.empty())
            mesh.texture_path = asset_dir + mesh.texture_path;

        float3 col = mat_color.count(mat) ? mat_color[mat] : make_float3(0.8f, 0.8f, 0.8f);

        for (size_t i = 0; i < corners.size(); i += 3) {
            uint32_t base = (uint32_t)mesh.vertices.size();
            mesh.vertices.push_back({ corners[i + 0].pos, col, corners[i + 0].uv });
            mesh.vertices.push_back({ corners[i + 1].pos, col, corners[i + 1].uv });
            mesh.vertices.push_back({ corners[i + 2].pos, col, corners[i + 2].uv });
            mesh.indices.push_back(make_uint3(base, base + 1, base + 2));
        }

        meshes.push_back(std::move(mesh));
    }

    std::cout << "[OBJ] Loaded " << meshes.size() << " material groups\n";
    for (auto& m : meshes) {
        std::cout << "  '" << m.material_name << "': "
            << m.vertices.size() << " verts, "
            << m.indices.size() << " tris, "
            << (m.is_glass ? "GLASS" : "solid")
            << (m.texture_path.empty() ? "" : " tex=" + m.texture_path)
            << "\n";
    }
    return meshes;
}

// ------------------------------------------------------------------
// Merge all ObjMeshes into one flat buffer with per-material SBT indexing.
// SBT records are assigned in the order materials appear in the ObjMesh vector.
// ------------------------------------------------------------------
inline MergedObjMesh mergeObjMeshes(const std::vector<ObjMesh>& meshes)
{
    MergedObjMesh result;

    for (uint32_t mat_idx = 0; mat_idx < meshes.size(); ++mat_idx) {
        const auto& mesh = meshes[mat_idx];

        // Record material info for SBT
        MergedObjMesh::MaterialInfo mat_info;
        mat_info.name = mesh.material_name;
        mat_info.color = mesh.vertices.empty() ? make_float3(0.8f, 0.8f, 0.8f)
            : mesh.vertices[0].color;
        mat_info.ior = mesh.ior;
        mat_info.is_glass = mesh.is_glass;
        mat_info.texture_path = mesh.texture_path;
        result.materials.push_back(mat_info);

        // Merge vertices and indices, assigning SBT record index
        uint32_t vertex_base = (uint32_t)result.vertices.size();
        for (const auto& v : mesh.vertices) {
            result.vertices.push_back(v);
        }

        for (const auto& tri : mesh.indices) {
            result.indices.push_back(make_uint3(
                tri.x + vertex_base, tri.y + vertex_base, tri.z + vertex_base));

            //// DIAGNOSTIC: Route textured materials (mat_idx 3, 4) to glass SBT record (1)
            //uint32_t sbt_idx = mat_idx;
            //if (!mesh.texture_path.empty()) {
            //    sbt_idx = 1;  // Point to glass shader (SBT index 1)
            //}
            //result.sbt_index_buffer.push_back(sbt_idx);
            result.sbt_index_buffer.push_back(mat_idx);
        }
    }

    std::cout << "[MERGE] Total: " << result.vertices.size() << " verts, "
        << result.indices.size() << " tris, "
        << result.materials.size() << " materials\n";
    for (size_t i = 0; i < result.materials.size(); ++i) {
        const auto& m = result.materials[i];
        std::cout << "  [SBT " << i << "] '" << m.name << "': "
            << (m.is_glass ? "GLASS" : "solid")
            << (m.texture_path.empty() ? "" : " tex=" + m.texture_path)
            << "\n";
    }
    return result;
}