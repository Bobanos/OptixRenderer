#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <cuda_runtime.h>

#include "optix_params.h"

// One mesh group per material found in the OBJ
struct ObjMesh {
    std::string             material_name;
    std::vector<ColoredVertex> vertices;
    std::vector<uint3>      indices;
    bool                    is_glass = false;
    float                   ior = 1.0f;
};

// MTL material properties we care about
struct MtlMaterial {
    float3 kd = { 0.8f, 0.8f, 0.8f }; // Diffuse color
    float  ni = 1.0f;                  // Index of refraction
    int    illum = 2;                     // Illumination model (3 = glass)
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
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tok;
        ss >> tok;

        if (tok == "newmtl") {
            ss >> current;
            mats[current] = {};
        }
        else if (tok == "Kd") {
            ss >> mats[current].kd.x >> mats[current].kd.y >> mats[current].kd.z;
        }
        else if (tok == "Ni") {
            ss >> mats[current].ni;
        }
        else if (tok == "illum") {
            ss >> mats[current].illum;
        }
    }
    return mats;
}

inline std::vector<ObjMesh> loadObj(const std::string& obj_path,
    const std::string& mtl_path)
{
    // Load materials first
    auto materials = loadMtl(mtl_path);

    std::ifstream f(obj_path);
    if (!f) throw std::runtime_error("Failed to open OBJ: " + obj_path);

    // All positions from the whole file (OBJ indices are global)
    std::vector<float3> raw_positions;

    // Triangle soup per material: list of (pos, color) triplets
    std::unordered_map<std::string, std::vector<float3>> mat_positions;
    std::unordered_map<std::string, float3>              mat_color;
    std::unordered_map<std::string, float>               mat_ior;
    std::unordered_map<std::string, bool>                mat_glass;

    std::string current_mat = "default";
    std::string line;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tok;
        ss >> tok;

        if (tok == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            raw_positions.push_back(make_float3(x, y, z));
        }
        else if (tok == "usemtl") {
            ss >> current_mat;
            if (mat_positions.find(current_mat) == mat_positions.end()) {
                // First time seeing this material — record its properties
                if (materials.count(current_mat)) {
                    auto& m = materials[current_mat];
                    mat_color[current_mat] = m.kd;
                    mat_ior[current_mat] = m.ni;
                    mat_glass[current_mat] = (m.illum == 3 || m.ni > 1.01f);
                }
                else {
                    mat_color[current_mat] = make_float3(0.8f, 0.8f, 0.8f);
                    mat_ior[current_mat] = 1.0f;
                    mat_glass[current_mat] = false;
                }
            }
        }
        else if (tok == "f") {
            // Parse face — may be triangle or quad, fan-triangulate
            std::vector<int> face_vi;
            std::string vtok;
            while (ss >> vtok) {
                // Format: v, v/vt, v/vt/vn, v//vn — we only need v
                int vi = std::stoi(vtok.substr(0, vtok.find('/')));
                if (vi < 0)
                    vi = (int)raw_positions.size() + vi + 1;
                face_vi.push_back(vi - 1); // OBJ is 1-indexed
            }
            // Fan triangulation
            for (int i = 1; i + 1 < (int)face_vi.size(); i++) {
                mat_positions[current_mat].push_back(raw_positions[face_vi[0]]);
                mat_positions[current_mat].push_back(raw_positions[face_vi[i]]);
                mat_positions[current_mat].push_back(raw_positions[face_vi[i + 1]]);
            }
        }
    }

    // Build indexed ObjMesh per material
    std::vector<ObjMesh> meshes;
    for (auto& [mat, soup] : mat_positions) {
        if (soup.empty()) continue;

        ObjMesh mesh;
        mesh.material_name = mat;
        mesh.is_glass = mat_glass.count(mat) ? mat_glass[mat] : false;
        mesh.ior = mat_ior.count(mat) ? mat_ior[mat] : 1.0f;

        float3 col = mat_color.count(mat) ? mat_color[mat] : make_float3(0.8f, 0.8f, 0.8f);

        // Build indexed mesh from triangle soup
        // Simple approach: just use unindexed (one vertex per triangle corner)
        // This avoids hash collisions from shared vertices with different normals
        for (size_t i = 0; i < soup.size(); i += 3) {
            uint32_t base = (uint32_t)mesh.vertices.size();
            mesh.vertices.push_back({ soup[i + 0], col });
            mesh.vertices.push_back({ soup[i + 1], col });
            mesh.vertices.push_back({ soup[i + 2], col });
            mesh.indices.push_back(make_uint3(base, base + 1, base + 2));
        }

        meshes.push_back(std::move(mesh));
    }

    std::cout << "[OBJ] Loaded " << meshes.size()
        << " material groups from: " << obj_path << "\n";
    for (auto& m : meshes) {
        std::cout << "  '" << m.material_name << "': "
            << m.vertices.size() << " verts, "
            << m.indices.size() << " tris, "
            << (m.is_glass ? "GLASS" : "solid")
            << " ior=" << m.ior << "\n";
    }

    return meshes;
}