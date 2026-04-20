#pragma once

#include <string>
#include <vector>
#include "optix_params.h"
#include "obj_loader.h"

#define M_PI 3.14159265f

enum class SceneID {
    ALLIED_AVENGER = 0,
    GEOSPHERE = 1,
    SPITFIRE = 2
};

struct SceneData {
    SceneID id;
    std::string name;
    std::string obj_path;
    std::string mtl_path;
    std::string envmap_path;

    // Scene-specific settings
    float3 camera_position;
    float3 camera_lookat;
    float3 camera_up;
    float camera_vfov;

    float3 object_position;
    float3 object_scale;
    float object_rotation_x;
    float object_rotation_y;
    float object_rotation_z;

    // Lighting setup
    int num_lights;
    Light lights[4];
    float envmap_scale;
    float envmap_exposure;
};

class SceneManager {
public:
    static SceneData getSceneConfig(SceneID id) {
        switch (id) {
        case SceneID::ALLIED_AVENGER:
            return getAliedAvengerScene();
        case SceneID::GEOSPHERE:
            return getGeosphereScene();
        case SceneID::SPITFIRE:
            return getSpitfireScene();
        default:
            return getAliedAvengerScene();
        }
    }

private:
    static SceneData getAliedAvengerScene() {
        SceneData scene;
        scene.id = SceneID::ALLIED_AVENGER;
        scene.name = "Allied Avenger";
        scene.obj_path = "assets/6887_allied_avenger.obj";
        scene.mtl_path = "assets/6887_allied_avenger.mtl";
        scene.envmap_path = "assets/golden_gate_hills_2k.hdr";

        scene.camera_position = make_float3(-0.7f, 3.0f, 8.0f);
        scene.camera_lookat = make_float3(0.0f, 0.0f, 0.0f);
        scene.camera_up = make_float3(0.0f, 1.0f, 0.0f);
        scene.camera_vfov = 60.0f;

        scene.object_scale = make_float3(0.05f, 0.05f, 0.05f);
        scene.object_position = make_float3(0.0f, 1.0f, 0.0f);
        scene.object_rotation_x = -M_PI / 2.0f;
        scene.object_rotation_y = -M_PI / 3.0f;
        scene.object_rotation_z = 0.0f;

        // Point Light 1
        scene.lights[0].type = 0;
        scene.lights[0].position_or_direction = make_float3(2.0f, 3.0f, 2.0f);
        scene.lights[0].color = make_float3(1.0f, 0.4f, 0.4f);

        // Point Light 2
        scene.lights[1].type = 0;
        scene.lights[1].position_or_direction = make_float3(2.0f, 3.0f, -2.0f);
        scene.lights[1].color = make_float3(0.4f, 0.4f, 1.0f);

        // Directional Light
        scene.lights[2].type = 1;
        scene.lights[2].position_or_direction = normalize(make_float3(0.0f, -1.0f, -0.2f));
        scene.lights[2].color = make_float3(0.5f, 0.5f, 0.5f);

        scene.num_lights = 3;
        scene.envmap_scale = 1.0f;
        scene.envmap_exposure = 0.0f;

        return scene;
    }

    static SceneData getGeosphereScene() {
        SceneData scene;
        scene.id = SceneID::GEOSPHERE;
        scene.name = "Geosphere (Furnace Test)";
        //scene.obj_path = "assets/geosphere_lambert/geosphere_lambert.obj";
        //scene.mtl_path = "assets/geosphere_lambert/geosphere_lambert.mtl";
        scene.obj_path = "assets/geosphere_glass/geosphere.obj";
        scene.mtl_path = "assets/geosphere_glass/geosphere_glass.mtl";
        scene.envmap_path = ""; // No env map for furnace test

        // Centered view
        scene.camera_position = make_float3(0.0f, 1.0f, 4.0f);
        scene.camera_lookat = make_float3(0.0f, 0.0f, 0.0f);
        scene.camera_up = make_float3(0.0f, 1.0f, 0.0f);
        scene.camera_vfov = 60.0f;

        // Object transform for furnace test
        scene.object_scale = make_float3(1.0f, 1.0f, 1.0f);
        scene.object_position = make_float3(0.0f, 0.0f, 0.0f);
        scene.object_rotation_x = 0.0f;
        scene.object_rotation_y = 0.0f;
        scene.object_rotation_z = 0.0f;

        // Uniform white lighting for furnace test
        scene.lights[0].type = 1;  // Directional
        scene.lights[0].position_or_direction = normalize(make_float3(0.0f, 1.0f, 0.0f));
        scene.lights[0].color = make_float3(1.0f, 1.0f, 1.0f);

        scene.num_lights = 0;
        scene.envmap_scale = 1.0f;
        scene.envmap_exposure = 0.0f;

        return scene;
    }

    static SceneData getSpitfireScene() {
        SceneData scene;
        scene.id = SceneID::SPITFIRE;
        scene.name = "Spitfire";
        scene.obj_path = "assets/spitfire/spitfire.obj";
        scene.mtl_path = "assets/spitfire/spitfire.mtl";
        scene.envmap_path = "assets/golden_gate_hills_2k.hdr";

        scene.camera_position = make_float3(-0.7f, 3.0f, 8.0f);
        scene.camera_lookat = make_float3(0.0f, 0.0f, 0.0f);
        scene.camera_up = make_float3(0.0f, 1.0f, 0.0f);
        scene.camera_vfov = 60.0f;

        scene.object_scale = make_float3(0.05f, 0.05f, 0.05f);
        scene.object_position = make_float3(0.0f, 1.0f, 0.0f);
        scene.object_rotation_x = -M_PI / 2.0f;
        scene.object_rotation_y = -M_PI / 3.0f;
        scene.object_rotation_z = 0.0f;

        scene.num_lights = 0;
        scene.envmap_scale = 1.0f;
        scene.envmap_exposure = 0.0f;

        return scene;
    }
};