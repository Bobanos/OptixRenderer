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

struct SceneObject {
    std::string name;
    std::string obj_path;
    std::string mtl_path;

    float3 position;
    float3 scale;
    float rotation_x;
    float rotation_y;
    float rotation_z;

    bool is_emissive = false;           
    float3 emissive_color = { 0, 0, 0 };
    float emissive_intensity = 0.0f;   
};

struct SceneData {
    SceneID id;
    std::string name;
    std::string envmap_path;

    // Camera settings
    float3 camera_position;
    float3 camera_lookat;
    float3 camera_up;
    float camera_vfov;

    // Lighting setup
    int num_lights;
    Light lights[4];
    float envmap_scale;
    float envmap_exposure;

    // Multiple objects in scene
    std::vector<SceneObject> objects;
};

class SceneManager {
public:
    static SceneData getSceneConfig(SceneID id) {
        switch (id) {
        case SceneID::ALLIED_AVENGER:
            return getAlliedAvengerScene();
        case SceneID::GEOSPHERE:
            return getGeosphereScene();
        case SceneID::SPITFIRE:
            return getSpitfireScene();
        default:
            return getAlliedAvengerScene();
        }
    }

private:
    static SceneData getAlliedAvengerScene() {
        SceneData scene;
        scene.id = SceneID::ALLIED_AVENGER;
        scene.name = "Allied Avenger";
        scene.envmap_path = "assets/golden_gate_hills_2k.hdr";

        scene.camera_position = make_float3(-0.7f, 3.0f, 8.0f);
        scene.camera_lookat = make_float3(0.0f, 0.0f, 0.0f);
        scene.camera_up = make_float3(0.0f, 1.0f, 0.0f);
        scene.camera_vfov = 60.0f;

        // Single object
        SceneObject obj;
        obj.name = "Allied Avenger";
        obj.obj_path = "assets/6887_allied_avenger.obj";
        obj.mtl_path = "assets/6887_allied_avenger.mtl";
        obj.scale = make_float3(0.05f, 0.05f, 0.05f);
        obj.position = make_float3(0.0f, 1.0f, 0.0f);
        obj.rotation_x = -M_PI / 2.0f;
        obj.rotation_y = -M_PI / 3.0f;
        obj.rotation_z = 0.0f;
        obj.is_emissive = false;
        scene.objects.push_back(obj);

        // Single object
        SceneObject obj2;
        obj2.name = "Spitfire";
        obj2.obj_path = "assets/spitfire/spitfire.obj";
        obj2.mtl_path = "assets/spitfire/spitfire.mtl";
        obj2.scale = make_float3(0.01f, 0.01f, 0.01f);
        obj2.position = make_float3(0.0f, 0.0f, 0.0f);
        obj2.rotation_x = 0.0f;
        obj2.rotation_y = 0.0f;
        obj2.rotation_z = 0.0f;
        obj2.is_emissive = false;
        //scene.objects.push_back(obj2);

        SceneObject obj3;
        obj3.name = "Geosphere";
        obj3.obj_path = "assets/geosphere_glass/geosphere.obj";
        obj3.mtl_path = "assets/geosphere_glass/geosphere_glass.mtl";
        obj3.scale = make_float3(1.0f, 1.0f, 1.0f);
        obj3.position = make_float3(2.0f, 0.0f, 0.0f);
        obj3.rotation_x = 0.0f;
        obj3.rotation_y = 0.0f;
        obj3.rotation_z = 0.0f;
        obj3.is_emissive = false;
        //scene.objects.push_back(obj3);  

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

        scene.num_lights = 0;
        scene.envmap_scale = 1.0f;
        scene.envmap_exposure = 0.0f;

        return scene;
    }

    static SceneData getGeosphereScene() {
        SceneData scene;
        scene.id = SceneID::GEOSPHERE;
        scene.name = "Geosphere (Furnace Test)";
        scene.envmap_path = "";

        scene.camera_position = make_float3(0.0f, 1.0f, 4.0f);
        scene.camera_lookat = make_float3(0.0f, 0.0f, 0.0f);
        scene.camera_up = make_float3(0.0f, 1.0f, 0.0f);
        scene.camera_vfov = 60.0f;

        // Single object
        SceneObject obj;
        obj.name = "Geosphere";
        //obj.obj_path = "assets/geosphere_glass/geosphere.obj";
        //obj.mtl_path = "assets/geosphere_glass/geosphere_glass.mtl";
        obj.obj_path = "assets/geosphere_lambert/geosphere_lambert.obj";
        obj.mtl_path = "assets/geosphere_lambert/geosphere_lambert.mtl";
        obj.scale = make_float3(1.0f, 1.0f, 1.0f);
        obj.position = make_float3(0.0f, 0.0f, 0.0f);
        obj.rotation_x = 0.0f;
        obj.rotation_y = 0.0f;
        obj.rotation_z = 0.0f;
        obj.is_emissive = false;
        scene.objects.push_back(obj);

        scene.lights[0].type = 1;
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
        scene.envmap_path = "assets/golden_gate_hills_2k.hdr";

        scene.camera_position = make_float3(-0.7f, 3.0f, 8.0f);
        scene.camera_lookat = make_float3(0.0f, 0.0f, 0.0f);
        scene.camera_up = make_float3(0.0f, 1.0f, 0.0f);
        scene.camera_vfov = 60.0f;

        // Single object
        SceneObject obj;
        obj.name = "Spitfire";
        obj.obj_path = "assets/spitfire/spitfire.obj";
        obj.mtl_path = "assets/spitfire/spitfire.mtl";
        obj.scale = make_float3(0.01f, 0.01f, 0.01f);
        obj.position = make_float3(0.0f, 0.0f, 0.0f);
        obj.rotation_x = 0.0f;
        obj.rotation_y = 0.0f;
        obj.rotation_z = 0.0f;
        obj.is_emissive = false;
        scene.objects.push_back(obj);

        scene.num_lights = 0;
        scene.envmap_scale = 1.0f;
        scene.envmap_exposure = 0.0f;

        return scene;
    }
};