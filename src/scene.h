#pragma once

#include <string>
#include <vector>
#include "optix_params.h"
#include "obj_loader.h"

#define M_PI 3.14159265f

enum class SceneID {
    ALLIED_AVENGER = 0,
    GEOSPHERE = 1,
    SPITFIRE = 2,
    STREET = 3,
    SPITFIRE_COMPANY = 4,
    CORNELL_BOX = 5,
    COUNT
};

struct SceneObject {
    std::string name;
    std::string obj_path;
    std::string base_dir;

    float3 position = make_float3(1.0f, 1.0f, 1.0f);
	float3 scale = make_float3(1.0f, 1.0f, 1.0f);
	float rotation_x = 0.0f;
	float rotation_y = 0.0f;
	float rotation_z = 0.0f;
};

struct SceneData {
    SceneID id;
    std::string name;
	std::string envmap_path = "";
    float3 background_color = make_float3(1.0f, 1.0f, 1.0f); // Default background thats displayed if no env map

    // Camera settings
	float3 camera_position = make_float3(0.0f, 1.0f, 4.0f);
	float3 camera_front = make_float3(0.0f, 0.0f, 1.0f);
	float3 camera_up = make_float3(0.0f, 1.0f, 0.0f);
	float camera_vfov = 60.0f;

	float envmap_scale = 1.0f;
	float envmap_exposure = 0.0f;

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
		case SceneID::STREET:
			return getStreetScene();
		case SceneID::SPITFIRE_COMPANY:
			return getSpitfireCompanyScene();
        case SceneID::CORNELL_BOX:
			return getCornellBoxScene();
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

        scene.camera_position = make_float3(-0.77f, 3.03f, 8.62f);
        scene.camera_front = make_float3(0.13f, -0.04f, -0.99f);

        SceneObject obj;
        obj.name = "Allied Avenger";
        obj.obj_path = "assets/allied_avenger/6887_allied_avenger.obj";
        obj.base_dir = "assets/allied_avenger";
        obj.scale = make_float3(0.05f, 0.05f, 0.05f);
        obj.position = make_float3(0.0f, 1.0f, 0.0f);
        obj.rotation_x = -M_PI / 2.0f;
        obj.rotation_y = -M_PI / 3.0f;
        obj.rotation_z = 0.0f;
        scene.objects.push_back(obj);

        SceneObject obj2;
        obj2.name = "Spitfire";
        obj2.obj_path = "assets/spitfire/spitfire.obj";
        obj2.base_dir = "assets/spitfire";
        obj2.scale = make_float3(0.01f, 0.01f, 0.01f);
        obj2.position = make_float3(0.0f, 0.0f, 0.0f);
        obj2.rotation_x = 0.0f;
        obj2.rotation_y = 0.0f;
        obj2.rotation_z = 0.0f;
        scene.objects.push_back(obj2);

        SceneObject obj3;
        obj3.name = "Geosphere";
        obj3.obj_path = "assets/geosphere_glass/geosphere.obj";
        obj3.base_dir = "assets/geosphere_glass";
        obj3.scale = make_float3(1.0f, 1.0f, 1.0f);
        obj3.position = make_float3(2.0f, 0.0f, 0.0f);
        obj3.rotation_x = 0.0f;
        obj3.rotation_y = 0.0f;
        obj3.rotation_z = 0.0f;
        //scene.objects.push_back(obj3);  

        scene.envmap_scale = 1.0f;
        scene.envmap_exposure = 0.0f;

        return scene;
    }

    static SceneData getGeosphereScene() {
        SceneData scene;
        scene.id = SceneID::GEOSPHERE;
        scene.name = "Geosphere (Furnace Test)";
        scene.envmap_path = "";
        scene.background_color = make_float3(1.0f, 1.0f, 1.0f);

        scene.camera_position = make_float3(0.00f, 0.70f, 2.50f);
        scene.camera_front = make_float3(-0.03f, -0.28f, -0.96f);

        SceneObject obj;
        obj.name = "Geosphere";
        obj.obj_path = "assets/geosphere_lambert/geosphere_lambert.obj";
        obj.base_dir = "assets/geosphere_lambert";
        obj.scale = make_float3(1.0f, 1.0f, 1.0f);
        obj.position = make_float3(0.0f, 0.0f, 0.0f);
        obj.rotation_x = 0.0f;
        obj.rotation_y = 0.0f;
        obj.rotation_z = 0.0f;
        scene.objects.push_back(obj);

        scene.envmap_scale = 1.0f;
        scene.envmap_exposure = 0.0f;

        return scene;
    }

    static SceneData getSpitfireScene() {
        SceneData scene;
        scene.id = SceneID::SPITFIRE;
        scene.name = "Spitfire";
        scene.envmap_path = "assets/moonless_golf_2k.hdr";

        scene.camera_position = make_float3(-0.70f, 3.00f, 8.00f);
        scene.camera_front = make_float3(0.59f, -0.27f, -0.76f);

        SceneObject obj;
        obj.name = "Spitfire";
        obj.obj_path = "assets/spitfire/spitfire.obj";
        obj.base_dir = "assets/spitfire";
        obj.scale = make_float3(0.01f, 0.01f, 0.01f);
        obj.position = make_float3(0.0f, 0.0f, 0.0f);
        obj.rotation_x = 0.0f;
        obj.rotation_y = 0.0f;
        obj.rotation_z = 0.0f;
        scene.objects.push_back(obj);

        scene.envmap_scale = 1.0f;
        scene.envmap_exposure = 0.0f;

        return scene;
    }

    static SceneData getStreetScene() {
        SceneData scene;
        scene.id = SceneID::STREET;
        scene.name = "LumberYard Bistro";
        scene.envmap_path = "assets/golden_gate_hills_2k.hdr";
        scene.camera_position = make_float3(-8.0f, 6.0f, 6.0f);
        scene.camera_front = make_float3(0.0f, -3.0f, 0.0f);

        SceneObject street;
        street.name = "Street";
        street.obj_path = "assets/lumberyard/exterior.obj";
        street.base_dir = "assets/lumberyard";
		street.scale = make_float3(0.01f, 0.01f, 0.01f);
        scene.objects.push_back(street);

        SceneObject room;
        room.name = "Room";
        room.obj_path = "assets/lumberyard/interior.obj";
        room.base_dir = "assets/lumberyard";
		room.scale = make_float3(0.01f, 0.01f, 0.01f);
        //scene.objects.push_back(room);

		return scene;
    }

    static SceneData getSpitfireCompanyScene() {
        SceneData scene;
        scene.id = SceneID::SPITFIRE_COMPANY;
        scene.name = "Spitfire Company";
        scene.envmap_path = "assets/golden_gate_hills_2k.hdr";

        scene.camera_position = make_float3(4.39f, 2.81f, 7.44f);
        scene.camera_front = make_float3(-0.29f, -0.20f, -0.94f);

        SceneObject obj;
		obj.name = "Spitfire Company";
		obj.obj_path = "assets/spitfire_company/spitfire_company.obj";
		obj.base_dir = "assets/spitfire_company";
		obj.scale = make_float3(0.01f, 0.01f, 0.01f);
        
        scene.objects.push_back(obj);
        return scene;
	}

    static SceneData getCornellBoxScene() {
        SceneData scene;
        scene.id = SceneID::CORNELL_BOX;
        scene.name = "Cornell Box";
        scene.background_color = make_float3(0.0f, 0.0f, 0.0f);

        scene.camera_position = make_float3(0.96f, 1.27f, 1.45f);
        scene.camera_front = make_float3(0.02f, -0.02f, -1.00f);


        SceneObject obj;
        obj.name = "Cornell Box";
        obj.obj_path = "assets/cornell_box/cornell-box.obj";
        obj.base_dir = "assets/cornell_box";
        obj.scale = make_float3(0.1f, 0.1f, 0.1f);

        scene.objects.push_back(obj);
        return scene;
	}
};