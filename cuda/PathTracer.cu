#include <optix.h>
#include <optix_device.h>

#include "optix_params.h"
#include "float3_math.h"

extern "C" {
__constant__ Params params;
}

#define M_PI       3.14159265358979323846f