#ifndef RENDER_H
#define RENDER_H

// CUDA Runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <math.h>

// Forward declaration of any kernels and device/host functions
__global__ void render(float* AVal, float* BVal, float* R1Val, float* R2Val, float* XPosVal, float* YPosVal, float* theta, float* phi, char* buffer, float* zBuffer, int* width, int* height, Light* light);

// Function prototypes
cudaError_t RenderDonut(float A, float B, float R1, float R2, float XPos, float YPos, float* theta, float* phi, int thetaSize, int phiSize, char* buffer, int bufferSize, float* zBuffer, int width, int height, Light* lightSource);

#endif // RENDER_H
