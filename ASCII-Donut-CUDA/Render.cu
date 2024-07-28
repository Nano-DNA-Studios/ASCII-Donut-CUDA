#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <Windows.h>
#include "Light.h"
#include "Render.cuh"
#include <math_functions.h>
#include <device_atomic_functions.h>  // For atomic functions
#include <device_functions.h>         // For device intrinsic functions

//extern "C" __declspec(dllimport) cudaError_t static AssignMemory(T * *variable, int size = 1)

 
//Add Z Buffer in the future

cudaError_t RenderDonut(float A, float B, float R1, float R2, float XPos, float YPos, float* theta, float* phi, int thetaSize, int phiSize, char* buffer, int bufferSize, float* zBuffer, int width, int height, Light* lightSource);

__device__ float clamp(float n, float lower, float upper) {
	return max(lower, min(n, upper));
}

// Helper function to perform atomicMax on floating point values in CUDA
__device__ bool atomicMaxFloat(float* address, float val)
{
	int* address_as_int = (int*)address;
	int old = *address_as_int;
	int assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old) < val;
}

__global__ void render(float* AVal, float* BVal, float* R1Val, float* R2Val, float* XPosVal, float* YPosVal, float* theta, float* phi, char* buffer, float* zBuffer, int* width, int* height, Light* light)
{
	char _luminenceVals[12] = { '.', ',', '-', '~', ':',';', '=','!', '*', '#', '$', '@' };

	int _luminenceValuesLength = 12;

	int thetaIndex = blockIdx.x;
	int phiIndex = threadIdx.x;

	float A = *AVal;
	float B = *BVal;
	float R1 = *R1Val;
	float R2 = *R2Val;
	float XPos = *XPosVal;
	float YPos = *YPosVal;
	int _width = *width;
	int _height = *height;

	int sizeOfScreen = _width * _height;

	float K2 = 10;
	float K1 = _width * K2 * 3 / (24 * (R1 + R2));

	float cosA = cosf(A);
	float sinA = sinf(A);
	float cosB = cosf(B);
	float sinB = sinf(B);

	float cosTheta = cosf(theta[thetaIndex]);
	float sinTheta = sinf(theta[thetaIndex]);

	float circleX = R2 + R1 * cosTheta;
	float circleY = R1 * sinTheta;

	float cosPhi = cosf(phi[phiIndex]);
	float sinPhi = sinf(phi[phiIndex]);

	float x1 = cosB * cosPhi + sinB * sinA * sinPhi;
	float x2 = cosA * sinB;

	float y1 = sinB * cosPhi - cosB * sinA * sinPhi;
	float y2 = cosA * cosB;

	float z1 = cosA * sinPhi;
	float z2 = sinA;

	float x = XPos + circleX * x1 - circleY * x2;
	float y = YPos + circleX * y1 + circleY * y2;
	float z = K2 + circleX * z1 + circleY * z2;

	float ooz = 1 / z;

	int xp = (int)(_width / 2 + K1 * ooz * x);
	int yp = (int)(_height / 2 - K1 * ooz * y);

	xp = clamp(xp, 0, _width - 1);
	yp = clamp(yp, 0, _height - 1);

	int idx = xp + yp * _width;

	float nx = cosTheta;
	float ny = sinTheta;

	float Nx = nx * x1 - ny * x2;
	float Ny = nx * y1 + ny * y2;
	float Nz = nx * z1 + ny * z2;

	float mag = sqrtf(Nx * Nx + Ny * Ny + Nz * Nz);

	Nx = Nx / mag;
	Ny = Ny / mag;
	Nz = Nz / mag;

	float Luminence = Nx * light->NormalizedX + Ny * light->NormalizedY + Nz * light->NormalizedZ;

	if (atomicMaxFloat(&zBuffer[idx], ooz))
	{
		Luminence = Nx * light->NormalizedX + Ny * light->NormalizedY + Nz * light->NormalizedZ;
		int luminance_index = (int)((Luminence + 1) * (_luminenceValuesLength / 2) * light->GetIntensity(x, y, z));
		luminance_index = clamp(luminance_index, 0, _luminenceValuesLength - 1);

		buffer[idx] = _luminenceVals[luminance_index];
	}
}

template <typename T>
cudaError_t static AssignMemory(T** variable, int size = 1)
{
	cudaError_t cudaStatus;
	// Allocate memory for the type T, not just float
	cudaStatus = cudaMalloc((void**)variable, size * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	return cudaStatus;

Error:
	// You might want to add clean-up code here if needed
	cudaFree(variable);
	return cudaStatus;
}

template <typename T>
/// <summary>
/// Get the variable from the GPU memory to the CPU memory
/// </summary>
cudaError_t static GetVariable(T* hostVariable, T* deviceVariable, int size = 1)
{
	cudaError_t cudaStatus;

	// Copy data from device to host
	cudaStatus = cudaMemcpy(hostVariable, deviceVariable, size * sizeof(T), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	return cudaStatus;
}

template <typename T>
cudaError_t static AssignVariable(T** variable, T* assignedValue, int size = 1)
{
	cudaError_t cudaStatus;
	// Allocate memory for the type T, not just float
	cudaStatus = cudaMalloc((void**)variable, size * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(*variable, assignedValue, size * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	return cudaStatus;

Error:
	// You might want to add clean-up code here if needed
	cudaFree(variable);
	return cudaStatus;
}

cudaError_t RenderDonut(float A, float B, float R1, float R2, float XPos, float YPos, float* theta, float* phi, int thetaSize, int phiSize, char* buffer, int bufferSize, float* zBuffer, int width, int height, Light* lightSource)
{
	float* kernel_A = 0;
	float* kernel_B = 0;
	float* kernel_R1 = 0;
	float* kernel_R2 = 0;
	float* kernel_XPos = 0;
	float* kernel_YPos = 0;
	int* kernel_width = 0;
	int* kernel_height = 0;
	float* kernel_theta = 0;
	float* kernel_phi = 0;
	char* kernel_buffer = 0;
	float* kernel_zBuffer = 0;
	Light* kernel_light = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//Assign all the Variables and Copy to GPU memory
	cudaStatus = AssignVariable(&kernel_A, &A);
	cudaStatus = AssignVariable(&kernel_B, &B);
	cudaStatus = AssignVariable(&kernel_R1, &R1);
	cudaStatus = AssignVariable(&kernel_R2, &R2);
	cudaStatus = AssignVariable(&kernel_XPos, &XPos);
	cudaStatus = AssignVariable(&kernel_YPos, &YPos);
	cudaStatus = AssignVariable(&kernel_width, &width);
	cudaStatus = AssignVariable(&kernel_height, &height);

	cudaStatus = AssignVariable(&kernel_theta, theta, thetaSize);
	cudaStatus = AssignVariable(&kernel_phi, phi, phiSize);
	cudaStatus = AssignVariable(&kernel_light, lightSource, 1);

	cudaStatus = AssignVariable(&kernel_buffer, buffer, bufferSize);
	cudaStatus = AssignVariable(&kernel_zBuffer, zBuffer, bufferSize);

	render << <thetaSize, phiSize >> > (kernel_A, kernel_B, kernel_R1, kernel_R2, kernel_XPos, kernel_YPos, kernel_theta, kernel_phi, kernel_buffer, kernel_zBuffer, kernel_width, kernel_height, kernel_light);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "render launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
	}

	// Copy the Result of the buffer on the GPU to the buffer on the CPU
	cudaStatus = GetVariable(buffer, kernel_buffer, bufferSize);

	//Free the Memory to be safe
	cudaFree(kernel_A);
	cudaFree(kernel_B);
	cudaFree(kernel_R1);
	cudaFree(kernel_R2);
	cudaFree(kernel_XPos);
	cudaFree(kernel_YPos);
	cudaFree(kernel_width);
	cudaFree(kernel_height);
	cudaFree(kernel_theta);
	cudaFree(kernel_phi);
	cudaFree(kernel_buffer);

	return cudaStatus;
}










