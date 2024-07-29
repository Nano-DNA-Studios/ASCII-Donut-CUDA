#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MemoryManagement.h"

template <typename T>
cudaError_t static AssignMemory(T** variable, int size)
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
cudaError_t static GetVariable(T* hostVariable, T* deviceVariable, int size)
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
cudaError_t static AssignVariable(T** variable, T* assignedValue, int size)
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