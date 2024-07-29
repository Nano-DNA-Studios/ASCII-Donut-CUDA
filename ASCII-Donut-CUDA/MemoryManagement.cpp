#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MemoryManagement.h"

#ifndef MY_HEADER_FILE_H
#define MY_HEADER_FILE_H


cudaError_t static AssignMemory(void** variable, size_t typeSize, int size)
{
	cudaError_t cudaStatus;
	// Allocate memory for the type T, not just float
	cudaStatus = cudaMalloc(variable, size * typeSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(variable);
	}

	return cudaStatus;
}

cudaError_t static GetVariable(void* hostVariable, void* deviceVariable, size_t typeSize, int size)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMemcpy(hostVariable, deviceVariable, size * typeSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	return cudaStatus;

}

cudaError_t static AssignVariable(void** variable, void* assignedValue, size_t typeSize, int size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(variable, size * typeSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(*variable, assignedValue, size * typeSize, cudaMemcpyHostToDevice);
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

#endif