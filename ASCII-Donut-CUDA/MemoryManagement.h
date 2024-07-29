#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

template <typename T>
cudaError_t AssignMemory(T** variable, int size = 1);

template <typename T>
/// <summary>
/// Get the variable from the GPU memory to the CPU memory
/// </summary>
cudaError_t GetVariable(T* hostVariable, T* deviceVariable, int size = 1);

template <typename T>
cudaError_t AssignVariable(T** variable, T* assignedValue, int size = 1);