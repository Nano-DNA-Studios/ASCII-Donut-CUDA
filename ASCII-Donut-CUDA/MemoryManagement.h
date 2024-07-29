#ifndef MEMORY_MANAGE_H
#define MEMORY_MANAGE_H


#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

cudaError_t static AssignMemory(void** variable, size_t typeSize, int size = 1);

cudaError_t static GetVariable(void* hostVariable, void* deviceVariable, size_t typeSize, int size = 1);

cudaError_t static AssignVariable(void** variable, void* assignedValue, size_t typeSize, int size = 1);

#endif // MEMORY_MANAGE_H