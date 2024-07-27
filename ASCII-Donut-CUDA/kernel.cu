#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

cudaError_t addWithCuda(float* c, float* a, float* b, unsigned int size);

__global__ void addKernel(float* c, const float* a, const float* b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	c[i] = cosf(a[i]) + sinf(b[i]);

}
template <typename T>
cudaError_t AssignMemory(T** variable, int size = 1)
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
cudaError_t AssignVariable(T** variable, T* assignedValue, int size = 1)
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


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float* c,  float* a,  float* b, unsigned int size)
{
	float* dev_a = 0;
	float* dev_b = 0;
	float* dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	/*cudaStatus = AssignMemory(&dev_c, size);

	float* aCopy = (float*)a;

	cudaStatus = AssignVariable(&dev_a, aCopy, size);

	float* bCopy = (float*)b;

	cudaStatus = AssignVariable(&dev_b, bCopy, size);*/

	// Allocate memory for dev_c, dev_a, dev_b
	cudaStatus = AssignMemory(&dev_c, size);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	cudaStatus = AssignVariable(&dev_a, a, size);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	cudaStatus = AssignVariable(&dev_b, b, size);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	//// Allocate GPU buffers for three vectors (two input, one output).
	//cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	//cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	//cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	//// Copy input vectors from host memory to GPU buffers.
	//cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	goto Error;
	//}

	//cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	goto Error;
	//}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <50, 1024 >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

