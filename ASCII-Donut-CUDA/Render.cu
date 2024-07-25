#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

cudaError_t RenderDonut(float A, float B, float R1, float R2, float XPos, float YPos, float* theta, float* phi, char* buffer);

__global__ void renderKernel(float A, float B, float R1, float R2, float XPos, float YPos, float* theta, float* phi, char* buffer)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;


	float cosA = cosf(A);
	float sinA = sinf(A);
	float cosB = cosf(B);
	float sinB = sinf(B);


    /*for (float theta = 0; theta < 2 * pi; theta += thetaStep)
    {
        float cosTheta = cos(theta);
        float sinTheta = sin(theta);

        float circleX = R2 + R1 * cosTheta;
        float circleY = R1 * sinTheta;

        for (float phi = 0; phi < 2 * pi; phi += phiStep)
        {
            float cosPhi = cos(phi);
            float sinPhi = sin(phi);

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

            float Luminence = Nx * LightSource->NormalizedX + Ny * LightSource->NormalizedY + Nz * LightSource->NormalizedZ;

            if (idx > sizeOfScreen || idx < 0)
                continue;

            if (ooz > zBuffer[idx])
            {
                int luminance_index = (int)((Luminence + 1) * (_luminenceValuesLength) / (2) * LightSource->GetIntensity(x, y, z));
                luminance_index = clamp(luminance_index, 0, _luminenceValuesLength - 1);
                zBuffer[idx] = ooz;
                _buf[idx] = _luminenceVals[luminance_index];
            }
        }
    }*/



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
//Add Z Buffer in the future

cudaError_t RenderDonut(float A, float B, float R1, float R2, float XPos, float YPos, float* theta, float* phi, char* buffer)
{
	float kernel_A = 0;
	float kernel_B = 0;
	float kernel_R1 = 0;
	float kernel_R2 = 0;
	float kernel_XPos = 0;
	float kernel_YPos = 0;

	float* kernel_theta = 0;
	float* kernel_phi = 0;
	char* kernel_buffer = 0;


	float* dev_a = 0;
	float* dev_b = 0;
	float* dev_c = 0;
	cudaError_t cudaStatus;



	//cudaStatus = cudaSetDevice(0);
	//if (cudaStatus != cudaSuccess)
	//{
	//	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	//	goto Error;
	//}

	//cudaStatus = cudaMalloc((void**)&kernel_A, sizeof(float));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	//cudaStatus = cudaMalloc((void**)&kernel_B, sizeof(float));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	//cudaStatus = cudaMalloc((void**)&kernel_R1, sizeof(float));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}

	//cudaStatus = cudaMalloc((void**)&kernel_R2, sizeof(float));
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMalloc failed!");
	//	goto Error;
	//}



	//// Choose which GPU to run on, change this on a multi-GPU system.
	//cudaStatus = cudaSetDevice(0);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	//	goto Error;
	//}

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

	//// Launch a kernel on the GPU with one thread for each element.
	////addKernel << <50, 1024 >> > (dev_c, dev_a, dev_b);

	//// Check for any errors launching the kernel
	//cudaStatus = cudaGetLastError();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	//	goto Error;
	//}

	//// cudaDeviceSynchronize waits for the kernel to finish, and returns
	//// any errors encountered during the launch.
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	//	goto Error;
	//}

	//// Copy output vector from GPU buffer to host memory.
	//cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	goto Error;
	//}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}










