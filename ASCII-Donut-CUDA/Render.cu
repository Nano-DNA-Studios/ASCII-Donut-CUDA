#include "Render.cuh"
#include <Windows.h>
#include "Light.h"
#include <math_functions.h>
#include <device_atomic_functions.h>  // For atomic functions
#include <device_functions.h>         // For device intrinsic functions
#include "MemoryManagement.cpp"

cudaError_t RenderDonut(float A, float B, float R1, float R2, float XPos, float YPos, float* theta, float* phi, int thetaSize, int phiSize, char* buffer, int bufferSize, float* zBuffer, int width, int height, Light* lightSource);

/// <summary>
/// Clamp function that is acceessible to the GPU
/// </summary>
/// <param name="n"> The Value to Clamp </param>
/// <param name="lower"> The Lower bound to Clamp to </param>
/// <param name="upper"> The Upper bound to Clamp to </param>
/// <returns></returns>
__device__ float clamp(float n, float lower, float upper) {
	return max(lower, min(n, upper));
}

/// <summary>
/// Function that performs a Float Atomix Max Operation (Determines Max Value accross all GPU threads)
/// </summary>
/// <param name="address"> Address of the Value to Compare </param>
/// <param name="val"> The New Value to Compare at the address </param>
/// <returns> True if the new Value is larger than the one at the address</returns>
__device__ bool AtomicMaxFloat(float* address, float val)
{
	//Convert the float* to a int*, and dereference the value (Get the Value), create a empty variable
	int* address_as_int = (int*)address;
	int old = *address_as_int;
	int assumed;

	//Do While loop, set Assumed to the last value, do a atomic compare to get the max value check on the newest value if it's changed, repeat while they aren't the same
	do {
		assumed = old;
		old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);

	//Return true if the Value assigned is larger than the value in the pointer
	return __int_as_float(old) < val;
}

/// <summary>
/// Renders the Donut
/// </summary>
/// <param name="AVal"> The Rotation on the X Axis </param>
/// <param name="BVal"> The Rotation on the Y Axis </param>
/// <param name="R1Val"> The Radius of Donut ring </param>
/// <param name="R2Val"> The Radius of the Donut </param>
/// <param name="XPosVal"> The X position of the Donut </param>
/// <param name="YPosVal"> The Y position of the Donut </param>
/// <param name="theta"> The Angles for each point on the Donut Ring that will be rendered </param>
/// <param name="phi"> The Angles for each point on the Donut that will be rendered </param>
/// <param name="buffer"> The ASCII Buffer, array of characters that will be displayed to the screen </param>
/// <param name="zBuffer"> The Z Values of the Buffer </param>
/// <param name="width"> The Screen Width </param>
/// <param name="height"> The Screen Height </param>
/// <param name="light"> The Light Source for the Donut </param>
/// <returns></returns>
__global__ void Render(float* AVal, float* BVal, float* R1Val, float* R2Val, float* XPosVal, float* YPosVal, float* theta, float* phi, char* buffer, float* zBuffer, int* width, int* height, Light* light)
{
	//Get the luminence values
	char _luminenceVals[12] = { '.', ',', '-', '~', ':',';', '=','!', '*', '#', '$', '@' };
	int _luminenceValuesLength = 12;

	//Get the theta and phi index
	int thetaIndex = blockIdx.x;
	int phiIndex = threadIdx.x;

	//Dereference the Pointers
	float A = *AVal;
	float B = *BVal;
	float R1 = *R1Val;
	float R2 = *R2Val;
	float XPos = *XPosVal;
	float YPos = *YPosVal;
	int _width = *width;
	int _height = *height;

	int sizeOfScreen = _width * _height;

	//Get perspective values
	float K2 = 10;
	float K1 = _width * K2 * 3 / (24 * (R1 + R2));

	//Get Cosine and Sin Caches
	float cosA = cosf(A);
	float sinA = sinf(A);
	float cosB = cosf(B);
	float sinB = sinf(B);

	float cosTheta = cosf(theta[thetaIndex]);
	float sinTheta = sinf(theta[thetaIndex]);

	float cosPhi = cosf(phi[phiIndex]);
	float sinPhi = sinf(phi[phiIndex]);

	//Calculate the X and Y positions of the Torus Body (The Circle of the Donut Ring) (  ()---()  )
	float circleX = R2 + R1 * cosTheta;
	float circleY = R1 * sinTheta;

	//Calculate the first and second Term of the X position
	float x1 = cosB * cosPhi + sinB * sinA * sinPhi;
	float x2 = cosA * sinB;

	//Calculate the first and second Term of the Y Position
	float y1 = sinB * cosPhi - cosB * sinA * sinPhi;
	float y2 = cosA * cosB;

	//Calculate first and second Term of the Z Position
	float z1 = cosA * sinPhi;
	float z2 = sinA;

	//Calculate the X, Y, Z Positions
	float x = XPos + circleX * x1 - circleY * x2;
	float y = YPos + circleX * y1 + circleY * y2;
	float z = K2 + circleX * z1 + circleY * z2;

	//Calculate the inverse distance from the camera
	float ooz = 1 / z;

	//Get the x and y pixel positions for the point, clamp their values to within the screen
	int xp = (int)(_width / 2 + K1 * ooz * x);
	int yp = (int)(_height / 2 - K1 * ooz * y);

	xp = clamp(xp, 0, _width - 1);
	yp = clamp(yp, 0, _height - 1);

	//Calculate th Pixel Index
	int idx = xp + yp * _width;

	//Calculate the Normal Vector to the Donut
	float nx = cosTheta;
	float ny = sinTheta;

	float Nx = nx * x1 - ny * x2;
	float Ny = nx * y1 + ny * y2;
	float Nz = nx * z1 + ny * z2;

	//Get Magnitude and Normalize the Vector
	float mag = sqrtf(Nx * Nx + Ny * Ny + Nz * Nz);

	Nx = Nx / mag;
	Ny = Ny / mag;
	Nz = Nz / mag;

	//Dot product of the Normal Vector and the Normalized Vector from 0 -> light position to determine how bright the point is
	float Luminence = Nx * light->NormalizedX + Ny * light->NormalizedY + Nz * light->NormalizedZ;

	//Make atomic compare and determine if pixel is closest to the screen, so that we aren't writing a pixel that is blocked
	if (AtomicMaxFloat(&zBuffer[idx], ooz))
	{
		//Get the character index from the list for the ASCII Pixel
		int luminance_index = (int)((Luminence + 1) * (_luminenceValuesLength / 2) * light->GetIntensity(x, y, z));
		luminance_index = clamp(luminance_index, 0, _luminenceValuesLength - 1);

		//Write the Pixel to the screen
		buffer[idx] = _luminenceVals[luminance_index];
	}
}

/// <summary>
/// CUDA function for Rendering a Donut
/// </summary>
/// <param name="A"> The Rotation on the X Axis </param>
/// <param name="B"> The Rotation on the Y Axis </param>
/// <param name="R1"> The Radius of Donut ring </param>
/// <param name="R2"> The Radius of the Donut </param>
/// <param name="XPos"> The X position of the Donut </param>
/// <param name="YPos"> The Y position of the Donut </param>
/// <param name="theta"> The Angles for each point on the Donut Ring that will be rendered </param>
/// <param name="phi"> The Angles for each point on the Donut that will be rendered </param>
/// <param name="thetaSize"> The number of points that will be rendered on the donut ring </param>
/// <param name="phiSize"> The number of points that will be rendered on the donut </param>
/// <param name="buffer"> The ASCII Buffer, array of characters that will be displayed to the screen </param>
/// <param name="bufferSize"> The Size of the Buffer </param>
/// <param name="zBuffer"> The Z Values of the Buffer </param>
/// <param name="width"> The Screen Width </param>
/// <param name="height"> The Screen Height </param>
/// <param name="lightSource"> The Light Source for the Donut </param>
/// <returns> Returns the Results of CUDA Errors </returns>
cudaError_t RenderDonut(float A, float B, float R1, float R2, float XPos, float YPos, float* theta, float* phi, int thetaSize, int phiSize, char* buffer, int bufferSize, float* zBuffer, int width, int height, Light* lightSource)
{
	//Initialize pointers
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

	//Get GPU
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//Assign all the Variables and Copy to GPU memory
	cudaStatus = AssignVariable((void**)&kernel_A, &A, sizeof(float));
	cudaStatus = AssignVariable((void**)&kernel_B, &B, sizeof(float));
	cudaStatus = AssignVariable((void**)&kernel_R1, &R1, sizeof(float));
	cudaStatus = AssignVariable((void**)&kernel_R2, &R2, sizeof(float));
	cudaStatus = AssignVariable((void**)&kernel_XPos, &XPos, sizeof(float));
	cudaStatus = AssignVariable((void**)&kernel_YPos, &YPos, sizeof(float));
	cudaStatus = AssignVariable((void**)&kernel_width, &width, sizeof(int));
	cudaStatus = AssignVariable((void**)&kernel_height, &height, sizeof(int));
								
	cudaStatus = AssignVariable((void**)&kernel_theta, theta, sizeof(float), thetaSize);
	cudaStatus = AssignVariable((void**)&kernel_phi, phi, sizeof(float), phiSize);
	cudaStatus = AssignVariable((void**)&kernel_light, lightSource, sizeof(Light), 1);
								
	cudaStatus = AssignVariable((void**)&kernel_buffer, buffer, sizeof(char), bufferSize);
	cudaStatus = AssignVariable((void**)&kernel_zBuffer, zBuffer, sizeof(float), bufferSize);

	//Run the Render function
	Render << <thetaSize, phiSize >> > (kernel_A, kernel_B, kernel_R1, kernel_R2, kernel_XPos, kernel_YPos, kernel_theta, kernel_phi, kernel_buffer, kernel_zBuffer, kernel_width, kernel_height, kernel_light);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "render launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
	}

	//Wait for the Render function to finish
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
	cudaFree(kernel_zBuffer);
	cudaFree(kernel_light);

	return cudaStatus;
}