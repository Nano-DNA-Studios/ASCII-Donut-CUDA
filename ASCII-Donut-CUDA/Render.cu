#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>


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










