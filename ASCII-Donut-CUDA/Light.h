#pragma once

#include <algorithm>
#include <cmath>
#include <crt/host_defines.h>


class Light
{
public:

	float X, Y ,Z;

	int Strength;

	float Magnitude;

	float NormalizedX, NormalizedY, NormalizedZ;

private:

	const float pi = 3.141f;

public:
	__host__ __device__ Light(float x, float y, float z, int strength) : X(x), Y(y), Z(z), Strength(strength)
	{
		Magnitude = sqrtf(X * X + Y * Y + Z * Z);
		NormalizedX = Magnitude > 0 ? X / Magnitude : X;
		NormalizedY = Magnitude > 0 ? Y / Magnitude : Y;
		NormalizedZ = Magnitude > 0 ? Z / Magnitude : Z;
	}
	
	/// <summary>
		/// Calculates the intensity of the light on an object at a given point based off the Inverse Square Law
		/// </summary>
		/// <param name="ox"> Objects X Position </param>
		/// <param name="oy"> Objects Y Position </param>
		/// <param name="oz"> Objects Z Position </param>
		/// <returns> The Intensity Value at a point on an object. </returns>
	__host__ __device__ float GetIntensity(float ox, float oy, float oz)
	{
		float rx = ox - X;
		float ry = oy - Y;
		float rz = oz - Z;

		float r = sqrtf(rx * rx + ry * ry + rz + rz);

		return r > 0 ? (float)Strength / (4 * pi * r) : (float)Strength / (4 * pi);
	}

	__host__ __device__ void UpdatePosition(float x, float y, float z)
	{
		X = x;
		Y = y;
		Z = z;
		Magnitude = sqrtf(X * X + Y * Y + Z * Z);
		NormalizedX = Magnitude > 0 ? X / Magnitude : X;
		NormalizedY = Magnitude > 0 ? Y / Magnitude : Y;
		NormalizedZ = Magnitude > 0 ? Z / Magnitude : Z;
	}

	__host__ __device__ void UpdateStrength(float strength)
	{
		Strength = strength;
	}
};

