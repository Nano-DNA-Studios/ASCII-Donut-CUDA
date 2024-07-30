// ASCII-Donut-CPP.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <cuda_runtime.h>
#include <iostream>
#include "Donut.h"

int main()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout << "Device Number: " << i << std::endl;
		std::cout << "  Device name: " << prop.name << std::endl;
		std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
		std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
		std::cout << "  Max block dimensions: [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
		std::cout << "  Max grid dimensions: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
	}

	std::cout << "Resize your console, once you're ready input (Y) : ";

	char reply;

	std::cin >> reply;

	reply = tolower(reply);

	system("cls");

	if (reply == 'y')
	{
		Light* light = new Light(0, 1, -1, 50);

		Donut* donut = new Donut(light);

		donut->A = 0;
		donut->B = 0;

		float counter = 0;

		while (true)
		{
			donut->A += 0.002f;
			donut->B += 0.002f;

			donut->XPos = sinf(counter) * 5;
			donut->YPos = cosf(counter);

			counter += 0.01f;

			donut->Render();
		}
	}
}
