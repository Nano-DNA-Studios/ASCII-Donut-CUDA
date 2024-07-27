// ASCII-Donut-CPP.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <cuda_runtime.h>
#include <iostream>
#include "Donut.h"

int main()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	//for (int i = 0; i < nDevices; i++) {
	//	cudaDeviceProp prop;
	//	cudaGetDeviceProperties(&prop, i);
	//	std::cout << "Device Number: " << i << std::endl;
	//	std::cout << "  Device name: " << prop.name << std::endl;
	//	std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
	//	std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
	//	std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
	//	std::cout << "  Max block dimensions: [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
	//	std::cout << "  Max grid dimensions: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
	//}

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

		while (true)
		{
			donut->A += 0.002f;
			donut->B += 0.002f;
			donut->Render();
		}
	}
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
