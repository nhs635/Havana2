
#include "CudaCircularize.cuh"
#include "Common/CudaErrorCheck.cuh"

#include <Havana2/Configuration.h>

#ifdef CUDA_ENABLED

#include <math_functions.h>
#include <math_constants.h>


// GPU kernels
__global__ void getInterpolationMap(float* rho, float* theta, int radius, int width)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * 2 * radius;

	if ((x < 2 * radius) && (y < 2 * radius))
	{
		// Set meshgrid
		int xr = radius - x;
		int yr = y - radius;

		// Rho : Interpolation Map
		rho[offset] = ((float)radius - 1.0f) / radius * hypotf(xr, yr);
		
		// Theta : Interpolation Map
		theta[offset] = ((float)width - 1.0f) / (2.0f * CUDART_PI_F) * (CUDART_PI_F + atan2f(yr, xr));
	}
}

__global__ void interpolation(uint8_t* pRect, uint8_t* pCirc, float* rho, float* theta, 
							  int radius, int circ_center, int width, int height, int channels)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	int offset = x + y * 2 * radius;

	if ((x < 2 * radius) && (y < 2 * radius))
	{
		int i = theta[offset];
		int j = circ_center + rho[offset];

		if (j < circ_center + radius)
		{
			// Bilinear
			int i0 = floorf(i); int j0 = floorf(j);
			int i1 = i0 + 1; int j1 = j0 + 1;

			for (int c = 0; c < channels; c++)
			{
				float f00 = pRect[(i0 + j0 * width) * channels + c];
				float f10 = pRect[(i1 + j0 * width) * channels + c];
				float f01 = pRect[(i0 + j1 * width) * channels + c];
				float f11 = pRect[(i1 + j1 * width) * channels + c];

				float f0 = (f01 - f00) / (j1 - j0) * (j - j0) + f00;
				float f1 = (f11 - f10) / (j1 - j0) * (j - j0) + f10;
				pCirc[offset * channels + c] = (f1 - f0) / (i1 - i0) * (i - i0) + f0;
			}

			// Nearest
			///for (int c = 0; c < channels; c++)
			///	pCirc[offset * channels + c] = pRect[(int)(roundf(i) + roundf(j) * width) * channels + c];
		}
		else
			for (int c = 0; c < channels; c++)
				pCirc[offset * channels + c] = 0;
	}
	
}



CudaCircularize::CudaCircularize(int _radius, int _width, int _height) :
	radius(_radius), diameter(2 * _radius), width(_width), height(_height)
{
	// Memory Allocation
	CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceRho, sizeof(float) * diameter * diameter));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceTheta, sizeof(float) * diameter * diameter));

	CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceRect, sizeof(uint8_t) * width * height));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceCirc, sizeof(uint8_t) * diameter * diameter));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceRectRGB, sizeof(uint8_t) * 3 * width * height));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceCircRGB, sizeof(uint8_t) * 3 * diameter * diameter));

	// Grid and Block Dimensions
	blocksPerGrid = dim3((diameter + N_THREADS - 1) / N_THREADS, (diameter + N_THREADS - 1) / N_THREADS);
	threadsPerBlock = dim3(N_THREADS, N_THREADS);

	// Interpolation Map
	getInterpolationMap << < blocksPerGrid, threadsPerBlock >> > (deviceRho, deviceTheta, radius, width);
}

CudaCircularize::~CudaCircularize()
{
	// Memory Deallocation
	CUDA_CHECK_ERROR(cudaFree(deviceRho));
	CUDA_CHECK_ERROR(cudaFree(deviceTheta));

	CUDA_CHECK_ERROR(cudaFree(deviceRect));
	CUDA_CHECK_ERROR(cudaFree(deviceCirc));
	CUDA_CHECK_ERROR(cudaFree(deviceRectRGB));
	CUDA_CHECK_ERROR(cudaFree(deviceCircRGB));
}


void CudaCircularize::operator()(uint8_t* pRect, uint8_t* pCirc, int circ_center)
{	
	// Transfer to Device 	
	CUDA_CHECK_ERROR(cudaMemcpy(deviceRect, pRect, sizeof(uint8_t) * width * height, cudaMemcpyHostToDevice));

	// Circularizing
	interpolation << < blocksPerGrid, threadsPerBlock >> > (deviceRect, deviceCirc, deviceRho, deviceTheta, radius, circ_center, width, height);

	// Transfer to Host
	CUDA_CHECK_ERROR(cudaMemcpy(pCirc, deviceCirc, sizeof(uint8_t) * diameter * diameter, cudaMemcpyDeviceToHost));
}

void CudaCircularize::operator()(uint8_t* pRectRGB, uint8_t* pCircRGB, const char* rgb, int circ_center)
{
	// Transfer to Device 	
	CUDA_CHECK_ERROR(cudaMemcpy(deviceRectRGB, pRectRGB, sizeof(uint8_t) * 3 * width * height, cudaMemcpyHostToDevice));

	// Circularizing
	interpolation << < blocksPerGrid, threadsPerBlock >> > (deviceRectRGB, deviceCircRGB, deviceRho, deviceTheta, radius, circ_center, width, height, 3);

	// Transfer to Host
	CUDA_CHECK_ERROR(cudaMemcpy(pCircRGB, deviceCircRGB, sizeof(uint8_t) * 3 * diameter * diameter, cudaMemcpyDeviceToHost));

	(void)rgb;
}

#endif