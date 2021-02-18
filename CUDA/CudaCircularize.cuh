#ifndef _CUDA_CIRCULARIZE_CUH_
#define _CUDA_CIRCULARIZE_CUH_

#include <Havana2/Configuration.h>

#ifdef CUDA_ENABLED

#include <iostream>
#include <cuda/cuda_runtime.h>

#define N_THREADS 16

// GPU kernels
static __global__ void getInterpolationMap(float* rho, float* theta, int radius, int width);
static __global__ void interpolation(uint8_t* pRect, uint8_t* pCirc, float* rho, float* theta, 
									 int radius, int circ_center, int width, int height, int channels = 1);


class CudaCircularize
{
public:
	CudaCircularize(int _radius, int _width, int _height);
	~CudaCircularize();

public:
	void operator()(uint8_t* pRect, uint8_t* pCirc, int circ_center);
	void operator()(uint8_t* pRect, uint8_t* pCirc, const char* rgb, int circ_center);
	
private:
	float* deviceRho;
	float* deviceTheta;

	uint8_t* deviceRect;
	uint8_t* deviceCirc;
	uint8_t* deviceRectRGB;
	uint8_t* deviceCircRGB;

	dim3 blocksPerGrid;
	dim3 threadsPerBlock;

private:
	int radius, diameter;
	int width, height;  // width: nalines
};

#endif

#endif