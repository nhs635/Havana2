#ifndef _CUDA_ERROR_CHECK_CUH_
#define _CUDA_ERROR_CHECK_CUH_

#include <iostream>

#include <cuda/cuda_runtime.h>
#include <cuda/cufft.h>


// CUDA Error Check Functions
static void cudaCheckError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		//exit(EXIT_FAILURE);
	}
}
#define CUDA_CHECK_ERROR(err) (cudaCheckError(err, __FILE__, __LINE__ ))

static void cufftCheckError(cufftResult err, const char *file, int line)
{
	const char* err_code;

	switch (err)
	{
	case CUFFT_SUCCESS: err_code = "CUFFT_SUCCESS";
	case CUFFT_INVALID_PLAN: err_code = "CUFFT_INVALID_PLAN";
	case CUFFT_ALLOC_FAILED: err_code = "CUFFT_ALLOC_FAILED";
	case CUFFT_INVALID_TYPE: err_code = "CUFFT_INVALID_TYPE";
	case CUFFT_INVALID_VALUE: err_code = "CUFFT_INVALID_VALUE";
	case CUFFT_INTERNAL_ERROR: err_code = "CUFFT_INTERNAL_ERROR";
	case CUFFT_EXEC_FAILED: err_code = "CUFFT_EXEC_FAILED";
	case CUFFT_SETUP_FAILED: err_code = "CUFFT_SETUP_FAILED";
	case CUFFT_INVALID_SIZE: err_code = "CUFFT_INVALID_SIZE";
	case CUFFT_UNALIGNED_DATA: err_code = "CUFFT_UNALIGNED_DATA";
	default: err_code = "<unknown>";
	}

	if (err != CUFFT_SUCCESS)
	{
		fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", err, err_code, __FILE__, __LINE__);
		//exit(EXIT_FAILURE);
	}
}
#define CUFFT_CHECK_ERROR(err) (cufftCheckError(err, __FILE__, __LINE__))

#endif
