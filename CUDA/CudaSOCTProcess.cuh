#ifndef _CUDA_SOCT_PROCESS_H_
#define _CUDA_SOCT_PROCESS_H_

#include <Havana2/Configuration.h>

#ifdef CUDA_ENABLED

#include <DataProcess/OCTProcess/OCTProcess.h>

#include <iostream>
#include <cuda/cuda_runtime.h>
#include <cuda/cufft.h>

#include <QString>
#include <QFile>


// GPU kernels for sub-functions
#ifndef K_CLOCKING
static __device__ inline void complexMul(cuComplex* pDst, cuComplex* pSrc1, cuComplex* pSrc2);
#else
static __device__ inline void realComplexMul(cuComplex* pDst, float* pSrc1, cuComplex* pSrc2);
#endif
static __global__ void preprocessing(float* fringeF32, ushort* fringeU16, int width, int widthFFT);
#ifndef K_CLOCKING
static __global__ void demodulation(cuComplex* pDemodul, cuComplex* pModul, int width);
static __global__ void calibration(cuComplex* pCalib, cuComplex* pSignal, int width, int widthFFT);
#else
static __global__ void dispersion_compensation(cuComplex* pComp, float* pSignal, int width, int widthFFT);
#endif
static __global__ void logScaling(float* pScaled, cuComplex* pcuComplex, int width, bool is_linear = false);


class CudaSOCTProcess : public OCTProcess
{
// Member Methods
public: // Construtor & Destructor
	CudaSOCTProcess(int _nScans, int _nAlines);
	~CudaSOCTProcess();

private: // Not to call copy constructor and copy assignment operator
	CudaSOCTProcess(const CudaSOCTProcess&);
	CudaSOCTProcess& operator=(const CudaSOCTProcess&);

public:
    // Generate SOCT image
	void operator()(float* spectra, uint16_t* fringe);
	
    // For initiation
	void initialize();

	// For calibration data transfer
	void transferCalibData();

private:
	// Initialization functions
	void allocateMemory();
	void setGridBlockDimension();
	void setCudaStream();
	void setCufftPlan();
	void freeMemory();


// Member Variables
private:
	// Host & Device Buffers
	ushort* deviceRawFringeU16;
	float* deviceRawFringeF32;
#ifndef K_CLOCKING
	cuComplex* deviceBScan;
	cuComplex* deviceDemodulBScan;
	cuComplex* deviceDemodulSignal;
#endif
	cuComplex* deviceCalibSignal;
	cuComplex* deviceCalibBScan;
	float* deviceScaledBScan;

	// Grid and Block Dimensions
	dim3 blocksPerGrid[3];
	dim3 threadsPerBlock;

	// CUDA Streams
	cudaStream_t stream[N_CUDA_STREAMS];

	// CUFFT Plans
#ifndef K_CLOCKING
	cufftHandle cufftPlan_R2C[N_CUDA_STREAMS]; // Real To Complex
	cufftHandle cufftPlan_C2C_Inv[N_CUDA_STREAMS]; // Inv: Complex To Complex
#endif
	cufftHandle cufftPlan_C2C_Fwd[N_CUDA_STREAMS]; // Fwd: Complex To Complexs

private:
	// Size Variables
	int nScans, n2Scans;
	int nScansFFT, n2ScansFFT;
	int nAlines;
};

#endif

#endif