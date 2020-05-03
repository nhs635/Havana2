
#include "CudaOCTProcess.cuh"

#ifdef CUDA_ENABLED

#include <cuda/math_functions.h>
#include "Common/CudaErrorCheck.cuh"


// Constant Memory for Calibration Data
#define _64KB_ 64 * 1024
__device__ __constant__ unsigned char cudaConstMem[_64KB_];


// GPU kernels for sub-functions
__device__ void complexMul(cuComplex* pDst, cuComplex* pSrc1, cuComplex* pSrc2)
{
	(*pDst).x = (*pSrc1).x * (*pSrc2).x - (*pSrc1).y * (*pSrc2).y;
	(*pDst).y = (*pSrc1).x * (*pSrc2).y + (*pSrc1).y * (*pSrc2).x;
}

__global__ void preprocessing(float* fringeF32, ushort* fringeU16, int width, int widthFFT)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < width)
	{
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		int offset0 = x + y * width;
		int offset1 = x + y * widthFFT; // Automatical zero-padding

		float* bg = (float*)cudaConstMem; // OCT interferogram background (from constant memory)
		float* win = (float*)cudaConstMem + width; // Hanning window (from constant memory)

		// single-precision conversion & zero-padding
		// Background subtraction & hann windowing
		fringeF32[offset1] = win[x] * ((float)fringeU16[offset0] - bg[x]);		
	}
}

__global__ void demodulation(cuComplex* pDemodul, cuComplex* pModul, int width)
{
#ifdef FREQ_SHIFTING
	int x0 = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int x1 = ((width - x0) + width / 4) % width;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset0 = x0 + y * width;
	int offset1 = x1 + y * width;

	pDemodul[offset1].x = (pModul[offset0].x) / width;
	pDemodul[offset1].y = (-pModul[offset0].y) / width;
#else
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * width;

	if (x < width / 2)
	{
		pDemodul[offset].x = pModul[offset].x / width;
		pDemodul[offset].y = pModul[offset].y / width;
	}
	else
	{
		pDemodul[offset].x = 0;
		pDemodul[offset].y = 0;
	}
#endif
}

__global__ void calibration(cuComplex* pCalib, cuComplex* pSignal, int width, int widthFFT)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < width)
	{
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		float* calib_idx = (float*)cudaConstMem + 4 * width; // k-linearization index (from constant memory)
		float* calib_weight = (float*)cudaConstMem + 5 * width; // k-linearization weight (from constant memory)
		cuComplex* disp_comp = (cuComplex*)cudaConstMem + 3 * width; // dispersion compensation (from constant memory)

		int offset0 = x + y * widthFFT / 2;
		int offset1 = (int)calib_idx[x] + y * widthFFT;

		// k-linearization
		cuComplex temp;
		temp.x = calib_weight[x] * pSignal[offset1].x + (1 - calib_weight[x]) * pSignal[offset1 + 1].x;
		temp.y = calib_weight[x] * pSignal[offset1].y + (1 - calib_weight[x]) * pSignal[offset1 + 1].y;

		// dispersion compensation
		complexMul(&pCalib[offset0], &temp, &disp_comp[x]);
	}
}

__global__ void logScaling(float* pScaled, cuComplex* pComplex, int width, bool is_linear)
{
	int x0 = threadIdx.x + blockIdx.x * blockDim.x;
#ifdef FREQ_SHIFTING
	int x1 = (x0 + width / 2) % width;
#else
	int x1 = x0;
#endif
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset0 = x0 + y * blockDim.x * gridDim.x;
	int offset1 = x1 + y * blockDim.x * gridDim.x;

	// log Scaling
	if (!is_linear)
		pScaled[offset1] = 10 * __log10f(pComplex[offset0].x * pComplex[offset0].x + pComplex[offset0].y * pComplex[offset0].y);
	else
		pScaled[offset1] = (pComplex[offset0].x * pComplex[offset0].x + pComplex[offset0].y * pComplex[offset0].y);
}



// Class member function definition
CudaOCTProcess:: CudaOCTProcess(int _nScans, int _nAlines) :
	OCTProcess(_nScans, _nAlines),
	nScans(_nScans), n2Scans(_nScans / 2),
	nScansFFT((int)(1 << (int)ceil(log2(_nScans)))),
	n2ScansFFT((int)(1 << (int)ceil(log2(_nScans / 2)))),
	nAlines(_nAlines)
{
}

CudaOCTProcess::~CudaOCTProcess()
{
	// Free Memories
	freeMemory();
}



void CudaOCTProcess::operator()(float* img, uint16_t* fringe)
{
    int transfer_nAlines = nAlines / N_CUDA_STREAMS / N_CUDA_PARTITIONS;
    for (int i = 0; i < N_CUDA_PARTITIONS; i++)
	{
		// Transfer to Device ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for (int j = 0; j < N_CUDA_STREAMS; j++)
			CUDA_CHECK_ERROR(cudaMemcpyAsync(deviceRawFringeU16 + nScans * j * transfer_nAlines, (ushort*)fringe + nScans * (i + N_CUDA_PARTITIONS * j) * transfer_nAlines,
				sizeof(ushort) * nScans * transfer_nAlines, cudaMemcpyHostToDevice, stream[j]));

		// Preprocessing /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (int j = 0; j < N_CUDA_STREAMS; j++)
			preprocessing << < blocksPerGrid[0], threadsPerBlock, 0, stream[j] >> >
				(deviceRawFringeF32 + nScansFFT * j * transfer_nAlines, deviceRawFringeU16 + nScans * j * transfer_nAlines, nScans, nScansFFT);

		// Real-To-Complex FFT ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (int j = 0; j < N_CUDA_STREAMS; j++)
			CUFFT_CHECK_ERROR(cufftExecR2C(cufftPlan_R2C[j],
				deviceRawFringeF32 + nScansFFT * j * transfer_nAlines, deviceBScan + nScansFFT * j * transfer_nAlines));

		// Demodulation ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (int j = 0; j < N_CUDA_STREAMS; j++)
			demodulation << < blocksPerGrid[1], threadsPerBlock, 0, stream[j] >> >
				(deviceDemodulBScan + nScansFFT * j * transfer_nAlines, deviceBScan + nScansFFT * j * transfer_nAlines, nScansFFT);

		// Complex-To-Complex IFFT ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (int j = 0; j < N_CUDA_STREAMS; j++)
			CUFFT_CHECK_ERROR(cufftExecC2C(cufftPlan_C2C_Inv[j],
				deviceDemodulBScan + nScansFFT * j * transfer_nAlines, deviceDemodulSignal + nScansFFT * j * transfer_nAlines, CUFFT_INVERSE));

		// k-linearization & Dispersion Compensation //////////////////////////////////////////////////////////////////////////////////////////////////////
        for (int j = 0; j < N_CUDA_STREAMS; j++)
			calibration << < blocksPerGrid[2], threadsPerBlock, 0, stream[j] >> >
				(deviceCalibSignal + n2ScansFFT * j * transfer_nAlines, deviceDemodulSignal + nScansFFT * j * transfer_nAlines, n2Scans, nScansFFT);

		// Complex-To-Complex FFT ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (int j = 0; j < N_CUDA_STREAMS; j++)
			CUFFT_CHECK_ERROR(cufftExecC2C(cufftPlan_C2C_Fwd[j],
				deviceCalibSignal + n2ScansFFT * j * transfer_nAlines, deviceCalibBScan + n2ScansFFT * j * transfer_nAlines, CUFFT_FORWARD));

		// Scaling ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////		
        for (int j = 0; j < N_CUDA_STREAMS; j++)
			logScaling << < blocksPerGrid[1], threadsPerBlock, 0, stream[j] >> >
				(deviceScaledBScan + n2ScansFFT * j * transfer_nAlines, deviceCalibBScan + n2ScansFFT * j * transfer_nAlines, n2ScansFFT);

		// Transfer to Host //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (int j = 0; j < N_CUDA_STREAMS; j++)
            CUDA_CHECK_ERROR(cudaMemcpyAsync(img + n2ScansFFT * (i + N_CUDA_PARTITIONS * j) * transfer_nAlines, deviceScaledBScan + n2ScansFFT * j * transfer_nAlines,
				sizeof(float) * n2ScansFFT * transfer_nAlines, cudaMemcpyDeviceToHost, stream[j]));
	}
    for (int i = 0; i < N_CUDA_STREAMS; i++)
		CUDA_CHECK_ERROR(cudaStreamSynchronize(stream[i]));
}

void CudaOCTProcess::operator()(float* img, uint16_t* fringe, const char* linear)
{
	int transfer_nAlines = nAlines / N_CUDA_STREAMS / N_CUDA_PARTITIONS;
	for (int i = 0; i < N_CUDA_PARTITIONS; i++)
	{
		// Transfer to Device ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for (int j = 0; j < N_CUDA_STREAMS; j++)
			CUDA_CHECK_ERROR(cudaMemcpyAsync(deviceRawFringeU16 + nScans * j * transfer_nAlines, (ushort*)fringe + nScans * (i + N_CUDA_PARTITIONS * j) * transfer_nAlines,
				sizeof(ushort) * nScans * transfer_nAlines, cudaMemcpyHostToDevice, stream[j]));

		// Preprocessing /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for (int j = 0; j < N_CUDA_STREAMS; j++)
			preprocessing << < blocksPerGrid[0], threadsPerBlock, 0, stream[j] >> >
				(deviceRawFringeF32 + nScansFFT * j * transfer_nAlines, deviceRawFringeU16 + nScans * j * transfer_nAlines, nScans, nScansFFT);

		// Real-To-Complex FFT ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for (int j = 0; j < N_CUDA_STREAMS; j++)
			CUFFT_CHECK_ERROR(cufftExecR2C(cufftPlan_R2C[j],
				deviceRawFringeF32 + nScansFFT * j * transfer_nAlines, deviceBScan + nScansFFT * j * transfer_nAlines));

		// Demodulation ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for (int j = 0; j < N_CUDA_STREAMS; j++)
			demodulation << < blocksPerGrid[1], threadsPerBlock, 0, stream[j] >> >
				(deviceDemodulBScan + nScansFFT * j * transfer_nAlines, deviceBScan + nScansFFT * j * transfer_nAlines, nScansFFT);

		// Complex-To-Complex IFFT ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for (int j = 0; j < N_CUDA_STREAMS; j++)
			CUFFT_CHECK_ERROR(cufftExecC2C(cufftPlan_C2C_Inv[j],
				deviceDemodulBScan + nScansFFT * j * transfer_nAlines, deviceDemodulSignal + nScansFFT * j * transfer_nAlines, CUFFT_INVERSE));

		// k-linearization & Dispersion Compensation //////////////////////////////////////////////////////////////////////////////////////////////////////
		for (int j = 0; j < N_CUDA_STREAMS; j++)
			calibration << < blocksPerGrid[2], threadsPerBlock, 0, stream[j] >> >
				(deviceCalibSignal + n2ScansFFT * j * transfer_nAlines, deviceDemodulSignal + nScansFFT * j * transfer_nAlines, n2Scans, nScansFFT);

		// Complex-To-Complex FFT ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for (int j = 0; j < N_CUDA_STREAMS; j++)
			CUFFT_CHECK_ERROR(cufftExecC2C(cufftPlan_C2C_Fwd[j],
				deviceCalibSignal + n2ScansFFT * j * transfer_nAlines, deviceCalibBScan + n2ScansFFT * j * transfer_nAlines, CUFFT_FORWARD));

		// Scaling ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////		
		for (int j = 0; j < N_CUDA_STREAMS; j++)
			logScaling << < blocksPerGrid[1], threadsPerBlock, 0, stream[j] >> >
				(deviceScaledBScan + n2ScansFFT * j * transfer_nAlines, deviceCalibBScan + n2ScansFFT * j * transfer_nAlines, n2ScansFFT, true);

		// Transfer to Host //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for (int j = 0; j < N_CUDA_STREAMS; j++)
			CUDA_CHECK_ERROR(cudaMemcpyAsync(img + n2ScansFFT * (i + N_CUDA_PARTITIONS * j) * transfer_nAlines, deviceScaledBScan + n2ScansFFT * j * transfer_nAlines,
				sizeof(float) * n2ScansFFT * transfer_nAlines, cudaMemcpyDeviceToHost, stream[j]));
	}
	for (int i = 0; i < N_CUDA_STREAMS; i++)
		CUDA_CHECK_ERROR(cudaStreamSynchronize(stream[i]));
	
	(void)linear;
}


void CudaOCTProcess::initialize()
{
	transferCalibData();
	allocateMemory();
	setGridBlockDimension();
	setCudaStream();
	setCufftPlan();
}


void CudaOCTProcess::transferCalibData()
{
	// Transfer OCT Calibration Data to Device as Constant Memory
	int tempOffset = 0;
	unsigned char* temp = (unsigned char*)malloc(sizeof(float) * 4 * nScans);
	memcpy(temp + tempOffset, this->getBg(), sizeof(float) * nScans);  tempOffset += sizeof(float) * nScans;
	memcpy(temp + tempOffset, this->getWin(), sizeof(float) * nScans);  tempOffset += sizeof(float) * nScans;
	memcpy(temp + tempOffset, this->getCalibIndex(), sizeof(float) * n2Scans); tempOffset += sizeof(float) * n2Scans;
	memcpy(temp + tempOffset, this->getCalibWeight(), sizeof(float) * n2Scans); tempOffset += sizeof(float) * n2Scans;
	memcpy(temp + tempOffset, this->getDispComp(), sizeof(cuComplex) * n2Scans); tempOffset += sizeof(cuComplex) * n2Scans;

	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(cudaConstMem, temp, tempOffset)); free(temp);
}


void CudaOCTProcess::allocateMemory()
{
	// Set Host & Device Memories
    CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceRawFringeU16, sizeof(ushort2) * nScans * nAlines / N_CUDA_PARTITIONS));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceRawFringeF32, sizeof(float) * nScansFFT * nAlines / N_CUDA_PARTITIONS));
    CUDA_CHECK_ERROR(cudaMemset(deviceRawFringeF32, 0, sizeof(float) * nScansFFT * nAlines / N_CUDA_PARTITIONS));

    CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceBScan, sizeof(cuComplex) * nScansFFT * nAlines / N_CUDA_PARTITIONS));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceDemodulBScan, sizeof(cuComplex) * nScansFFT * nAlines / N_CUDA_PARTITIONS));
    CUDA_CHECK_ERROR(cudaMemset(deviceBScan, 0, sizeof(cuComplex) * nScansFFT * nAlines / N_CUDA_PARTITIONS));
    CUDA_CHECK_ERROR(cudaMemset(deviceDemodulBScan, 0, sizeof(cuComplex) * nScansFFT * nAlines / N_CUDA_PARTITIONS));

    CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceDemodulSignal, sizeof(cuComplex) * nScansFFT * nAlines / N_CUDA_PARTITIONS));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceCalibSignal, sizeof(cuComplex) * n2ScansFFT * nAlines / N_CUDA_PARTITIONS));
    CUDA_CHECK_ERROR(cudaMemset(deviceCalibSignal, 0, sizeof(cuComplex) * n2ScansFFT * nAlines / N_CUDA_PARTITIONS));

    CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceCalibBScan, sizeof(cuComplex) * n2ScansFFT * nAlines / N_CUDA_PARTITIONS));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&deviceScaledBScan, sizeof(float) * n2ScansFFT * nAlines / N_CUDA_PARTITIONS));
}

void CudaOCTProcess::setGridBlockDimension()
{
	// Grid and Block Dimension 
    blocksPerGrid[0] = dim3((nScans + N_CUDA_THREADS - 1) / N_CUDA_THREADS, nAlines / N_CUDA_PARTITIONS / N_CUDA_STREAMS / N_CUDA_THREADS);
    blocksPerGrid[1] = dim3((n2ScansFFT + N_CUDA_THREADS - 1) / N_CUDA_THREADS, nAlines / N_CUDA_PARTITIONS / N_CUDA_STREAMS / N_CUDA_THREADS);
    blocksPerGrid[2] = dim3((n2Scans + N_CUDA_THREADS - 1) / N_CUDA_THREADS, nAlines / N_CUDA_PARTITIONS / N_CUDA_STREAMS / N_CUDA_THREADS);
    threadsPerBlock = dim3(N_CUDA_THREADS, N_CUDA_THREADS);
}

void CudaOCTProcess::setCudaStream()
{
	// CUDA Stream 
    for (int i = 0; i < N_CUDA_STREAMS; i++)
		CUDA_CHECK_ERROR(cudaStreamCreate(&stream[i]));
}

void CudaOCTProcess::setCufftPlan()
{
	// FFT Plans 
	int rank = 1;
	int n1[] = { nScansFFT };
	int n2[] = { n2ScansFFT };
	int istride = 1, ostride = 1;
	int idist1 = nScansFFT, odist1 = idist1;
	int idist2 = n2ScansFFT, odist2 = idist2;
	int inembed[] = { 0 }, onembed[] = { 0 };
    int batch = nAlines / N_CUDA_PARTITIONS / N_CUDA_STREAMS;
	
    for (int i = 0; i < N_CUDA_STREAMS; i++)
	{
		CUFFT_CHECK_ERROR(cufftPlanMany(&cufftPlan_R2C[i], rank, n1, inembed, istride, idist1, onembed, ostride, odist1, CUFFT_R2C, batch));
		CUFFT_CHECK_ERROR(cufftSetStream(cufftPlan_R2C[i], stream[i]));

		CUFFT_CHECK_ERROR(cufftPlanMany(&cufftPlan_C2C_Inv[i], rank, n1, inembed, istride, idist1, onembed, ostride, odist1, CUFFT_C2C, batch));
		CUFFT_CHECK_ERROR(cufftSetStream(cufftPlan_C2C_Inv[i], stream[i]));

		CUFFT_CHECK_ERROR(cufftPlanMany(&cufftPlan_C2C_Fwd[i], rank, n2, inembed, istride, idist2, onembed, ostride, odist2, CUFFT_C2C, batch));
		CUFFT_CHECK_ERROR(cufftSetStream(cufftPlan_C2C_Fwd[i], stream[i]));
	}
}


void CudaOCTProcess::freeMemory()
{
	// Free Objects and Memories
    for (int i = 0; i < N_CUDA_STREAMS; i++)
	{
		CUDA_CHECK_ERROR(cudaStreamDestroy(stream[i]));
		CUFFT_CHECK_ERROR(cufftDestroy(cufftPlan_R2C[i]));
		CUFFT_CHECK_ERROR(cufftDestroy(cufftPlan_C2C_Inv[i]));
		CUFFT_CHECK_ERROR(cufftDestroy(cufftPlan_C2C_Fwd[i]));
	}

	CUDA_CHECK_ERROR(cudaFree(deviceRawFringeU16));
	CUDA_CHECK_ERROR(cudaFree(deviceRawFringeF32));

	CUDA_CHECK_ERROR(cudaFree(deviceBScan));
	CUDA_CHECK_ERROR(cudaFree(deviceDemodulBScan));

	CUDA_CHECK_ERROR(cudaFree(deviceDemodulSignal));
	CUDA_CHECK_ERROR(cudaFree(deviceCalibSignal));

	CUDA_CHECK_ERROR(cudaFree(deviceCalibBScan));
	CUDA_CHECK_ERROR(cudaFree(deviceScaledBScan));
}

#endif
