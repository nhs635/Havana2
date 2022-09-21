#ifndef _OCT_PROCESS_H_
#define _OCT_PROCESS_H_

#include <Havana2/Configuration.h>

#include <iostream>
#include <thread>
#include <complex>

#include <QString>
#include <QFile>

#include <ipps.h>
#include <ippvm.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <Common/array.h>
#include <Common/callback.h>
using namespace np;

#define CH_1 1
#define CH_2 2

struct FFT_R2C // 1D Fourier transformation for real signal (only for forward transformation)
{
public:
	FFT_R2C() :
        pFFTSpec(nullptr), pMemSpec(nullptr), pMemInit(nullptr), pMemBuffer(nullptr)
	{
	}

	~FFT_R2C()
	{
		if (pMemSpec) { ippsFree(pMemSpec); pMemSpec = nullptr; }
		if (pMemInit) { ippsFree(pMemInit); pMemInit = nullptr; }
		if (pMemBuffer) { ippsFree(pMemBuffer); pMemBuffer = nullptr; }
	}

	void operator() (Ipp32fc* dst, const Ipp32f* src, int line = 0)
	{
		ippsFFTFwd_RToPerm_32f(src, &temp(0, line), pFFTSpec, pMemBuffer);
		ippsConjPerm_32fc(&temp(0, line), dst, temp.size(0));
	}

	void initialize(int length, int line)
	{
		// init FFT spec
		const int ORDER = (int)(ceil(log2(length)));
		temp = FloatArray2(1 << ORDER, line); // for TBB operation to avoid thread risk

		int sizeSpec, sizeInit, sizeBuffer;
		ippsFFTGetSize_R_32f(ORDER, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuffer);

		pMemSpec = ippsMalloc_8u(sizeSpec);
		pMemInit = ippsMalloc_8u(sizeInit);
		pMemBuffer = ippsMalloc_8u(sizeBuffer);

		ippsFFTInit_R_32f(&pFFTSpec, ORDER, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, pMemSpec, pMemInit);
	}

private:
	IppsFFTSpec_R_32f* pFFTSpec;
    Ipp8u* pMemSpec;
    Ipp8u* pMemInit;
    Ipp8u* pMemBuffer;

	FloatArray2 temp;
};

struct FFT_C2C // 1D Fourier transformation for complex signal (for both forward and inverse transformation)
{
	FFT_C2C() :
        pFFTSpec(nullptr), pMemSpec(nullptr), pMemInit(nullptr), pMemBuffer(nullptr)
	{
	}

	~FFT_C2C()
	{
		if (pMemSpec) { ippsFree(pMemSpec); pMemSpec = nullptr; }
		if (pMemInit) { ippsFree(pMemInit); pMemInit = nullptr; }
		if (pMemBuffer) { ippsFree(pMemBuffer); pMemBuffer = nullptr; }
	}

	void forward(Ipp32fc* dst, const Ipp32fc* src)
	{
		ippsFFTFwd_CToC_32fc(src, dst, pFFTSpec, pMemBuffer);
	}

	void inverse(Ipp32fc* dst, const Ipp32fc* src)
	{
		ippsFFTInv_CToC_32fc(src, dst, pFFTSpec, pMemBuffer);
	}

	void initialize(int length)
	{
		const int ORDER = (int)(ceil(log2(length)));

		int sizeSpec, sizeInit, sizeBuffer;
		ippsFFTGetSize_C_32fc(ORDER, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuffer);

		pMemSpec = ippsMalloc_8u(sizeSpec);
		pMemInit = ippsMalloc_8u(sizeInit);
		pMemBuffer = ippsMalloc_8u(sizeBuffer);

		ippsFFTInit_C_32fc(&pFFTSpec, ORDER, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, pMemSpec, pMemInit);
	}

private:
	IppsFFTSpec_C_32fc* pFFTSpec;
    Ipp8u* pMemSpec;
    Ipp8u* pMemInit;
    Ipp8u* pMemBuffer;
};


class OCTProcess
{
// Methods
public: // Constructor & Destructor
	explicit OCTProcess(int nScans, int nAlines);
	~OCTProcess();
    
protected: // Not to call copy constrcutor and copy assignment operator
    OCTProcess(const OCTProcess&);
    OCTProcess& operator=(const OCTProcess&);
	
public:
	// Generate OCT image
	void operator()(float* img, uint16_t* fringe);
	void operator()(float* img, uint16_t* fringe, const char* linear);
	   
	// For calibration
    void setBg(const Uint16Array2& frame);
    void setFringe(const Uint16Array2& frame, int ch);

    float* getBg() { return bg.raw_ptr(); }
    float* getBg0() { return bg0.raw_ptr(); }
    float* getFringe(int ch) { return &fringe(0, ch); }
	float* getWin() { return win.raw_ptr(); }
#ifndef K_CLOCKING
	float* getCalibIndex() { return calib_index.raw_ptr(); }
	float* getCalibWeight() { return calib_weight.raw_ptr(); }
#endif
	auto getDispComp() { return dispersion1.raw_ptr(); }

#ifndef K_CLOCKING
    void generateCalibration(int discom_val = 0);
#endif
	void removeCalibration();

    void changeDiscomValue(int discom_val = 0);
#ifndef K_CLOCKING
    //void saveCalibration(const char* calibpath = "calibration.dat");	
	void saveCalibration(QString calibpath = "calibration.dat");
    //void loadCalibration(int ch = CH_1, const char* calibpath = "calibration.dat", const char* bgpath = "bg.bin");
	void loadCalibration(int ch = CH_1, QString calibpath = "calibration.dat", QString bgpath = "bg.bin", bool erasmus = false);
#endif

	// For calibration dialog
	callback2<float*, const char*> drawGraph;
	callback2<int&, int&> waitForRange;
	callback<void> endCalibration;
    
// Variables
protected:
    // FFT objects
#ifndef K_CLOCKING
    FFT_R2C fft1; // fft
    FFT_C2C fft2; // ifft
#endif
    FFT_C2C fft3; // fft
    
    // Size variables
    IppiSize raw_size;
#ifndef K_CLOCKING
    IppiSize raw2_size;
#endif
    IppiSize fft_size;
    IppiSize fft2_size;
    
    // OCT image processing buffer
    FloatArray2 signal;
    ComplexFloatArray2 complex_signal;
#ifndef K_CLOCKING
    ComplexFloatArray2 complex_resamp;
#endif
    ComplexFloatArray2 fft_complex;
    ComplexFloatArray2 fft2_complex;
    FloatArray2 fft2_linear;
    
    // Calibration varialbes
    FloatArray bg0;
    FloatArray bg;
    FloatArray2 fringe;
#ifndef K_CLOCKING
    FloatArray calib_index;
    FloatArray calib_weight;
#endif
    FloatArray win;
#ifndef K_CLOCKING
    ComplexFloatArray dispersion;
    ComplexFloatArray discom;
#endif
    ComplexFloatArray dispersion1;
};

#endif
