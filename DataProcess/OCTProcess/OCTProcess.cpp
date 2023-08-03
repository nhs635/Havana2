
#include "OCTProcess.h"
#include <Common/basic_functions.h>


OCTProcess::OCTProcess(int nScans, int nAlines) :
    raw_size({ nScans, nAlines }),
#ifndef K_CLOCKING
    raw2_size({ raw_size.width / 2, nAlines }),
#endif
	fft_size({ (int)exp2(ceil(log2((double)nScans))), nAlines }),
    fft2_size({ fft_size.width / 2, nAlines }),
	
	signal(fft_size.width, fft_size.height),
	complex_signal(fft_size.width, fft_size.height),
#ifndef K_CLOCKING
    complex_resamp(fft2_size.width, fft2_size.height),
#endif
	fft_complex(fft_size.width, fft_size.height),
    fft2_complex(fft2_size.width, fft2_size.height),
	fft2_linear(fft2_size.width, fft2_size.height), 	

    bg0(raw_size.width),
	bg(raw_size.width),
    fringe(raw_size.width, 2),
#ifndef K_CLOCKING
	calib_index(raw2_size.width),
    calib_weight(raw2_size.width),
#endif
    win(raw_size.width),

#ifndef K_CLOCKING
    dispersion(raw2_size.width),
    discom(raw2_size.width),
    dispersion1(raw2_size.width)
#else
	dispersion1(raw_size.width)
#endif
{   
#ifndef K_CLOCKING
	fft1.initialize(raw_size.width, raw_size.height);
	fft2.initialize(fft_size.width);
	fft3.initialize(fft2_size.width);
#else
	fft3.initialize(fft_size.width);
#endif

	memset(bg0.raw_ptr(), 0, sizeof(float) * bg0.length());
	memset(bg.raw_ptr(), 0, sizeof(float) * bg.length());
    memset(fringe.raw_ptr(), 0, sizeof(float) * fringe.length());

    memset(signal.raw_ptr(), 0, sizeof(float) * signal.length());
#ifndef K_CLOCKING
    memset(complex_resamp.raw_ptr(), 0, sizeof(float) * 2 * complex_resamp.length());
#else
	memset(complex_signal.raw_ptr(), 0, sizeof(float) * 2 * complex_signal.length());
#endif
    memset(fft_complex.raw_ptr(), 0, sizeof(float) * 2 * fft_complex.length());
	for (int i = 0; i < raw_size.width; i++)
	{
        bg0(i) = (float)(POWER_2(15));
        bg(i)  = (float)(POWER_2(15));
		win(i) = (float)(1 - cos(IPP_2PI * i / (raw_size.width - 1))) / 2; // Hann Window
	}

#ifndef K_CLOCKING
	for (int i = 0; i < raw2_size.width; i++)
	{
		calib_index(i) = (float)i * 2.0f;
		calib_weight(i) = 1;
		dispersion(i) = { 1, 0 };
		discom(i) = { 1, 0 };
		dispersion1(i) = { 1, 0 };
	}
#else
	for (int i = 0; i < raw_size.width; i++)
	{
		dispersion1(i) = { 1, 0 };
	}
#endif
}


OCTProcess::~OCTProcess()
{
}


/* OCT Image */
void OCTProcess::operator() (float* img, uint16_t* fringe)
{		
	tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)raw_size.height),
		[&](const tbb::blocked_range<size_t>& r) {
		//for (int i = 0; i < raw_size.height; i++)
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			int r1 = raw_size.width * (int)i;
			int f1 = fft_size.width * (int)i;
			int f2 = fft2_size.width * (int)i;

			// 1. Single Precision Conversion & Zero Padding (3 msec)
			ippsConvert_16u32f(fringe + r1, signal.raw_ptr() + f1, raw_size.width);

			// 2. BG Subtraction & Hanning Windowing (10 msec)
			ippsSub_32f_I(bg, signal.raw_ptr() + f1, raw_size.width);
			ippsMul_32f_I(win, signal.raw_ptr() + f1, raw_size.width);

#ifndef K_CLOCKING
			// 3. Fourier transform (21 msec)
			fft1((Ipp32fc*)(fft_complex.raw_ptr() + f1), signal.raw_ptr() + f1, (int)i);

#ifdef FREQ_SHIFTING
			// 4. Circshift by nfft/4 & Remove virtual peak & Inverse Fourier transform (5+19 sec)
			std::rotate(fft_complex.raw_ptr() + f1, fft_complex.raw_ptr() + f1 + 3 * fft_size.width / 4, fft_complex.raw_ptr() + f1 + fft_size.width);
			ippsSet_32f(0.0f, (Ipp32f*)(fft_complex.raw_ptr() + f1 + fft_size.width / 4), fft_size.width);
#else
			// 4. Remove virtual peak & Inverse Fourier transform (5+19 sec)
			ippsSet_32f(0.0f, (Ipp32f*)(fft_complex.raw_ptr() + f1 + fft_size.width / 2), fft_size.width);
#endif
			fft2.inverse((Ipp32fc*)(complex_signal.raw_ptr() + f1), (const Ipp32fc*)(fft_complex.raw_ptr() + f1));

			// 5. k linear resampling (5 msec)		
			bf::LinearInterp_32fc((const Ipp32fc*)(complex_signal.raw_ptr() + f1), (Ipp32fc*)(complex_resamp.raw_ptr() + f2),
				raw2_size.width, calib_index.raw_ptr(), calib_weight.raw_ptr());

			// 6. Dispersion compensation (4 msec)
			ippsMul_32fc_I((const Ipp32fc*)dispersion1.raw_ptr(), (Ipp32fc*)(complex_resamp.raw_ptr() + f2), raw2_size.width);

			// 7. Fourier transform (9 msec)
			fft3.forward((Ipp32fc*)(fft2_complex.raw_ptr() + f2), (const Ipp32fc*)(complex_resamp.raw_ptr() + f2));

			// 8. dB Scaling (5 msec)
			ippsPowerSpectr_32fc((const Ipp32fc*)(fft2_complex.raw_ptr() + f2), fft2_linear.raw_ptr() + f2, fft2_size.width);
			ippsLog10_32f_A11(fft2_linear.raw_ptr() + f2, img + f2, fft2_size.width);
			ippsMulC_32f_I(10.0f, img + f2, fft2_size.width);

#ifdef FREQ_SHIFTING
			// 9. Circshift by -nfft/2 (1 msec)
			std::rotate(img + f2, img + f2 + fft2_size.width / 2, img + f2 + fft2_size.width);
#endif
#else
			// 3. Dispersion compensation
			ippsMul_32f32fc((const Ipp32f*)signal.raw_ptr() + f1, (const Ipp32fc*)dispersion1.raw_ptr(), (Ipp32fc*)complex_signal.raw_ptr() + f1, raw_size.width);

			// 4. Fourier transform
			fft3.forward((Ipp32fc*)(fft_complex.raw_ptr() + f1), (const Ipp32fc*)(complex_signal.raw_ptr() + f1));

			// 5. dB Scaling
			ippsPowerSpectr_32fc((const Ipp32fc*)(fft_complex.raw_ptr() + f1), fft2_linear.raw_ptr() + f2, fft2_size.width);
			ippsLog10_32f_A11(fft2_linear.raw_ptr() + f2, img + f2, fft2_size.width);
			ippsMulC_32f_I(10.0f, img + f2, fft2_size.width);
#endif
		}
	});
}

void OCTProcess::operator()(float* lin_img, uint16_t* fringe, const char* linear)
{
//	tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)raw_size.height),
//		[&](const tbb::blocked_range<size_t>& r) {
//		for (size_t i = r.begin(); i != r.end(); ++i)
//		{
//			int r1 = raw_size.width * (int)i;
//			int f1 = fft_size.width * (int)i;
//			int f2 = fft2_size.width * (int)i;
//
//			// 1. Single Precision Conversion & Zero Padding (3 msec)
//			ippsConvert_16u32f(fringe + r1, signal.raw_ptr() + f1, raw_size.width);
//
//			// 2. BG Subtraction & Hanning Windowing (10 msec)
//			ippsSub_32f_I(bg, signal.raw_ptr() + f1, raw_size.width);
//			ippsMul_32f_I(win, signal.raw_ptr() + f1, raw_size.width);
//
//			// 3. Fourier transform (21 msec)
//			fft1((Ipp32fc*)(fft_complex.raw_ptr() + f1), signal.raw_ptr() + f1, (int)i);
//
//#ifdef FREQ_SHIFTING
//            // 4. Circshift by nfft/4 & Remove virtual peak & Inverse Fourier transform (5+19 sec)
//            std::rotate(fft_complex.raw_ptr() + f1, fft_complex.raw_ptr() + f1 + 3 * fft_size.width / 4, fft_complex.raw_ptr() + f1 + fft_size.width);
//            ippsSet_32f(0.0f, (Ipp32f*)(fft_complex.raw_ptr() + f1 + fft_size.width / 4), fft_size.width);
//#else
//            // 4. Remove virtual peak & Inverse Fourier transform (5+19 sec)
//            ippsSet_32f(0.0f, (Ipp32f*)(fft_complex.raw_ptr() + f1 + fft_size.width / 2), fft_size.width);
//#endif
//			fft2.inverse((Ipp32fc*)(complex_signal.raw_ptr() + f1), (const Ipp32fc*)(fft_complex.raw_ptr() + f1));
//
//			// 5. k linear resampling (5 msec)		
//			bf::LinearInterp_32fc((const Ipp32fc*)(complex_signal.raw_ptr() + f1), (Ipp32fc*)(complex_resamp.raw_ptr() + f2),
//				raw2_size.width, calib_index.raw_ptr(), calib_weight.raw_ptr());
//
//			// 6. Dispersion compensation (4 msec)
//			ippsMul_32fc_I((const Ipp32fc*)dispersion1.raw_ptr(), (Ipp32fc*)(complex_resamp.raw_ptr() + f2), raw2_size.width);
//
//			// 7. Fourier transform (9 msec)
//			fft3.forward((Ipp32fc*)(fft2_complex.raw_ptr() + f2), (const Ipp32fc*)(complex_resamp.raw_ptr() + f2));
//
//			// 8. Power spectrum (2 msec)
//			ippsPowerSpectr_32fc((const Ipp32fc*)(fft2_complex.raw_ptr() + f2), lin_img + f2, fft2_size.width);
//
//#ifdef FREQ_SHIFTING
//			// 9. Circshift by -nfft/2 (1 msec)
//			std::rotate(lin_img + f2, lin_img + f2 + fft2_size.width / 2, lin_img + f2 + fft2_size.width);
//#endif
//		}
//	});
//
//    (void)linear;
}


/* OCT Calibration */
void OCTProcess::setBg(const Uint16Array2& frame)
{
    int N = 50;

    for (int i = 0; i < frame.size(0); i++)
    {
        bg(i) = 0;
        for (int j = 0; j < N; j++)
            bg(i) += (float)frame(i, j);
        bg(i) /= N;
    }
}


void OCTProcess::setFringe(const Uint16Array2& frame, int ch)
{
    for (int i = 0; i < frame.size(0); i++)
        fringe(i, ch) = (float)frame(i, 0);
}

#ifndef K_CLOCKING
void OCTProcess::generateCalibration(int discom_val)
{
    std::thread calib([&, discom_val]() {

        // buffer
        np::Array<float, 2> fringe1(fft_size.width, 2);
        memset(fringe1.raw_ptr(), 0, sizeof(float) * fft_size.width * 2);
        np::Array<std::complex<float>, 2> res(fft_size.width, 2);
        np::Array<float, 2> res1(fft_size.width, 2);
        np::Array<float, 2> res2(fft_size.width, 2);
        np::Array<std::complex<float>, 2> res3(fft_size.width, 2);
        np::Array<float, 2> phase(raw_size.width, 2);

        for (int ch = 0; ch < 2; ch++)
        {
            // 0. BG removal & windowing
            for (int i = 0; i < raw_size.width; i++)
                fringe1(i, ch) = (fringe(i, ch) - bg(i)) * win(i);

            // 1. FFT of raw signal
            fft1((Ipp32fc*)&res(0, ch), &fringe1(0, ch));
            ippsPowerSpectr_32fc((const Ipp32fc*)&res(0, ch), (Ipp32f*)&res1(0, ch), fft_size.width);
            ippsLog10_32f_A11((const Ipp32f*)&res1(0, ch), (Ipp32f*)&res2(0, ch), fft_size.width);
            ippsMulC_32f_I(10.0f, (Ipp32f*)&res2(0, ch), fft_size.width);

            char title[50];
            sprintf(title, "FFT of d%d", ch + 1);
            drawGraph(&res2(0, ch), title);

            // 2. Maximum detection & Non-peak signal elimination
            int start1 = 0, end1 = 0;
			waitForRange(start1, end1);
			//printf("[%d %d]\n", start1, end1);

            np::Array<float> mask(fft_size.width);
            ippsZero_32f(mask, fft_size.width);
            ippsSet_32f(1.0f, &mask(start1), end1 - start1 + 1);

            ippsMul_32f32fc_I(mask.raw_ptr(), (Ipp32fc*)&res(0, ch), fft_size.width);

#ifdef FREQ_SHIFTING
            // 3. Frequency shifting effect removal
            std::rotate(&res(0, ch), &res(3 * fft_size.width / 4, ch), &res(fft_size.width, ch));
            //ippsSet_32f(0.0f, (Ipp32f*)&res(fft_size.width / 4, ch), fft_size.width); // should be removed?
#endif

            // 4. IFFT of the signal & Phase extraction
            fft2.inverse((Ipp32fc*)&res3(0, ch), (Ipp32fc*)&res(0, ch));

            ippsPhase_32fc((const Ipp32fc*)&res3(0, ch), &phase(0, ch), raw_size.width);
            bf::UnwrapPhase_32f(&phase(0, ch), raw_size.width);
        }

        // 5. calib_index & calib_weight generation
        np::Array<float> index(raw_size.width); // 0 ~ nScans
        np::Array<float> lin_phase(raw2_size.width);
        np::Array<float> new_index(raw2_size.width);

        ippsSub_32f_I(&phase(0, 0), &phase(0, 1), raw_size.width); // phase(:,1) - phase(:,0)
        ippsAbs_32f_I(&phase(0, 1), raw_size.width); // absolute value

        bf::LineSpace_32f(0.0f, (float)(raw_size.width - 1), raw_size.width, index.raw_ptr());
        bf::LineSpace_32f(phase(0, 1), phase(raw_size.width - 1, 1), raw2_size.width, lin_phase.raw_ptr());

		//FILE* pFile;
		//fopen_s(&pFile, "test.dat", "wb");
		//fwrite(&phase(0, 1), sizeof(float), raw_size.width, pFile);
		//fwrite(index.raw_ptr(), sizeof(float), raw_size.width, pFile);
		//fwrite(lin_phase.raw_ptr(), sizeof(float), raw_size.width, pFile);
		//fclose(pFile);

        bf::Interpolation_32f(&phase(0, 1), index.raw_ptr(), lin_phase.raw_ptr(), raw_size.width, raw2_size.width, new_index.raw_ptr());

        float temp;
        for (int i = 0; i < raw2_size.width; i++)
        {
            temp = floor(new_index(i));
            calib_weight(i) = 1.0f - (new_index(i) - temp);
            calib_index(i) = temp;
        }

        // 6. Dispersion compensation
        np::Array<float> index1(raw2_size.width); // 0 ~ nScans/2
        np::Array<float> lin_phase1(raw2_size.width);
        np::Array<float> lin_phase2(raw2_size.width);
        bf::LineSpace_32f(0.0f, (float)(raw2_size.width - 1), raw2_size.width, index1.raw_ptr());

        Ipp32f offset, slope;
        bf::Interpolation_32f(index.raw_ptr(), &phase(0, 0), new_index.raw_ptr(), raw_size.width, raw2_size.width, lin_phase1.raw_ptr()); // phase with disp : lin_phase1
        bf::LinearRegression_32f(index1.raw_ptr(), lin_phase1.raw_ptr(), raw2_size.width, offset, slope); // poly fit
        ippsVectorSlope_32f(lin_phase2.raw_ptr(), raw2_size.width, offset, slope); // poly val (phase without disp : lin_phase2)

        float* temp_disp = new float[raw2_size.width];
        float* temp_filt  = new float[raw2_size.width];
		for (int i = 0; i < raw2_size.width; i++)
			temp_disp[i] = lin_phase2(i) - lin_phase1(i);     

		bf::movingAverage_32f(temp_disp, temp_filt, raw2_size.width, 15);
        //bf::polynomialFitting_32f(temp_disp, temp_fit, raw2_size.width, 4); // 4th polynomial fitting

        for (int i = 0; i < raw2_size.width; i++)
            dispersion(i) = { cosf(temp_filt[i]), sinf(temp_filt[i]) };
		
        changeDiscomValue(discom_val);

        delete[] temp_disp;
        delete[] temp_filt;

        saveCalibration();
		endCalibration();

        printf("Successfully calibrated!\n");
    });

    calib.detach();
}
#endif

void OCTProcess::removeCalibration()
{
	for (int i = 0; i < raw_size.width; i++)
	{
		bg0(i) = (float)(POWER_2(15));
		bg(i) = (float)(POWER_2(15));
	}

#ifndef K_CLOCKING
	for (int i = 0; i < raw2_size.width; i++)
	{
		calib_index(i) = (float)i * 2.0f;
		calib_weight(i) = 1;
		dispersion(i) = { 1, 0 };
		discom(i) = { 1, 0 };
		dispersion1(i) = { 1, 0 };
	}
#else
	for (int i = 0; i < raw_size.width; i++)
	{
		dispersion1(i) = { 1, 0 };
	}
#endif
}


void OCTProcess::changeDiscomValue(int discom_val)
{
	double temp;
#ifndef K_CLOCKING
	for (int i = 0; i < raw2_size.width; i++)
	{
		temp = (((double)i - (double)raw2_size.width / 2.0f) / (double)raw2_size.width); // *((i - raw2_size.width / 2) / raw2_size.width);
		discom(i) = { (float)cos((double)discom_val*temp*temp), (float)sin((double)discom_val*temp*temp) };
	}
	ippsMul_32fc((Ipp32fc*)dispersion.raw_ptr(), (Ipp32fc*)discom.raw_ptr(), (Ipp32fc*)dispersion1.raw_ptr(), raw2_size.width);
#else
	for (int i = 0; i < raw_size.width; i++)
	{
		temp = (((double)i - (double)raw_size.width / 2.0f) / (double)raw_size.width); // *((i - raw2_size.width / 2) / raw2_size.width);
		dispersion1(i) = { (float)cos((double)discom_val*temp*temp), (float)sin((double)discom_val*temp*temp) };
	}	
#endif
}


#ifndef K_CLOCKING
//void OCTProcess::saveCalibration(const char* calibpath)
//{
//    size_t sizeWrote, sizeTotalWrote = 0;
//
//    // create file (calibration)
//    FILE* pCalibFile = nullptr;
//
//    pCalibFile = fopen(calibpath, "wb");
//    if (pCalibFile != nullptr)
//    {
//        // calib_index
//        sizeWrote = fwrite(calib_index.raw_ptr(), sizeof(float), raw2_size.width, pCalibFile);
//        sizeTotalWrote += sizeWrote;
//
//        // calib_weight
//        sizeWrote = fwrite(calib_weight.raw_ptr(), sizeof(float), raw2_size.width, pCalibFile);
//        sizeTotalWrote += sizeWrote;
//
//        // dispersion compensation real
//        Ipp32f* real = ippsMalloc_32f(raw2_size.width);
//        ippsReal_32fc((const Ipp32fc*)dispersion.raw_ptr(), real, raw2_size.width);
//        sizeWrote = fwrite(real, sizeof(float), raw2_size.width, pCalibFile);
//        ippsFree(real);
//        sizeTotalWrote += sizeWrote;
//
//        // dispersion compensation imag
//        Ipp32f* imag = ippsMalloc_32f(raw2_size.width);
//        ippsImag_32fc((const Ipp32fc*)dispersion.raw_ptr(), imag, raw2_size.width);
//        sizeWrote = fwrite(imag, sizeof(float), raw2_size.width, pCalibFile);
//        ippsFree(imag);
//        sizeTotalWrote += sizeWrote;
//
//        fclose(pCalibFile);
//    }
//    else
//        printf("Calibration data cannot be saved.\n");
//}

void OCTProcess::saveCalibration(QString calibpath)
{
	qint64 sizeWrote, sizeTotalWrote = 0;
	
	// create file (calibration)
	QFile calibFile(calibpath);
	if (false != calibFile.open(QFile::WriteOnly))
	{
		// calib_index
		sizeWrote = calibFile.write(reinterpret_cast<char*>(calib_index.raw_ptr()), sizeof(float) * raw2_size.width);
		sizeTotalWrote += sizeWrote;

		// calib_weight
		sizeWrote = calibFile.write(reinterpret_cast<char*>(calib_weight.raw_ptr()), sizeof(float) * raw2_size.width);
		sizeTotalWrote += sizeWrote;

		// dispersion compensation real
		Ipp32f* real = ippsMalloc_32f(raw2_size.width);
		ippsReal_32fc((const Ipp32fc*)dispersion.raw_ptr(), real, raw2_size.width);
		sizeWrote = calibFile.write(reinterpret_cast<char*>(real), sizeof(float) * raw2_size.width);
		ippsFree(real);
		sizeTotalWrote += sizeWrote;

		// dispersion compensation imag
		Ipp32f* imag = ippsMalloc_32f(raw2_size.width);
		ippsImag_32fc((const Ipp32fc*)dispersion.raw_ptr(), imag, raw2_size.width);
		sizeWrote = calibFile.write(reinterpret_cast<char*>(imag), sizeof(float) * raw2_size.width);
		ippsFree(imag);
		sizeTotalWrote += sizeWrote;

		calibFile.close();
	}
	else
		printf("Calibration data cannot be saved.\n");
}


//void OCTProcess::loadCalibration(int ch, const char* calibpath, const char* bgpath)
//{
//	size_t sizeRead, sizeTotalRead = 0;
//	
//    printf("\n//// Load OCT Calibration Data ////\n");
//    
//	// create file (background)
//    FILE* pBgFile = nullptr;
//    Uint16Array2 frame;
//    
//    #if defined(BOP_SYSTEM)
//        frame = Uint16Array2(raw_size.width, raw_size.height);
//    #elif defined(KAIST_SYSTEM)
//        frame = Uint16Array2(2 * raw_size.width, raw_size.height);
//    #endif
//    
//    Uint16Array2 frame1(raw_size.width, raw_size.height);
//    Uint16Array2 frame2(raw_size.width, raw_size.height); 
//	
//    pBgFile = fopen(bgpath, "rb");
//	if (pBgFile != nullptr)
//	{
//		// background
//		if (ch == CH_1)
//			sizeRead = fread(frame.raw_ptr(), sizeof(uint16_t), frame.length(), pBgFile);
//		else if (ch == CH_2)
//		{
//			fseek(pBgFile, sizeof(uint16_t) * frame.length(), SEEK_SET);
//			sizeRead = fread(frame.raw_ptr(), sizeof(uint16_t), frame.length(), pBgFile);
//		}
//
//        #if defined(BOP_SYSTEM)
//            memcpy(frame1.raw_ptr(), frame.raw_ptr(), frame1.length());
//        #elif defined(KAIST_SYSTEM)
//            ippsCplxToReal_16sc((Ipp16sc*)frame.raw_ptr(), (Ipp16s*)frame1.raw_ptr(), (Ipp16s*)frame2.raw_ptr(), frame1.length());
//        #endif
//        printf("Background data is successfully loaded.[%zu]\n", sizeRead);
//        
//		if (sizeRead)
//		{
//			if (frame1.size(1) > 50)
//			{
//				int N = 50;
//				for (int i = 0; i < frame1.size(0); i++)
//				{
//					bg(i) = 0;
//					for (int j = 0; j < N; j++)
//						bg(i) += (float)frame1(i, j);
//					bg(i) /= N;
//				}
//			}
//			else
//			{
//				printf("nAlines should be larger than 50.\n");
//				return; 
//			}
//		}
//        
//        fclose(pBgFile);
//	}
//    else
//        printf("Background data cannot be loaded.\n");
//    
//	// create file (calibration)
//    FILE* pCalibFile = nullptr;   
//    
//    pCalibFile = fopen(calibpath, "rb");
//	if (pCalibFile != nullptr)
//	{
//		// calib_index
//        #if defined(BOP_SYSTEM)
//            sizeRead = fread(calib_index.raw_ptr(), sizeof(float), raw2_size.width, pCalibFile);
//        #elif defined(KAIST_SYSTEM)
//            Ipp32s* ind = ippsMalloc_32s(raw2_size.width);
//            sizeRead = fread(ind, sizeof(int), raw2_size.width, pCalibFile);
//            ippsConvert_32s32f(ind, calib_index.raw_ptr(), raw2_size.width);
//            ippsFree(ind);
//            ippsFlip_32f_I(calib_index.raw_ptr(), raw2_size.width);
//        #endif
//        sizeTotalRead += sizeRead;
//        printf("Calibration index is successfully loaded.[%zu]\n", sizeRead);
//
//		// calib_weight
//        sizeRead = fread(calib_weight.raw_ptr(), sizeof(float), raw2_size.width, pCalibFile);
//        #if defined(KAIST_SYSTEM)
//            ippsFlip_32f_I(calib_weight.raw_ptr(), raw2_size.width);
//        #endif
//        sizeTotalRead += sizeRead;
//        printf("Calibration weight is successfully loaded.[%zu]\n", sizeRead);
//
//		// dispersion compensation real
//		Ipp32f* real = ippsMalloc_32f(raw2_size.width);
//        sizeRead = fread(real, sizeof(float), raw2_size.width, pCalibFile);
//        #if defined(KAIST_SYSTEM)
//            ippsFlip_32f_I(real, raw2_size.width);
//        #endif
//        sizeTotalRead += sizeRead;
//        printf("Dispersion data (real) is successfully loaded.[%zu]\n", sizeRead);
//		
//		// dispersion compensation imag
//		Ipp32f* imag = ippsMalloc_32f(raw2_size.width);
//        sizeRead = fread(imag, sizeof(float), raw2_size.width, pCalibFile);
//        #if defined(KAIST_SYSTEM)
//            ippsFlip_32f_I(imag, raw2_size.width);
//        #endif
//        sizeTotalRead += sizeRead;
//        printf("Dispersion data (imag) is successfully loaded.[%zu]\n", sizeRead);
//        
//		// dispersion compensation
//		ippsRealToCplx_32f(real, imag, (Ipp32fc*)dispersion.raw_ptr(), raw2_size.width);			
//		ippsFree(real); ippsFree(imag);	
//        changeDiscomValue(0);
//
//		fclose(pCalibFile);
//    }
//    else
//        printf("Calibration data cannot be loaded.\n");
//    
//    printf("\n");
//}

void OCTProcess::loadCalibration(int ch, QString calibpath, QString bgpath, bool erasmus)
{
	qint64 sizeRead, sizeTotalRead = 0;

	printf("\n//// Load OCT Calibration Data ////\n");
	if (erasmus)
		printf("this data is acquired by erasmus.\n");

	// create file (background)
	QFile bgFile(bgpath);

	Uint16Array2 frame;

	if (!erasmus)
		frame = Uint16Array2(raw_size.width, raw_size.height);	
	else	
		frame = Uint16Array2(2 * raw_size.width, raw_size.height);

	Uint16Array2 frame1(raw_size.width, raw_size.height);
	Uint16Array2 frame2(raw_size.width, raw_size.height);

	if (false != bgFile.open(QIODevice::ReadOnly))
	{
		// background
		if (ch == CH_1)
			sizeRead = bgFile.read(reinterpret_cast<char*>(frame.raw_ptr()), sizeof(uint16_t) * frame.length());
		else if (ch == CH_2)
		{
			bgFile.reset();
			bgFile.seek(sizeof(uint16_t) * frame.length());
			sizeRead = bgFile.read(reinterpret_cast<char*>(frame.raw_ptr()), sizeof(uint16_t) * frame.length());
		}

		if (!erasmus)
			memcpy(frame1.raw_ptr(), frame.raw_ptr(), frame1.length());
		else
			ippsCplxToReal_16sc((Ipp16sc*)frame.raw_ptr(), (Ipp16s*)frame1.raw_ptr(), (Ipp16s*)frame2.raw_ptr(), frame1.length());

		printf("Background data is successfully loaded.[%zu]\n", sizeRead);

		if (sizeRead)
		{
			if (frame1.size(1) > 50)
			{
				int N = 50;
				for (int i = 0; i < frame1.size(0); i++)
				{
					bg(i) = 0;
					for (int j = 0; j < N; j++)
						bg(i) += (float)frame1(i, j);
					bg(i) /= N;
				}
			}
			else
			{
				printf("nAlines should be larger than 50.\n");
				return;
			}
		}

		bgFile.close();
	}
	else
		printf("Background data cannot be loaded.\n");

	// create file (calibration)
	QFile calibFile(calibpath);
	if (false != calibFile.open(QFile::ReadOnly))
	{
		// calib_index
		if (!erasmus)
			sizeRead = calibFile.read(reinterpret_cast<char*>(calib_index.raw_ptr()), sizeof(float) * raw2_size.width);
		else
		{
			Ipp32s* ind = ippsMalloc_32s(raw2_size.width);
			sizeRead = calibFile.read(reinterpret_cast<char*>(ind), sizeof(int) * raw2_size.width);
			ippsConvert_32s32f(ind, calib_index.raw_ptr(), raw2_size.width);
			ippsFree(ind);
			ippsFlip_32f_I(calib_index.raw_ptr(), raw2_size.width);
		}
		sizeTotalRead += sizeRead;
		printf("Calibration index is successfully loaded.[%zu]\n", sizeRead);

		// calib_weight
		sizeRead = calibFile.read(reinterpret_cast<char*>(calib_weight.raw_ptr()), sizeof(float) * raw2_size.width);
		if (erasmus)
			ippsFlip_32f_I(calib_weight.raw_ptr(), raw2_size.width);
		sizeTotalRead += sizeRead;
		printf("Calibration weight is successfully loaded.[%zu]\n", sizeRead);

		// dispersion compensation real
		Ipp32f* real = ippsMalloc_32f(raw2_size.width);
		sizeRead = calibFile.read(reinterpret_cast<char*>(real), sizeof(float) * raw2_size.width);
		if (erasmus)
			ippsFlip_32f_I(real, raw2_size.width);
		sizeTotalRead += sizeRead;
		printf("Dispersion data (real) is successfully loaded.[%zu]\n", sizeRead);

		// dispersion compensation imag
		Ipp32f* imag = ippsMalloc_32f(raw2_size.width);
		sizeRead = calibFile.read(reinterpret_cast<char*>(imag), sizeof(float) * raw2_size.width);
		if (erasmus)
			ippsFlip_32f_I(imag, raw2_size.width);
		sizeTotalRead += sizeRead;
		printf("Dispersion data (imag) is successfully loaded.[%zu]\n", sizeRead);

		// dispersion compensation
		ippsRealToCplx_32f(real, imag, (Ipp32fc*)dispersion.raw_ptr(), raw2_size.width);
		ippsFree(real); ippsFree(imag);
		changeDiscomValue(0);

		calibFile.close();
	}
	else
		printf("Calibration data cannot be loaded.\n");

	printf("\n");
}
#endif