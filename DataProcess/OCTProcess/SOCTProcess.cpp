
#include "SOCTProcess.h"
#include <Common/basic_functions.h>

#include <fstream>


SOCTProcess::SOCTProcess(int nScans, int nAlines, int nwin, int noverlap) :
	OCTProcess(nScans, nAlines),
#ifndef K_CLOCKING
	nk((int)((raw2_size.width - noverlap) / (nwin - noverlap))),
#else
	nk((int)((raw_size.width - noverlap) / (nwin - noverlap))),
#endif
	nincrement(nwin - noverlap)
{
#ifndef K_CLOCKING
	win_repmat = FloatArray2(fft2_size.width, nk);

	// SOCT window
	memset(win_repmat, 0, sizeof(float) * win_repmat.length());
	for (int i = 0; i < nwin; i++)
	{
		float win = (float)(1 - cos(IPP_2PI * i / (nwin - 1))) / 2; // Hann Window
		for (int j = 0; j < nk; j++)
			win_repmat(i + j * nincrement, j) = win;
	}
#else
	win_repmat = FloatArray2(fft_size.width, nk);

	// SOCT window
	memset(win_repmat, 0, sizeof(float) * win_repmat.length());
	for (int i = 0; i < nwin; i++)
	{
		float win = (float)(1 - cos(IPP_2PI * i / (nwin - 1))) / 2; // Hann Window
		for (int j = 0; j < nk; j++)
			win_repmat(i + j * nincrement, j) = win;
	}
#endif
}


SOCTProcess::~SOCTProcess()
{
}


/* SOCT Image */
void SOCTProcess::operator()(std::vector<np::FloatArray2> spectra, uint16_t* fringe)
{
	tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)raw_size.height),
		[&](const tbb::blocked_range<size_t>& r) {

		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			int r1 = raw_size.width * (int)i;
			int f1 = fft_size.width * (int)i;
			int f2 = fft2_size.width * (int)i;

			// 1. Single Precision Conversion & Zero Padding (3 msec)
			ippsConvert_16u32f(fringe + r1, signal.raw_ptr() + f1, raw_size.width);

			// 2. BG Subtraction
			ippsSub_32f_I(bg, signal.raw_ptr() + f1, raw_size.width);

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

			// Spectroscopic OCT calculation
			ComplexFloatArray2 complex_resamp0(fft2_size.width, nk);
			ComplexFloatArray2 stft_complex0(fft2_size.width, nk);

			tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)nk),
				[&, i, f2, complex_resamp0, stft_complex0](const tbb::blocked_range<size_t>& r) {

				for (size_t j = r.begin(); j != r.end(); ++j)
				{
					int f3 = fft2_size.width * (int)j;

					// Copy data										
					memcpy((Ipp32fc*)complex_resamp0.raw_ptr() + f3, (Ipp32fc*)(complex_resamp.raw_ptr() + f2), 2 * sizeof(float) * fft2_size.width);

					// Hanning windowing
					ippsMul_32f32fc_I(win_repmat + f3, (Ipp32fc*)complex_resamp0.raw_ptr() + f3, fft2_size.width);

					// 7. Fourier transform (9 msec)					
					fft3.forward((Ipp32fc*)stft_complex0.raw_ptr() + f3, (const Ipp32fc*)complex_resamp0.raw_ptr() + f3);

					// 8. Power spetrum (5 msec)					
					ippsPowerSpectr_32fc((Ipp32fc*)stft_complex0.raw_ptr() + f3, spectra.at((int)i) + f3, fft2_size.width);

#ifdef FREQ_SHIFTING
					// 9. Circshift by -nfft/2 (1 msec)
					std::rotate(spectra.at((int)i) + f3, spectra.at((int)i) + f3 + fft2_size.width / 2, spectra.at((int)i) + f3 + fft2_size.width);
#endif
				}
			});
#else
			// 3. Dispersion compensation
			ippsMul_32f32fc((const Ipp32f*)signal.raw_ptr() + f1, (const Ipp32fc*)dispersion1.raw_ptr(), (Ipp32fc*)complex_signal.raw_ptr() + f1, raw_size.width);

			// Spectroscopic OCT calculation
			ComplexFloatArray2 complex_signal0(fft_size.width, nk);
			ComplexFloatArray2 stft_complex0(fft_size.width, nk);

			tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)nk),
				[&, i, f2, complex_signal0, stft_complex0](const tbb::blocked_range<size_t>& r) {

				for (size_t j = r.begin(); j != r.end(); ++j)
				{
					int f3 = fft_size.width * (int)j;
					int f4 = fft2_size.width * (int)j;

					// Copy data										
					memcpy((Ipp32fc*)complex_signal0.raw_ptr() + f3, (Ipp32fc*)(complex_signal.raw_ptr() + f1), 2 * sizeof(float) * fft_size.width);

					// Hanning windowing
					ippsMul_32f32fc_I(win_repmat + f3, (Ipp32fc*)complex_signal0.raw_ptr() + f3, fft_size.width);

					// 7. Fourier transform (9 msec)					
					fft3.forward((Ipp32fc*)stft_complex0.raw_ptr() + f3, (const Ipp32fc*)complex_signal0.raw_ptr() + f3);

					// 8. Power spetrum (5 msec)					
					ippsPowerSpectr_32fc((Ipp32fc*)stft_complex0.raw_ptr() + f3, spectra.at((int)i) + f4, fft2_size.width);
				}
			});
#endif

			DidProcess();
		}
	});
}

void SOCTProcess::db_scaling(std::vector<np::FloatArray2> spectra)
{
	tbb::parallel_for(tbb::blocked_range<size_t>(0, spectra.size()),
		[&](const tbb::blocked_range<size_t>& r) {

		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			FloatArray2 stft_linear(fft2_size.width, nk);

			tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)nk),
				[&, i, stft_linear](const tbb::blocked_range<size_t>& r) {

				for (size_t j = r.begin(); j != r.end(); ++j)
				{
					int f3 = fft2_size.width * (int)j;

					memcpy((float*)stft_linear.raw_ptr() + f3, (const float*)spectra.at((int)i) + f3, sizeof(float) * fft2_size.width);
					ippsLog10_32f_A11(stft_linear.raw_ptr() + f3, spectra.at((int)i) + f3, fft2_size.width);
					ippsMulC_32f_I(10.0f, spectra.at((int)i) + f3, fft2_size.width);
				}
			});
		}
	});

}

void SOCTProcess::spectrum_averaging(std::vector<np::FloatArray2>& spectra, np::Uint16Array& contour,
	int in_plane, int out_of_plane, int roi_depth)
{
	// Copy original spectra to temp memories, set original spectra as zero mat
	std::vector<np::FloatArray2> spectra_temp;
	for (int i = 0; i < spectra.size(); i++)
	{
		FloatArray2 temp(spectra.at(0).size(0), spectra.at(0).size(1));
		memcpy(temp, spectra.at(i), sizeof(float) * temp.length());
		spectra_temp.push_back(temp);
		memset(spectra.at(i), 0, sizeof(float) * spectra.at(i).length());
	}
		
	int nalines = raw_size.height / (2 * out_of_plane + 1);

	tbb::parallel_for(tbb::blocked_range<size_t>(out_of_plane * nalines, (out_of_plane + 1) * nalines),
		[&](const tbb::blocked_range<size_t>& r) {

		for (size_t i = r.begin(); i != r.end(); ++i)	
		{
			// Check surface 
#ifndef OCT_VERTICAL_MIRRORING
			int ref = (int)contour.at((int)i);
			int surface = (roi_depth == 0) ? 0 : ref - SURFACE_OFFSET;
#else
			int ref = fft2_size.width - (int)contour.at((int)i);
			int surface = (roi_depth == 0) ? 0 : ref - roi_depth + SURFACE_OFFSET;
#endif
			if (surface < 0) surface = 0;
			int n_avg = 0;

			// in case of non-circular range
			int in_start = ((i - in_plane) >= out_of_plane * nalines) ? (i - in_plane) : (out_of_plane * nalines);
			int in_end = ((i + in_plane) < (out_of_plane + 1) * nalines) ? (i + in_plane) : ((out_of_plane + 1) * nalines - 1);
			in_end++;

			// in plane averaging
			tbb::parallel_for(tbb::blocked_range<size_t>(in_start, in_end),
				[&](const tbb::blocked_range<size_t>& t) {

				for (size_t j = t.begin(); j != t.end(); ++j)
				{					
					// out of plane averaging
					//tbb::parallel_for(tbb::blocked_range<int>(-out_of_plane, out_of_plane + 1),
					//	[&](const tbb::blocked_range<int>& u) {

						//for (int o = u.begin(); o != u.end(); ++o)
					for (int o = -out_of_plane; o <= out_of_plane; o++)
						{
							int j0 = j + o * nalines;

#ifndef OCT_VERTICAL_MIRRORING
							int cur = (int)contour.at((int)j0);
#else
							int cur = fft2_size.width - (int)contour.at(j0);
#endif						
							FloatArray2 temp(spectra_temp.at(j0).size(0), spectra_temp.at(j0).size(1));
							memset(temp, 0, sizeof(float) * temp.length());

							if (ref > cur)
								ippiCopy_32f_C1R(&spectra_temp.at(j0)(surface, 0), sizeof(float) * temp.size(0),
									&temp(surface + (ref - cur), 0), sizeof(float) * temp.size(0),
									{ (surface == 0 ? temp.size(0) : roi_depth) - (ref - cur), temp.size(1) });
							else
								ippiCopy_32f_C1R(&spectra_temp.at(j0)(surface + (cur - ref), 0), sizeof(float) * temp.size(0),
									&temp(surface, 0), sizeof(float) * temp.size(0),
									{ (surface == 0 ? temp.size(0) : roi_depth) - (cur - ref), temp.size(1) });

							ippsAdd_32f_I(temp, spectra.at((int)i), spectra.at((int)i).length());

							n_avg++;
						}
					//});
				}
			});

			ippsDivC_32f_I(n_avg, spectra.at((int)i), spectra.at((int)i).length());

			DidProcess();
		}
	});
	
	std::vector<np::FloatArray2> clear_vector;
	clear_vector.swap(spectra_temp);

	spectra.erase(spectra.end() - out_of_plane * nalines, spectra.end());
	spectra.erase(spectra.begin(), spectra.begin() + out_of_plane * nalines);	
}