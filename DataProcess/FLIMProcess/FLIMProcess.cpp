
#include "FLIMProcess.h"

#ifdef OCT_FLIM

FLIMProcess::FLIMProcess()
{
}

FLIMProcess::~FLIMProcess()
{
}


void FLIMProcess::operator() (FloatArray2& intensity, FloatArray2& mean_delay, FloatArray2& lifetime, Uint16Array2& pulse)
{
	// 1. Crop and resize pulse data
	_resize(pulse, _params);
	
	// 2. Get intensity
	_intensity(_resize, intensity);

	// 3. Get lifetime
	_lifetime(_resize, _params, intensity, mean_delay, lifetime);
}



#ifdef OCT_FLIM
void FLIMProcess::setParameters(Configuration* pConfig)
{
#elif defined (STANDALONE_OCT)
void FLIMProcess::setParameters(Configuration*)
{
#endif
#ifdef OCT_FLIM
	_params.bg = pConfig->flimBg;
	_params.act_ch = pConfig->flimCh;
	_params.pre_trig = pConfig->preTrigSamps;

#if PX14_ENABLE
	_params.samp_intv = 1000.0f / (float)ADC_RATE;
#else
	_params.samp_intv = 1000.0f / 340.0f;
#endif
	_params.width_factor = 2.0f;

	for (int i = 0; i < 4; i++)
	{
		_params.ch_start_ind[i] = pConfig->flimChStartInd[i];
		if (i != 0)
			_params.delay_offset[i - 1] = pConfig->flimDelayOffset[i - 1];
	}
	_params.ch_start_ind[4] = _params.ch_start_ind[3] + FLIM_CH_START_5;
#endif
}

//void FLIMProcess::saveMaskData(const char* maskpath)
//{
//	size_t sizeRead;
//
//	// create file (flim mask)
//	FILE* pMaskFile = nullptr;
//	pMaskFile = fopen(maskpath, "wb");
//	if (pMaskFile != nullptr)
//	{
//		sizeRead = fwrite(_resize.pMask, sizeof(float), _resize.nx, pMaskFile);
//		fclose(pMaskFile);
//	}
//}

void FLIMProcess::saveMaskData(QString maskpath)
{
	qint64 sizeRead;

	// create file (flim mask)
	QFile maskFile(maskpath);
	if (false != maskFile.open(QIODevice::WriteOnly))
	{
		sizeRead = maskFile.write(reinterpret_cast<char*>(_resize.pMask), sizeof(float) * _resize.nx);
		maskFile.close();
	}
}

//void FLIMProcess::loadMaskData(const char* maskpath)
//{
//	size_t sizeRead;
//
//    // create file (flim mask)
//    FILE* pMaskFile = nullptr;  
//    pMaskFile = fopen(maskpath, "rb");
//    if (pMaskFile != nullptr)
//    {
//        sizeRead = fread(_resize.pMask, sizeof(float), _resize.nx, pMaskFile);
//        fclose(pMaskFile);
//        
//        int start_count = 0, end_count = 0;
//        for (int i = 0; i < _resize.nx - 1; i++)
//        {
//            if (_resize.pMask[i + 1] - _resize.pMask[i] == -1)
//            {
//                start_count++;
//                if (start_count < 5)
//                    _resize.start_ind[start_count - 1] = i + 1;
//            }
//            if (_resize.pMask[i + 1] - _resize.pMask[i] == 1)
//            {
//                end_count++;
//                if (end_count < 5)
//                    _resize.end_ind[end_count - 1] = i;
//            }
//        }
//        
//        for (int i = 0; i < 4; i++)
//            printf("mask %d: [%d %d]\n", i + 1, _resize.start_ind[i], _resize.end_ind[i]);
//        
//        if ((start_count == 4) && (end_count == 4))
//            printf("Proper mask is selected!!\n");
//        else
//            printf("Improper mask: please modify the mask!\n");
//    }
//}

void FLIMProcess::loadMaskData(QString maskpath)
{
	qint64 sizeRead;

	// create file (flim mask)
	QFile maskFile(maskpath);
	if (false != maskFile.open(QIODevice::ReadOnly))
	{
		sizeRead = sizeRead = maskFile.read(reinterpret_cast<char*>(_resize.pMask), sizeof(float) * _resize.nx);
		maskFile.close();

		int start_count = 0, end_count = 0;
		for (int i = 0; i < _resize.nx - 1; i++)
		{
			if (_resize.pMask[i + 1] - _resize.pMask[i] == -1)
			{
				start_count++;
				if (start_count < 5)
					_resize.start_ind[start_count - 1] = i + 1;
			}
			if (_resize.pMask[i + 1] - _resize.pMask[i] == 1)
			{
				end_count++;
				if (end_count < 5)
					_resize.end_ind[end_count - 1] = i;
			}
		}

		for (int i = 0; i < 4; i++)
			printf("mask %d: [%d %d]\n", i + 1, _resize.start_ind[i], _resize.end_ind[i]);

		if ((start_count == 4) && (end_count == 4))
			printf("Proper mask is selected!!\n");
		else
			printf("Improper mask: please modify the mask!\n");
	}
}

#endif