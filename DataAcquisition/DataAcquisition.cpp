
#include "DataAcquisition.h"

#include <AlazarApi.h>


DataAcquisition::DataAcquisition(Configuration* pConfig)
#if PX14_ENABLE || ALAZAR_ENABLE
    : pDaq(nullptr)
#ifdef ALAZAR_NIRF_ACQUISITION
	, pDaqNirf(nullptr)
#endif
#endif
{
	m_pConfig = pConfig;
#if PX14_ENABLE
    pDaq = new SignatecDAQ;
#endif
#if ALAZAR_ENABLE
    pDaq = new AlazarDAQ;

#ifdef ALAZAR_NIRF_ACQUISITION
	pDaqNirf = new AlazarDAQ;
#endif

#endif
#if PX14_ENABLE || ALAZAR_ENABLE
	pDaq->DidStopData += [&]() { pDaq->_running = false; };

#ifdef ALAZAR_NIRF_ACQUISITION
	pDaqNirf->DidStopData += [&]() { pDaqNirf->_running = false; };
#endif
#endif
}

DataAcquisition::~DataAcquisition()
{
#if PX14_ENABLE || ALAZAR_ENABLE
    if (pDaq) delete pDaq;
#ifdef ALAZAR_NIRF_ACQUISITION
	if (pDaqNirf) delete pDaqNirf;
#endif
#endif
}


bool DataAcquisition::InitializeAcquistion()
{
#if PX14_ENABLE
    // Parameter settings for DAQ
	pDaq->AcqRate = ADC_RATE;
    pDaq->nChannels = m_pConfig->nChannels;
    pDaq->nScans = m_pConfig->nScans;
    pDaq->nAlines = m_pConfig->nAlines;
	pDaq->BootTimeBufIdx = m_pConfig->bootTimeBufferIndex;

    pDaq->VoltRange1 = m_pConfig->ch1VoltageRange + 1;
    pDaq->VoltRange2 = m_pConfig->ch2VoltageRange + 1;
    pDaq->PreTrigger = m_pConfig->preTrigSamps;

    // Initialization for DAQ
    if (!(pDaq->set_init()))
    {
        StopAcquisition();
        return false;
    }
	return true;    
#elif ALAZAR_ENABLE
    // Parameter settings for DAQ
	pDaq->SystemId = 1;
	pDaq->AcqRate = SAMPLE_RATE_500MSPS;
    pDaq->nChannels = m_pConfig->nChannels;
    pDaq->nScans = m_pConfig->nScans;
    pDaq->nAlines = m_pConfig->nAlines;

    pDaq->VoltRange1 = m_pConfig->ch1VoltageRange + 1;
    pDaq->VoltRange2 = m_pConfig->ch2VoltageRange + 1;
    pDaq->TriggerDelay = m_pConfig->triggerDelay;  ///// need to be revised!!!! 2020 02 20

	pDaq->UseExternalClock = USE_EXTERNAL_K_CLOCK;

    // Initialization for DAQ
    if (!(pDaq->set_init()))
    {
        StopAcquisition();
        return false;
    }

#ifdef ALAZAR_NIRF_ACQUISITION
	// Parameter settings for DAQ (NIRF)
	pDaqNirf->SystemId = 2;
	pDaqNirf->AcqRate = SAMPLE_RATE_50MSPS;
#ifndef TWO_CHANNEL_NIRF
	pDaqNirf->nChannels = 1;
#else
	pDaqNirf->nChannels = 2;
#endif
	pDaqNirf->nScans = NIRF_SCANS;
	pDaqNirf->nAlines = m_pConfig->nAlines;

	pDaqNirf->VoltRange1 = INPUT_RANGE_PM_4_V;
	pDaqNirf->VoltRange2 = INPUT_RANGE_PM_4_V;

	// Initialization for DAQ (NIRF)
	if (!(pDaqNirf->set_init()))
	{
		StopAcquisition();
		return false;
	}
#endif
    return true;
#else
    return false;
#endif
}

bool DataAcquisition::StartAcquisition()
{
#if PX14_ENABLE
    // Parameter settings for DAQ
	pDaq->VoltRange1 = m_pConfig->ch1VoltageRange + 1;
	pDaq->VoltRange2 = m_pConfig->ch2VoltageRange + 1;
	pDaq->PreTrigger = m_pConfig->preTrigSamps;

    // Start acquisition
    if (!(pDaq->startAcquisition()))
    {
        StopAcquisition();
        return false;
    }
    return true;
#elif ALAZAR_ENABLE
    // Parameter settings for DAQ
    pDaq->VoltRange1 = m_pConfig->ch1VoltageRange + 1;
    pDaq->VoltRange2 = m_pConfig->ch2VoltageRange + 1;
    pDaq->TriggerDelay = m_pConfig->triggerDelay;

    // Start acquisition
    if (!(pDaq->startAcquisition()))
    {
        StopAcquisition();
        return false;
    }

#ifdef ALAZAR_NIRF_ACQUISITION
	if (!(pDaqNirf->startAcquisition()))
	{
		StopAcquisition();
		return false;
	}
#endif

    return true;
#else
    return false;
#endif
}

void DataAcquisition::StopAcquisition()
{
#if PX14_ENABLE || ALAZAR_ENABLE
    // Stop thread
    pDaq->stopAcquisition();
#ifdef ALAZAR_NIRF_ACQUISITION
	pDaqNirf->stopAcquisition();
#endif
#endif
}

#if PX14_ENABLE
void DataAcquisition::GetBootTimeBufCfg(int idx, int& buffer_size)
{
	buffer_size = pDaq->getBootTimeBuffer(idx);
}

void DataAcquisition::SetBootTimeBufCfg(int idx, int buffer_size)
{
	pDaq->setBootTimeBuffer(idx, buffer_size);
}
#endif
