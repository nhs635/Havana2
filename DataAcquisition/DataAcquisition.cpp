
#include "DataAcquisition.h"


DataAcquisition::DataAcquisition(Configuration* pConfig)
#if PX14_ENABLE || ALAZAR_ENABLE
    : pDaq(nullptr)
#endif
{
	m_pConfig = pConfig;
#if PX14_ENABLE
    pDaq = new SignatecDAQ;
#endif
#if ALAZAR_ENABLE
    pDaq = new AlazarDAQ;
#endif
#if PX14_ENABLE || ALAZAR_ENABLE
	pDaq->DidStopData += [&]() { pDaq->_running = false; };
#endif
}

DataAcquisition::~DataAcquisition()
{
#if PX14_ENABLE || ALAZAR_ENABLE
    if (pDaq) delete pDaq;
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
///	pDaq->AcqRate = ADC_RATE;
    pDaq->nChannels = m_pConfig->nChannels;
    pDaq->nScans = m_pConfig->nScans;
    pDaq->nAlines = m_pConfig->nAlines;

    pDaq->VoltRange1 = m_pConfig->ch1VoltageRange + 1;
    pDaq->VoltRange2 = m_pConfig->ch2VoltageRange + 1;
    pDaq->TriggerDelay = 120; // m_pConfig->triggerDelay;  ///// need to be revised!!!! 2020 02 20

    // Initialization for DAQ
    if (!(pDaq->set_init()))
    {
        StopAcquisition();
        return false;
    }
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
