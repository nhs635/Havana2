#ifndef DATAACQUISITION_H
#define DATAACQUISITION_H

#include <QObject>

#include <Havana2/Configuration.h>

#if PX14_ENABLE
#include <DataAcquisition/SignatecDAQ/SignatecDAQ.h>
#endif
#if ALAZAR_ENABLE
#include <DataAcquisition/AlazarDAQ/AlazarDAQ.h>
#endif

#include <Common/array.h>
#include <Common/callback.h>


class DataAcquisition : public QObject
{
	Q_OBJECT

public:
    explicit DataAcquisition(Configuration*);
    virtual ~DataAcquisition();

public:
    bool InitializeAcquistion();
    bool StartAcquisition();
    void StopAcquisition();

public:
#if PX14_ENABLE
	void GetBootTimeBufCfg(int idx, int& buffer_size);
	void SetBootTimeBufCfg(int idx, int buffer_size);
#endif
	
#if PX14_ENABLE || ALAZAR_ENABLE
public:
    void ConnectDaqAcquiredData(const std::function<void(int, const np::Array<uint16_t, 2>&)> &slot) { pDaq->DidAcquireData += slot; }
    void ConnectDaqStopData(const std::function<void(void)> &slot) { pDaq->DidStopData += slot; }
    void ConnectDaqSendStatusMessage(const std::function<void(const char*)> &slot) { pDaq->SendStatusMessage += slot; }

#ifdef ALAZAR_NIRF_ACQUISITION
	void ConnectDaqNirfAcquiredData(const std::function<void(int, const np::Array<uint16_t, 2>&)> &slot) { pDaqNirf->DidAcquireData += slot; }
	void ConnectDaqNirfStopData(const std::function<void(void)> &slot) { pDaqNirf->DidStopData += slot; }
	void ConnectDaqNirfSendStatusMessage(const std::function<void(const char*)> &slot) { pDaqNirf->SendStatusMessage += slot; }
#endif
#endif

private:
	Configuration* m_pConfig;
#if PX14_ENABLE
	SignatecDAQ* pDaq;
#endif
#if ALAZAR_ENABLE
    AlazarDAQ* pDaq;
#ifdef ALAZAR_NIRF_ACQUISITION
	AlazarDAQ* pDaqNirf;
#endif
#endif
};

#endif // DATAACQUISITION_H
