#ifndef DATAACQUISITION_H
#define DATAACQUISITION_H

#include <QObject>

#include <Havana2/Configuration.h>

#if PX14_ENABLE
#include <DataAcquisition/SignatecDAQ/SignatecDAQ.h>
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
	void GetBootTimeBufCfg(int idx, int& buffer_size);
	void SetBootTimeBufCfg(int idx, int buffer_size);
	
#if PX14_ENABLE
public:
    void ConnectDaqAcquiredData(const std::function<void(int, const np::Array<uint16_t, 2>&)> &slot) { pDaq->DidAcquireData += slot; }
    void ConnectDaqStopData(const std::function<void(void)> &slot) { pDaq->DidStopData += slot; }
    void ConnectDaqSendStatusMessage(const std::function<void(const char*)> &slot) { pDaq->SendStatusMessage += slot; }
#endif

private:
	Configuration* m_pConfig;
#if PX14_ENABLE
	SignatecDAQ* pDaq;
#endif
};

#endif // DATAACQUISITION_H
