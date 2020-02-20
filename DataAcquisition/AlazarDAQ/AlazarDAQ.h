#ifndef _ALAZAR_DAQ_H_
#define _ALAZAR_DAQ_H_

#include <Havana2/Configuration.h>

#include <Common/array.h>
#include <Common/callback.h>

#include <iostream>
#include <array>
#include <thread>

#define BUFFER_COUNT 2
#define MAX_MSG_LENGTH 2000

typedef enum RETURN_CODE RETURN_CODE;
typedef void * HANDLE;

class AlazarDAQ
{
public:
#if ALAZAR_ENABLE
    explicit AlazarDAQ();
    virtual ~AlazarDAQ();

    // callbacks
    callback2<int, const np::Array<uint16_t, 2> &> DidAcquireData;
    callback<void> DidStopData;
    callback<const char*> SendStatusMessage;

public:
    bool initialize();
    bool set_init();

    bool startAcquisition();
    void stopAcquisition();

public:
    int nChannels, nScans, nAlines;
    unsigned long VoltRange1, VoltRange2;
//    int AcqRate;
    unsigned long TriggerDelay;
    bool UseExternalClock;
    bool UseAutoTrigger;

    bool _running;

private:
    bool _dirty;

	// thread
	std::thread _thread;
	void run();

private:
	// Handle of Alazar board
	HANDLE boardHandle;

	// Array of buffer pointers
    std::array<uint16_t *, BUFFER_COUNT> BufferArray;

    // Dump an error
    void dumpError(RETURN_CODE retCode, const char* pPreamble);
    void dumpErrorSystem(int res, const char* pPreamble);
#endif
};

#endif // ALAZAR_DAQ_H_
