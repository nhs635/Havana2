
#include "SyncFLIM.h"
#include <Havana2/Configuration.h>
#include <QtWidgets/QMessageBox.h>

#ifdef OCT_FLIM

#if NI_ENABLE
#include <NIDAQmx.h>
using namespace std;


SyncFLIM::SyncFLIM() :
	_taskHandle(nullptr),
	slow(4),
	counterChannel(NI_FLIM_SYNC_CHANNEL),
    sourceTerminal(NI_FLIM_SYNC_SOURCE)
{
}


SyncFLIM::~SyncFLIM()
{
	if (_taskHandle)
		DAQmxClearTask(_taskHandle);
}


bool SyncFLIM::initialize()
{	
	printf("Initializing NI Counter for synchronization of FLIM laser...\n");
		
	int lowTicks = (int)(ceil(slow / 2));
	int highTicks = (int)(floor(slow / 2));
	uint64_t sampsPerChan = 120000;
	int res;

	if ((res = DAQmxCreateTask("", &_taskHandle)) != 0)
	{
		dumpError(res, "ERROR: Failed to set NI Counter: ");
		return false;
	}

	if ((res = DAQmxCreateCOPulseChanTicks(_taskHandle, counterChannel, NULL, sourceTerminal, DAQmx_Val_Low, 0, lowTicks, highTicks)) != 0)
	{
		dumpError(res, "ERROR: Failed to set NI Counter: ");
		return false;
	}

	if ((res = DAQmxCfgImplicitTiming(_taskHandle, DAQmx_Val_ContSamps, sampsPerChan)) != 0)
	{
		dumpError(res, "ERROR: Failed to set NI Counter: ");
		return false;
	}

	printf("NI Counter for synchronization of FLIM laser is successfully initialized.\n");	

	return true;
}


void SyncFLIM::start()
{
	if (_taskHandle)
	{
		printf("NI Counter is issueing external triggers for synchronization of FLIM laser...\n");
		DAQmxStartTask(_taskHandle);
	}
}


void SyncFLIM::stop()
{
	if (_taskHandle)
	{
		printf("NI Counter is stopped.\n");
		DAQmxStopTask(_taskHandle);
		DAQmxClearTask(_taskHandle);
		_taskHandle = nullptr;
	}
}


void SyncFLIM::dumpError(int res, const char* pPreamble)
{	
	char errBuff[2048];
	if (res < 0)
		DAQmxGetErrorString(res, errBuff, 2048);	

	//QMessageBox::critical(nullptr, "Error", (QString)pPreamble + (QString)errBuff);
	printf("%s\n\n", ((QString)pPreamble + (QString)errBuff).toUtf8().data());
	
	if (_taskHandle)
	{
		DAQmxStopTask(_taskHandle);
		DAQmxClearTask(_taskHandle);
		_taskHandle = nullptr;
	}
}
#endif

#endif