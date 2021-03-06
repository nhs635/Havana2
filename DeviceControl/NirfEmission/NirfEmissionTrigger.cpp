
#include "NirfEmissionTrigger.h"

#include <Havana2/Configuration.h>
#include <QtWidgets/QMessageBox.h>

#ifdef OCT_NIRF_DISABLE

#if NI_ENABLE
#include <NIDAQmx.h>
using namespace std;


NirfEmissionTrigger::NirfEmissionTrigger() :
	_taskHandle(nullptr),
	nAlines(1024),
	counterChannel(NI_NIRF_ALINES_COUNTER),
    sourceTerminal(NI_NIRF_TRIGGER_SOURCE)
{
}


NirfEmissionTrigger::~NirfEmissionTrigger()
{
	if (_taskHandle) 
		DAQmxClearTask(_taskHandle);
}


bool NirfEmissionTrigger::initialize()
{
	printf("Initializing NI Counter for triggering of NIRF emission acquisition...\n");

	int lowTicks = nAlines / 2;
	int highTicks = lowTicks;
	uint64_t sampsPerChan = nAlines;
	int res;

	if ((res = DAQmxCreateTask("", &_taskHandle)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Counter: ");
		return false;
	}
	if ((res = DAQmxCreateCOPulseChanTicks(_taskHandle, counterChannel, NULL, sourceTerminal, DAQmx_Val_High, 0, lowTicks, highTicks)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Counter: ");
		return false;
	}

	if ((res = DAQmxCfgImplicitTiming(_taskHandle, DAQmx_Val_ContSamps, sampsPerChan)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Counter: ");
		return false;
	}

	printf("NI Counter for triggering of NIRF emission acquisition is successfully initialized.\n");

	return true;
}


void NirfEmissionTrigger::start()
{
	if (_taskHandle)
	{
		printf("NI Counter is issueing external triggers for triggering of NIRF emission acquisition...\n");
		DAQmxStartTask(_taskHandle);
	}
}


void NirfEmissionTrigger::stop()
{
	if (_taskHandle)
	{
		printf("NI Counter is stopped.\n");
		DAQmxStopTask(_taskHandle);
		DAQmxClearTask(_taskHandle);
		_taskHandle = nullptr;
	}
}


void NirfEmissionTrigger::dumpError(int res, const char* pPreamble)
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