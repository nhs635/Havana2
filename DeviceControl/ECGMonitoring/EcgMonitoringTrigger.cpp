
#include "EcgMonitoringTrigger.h"

#include <Havana2/Configuration.h>
#include <QtWidgets/QMessageBox.h>

#if NI_ENABLE
#include <NIDAQmx.h>
using namespace std;


EcgMonitoringTrigger::EcgMonitoringTrigger() :
	_taskHandle(nullptr),
	counterChannel(NI_ECG_TRIGGER_CHANNEL),
    sourceTerminal("20MHzTimebase")
{
}


EcgMonitoringTrigger::~EcgMonitoringTrigger()
{
	if (_taskHandle)
		DAQmxClearTask(_taskHandle);
}


bool EcgMonitoringTrigger::initialize()
{	
	printf("Initializing NI Counter for triggering of ECG monitoring...\n");
		
	int lowTicks = 10000000 / ECG_SAMPLING_RATE;
	int highTicks = lowTicks;
	uint64_t sampsPerChan = 100000;
	int res;

	if ((res = DAQmxCreateTask("", &_taskHandle)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Counter: ");
		return false;
	}

	if ((res = DAQmxCreateCOPulseChanTicks(_taskHandle, counterChannel, NULL, sourceTerminal, DAQmx_Val_Low, 0, lowTicks, highTicks)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Counter: ");
		return false;
	}

	if ((res = DAQmxCfgImplicitTiming(_taskHandle, DAQmx_Val_ContSamps, sampsPerChan)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Counter: ");
		return false;
	}

	printf("NI Counter for triggering of ECG monitoring is successfully initialized.\n");	

	return true;
}


void EcgMonitoringTrigger::start()
{
	if (_taskHandle)
	{
		printf("NI Counter is issueing external triggers for triggering of ECG monitoring...\n");
		DAQmxStartTask(_taskHandle);
	}
}


void EcgMonitoringTrigger::stop()
{
	if (_taskHandle)
	{
		printf("NI Counter is stopped.\n");
		DAQmxStopTask(_taskHandle);
		DAQmxClearTask(_taskHandle);
		_taskHandle = nullptr;
	}
}


void EcgMonitoringTrigger::dumpError(int res, const char* pPreamble)
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