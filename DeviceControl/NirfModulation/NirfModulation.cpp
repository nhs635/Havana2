
#include "NirfModulation.h"

#include <Havana2/Configuration.h>
#include <QtWidgets/QMessageBox.h>

#if NI_ENABLE
#ifdef OCT_NIRF
#ifdef TWO_CHANNEL_NIRF
#include <NIDAQmx.h>
using namespace std;


NirfModulation::NirfModulation() :
	_taskHandle(nullptr),
	nCh(2),
	max_rate(ALINE_RATE),
	physicalChannel(NI_NIRF_MODUL_CHANNEL),
    sourceTerminal(NI_NIRF_MODUL_SOURCE)
{
}


NirfModulation::~NirfModulation()
{
	if (_taskHandle) 
		DAQmxClearTask(_taskHandle);
}


bool NirfModulation::initialize()
{
	printf("Initializing NI Analog Output for triggering NIRF modulated excitation...\n");

	int res;

	int N = 1000;
	data = new double[nCh * N];
	for (int i = 0; i < N; i++)
	{
#if MODULATION_FREQ == 2
		// [1/2 modulation]
		data[i] = (i % 2 == 1) ? 10 : -10;
		if (nCh == 2)
			data[i + N] = (i % 2 == 1) ? -10 : 10;
#endif

#if MODULATION_FREQ == 4
		// [1/4 modulation]
		data[i] = (i % 4 == 0) ? 10 : -10;
		if (nCh == 2)
			data[i + N] = ((i + 2) % 4 == 0) ? 10 : -10;
#endif
	}

	if ((res = DAQmxCreateTask("", &_taskHandle)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Analog Output: ");
		return false;
	}
	if ((res = DAQmxCreateAOVoltageChan(_taskHandle, physicalChannel, "", -10.0, 10.0, DAQmx_Val_Volts, NULL)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Analog Output: ");
		return false;
	}
	if ((res = DAQmxCfgSampClkTiming(_taskHandle, sourceTerminal, max_rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, N)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Analog Output: ");
		return false;
	}
	if ((res = DAQmxWriteAnalogF64(_taskHandle, N, 0, 10.0, DAQmx_Val_GroupByChannel, data, NULL, NULL)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Analog Output: ");
		return false;
	}
	
	printf("NI Analog Output for triggering NIRF modulated excitation is successfully initialized.\n");

	return true;
}


void NirfModulation::start()
{
	if (_taskHandle)
	{
		printf("NI Analog Output is issueing external triggers for triggering NIRF modulated excitation...\n");
		DAQmxStartTask(_taskHandle);
	}
}


void NirfModulation::stop()
{
	if (_taskHandle)
	{
		printf("NI Analog Output is stopped.\n");
		DAQmxStopTask(_taskHandle);
		DAQmxClearTask(_taskHandle);
		_taskHandle = nullptr;
	}
}


void NirfModulation::dumpError(int res, const char* pPreamble)
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
#endif
