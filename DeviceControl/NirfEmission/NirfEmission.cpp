
#include "NirfEmission.h"
#include <Havana2/Configuration.h>
#include <QtWidgets/QMessageBox.h>

#ifdef OCT_NIRF

#if NI_ENABLE
#include <NIDAQmx.h>
using namespace std;


int32 CVICALLBACK EveryNCallback(TaskHandle taskHandle, int32 everyNsamplesEventType, uInt32 nSamples, void *callbackData);


NirfEmission::NirfEmission() :
	_taskHandle(nullptr),
	nAlines(1024),
	max_rate(120000.0),
	data(nullptr),
	physicalChannel(NI_NIRF_EMISSION_CHANNEL),
	sampleClockSource(NI_NIRF_TRIGGER_CHANNEL),
	triggerSource(NI_NIRF_TRIGGER_SOURCE)
{
}


NirfEmission::~NirfEmission()
{
	if (data)
	{
		delete[] data;
		data = nullptr;
	}
	if (_taskHandle) 
		DAQmxClearTask(_taskHandle);
}


bool NirfEmission::initialize()
{
	printf("Initializing NI Analog Input for NIRF Emission Acquisition...\n");

	int res;	
	data = new double[nAlines];
	
	/*********************************************/
	// Analog Input for NIRF Emission Acquisition
	/*********************************************/
	if ((res = DAQmxCreateTask("", &_taskHandle)) != 0)
	{
		dumpError(res, "ERROR: Failed to set NIRF emission acquisition: ");
		return false;
	}
	if ((res = DAQmxCreateAIVoltageChan(_taskHandle, physicalChannel, "", DAQmx_Val_NRSE, 0.0, 5.0, DAQmx_Val_Volts, NULL)) != 0)
	{
		dumpError(res, "ERROR: Failed to set NIRF emission acquisition: ");
		return false;
	}
	if ((res = DAQmxCfgSampClkTiming(_taskHandle, sampleClockSource, max_rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, nAlines)) != 0)
	{
		dumpError(res, "ERROR: Failed to set NIRF emission acquisition: ");
		return false;
	}
	if ((res = DAQmxCfgDigEdgeStartTrig(_taskHandle, triggerSource, DAQmx_Val_Rising)) != 0)
	{
		dumpError(res, "ERROR: Failed to set NIRF emission acquisition: ");
		return false;
	}
	if ((res = DAQmxRegisterEveryNSamplesEvent(_taskHandle, DAQmx_Val_Acquired_Into_Buffer, nAlines, 0, EveryNCallback, this)) != 0)
	{
		dumpError(res, "ERROR: Failed to set NIRF emission acquisition: ");
		return false;
	}

	printf("NI Analog Input for NIRF emission acquisition is successfully initialized.\n");	

	return true;
}


void NirfEmission::start()
{
	if (_taskHandle)
	{
		printf("NIRF emission is acquiring...\n");
		DAQmxStartTask(_taskHandle);
	}
}


void NirfEmission::stop()
{
	if (_taskHandle)
	{
		printf("NI Analog Input is stopped.\n");
		DAQmxStopTask(_taskHandle);
		DAQmxClearTask(_taskHandle);
		if (data)
		{
			delete[] data;
			data = nullptr;
		}
		_taskHandle = nullptr;
	}
}


void NirfEmission::dumpError(int res, const char* pPreamble)
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


int32 CVICALLBACK EveryNCallback(TaskHandle taskHandle, int32 everyNsamplesEventType, uInt32 nSamples, void *callbackData)
{
	

	return 0;
}
#endif

#endif