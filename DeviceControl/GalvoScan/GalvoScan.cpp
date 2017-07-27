
#include "GalvoScan.h"
#include <Havana2/Configuration.h>
#include <QtWidgets/QMessageBox.h>

#if NI_ENABLE
#include <NIDAQmx.h>
using namespace std;


GalvoScan::GalvoScan() :
	_taskHandle(nullptr),
	nAlines(1024),
	N(nAlines * 8),
	pp_voltage(2.0),
	offset(0.0),
	max_rate(1000.0),
	data(nullptr),
	physicalChannel(NI_GALVO_CHANNEL),
    sourceTerminal(NI_GAVLO_SOURCE)
{
}


GalvoScan::~GalvoScan()
{
	if (data)
	{
		delete[] data;
		data = nullptr;
	}
	if (_taskHandle) 
		DAQmxClearTask(_taskHandle);
}


bool GalvoScan::initialize()
{
	printf("Initializing NI Analog Output for galvano mirror...\n");

	int res;
	N = nAlines * 8;
	data = new double[N];

	for (int i = 0; i < N; i++)
	{
		double x = (double)i / (double)nAlines;
		data[i] = pp_voltage * (x - floor(x)) - pp_voltage / 2 + offset;
	}

	/*********************************************/
	// Scan Part
	/*********************************************/
	if ((res = DAQmxCreateTask("", &_taskHandle)) != 0)
	{
		dumpError(res, "ERROR: Failed to set galvoscanner: ");
		return false;
	}
	if ((res = DAQmxCreateAOVoltageChan(_taskHandle, physicalChannel, "", -10.0, 10.0, DAQmx_Val_Volts, NULL)) != 0)
	{
		dumpError(res, "ERROR: Failed to set galvoscanner: ");
		return false;
	}
	if ((res = DAQmxCfgSampClkTiming(_taskHandle, sourceTerminal, max_rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, N)) != 0)
	{
		dumpError(res, "ERROR: Failed to set galvoscanner: ");
		return false;
	}

	if ((res = DAQmxWriteAnalogF64(_taskHandle, N, FALSE, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel, data, NULL, NULL)) != 0)
	{
		dumpError(res, "ERROR: Failed to set galvoscanner: ");
		return false;
	}		

	printf("NI Analog Output for galvano mirror is successfully initialized.\n");	

	return true;
}


void GalvoScan::start()
{
	if (_taskHandle)
	{
		printf("Galvano mirror is scanning a sample...\n");
		DAQmxStartTask(_taskHandle);
	}
}


void GalvoScan::stop()
{
	if (_taskHandle)
	{
		printf("NI Analog Output is stopped.\n");
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


void GalvoScan::dumpError(int res, const char* pPreamble)
{	
	char errBuff[2048];
	if (res < 0)
		DAQmxGetErrorString(res, errBuff, 2048);

	//QMessageBox::critical(nullptr, "Error", (QString)pPreamble + (QString)errBuff);
	printf("%s\n\n", ((QString)pPreamble + (QString)errBuff).toUtf8().data());

	if (_taskHandle)
	{
		double data[1] = { 0.0 };
		DAQmxWriteAnalogF64(_taskHandle, 1, TRUE, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel, data, NULL, NULL);

		DAQmxStopTask(_taskHandle);
		DAQmxClearTask(_taskHandle);

		_taskHandle = nullptr;
	}
}
#endif