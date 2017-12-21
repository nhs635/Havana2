
#include "GalvoScan.h"
#include <Havana2/Configuration.h>
#include <QtWidgets/QMessageBox.h>

#ifdef GALVANO_MIRROR

#if NI_ENABLE
#include <NIDAQmx.h>
using namespace std;


int32 CVICALLBACK DoneCallback(TaskHandle taskHandle, int32 status, void *callbackData);


GalvoScan::GalvoScan() :
	_taskHandle(nullptr),
	nAlines(1024),
	nIter(1),
	pp_voltage_fast(2.0),
	offset_fast(0.0),
	pp_voltage_slow(2.0),
	offset_slow(0.0),
	step_slow(0.0),
	max_rate(118400.0),
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
	
	int sample_mode = (pp_voltage_slow == 0) ? DAQmx_Val_ContSamps : DAQmx_Val_FiniteSamps;

	nIter = int(ceil(pp_voltage_slow / step_slow)) + 1;
	int N = nAlines * nIter;
	if (sample_mode == DAQmx_Val_FiniteSamps) N = N + 1;

	data = new double[2 * N];
	for (int i = 0; i < N; i++)
	{
		double x = (double)i / (double)nAlines;
		if (nIter != 1)
			data[2 * i + 0] = pp_voltage_slow * (floor(x)) / double(nIter - 1) - pp_voltage_slow / 2 + offset_slow;
		else
			data[2 * i + 0] = offset_slow;
		data[2 * i + 1] = pp_voltage_fast * (x - floor(x)) - pp_voltage_fast / 2 + offset_fast;
	}
	if (sample_mode == DAQmx_Val_FiniteSamps)
	{
		data[2 * (N - 1) + 0] = offset_slow;
		data[2 * (N - 1) + 1] = offset_fast;
	}
	
	/*********************************************/
	// Scan Part
	/*********************************************/
	if ((res = DAQmxCreateTask("", &_taskHandle)) != 0)
	{
		dumpError(res, "ERROR: Failed to set galvoscanner1: ");
		return false;
	}
	if ((res = DAQmxCreateAOVoltageChan(_taskHandle, physicalChannel, "", -10.0, 10.0, DAQmx_Val_Volts, NULL)) != 0)
	{
		dumpError(res, "ERROR: Failed to set galvoscanner2: ");
		return false;
	}
	if ((res = DAQmxCfgSampClkTiming(_taskHandle, sourceTerminal, max_rate, DAQmx_Val_Rising, sample_mode, N)) != 0)
	{
		dumpError(res, "ERROR: Failed to set galvoscanner3: ");
		return false;
	}
	if ((res = DAQmxWriteAnalogF64(_taskHandle, N, FALSE, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByScanNumber, data, NULL, NULL)) != 0)
	{
		dumpError(res, "ERROR: Failed to set galvoscanner4: ");
		return false;
	}		
	if (sample_mode != DAQmx_Val_ContSamps)
	{
		if ((res = DAQmxRegisterDoneEvent(_taskHandle, 0, DoneCallback, this)) != 0)
		{
			dumpError(res, "ERROR: Failed to set galvoscanner5: ");
			return false;
		}
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


int32 CVICALLBACK DoneCallback(TaskHandle taskHandle, int32 status, void *callbackData)
{
	GalvoScan* pGalvoScan = (GalvoScan*)callbackData;

	pGalvoScan->stopScan();

	(void)taskHandle;
	(void)status;

	return 0;
}
#endif

#endif