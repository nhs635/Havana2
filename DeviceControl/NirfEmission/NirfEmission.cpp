
#include "NirfEmission.h"
#include <Havana2/Configuration.h>
#include <QtWidgets/QMessageBox.h>

#include <Common/array.h>

#ifdef OCT_NIRF

#if NI_ENABLE
#include <NIDAQmx.h>
using namespace std;

#include <ipps.h>

#include <chrono>
chrono::steady_clock::time_point startTime, endTime;


int32 CVICALLBACK EveryNCallback(TaskHandle taskHandle, int32 everyNsamplesEventType, uInt32 nSamples, void *callbackData);


NirfEmission::NirfEmission() :
	_taskHandle(nullptr),
	N(1),
	nCh(1),
	nAlines(1024),
	nAcqs(0),
	max_rate(120000.0),
	data(nullptr),
	physicalChannel(NI_NIRF_EMISSION_CHANNEL),
	sampleClockSource(NI_NIRF_TRIGGER_SOURCE),
	alinesTrigger(NI_NIRF_ALINES_SOURCE)
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
#ifdef TWO_CHANNEL_NIRF
	nCh = 2;
#endif
	data = new double[nCh * nAlines];
	N = 32;

	/*********************************************/
	// Analog Input for NIRF Emission Acquisition
	/*********************************************/
	if ((res = DAQmxCreateTask("", &_taskHandle)) != 0)
	{
		dumpError(res, "ERROR: Failed to set NIRF emission acquisition: ");
		return false;
	}
	if ((res = DAQmxCreateAIVoltageChan(_taskHandle, physicalChannel, "", DAQmx_Val_RSE, 0.0, 5.0, DAQmx_Val_Volts, NULL)) != 0)
	{
		dumpError(res, "ERROR: Failed to set NIRF emission acquisition: ");
		return false;
	}
	if ((res = DAQmxCfgSampClkTiming(_taskHandle, sampleClockSource, max_rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, N)) != 0)
	{
		dumpError(res, "ERROR: Failed to set NIRF emission acquisition: ");
		return false;
	}
	if ((res = DAQmxCfgDigEdgeStartTrig(_taskHandle, alinesTrigger, DAQmx_Val_Rising)) != 0)
	{
		dumpError(res, "ERROR: Failed to set NIRF emission acquisition: ");
		return false;
	}
	if ((res = DAQmxRegisterEveryNSamplesEvent(_taskHandle, DAQmx_Val_Acquired_Into_Buffer, N, 0, EveryNCallback, this)) != 0)
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

		int res;
		if ((res = DAQmxStartTask(_taskHandle)) != 0)
		{
			dumpError(res, "ERROR: Failed to set NIRF emission acquisition: ");
			return;
		}
		startTime = chrono::steady_clock::now();
	}
}


void NirfEmission::stop()
{
	if (_taskHandle)
	{
		printf("NI Analog Input is stopped.\n");
		DidStopData();
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

	printf("%s\n\n", ((QString)pPreamble + (QString)errBuff).toUtf8().data());
	SendStatusMessage(((QString)pPreamble + (QString)errBuff).toUtf8().data());

	if (_taskHandle)
	{
		DAQmxStopTask(_taskHandle);
		DAQmxClearTask(_taskHandle);
		_taskHandle = nullptr;
	}
}


int32 CVICALLBACK EveryNCallback(TaskHandle taskHandle, int32 everyNsamplesEventType, uInt32 nSamples, void *callbackData)
{
	NirfEmission* pNirfEmission = (NirfEmission*)callbackData;

	static int n = 0;
	int32 read;
	int32 res = DAQmxReadAnalogF64(taskHandle, pNirfEmission->N, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByScanNumber, pNirfEmission->data + n, pNirfEmission->nCh * pNirfEmission->N, &read, NULL);
	if (res != 0) printf("error\n");
	
	n = n + pNirfEmission->nCh * pNirfEmission->N;
	if (n == pNirfEmission->nCh * pNirfEmission->nAlines)
	{
		n = 0;
		pNirfEmission->nAcqs++;
		
		np::DoubleArray data(pNirfEmission->nCh * pNirfEmission->nAlines);
#ifndef TWO_CHANNEL_NIRF
		memcpy(data, pNirfEmission->data, sizeof(double) * pNirfEmission->nAlines);
#else
		ippsCplxToReal_64fc((const Ipp64fc*)pNirfEmission->data, data.raw_ptr(), data.raw_ptr() + pNirfEmission->nAlines, pNirfEmission->nAlines);
#endif
		pNirfEmission->DidAcquireData(pNirfEmission->nAcqs, data.raw_ptr());

		endTime = chrono::steady_clock::now();
		chrono::milliseconds elapsed = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);

		if (pNirfEmission->nAcqs % 500 == 0)
			printf("@@ NIRF time: %.2f sec // NIRF rate: %.2f acqs/sec\n", 
				(double)elapsed.count() / 1000.0, 1000.0 * (double)pNirfEmission->nAcqs / (double)elapsed.count());
	}

	(void)nSamples;
	(void)everyNsamplesEventType;

	return 0;
}
#endif

#endif