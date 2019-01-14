
#include "EcgMonitoring.h"
#include <Havana2/Configuration.h>
#include <QtWidgets/QMessageBox.h>

#ifdef ECG_TRIGGERING

#if NI_ENABLE
#include <NIDAQmx.h>
using namespace std;

int32 CVICALLBACK EveryNCallback(TaskHandle taskHandle, int32 everyNsamplesEventType, uInt32 nSamples, void *callbackData);


EcgMonitoring::EcgMonitoring() :
	_taskHandle(nullptr),
	prev_peak_pos(0),
	isRecording(false),
	physicalChannel(NI_ECG_CHANNEL),
	sourceTerminal(NI_ECG_TRIGGER_SOURCE)
{
	for (int i = 0; i < N_VIS_SAMPS_ECG; i++)
		deque_ecg.push_back(0.0);
}

EcgMonitoring::~EcgMonitoring()
{
	if (_taskHandle) 
		DAQmxClearTask(_taskHandle);
}


bool EcgMonitoring::initialize()
{
	printf("Initializing NI Analog Input for ECG monitoring...\n");

	int res;
	int N = 1;

	if ((res = DAQmxCreateTask("", &_taskHandle)) != 0)
	{
		dumpError(res, "ERROR: Failed to set ECG monitoring: ");
		return false;
	}
	if ((res = DAQmxCreateAIVoltageChan(_taskHandle, physicalChannel, "", DAQmx_Val_RSE, -5.0, 5.0, DAQmx_Val_Volts, NULL)) != 0)
	{
		dumpError(res, "ERROR: Failed to set ECG monitoring: ");
		return false;
	}
	if ((res = DAQmxCfgSampClkTiming(_taskHandle, sourceTerminal, ECG_SAMPLING_RATE, DAQmx_Val_Rising, DAQmx_Val_ContSamps, N)) != 0)
	{
		dumpError(res, "ERROR: Failed to set ECG monitoring: ");
		return false;
	}
	//if ((res = DAQmxCfgDigEdgeStartTrig(_taskHandle, sourceTerminal, DAQmx_Val_Rising)) != 0)
	//{
	//	dumpError(res, "ERROR: Failed to set ECG monitoring: ");
	//	return false;
	//}
	//if ((res = DAQmxSetStartTrigRetriggerable(_taskHandle, 1)) != 0)
	//{
	//	dumpError(res, "ERROR: Failed to set ECG monitoringn: ");
	//	return false;
	//}

	if ((res = DAQmxRegisterEveryNSamplesEvent(_taskHandle, DAQmx_Val_Acquired_Into_Buffer, N, 0, EveryNCallback, this)) != 0)
	{
		dumpError(res, "ERROR: Failed to set ECG monitoring: ");
		return false;
	}

	printf("NI Analog Input for ECG monitoring is successfully initialized.\n");	

	return true;
}


void EcgMonitoring::start()
{
	if (_taskHandle)
	{
		printf("ECG is monitoring...\n");
		DAQmxStartTask(_taskHandle);
	}
}


void EcgMonitoring::stop()
{
	if (_taskHandle)
	{
		printf("NI Analog Intput is stopped.\n");
		DAQmxStopTask(_taskHandle);
		DAQmxClearTask(_taskHandle);
		_taskHandle = nullptr;
	}
}


void EcgMonitoring::dumpError(int res, const char* pPreamble)
{	
	char errBuff[2048];
	if (res < 0)
		DAQmxGetErrorString(res, errBuff, 2048);

	printf("%s\n\n", ((QString)pPreamble + (QString)errBuff).toUtf8().data());
	//SendStatusMessage(((QString)pPreamble + (QString)errBuff).toUtf8().data());

	if (_taskHandle)
	{
		DAQmxStopTask(_taskHandle);
		DAQmxClearTask(_taskHandle);
		_taskHandle = nullptr;
	}
}


int32 CVICALLBACK EveryNCallback(TaskHandle taskHandle, int32 everyNsamplesEventType, uInt32 nSamples, void *callbackData)
{
	EcgMonitoring* pEcgMonitor = (EcgMonitoring*)callbackData;

	static int n = 0;
	double data;
	bool is_peak = false;

	int32 res;
	if ((res = DAQmxReadAnalogScalarF64(taskHandle, 0.0, &data, NULL)) != 0)
	{
		pEcgMonitor->dumpError(res, "ERROR: Failed to read ECG Monitoring: ");
		return -1;
	}

	n++;
	pEcgMonitor->deque_ecg.push_back(data);
	pEcgMonitor->deque_ecg.pop_front();
	if (pEcgMonitor->isRecording)
		pEcgMonitor->deque_record_ecg.push_back(data);

	double derivative[2];
	derivative[0] = pEcgMonitor->deque_ecg.at(N_VIS_SAMPS_ECG - 2) - pEcgMonitor->deque_ecg.at(N_VIS_SAMPS_ECG - 3);
	derivative[1] = pEcgMonitor->deque_ecg.at(N_VIS_SAMPS_ECG - 1) - pEcgMonitor->deque_ecg.at(N_VIS_SAMPS_ECG - 2);

	if (derivative[0] * derivative[1] < 0) // R peak detection condition 1 : is it peak?
	{
		if (pEcgMonitor->deque_ecg.at(N_VIS_SAMPS_ECG - 2) > ECG_THRES_VALUE) // R peak detection condition 2 : is it over a threshold value?
		{
			if (n - pEcgMonitor->prev_peak_pos > ECG_THRES_TIME) // R peak detection condition 3 : it it occurred after a threshold period of time?
			{
				pEcgMonitor->startRecording();
				printf("R peak detected! %.1f \n", (double)n / 1000.0);
				pEcgMonitor->deque_period.push_back(n - pEcgMonitor->prev_peak_pos);			
				pEcgMonitor->prev_peak_pos = n;

				if (pEcgMonitor->deque_period.size() == 7)
				{
					pEcgMonitor->deque_period.pop_front();

					double heart_interval = 0;
					for (int i = 0; i < 6; i++)
						heart_interval += (double)pEcgMonitor->deque_period.at(i) / 6.0;

					pEcgMonitor->heart_interval = heart_interval; // milliseconds
					pEcgMonitor->renewHeartRate(60.0 / heart_interval * 1000.0);
				}

				is_peak = true;
			}
		}
	}
	
	pEcgMonitor->acquiredData(data, is_peak);
	
    (void)nSamples;
    (void)everyNsamplesEventType;

	return 0;
}
#endif

#endif