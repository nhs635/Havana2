
#include "EcgMonitoring.h"
#include <Havana2/Configuration.h>
#include <QtWidgets/QMessageBox.h>

#if NI_ENABLE
#include <NIDAQmx.h>
using namespace std;


int32 CVICALLBACK EveryNCallback(TaskHandle taskHandle, int32 everyNsamplesEventType, uInt32 nSamples, void *callbackData);


EcgMonitoring::EcgMonitoring() :
	_taskHandle(nullptr),
	prev_peak_pos(0),
	physicalChannel(NI_ECG_CHANNEL),
    sourceTerminal(NI_ECG_TRIG_SOURCE)
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
	printf("Initializing NI Analog Output for galvano mirror...\n");

	int res;
	double max_rate = 10.0;
	int N = 1;

	if ((res = DAQmxCreateTask("", &_taskHandle)) != 0)
	{
		dumpError(res, "ERROR: Failed to set ECG Monitoring: ");
		return false;
	}
	if ((res = DAQmxCreateAIVoltageChan(_taskHandle, physicalChannel, "", DAQmx_Val_RSE, -10.0, 10.0, DAQmx_Val_Volts, NULL)) != 0)
	{
		dumpError(res, "ERROR: Failed to set ECG Monitoring: ");
		return false;
	}
	if ((res = DAQmxCfgSampClkTiming(_taskHandle, sourceTerminal, max_rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, N)) != 0)
	{
		dumpError(res, "ERROR: Failed to set ECG Monitoring: ");
		return false;
	}
	if ((res = DAQmxRegisterEveryNSamplesEvent(_taskHandle, DAQmx_Val_Acquired_Into_Buffer, N, 0, EveryNCallback, this)))
	{
		dumpError(res, "ERROR: Failed to set ECG Monitoring: ");
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
	EcgMonitoring* pEcgMonitor = (EcgMonitoring*)callbackData;

	static int n = 0;
	double data;

	DAQmxReadAnalogScalarF64(taskHandle, DAQmx_Val_WaitInfinitely, &data, NULL);
	n++;
	pEcgMonitor->deque_ecg.push_back(data);
	pEcgMonitor->deque_ecg.pop_front();

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

				if (pEcgMonitor->deque_period.size() == 6)
				{
					pEcgMonitor->deque_period.pop_front();

					double heart_period = 0;
					for (int i = 0; i < 5; i++)
						heart_period += (double)pEcgMonitor->deque_period.at(i) / 5.0;

					pEcgMonitor->renewHeartRate(60.0 / heart_period * 1000.0);
				}
			}
		}
	}
	
	pEcgMonitor->acquiredData(data);	
	
    (void)nSamples;
    (void)everyNsamplesEventType;

	return 0;
}
#endif