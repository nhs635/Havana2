
#include "Voltage800RPS.h"
#include <Havana2/Configuration.h>
#include <QtWidgets/QMessageBox.h>

#if NI_ENABLE
#include <NIDAQmx.h>
using namespace std;


Voltage800RPS::Voltage800RPS() :
	_taskHandle(nullptr),
	physicalChannel(NI_800RPS_CHANNEL)
{
}

Voltage800RPS::~Voltage800RPS()
{
	if (_taskHandle) 
		DAQmxClearTask(_taskHandle);
}


bool Voltage800RPS::initialize()
{
	printf("Initializing NI Analog Output for 800 rps motor control...\n");

	int res; 

	if ((res = DAQmxCreateTask("", &_taskHandle)) != 0)
	{
		dumpError(res, "ERROR: Failed to set 800 rps motor control: ");
		return false;
	}
	if ((res = DAQmxCreateAOVoltageChan(_taskHandle, physicalChannel, "", -10.0, 10.0, DAQmx_Val_Volts, NULL)) != 0)
	{
		dumpError(res, "ERROR: Failed to set 800 rps motor control: ");
		return false;
	}

	printf("NI Analog Output for 800 rps motor control is successfully initialized.\n");	

	return true;
}


bool Voltage800RPS::apply(double voltage)
{
	if (_taskHandle)
	{
		int res;
		
		if ((res = DAQmxStartTask(_taskHandle)))
		{
			dumpError(res, "ERROR: Failed to set 800 rps motor control: ");
			return false;
		}

		if ((res = DAQmxWriteAnalogScalarF64(_taskHandle, FALSE, DAQmx_Val_WaitInfinitely, voltage, NULL)))
		{
			dumpError(res, "ERROR: Failed to set 800 rps motor control: ");
			return false;
		}

		if ((res = DAQmxStopTask(_taskHandle)))
		{
			dumpError(res, "ERROR: Failed to set 800 rps motor control: ");
			return false;
		}

		printf("A voltage is applied for 800 rps motor control (%.1f V)...\n", voltage);

		return true;
	}

	return false;
}


void Voltage800RPS::dumpError(int res, const char* pPreamble)
{	
	char errBuff[2048];
	if (res < 0)
		DAQmxGetErrorString(res, errBuff, 2048);

	//QMessageBox::critical(nullptr, "Error", (QString)pPreamble + (QString)errBuff);
	printf("%s", ((QString)pPreamble + (QString)errBuff).toUtf8().data());

	if (_taskHandle)
	{
		DAQmxStopTask(_taskHandle);
		DAQmxClearTask(_taskHandle);
		_taskHandle = nullptr;
	}
}
#endif