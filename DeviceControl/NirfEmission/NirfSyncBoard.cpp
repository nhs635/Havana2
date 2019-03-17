
#include "NirfSyncBoard.h"

#include <Havana2/Configuration.h>
#include <QtWidgets/QMessageBox.h>

#ifdef OCT_NIRF
#ifdef NI_NIRF_SYNC

#if NI_ENABLE
#include <NIDAQmx.h>
using namespace std;


NirfSyncBoard::NirfSyncBoard() :
	_taskHandle(nullptr),
	value(0x00000000),
	lines(NI_NIRF_SYNC_PORT)
{
}


NirfSyncBoard::~NirfSyncBoard()
{
	if (_taskHandle) 
		DAQmxClearTask(_taskHandle);
}


bool NirfSyncBoard::initialize()
{
	printf("Initializing NI Digital Output for NIRF synchronization...\n");

	int32 written;
	int res;

	if ((res = DAQmxCreateTask("", &_taskHandle)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Digital Output: ");
		return false;
	}
	if ((res = DAQmxCreateDOChan(_taskHandle, lines, "", DAQmx_Val_ChanForAllLines)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Digital Output: ");
		return false;
	}
	if ((res = DAQmxWriteDigitalU32(_taskHandle, 1, TRUE, 10.0, DAQmx_Val_GroupByChannel, (const uInt32*)&value, &written, NULL)) != 0)
	{
		dumpError(res, "ERROR: Failed to initialize NI Digital Output: ");
		return false;
	}

	printf("NI Digital Output for NIRF synchronization is successfully initialized.\n");

	return true;
}


void NirfSyncBoard::start()
{
	if (_taskHandle)
	{
		printf("NI Digital Output is generating a digital signal...\n");
		DAQmxStartTask(_taskHandle);
	}
}


void NirfSyncBoard::stop()
{
	if (_taskHandle)
	{
		int32 written;
		value = 0x00;
		DAQmxWriteDigitalU32(_taskHandle, 1, TRUE, 10.0, DAQmx_Val_GroupByChannel, (const uInt32*)&value, &written, NULL);

		printf("NI Digital Output is stopped.\n");
		DAQmxStopTask(_taskHandle);
		DAQmxClearTask(_taskHandle);
		_taskHandle = nullptr;
	}
}


void NirfSyncBoard::dumpError(int res, const char* pPreamble)
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