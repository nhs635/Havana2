#ifndef GALVO_SCAN_H_
#define GALVO_SCAN_H_

#include <iostream>

#include <Common/callback.h>

typedef void *TaskHandle;

class GalvoScan
{
public:
	GalvoScan();
	~GalvoScan();
	
	int nAlines;	

	double pp_voltage_fast;
	double offset_fast;
	double pp_voltage_slow;
	double offset_slow;
	double step_slow;

	bool initialize();
	void start();
	void stop();
		
	callback<void> startScan;
	callback<void> stopScan;

private:	
	int nIter;

	double max_rate;
	double* data;
	
	const char* physicalChannel;
	const char* sourceTerminal;

	TaskHandle _taskHandle;
	void dumpError(int res, const char* pPreamble);
};

#endif // GALVO_SCAN_H_