#ifndef NIRF_EMISSION_H_
#define NIRF_EMISSION_H_

#include <iostream>

#include <Common/callback.h>

typedef void *TaskHandle;

class NirfEmission
{
public:
	NirfEmission();
	~NirfEmission();
	
	int nCh;
	int nAlines;		
	int nAcqs;
	double* data;

	bool initialize();
	void start();
	void stop();

	// callbacks
	callback2<int, const double*> DidAcquireData;
	callback<void> DidStopData;
	callback<const char*> SendStatusMessage;

	void dumpError(int res, const char* pPreamble);

private:
	double max_rate;
	
	const char* physicalChannel;
	const char* sampleClockSource;
	const char* alinesTrigger;

	TaskHandle _taskHandle;
};

#endif // NIRF_EMISSION_H_