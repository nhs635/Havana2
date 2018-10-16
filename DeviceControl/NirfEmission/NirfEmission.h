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
	
	int N;
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

private:
	double max_rate;
	
	const char* physicalChannel;
	const char* sampleClockSource;
	const char* alinesTrigger;

	TaskHandle _taskHandle;
	void dumpError(int res, const char* pPreamble);
};

#endif // NIRF_EMISSION_H_