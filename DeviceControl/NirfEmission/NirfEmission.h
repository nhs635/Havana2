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
	
	int nAlines;	

	bool initialize();
	void start();
	void stop();

private:	
	double max_rate;
	double* data;
	
	const char* physicalChannel;
	const char* sampleClockSource;
	const char* triggerSource;

	TaskHandle _taskHandle;
	void dumpError(int res, const char* pPreamble);
};

#endif // NIRF_EMISSION_H_