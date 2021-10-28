#ifndef NIRF_MODULATION_H_
#define NIRF_MODULATION_H_

#include <iostream>

typedef void *TaskHandle;

class NirfModulation
{
public:
	NirfModulation();
	~NirfModulation();
	
	int nCh;	
	double max_rate;

	bool initialize();
	void start();
	void stop();

private:		
	const char* physicalChannel;
	const char* sourceTerminal;

	double* data;

	TaskHandle _taskHandle;
	void dumpError(int res, const char* pPreamble);
};

#endif // NIRF_MODULATION_H_