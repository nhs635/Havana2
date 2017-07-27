#ifndef GALVO_SCAN_H_
#define GALVO_SCAN_H_

#include <iostream>

typedef void *TaskHandle;

class GalvoScan
{
public:
	GalvoScan();
	~GalvoScan();
	
	int nAlines;

	double pp_voltage;
	double offset;
	
	bool initialize();
	void start();
	void stop();
		
private:
	const char* physicalChannel;
	const char* sourceTerminal;
	
	double max_rate;

	int N;
	double* data;

	TaskHandle _taskHandle;
	void dumpError(int res, const char* pPreamble);
};

#endif // GALVO_SCAN_H_