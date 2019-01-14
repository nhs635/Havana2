#ifndef GAIN_CONTROL_H_
#define GAIN_CONTROL_H_

#include <iostream>

typedef void *TaskHandle;

class PmtGainControl
{
public:
	PmtGainControl();
	~PmtGainControl();

	double voltage1;
	double voltage2;

	bool initialize();
	void start();
	void stop();
		
private:
	const char* physicalChannel;
	const char* sourceTerminal;

	TaskHandle _taskHandle;	
	void dumpError(int res, const char* pPreamble);
};

#endif // GAIN_CONTROL_H_