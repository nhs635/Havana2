#ifndef NIRF_EMISSION_TRIGGER_H_
#define NIRF_EMISSION_TRIGGER_H_

#include <iostream>

typedef void *TaskHandle;

class NirfEmissionTrigger
{
public:
	NirfEmissionTrigger();
	~NirfEmissionTrigger();
	
	int nAlines;	

	bool initialize();
	void start();
	void stop();

private:		
	const char* syncPowerTerminal;
	const char* syncResetTerminal;
	const char* counterChannel;
	const char* sourceTerminal;

	TaskHandle _taskHandle;
	void dumpError(int res, const char* pPreamble);
};

#endif // NIRF_EMISSION_TRIGGER_H_