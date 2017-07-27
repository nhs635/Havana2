#ifndef ECG_MONITORING_TRIGGER_H_
#define ECG_MONITORING_TRIGGER_H_

#include <iostream>

typedef void *TaskHandle;

class EcgMonitoringTrigger
{
public:
	EcgMonitoringTrigger();
	~EcgMonitoringTrigger();
	
	bool initialize();
	void start();
	void stop();
		
private:
	const char* counterChannel;
	const char* sourceTerminal;

	TaskHandle _taskHandle;
	void dumpError(int res, const char* pPreamble);
};

#endif // ECG_MONITORING_TRIGGER_H_