#ifndef VOLTAGE_800RPS_H
#define VOLTAGE_800RPS_H

#include <iostream>

typedef void *TaskHandle;

class Voltage800RPS
{
public:
	Voltage800RPS();
	~Voltage800RPS();
	
	bool initialize();
	bool apply(double voltage);

private:
	const char* physicalChannel;
	
	TaskHandle _taskHandle;
	void dumpError(int res, const char* pPreamble);
};

#endif // VOLTAGE_800RPS_H