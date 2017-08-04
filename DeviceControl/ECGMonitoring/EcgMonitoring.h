#ifndef ECG_MONITORING_H_
#define ECG_MONITORING_H_

#include <iostream>
#include <deque>

#include <Common/callback.h>

typedef void *TaskHandle;

class EcgMonitoring
{
public:
	EcgMonitoring();
	~EcgMonitoring();
			
	bool initialize();
	void start();
	void stop();

	callback<double&> acquiredData;
	callback<void> startRecording;
	callback<double> renewHeartRate;
			
public:
	std::deque<double> deque_ecg;
	std::deque<double> deque_record_ecg;
	std::deque<int> deque_period;
	int prev_peak_pos;
	bool isRecording;

private:
	const char* physicalChannel;
	const char* sourceTerminal;

	TaskHandle _taskHandle;	
	void dumpError(int res, const char* pPreamble);
};

#endif // ECG_MONITORING_H_