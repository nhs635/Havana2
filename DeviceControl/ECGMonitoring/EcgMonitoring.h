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

	callback2<double&, bool&> acquiredData;
	callback<void> startRecording;
	callback<double> renewHeartRate;
	callback<const char*> SendStatusMessage;

	void dumpError(int res, const char* pPreamble);
			
public:
	std::deque<double> deque_ecg;
	std::deque<bool> deque_is_peak;
	std::deque<double> deque_record_ecg;
	std::deque<int> deque_period;
	int prev_peak_pos;
	double heart_interval;
	bool isRecording;

private:
	const char* physicalChannel;
	const char* sourceTerminal;

	TaskHandle _taskHandle;	
};

#endif // ECG_MONITORING_H_