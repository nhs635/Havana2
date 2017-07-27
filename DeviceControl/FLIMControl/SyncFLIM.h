#ifndef SYNC_FLIM_H_
#define SYNC_FLIM_H_

#include <iostream>

typedef void *TaskHandle;

class SyncFLIM
{
public:
	SyncFLIM();
	~SyncFLIM();
	
	int slow;
	const char* sourceTerminal;

	bool initialize();
	void start();
	void stop();
		
private:
	const char* counterChannel;

	TaskHandle _taskHandle;
	void dumpError(int res, const char* pPreamble);
};

#endif // SYNC_FLIM_H_