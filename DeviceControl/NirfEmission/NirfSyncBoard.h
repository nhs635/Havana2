#ifndef NIRF_SYNC_BOARD_H_
#define NIRF_SYNC_BOARD_H_

#include <iostream>

typedef void *TaskHandle;

class NirfSyncBoard
{
public:
	NirfSyncBoard();
	~NirfSyncBoard();
	
	bool initialize();
	void start();
	void stop();

	uint32_t value;

private:		
	const char* lines;

	TaskHandle _taskHandle;
	void dumpError(int res, const char* pPreamble);
};

#endif // NIRF_SYNC_BOARD_H_