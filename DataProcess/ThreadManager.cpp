
#include <DataProcess/ThreadManager.h>


ThreadManager::ThreadManager(const char* _threadID) :
    _running(false)
{
	memset(threadID, 0, MAX_LENGTH);
	memcpy(threadID, _threadID, strlen(_threadID));
}

ThreadManager::~ThreadManager()
{
    if (_thread.joinable())
    {
        _running = false;
        _thread.join();
    }
}


void ThreadManager::run()
{
    unsigned int frameIndex = 0;

    _running = true;
    while (_running)
        DidAcquireData(frameIndex++);
}

bool ThreadManager::startThreading()
{
    if (_thread.joinable())
    {
        char* msg = nullptr;
        sprintf(msg, "ERROR: %s thread is already running: ", threadID);
        dumpErrorSystem(-1, msg); //(::GetLastError(), msg);
        return false;
    }

    _thread = std::thread(&ThreadManager::run, this);

    printf("%s thread is started.\n", threadID);

    return true;
}

void ThreadManager::stopThreading()
{
    if (_thread.joinable())
    {
        DidStopData(); //_running = false;
        _thread.join();
    }

    printf("%s thread is finished normally.\n", threadID);
}


void ThreadManager::dumpErrorSystem(int res, const char* pPreamble)
{
    char* pErr = nullptr;
    char msg[MAX_LENGTH];
    memcpy(msg, pPreamble, strlen(pPreamble));

    sprintf(pErr, "Error code (%d)", res);
    strcat(msg, pErr);

    printf("%s\n", msg);
    SendStatusMessage(msg);
}

