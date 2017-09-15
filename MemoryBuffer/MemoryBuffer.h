#ifndef MEMORYBUFFER_H
#define MEMORYBUFFER_H

#include <QObject>

#include <iostream>
#include <thread>
#include <queue>

#include <Common/SyncObject.h>

class MainWindow;
class Configuration;
class QOperationTab;
class QDeviceControlTab;

class MemoryBuffer : public QObject
{
	Q_OBJECT

public:
    explicit MemoryBuffer(QObject *parent = 0);
    virtual ~MemoryBuffer();


public:
	// Memory allocation function (buffer for writing)
    void allocateWritingBuffer();

    // Data recording (transfer streaming data to writing buffer)
    bool startRecording();
    void stopRecording();

    // Data saving (save wrote data to hard disk)
    bool startSaving();

	// Circulation
	void circulation(int nFramesToCirc);

	// Buffer operation
	uint16_t* pop_front();
	void push_back(uint16_t* buffer);

private: // writing threading operation
	void write();

signals:
	void wroteSingleFrame(int);
	void finishedBufferAllocation();
	void finishedWritingThread(bool);

private:
	MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
	QOperationTab* m_pOperationTab;
	QDeviceControlTab* m_pDeviceControlTab;

public:
	bool m_bIsAllocatedWritingBuffer;
	bool m_bIsRecording;
	bool m_bIsSaved;
	int m_nRecordedFrames;

public:
	SyncObject<uint16_t> m_syncBuffering;

private:
    std::queue<uint16_t*> m_queueWritingBuffer; // writing buffer
	QString m_fileName;
};

#endif // MEMORYBUFFER_H
