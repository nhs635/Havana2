
#include "MemoryBuffer.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QOperationTab.h>
#include <Havana2/QDeviceControlTab.h>

#ifdef ECG_TRIGGERING
#include <DeviceControl/ECGMonitoring/EcgMonitoring.h>
#endif

#include <QtCore/QFile>
#include <QtWidgets/QMessageBox.h>

#include <iostream>
#include <deque>
#include <chrono>
#include <mutex>
#include <condition_variable>


MemoryBuffer::MemoryBuffer(QObject *parent) :
    QObject(parent),
	m_bIsAllocatedWritingBuffer(false), 
	m_bIsRecording(false), m_bIsSaved(false),
	m_nRecordedFrames(0)
{
	m_pOperationTab = (QOperationTab*)parent;
	m_pMainWnd = m_pOperationTab->getMainWnd();
	m_pConfig = m_pMainWnd->m_pConfiguration;
}

MemoryBuffer::~MemoryBuffer()
{
	for (int i = 0; i < WRITING_BUFFER_SIZE; i++)
	{
		if (!m_queueWritingBuffer.empty())
		{
			uint16_t* buffer = m_queueWritingBuffer.front();
			if (buffer)
			{
				m_queueWritingBuffer.pop();
				delete[] buffer;
			}
		}
	}
	printf("Writing buffers are successfully disallocated.\n");
}


void MemoryBuffer::allocateWritingBuffer()
{
	if (!m_bIsAllocatedWritingBuffer)
	{
		int nFrameSize = m_pConfig->nFrameSize;
		for (int i = 0; i < WRITING_BUFFER_SIZE; i++)
		{
			uint16_t* buffer = new uint16_t[nFrameSize];
			memset(buffer, 0, nFrameSize * sizeof(uint16_t));
			m_queueWritingBuffer.push(buffer);
			printf("\rAllocating the writing buffers... [%d / %d]", i + 1, WRITING_BUFFER_SIZE);
		}
		printf("\nWriting buffers are successfully allocated. [Number of buffers: %d]\n", WRITING_BUFFER_SIZE);
		printf("Now, recording process is available!\n");

		m_bIsAllocatedWritingBuffer = true;
	}

	emit finishedBufferAllocation();
}

bool MemoryBuffer::startRecording()
{
	// Check if the previous recorded data is saved
	if (!m_bIsSaved && (m_nRecordedFrames != 0))
	{
		QMessageBox MsgBox;
		MsgBox.setWindowTitle("Question");
		MsgBox.setIcon(QMessageBox::Question);
		MsgBox.setText("The previous recorded data is not saved.\nWhat would you like to do with this data?");
		MsgBox.setStandardButtons(QMessageBox::Ignore | QMessageBox::Discard | QMessageBox::Cancel);
		MsgBox.setDefaultButton(QMessageBox::Cancel);

		int resp = MsgBox.exec();
		switch (resp)
		{
		case QMessageBox::Ignore:
			break;
		case QMessageBox::Discard:
			m_bIsSaved = true;
			m_nRecordedFrames = 0;
			return false;
		case QMessageBox::Cancel:
			return false;
		default:
			break;
		}
	}

	// Start Recording
	printf("Data recording is started.\n");
	m_nRecordedFrames = 0;

	m_pDeviceControlTab = m_pMainWnd->m_pDeviceControlTab;
#ifdef ECG_TRIGGERING
	// ECG Triggering	
	m_pDeviceControlTab->m_bRpeakNotDetected = false;
	if (m_pDeviceControlTab->isEcgTriggered()) // When ECG is triggered
	{
		// Thread for ECG triggering failing case (R peak is not detect within 2 seconds)
		std::thread halt([&]() {
			std::this_thread::sleep_for(std::chrono::seconds(2));			

			if (!m_bIsRecording && (m_nRecordedFrames == 0))
			{
				m_pDeviceControlTab->m_bRpeakNotDetected = true;

				std::unique_lock<std::mutex> mlock(m_pDeviceControlTab->m_mtxRpeakDetected);
				mlock.unlock();
				m_pDeviceControlTab->m_cvRpeakDetected.notify_one();

				m_pMainWnd->m_pOperationTab->setRecordingButton(false);
				printf("R peak is not detected. Recording process is halted.\n");				
			}
		});
		halt.detach();

		// Wait until R peak is detected
		std::unique_lock<std::mutex> mlock(m_pDeviceControlTab->m_mtxRpeakDetected);
		m_pDeviceControlTab->m_cvRpeakDetected.wait(mlock);
		if (m_pDeviceControlTab->m_bRpeakNotDetected)
			return false; // Cancel recording process
		m_pDeviceControlTab->setEcgRecording(true);
	}
#endif

#ifdef PULLBACK_DEVICE
	// Pullback
	if (m_pDeviceControlTab->isZaberStageEnabled())
		m_pDeviceControlTab->pullback();
#endif
	
	// Start Recording
	m_bIsRecording = true;
	m_bIsSaved = false;

	// Thread for buffering transfered data (memcpy)
	std::thread	thread_buffering_data = std::thread([&]() {
		printf("Data buffering thread is started.\n");
		int nFrameSize = m_pConfig->nFrameSize;
		while (1)
		{
			const uint16_t* frame = m_syncBuffering.Queue_sync.pop();
			if (frame)
			{
				uint16_t* buffer = m_queueWritingBuffer.front();

				m_queueWritingBuffer.pop();
				memcpy(buffer, frame, sizeof(uint16_t) * nFrameSize);
				m_queueWritingBuffer.push(buffer);
			}
			else
				break;
		}
		printf("Data copying thread is finished.\n");
	});
	thread_buffering_data.detach();

	return true;
}

void MemoryBuffer::stopRecording()
{
	// Stop recording
	m_bIsRecording = false;
	#ifdef ECG_TRIGGERING
		if (m_pDeviceControlTab->isEcgTriggered()) // When ECG is triggered
			m_pDeviceControlTab->setEcgRecording(false);
	#endif
		
	if (m_nRecordedFrames != 0) // Not allowed when 'discard'
	{
		// Push nullptr to Buffering Queue
		m_syncBuffering.Queue_sync.push(nullptr);

		// Status update
		m_pConfig->nFrames = m_nRecordedFrames;
		uint64_t total_size = (uint64_t)m_nRecordedFrames * (uint64_t)(m_pConfig->nFrameSize * sizeof(uint16_t)) / (uint64_t)1024;
		printf("Data recording is finished normally. \n(Recorded frames: %d frames (%1.3f GB)\n", m_nRecordedFrames, (double)total_size / 1024.0 / 1024.0);
	}
}

bool MemoryBuffer::startSaving()
{
	// Get path to write
	m_fileName = QFileDialog::getSaveFileName(nullptr, "Save As", "", "OCT raw data (*.data)");
	if (m_fileName == "") return false;
	
	// Start writing thread
	std::thread _thread = std::thread(&MemoryBuffer::write, this);
	_thread.detach();

	return true;
}

void MemoryBuffer::circulation(int nFramesToCirc)
{
	for (int i = 0; i < nFramesToCirc; i++)
	{
		uint16_t* buffer = m_queueWritingBuffer.front();
		m_queueWritingBuffer.pop();
		m_queueWritingBuffer.push(buffer);
	}
}

uint16_t* MemoryBuffer::pop_front()
{
	uint16_t* buffer = m_queueWritingBuffer.front();
	m_queueWritingBuffer.pop();

	return buffer;
}

void MemoryBuffer::push_back(uint16_t* buffer)
{
	m_queueWritingBuffer.push(buffer);
}


void MemoryBuffer::write()
{	
	qint64 res, samplesToWrite = m_pConfig->nFrameSize / 8;

	if (QFile::exists(m_fileName))
	{
		printf("Havana2 does not overwrite a recorded data.\n");
		emit finishedWritingThread(true);
		return;
	}

	// Move to start point
	uint16_t* buffer = nullptr;
	for (int i = 0; i < WRITING_BUFFER_SIZE - m_nRecordedFrames; i++)
	{
		buffer = m_queueWritingBuffer.front();
		m_queueWritingBuffer.pop();
		m_queueWritingBuffer.push(buffer);
	}	

	// Writing 
	QFile file(m_fileName);
	if (file.open(QIODevice::WriteOnly))
	{
		for (int i = 0; i < m_nRecordedFrames; i++)
		{
			buffer = m_queueWritingBuffer.front();
			m_queueWritingBuffer.pop();

			for (int j = 0; j < 8; j++)
			{
				res = file.write(reinterpret_cast<char*>(buffer + j * samplesToWrite), sizeof(uint16_t) * samplesToWrite);
				if (!(res == sizeof(uint16_t) * samplesToWrite))
				{
					printf("Error occurred while writing...\n");
					emit finishedWritingThread(true);
					return;
				}
			}
			emit wroteSingleFrame(i);
			//printf("\r%dth frame is wrote... [%3.2f%%]", i + 1, 100 * (double)(i + 1) / (double)m_nRecordedFrames);

			m_queueWritingBuffer.push(buffer);
		}
		file.close();
	}
	else
	{
		printf("Error occurred during writing process.\n");
		return;
	}
	m_bIsSaved = true;

	// Move files
	QString fileTitle, filePath;
	for (int i = 0; i < m_fileName.length(); i++)
	{		
		if (m_fileName.at(i) == QChar('.')) fileTitle = m_fileName.left(i);
		if (m_fileName.at(i) == QChar('/')) filePath = m_fileName.left(i);
	}	

	m_pConfig->setConfigFile("Havana2.ini");
	if (false == QFile::copy("Havana2.ini", fileTitle + ".ini"))
		printf("Error occurred while copying configuration data.\n");

	if (false == QFile::copy("calibration.dat", fileTitle + ".calibration"))
		printf("Error occurred while copying calibration data.\n");
	if (false == QFile::copy("bg.bin", fileTitle + ".background"))
		printf("Error occurred while copying background data.\n");
	if (false == QFile::copy("d1.bin", filePath + "/d1.bin"))
		printf("Error occurred while copying d1 data.\n");
	if (false == QFile::copy("d2.bin", filePath + "/d2.bin"))
		printf("Error occurred while copying d2 data.\n");
#ifdef OCT_FLIM
	if (false == QFile::copy("flim_mask.dat", fileTitle + ".flim_mask"))
		printf("Error occurred while copying flim_mask data.\n");
#endif
#ifdef ECG_TRIGGERING
	auto pDequeEcg = m_pDeviceControlTab->getRecordedEcg();
	if (pDequeEcg->size() != 0)
	{
		QFile ecgFile(fileTitle + ".ecg");
		if (ecgFile.open(QIODevice::WriteOnly))
		{
			for (int i = 0; i < pDequeEcg->size(); i++)
				ecgFile.write(reinterpret_cast<char*>(&pDequeEcg->at(i)), sizeof(double));
		}
		ecgFile.close();
		std::deque<double>().swap(*pDequeEcg);
	}
#endif
	if (false == QFile::copy("Lumen_IP_havana2.m", filePath + "/Lumen_IP_havana2.m"))
		printf("Error occurred while copying MATLAB processing data.\n");
	
	// Send a signal to notify this thread is finished
	emit finishedWritingThread(false);

	// Status update
	printf("\nData saving thread is finished normally. (Saved frames: %d frames)\n", m_nRecordedFrames);
	QByteArray temp = m_fileName.toLocal8Bit();
	char* filename = temp.data();
	printf("[%s]\n", filename);
}
