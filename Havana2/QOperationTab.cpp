
#include "QOperationTab.h"

#include <Havana2/Configuration.h>
#include <Havana2/MainWindow.h>
#include <Havana2/QStreamTab.h>
#include <Havana2/QResultTab.h>
#include <Havana2/QDeviceControlTab.h>

#include <Havana2/Dialog/DigitizerSetupDlg.h>

#include <DataProcess/ThreadManager.h>

#include <DataAcquisition/DataAcquisition.h>
#include <MemoryBuffer/MemoryBuffer.h>

#include <iostream>
#include <thread>


QOperationTab::QOperationTab(QWidget *parent) :
    QDialog(parent), m_pDigitizerSetupDlg(nullptr)
{
	// Set main window objects
    m_pMainWnd = (MainWindow*)parent;
	m_pConfig = m_pMainWnd->m_pConfiguration;

	// Create data acquisition object
	m_pDataAcquisition = new DataAcquisition(m_pConfig);

	// Create memory buffer object
	m_pMemoryBuffer = new MemoryBuffer(this);

    // Create widgets for acquisition / recording / saving operation
    m_pToggleButton_Acquisition = new QPushButton(this);
    m_pToggleButton_Acquisition->setCheckable(true);
	m_pToggleButton_Acquisition->setMinimumSize(80, 30);
    m_pToggleButton_Acquisition->setText("Start &Acquisition");

	m_pPushButton_DigitizerSetup = new QPushButton(this);
	m_pPushButton_DigitizerSetup->setMinimumSize(80, 30);
	m_pPushButton_DigitizerSetup->setText("&Digitizer Setup...");

    m_pToggleButton_Recording = new QPushButton(this);
    m_pToggleButton_Recording->setCheckable(true);
	m_pToggleButton_Recording->setMinimumSize(110, 30);
    m_pToggleButton_Recording->setText("Start &Recording");
	m_pToggleButton_Recording->setDisabled(true);

    m_pToggleButton_Saving = new QPushButton(this);
    m_pToggleButton_Saving->setCheckable(true);
	m_pToggleButton_Saving->setMinimumSize(110, 30);
    m_pToggleButton_Saving->setText("&Save Recorded Data");
	m_pToggleButton_Saving->setDisabled(true);

    // Create a progress bar (general purpose?)
    m_pProgressBar = new QProgressBar(this);
    m_pProgressBar->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
	m_pProgressBar->setFormat("");
    m_pProgressBar->setValue(0);

    // Set layout
    m_pVBoxLayout = new QVBoxLayout;
    m_pVBoxLayout->setSpacing(1);
	
	QHBoxLayout *pHBoxLayout1 = new QHBoxLayout;
	pHBoxLayout1->setSpacing(1);

	pHBoxLayout1->addWidget(m_pToggleButton_Acquisition);
	pHBoxLayout1->addWidget(m_pToggleButton_Recording); 

	QHBoxLayout *pHBoxLayout2 = new QHBoxLayout;
	pHBoxLayout2->setSpacing(1);

	pHBoxLayout2->addWidget(m_pPushButton_DigitizerSetup);
	pHBoxLayout2->addWidget(m_pToggleButton_Saving);

    m_pVBoxLayout->addItem(pHBoxLayout1);
	m_pVBoxLayout->addItem(pHBoxLayout2);
    m_pVBoxLayout->addWidget(m_pProgressBar);
    m_pVBoxLayout->addStretch(1);

    setLayout(m_pVBoxLayout);


	// Connect signal and slot
    connect(m_pToggleButton_Acquisition, SIGNAL(toggled(bool)), this, SLOT(operateDataAcquisition(bool)));
	connect(m_pToggleButton_Recording, SIGNAL(toggled(bool)), this, SLOT(operateDataRecording(bool)));
	connect(m_pToggleButton_Saving, SIGNAL(toggled(bool)), this, SLOT(operateDataSaving(bool)));
	connect(m_pPushButton_DigitizerSetup, SIGNAL(clicked(bool)), this, SLOT(createDigitizerSetupDlg()));
	connect(m_pMemoryBuffer, SIGNAL(finishedBufferAllocation()), this, SLOT(setAcqRecEnable()));
	connect(m_pMemoryBuffer, SIGNAL(finishedWritingThread(bool)), this, SLOT(setSaveButtonDefault(bool)));
	connect(m_pMemoryBuffer, SIGNAL(wroteSingleFrame(int)), m_pProgressBar, SLOT(setValue(int)));
}

QOperationTab::~QOperationTab()
{
    if (m_pMemoryBuffer) delete m_pMemoryBuffer;
	if (m_pDataAcquisition) delete m_pDataAcquisition;
}


void QOperationTab::changedTab(bool change)
{	
	if (change)
		m_pToggleButton_Acquisition->setChecked(!change);

	m_pToggleButton_Acquisition->setDisabled(change);
	m_pPushButton_DigitizerSetup->setDisabled(change);
}

void QOperationTab::operateDataAcquisition(bool toggled)
{
	QStreamTab* pStreamTab = m_pMainWnd->m_pStreamTab;
	if (toggled) // Start Data Acquisition
	{
		if ((m_pDataAcquisition->InitializeAcquistion()) 
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
			&& (m_pMainWnd->m_pDeviceControlTab->initializeNiDaqAnalogInput())
#endif
#endif
			)
		{
			// Start Thread Process
			pStreamTab->m_pThreadVisualization->startThreading();
			pStreamTab->m_pThreadCh1Process->startThreading();
			pStreamTab->m_pThreadCh2Process->startThreading();
			pStreamTab->m_pThreadDeinterleave->startThreading();
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
			m_pMainWnd->m_pDeviceControlTab->startNiDaqAnalogInput();
#endif
#endif
			// Start Data Acquisition
			if (m_pDataAcquisition->StartAcquisition())
			{
				m_pToggleButton_Acquisition->setText("Stop &Acquisition");								

				m_pToggleButton_Acquisition->setDisabled(true);
				m_pToggleButton_Recording->setDisabled(true);
				m_pPushButton_DigitizerSetup->setDisabled(true);

				if (m_pDigitizerSetupDlg != nullptr)
					m_pDigitizerSetupDlg->close();

				std::thread allocate_writing_buffer([&]() {			
					m_pMemoryBuffer->allocateWritingBuffer();
				});
				allocate_writing_buffer.detach();
			}
		}
		else
			m_pToggleButton_Acquisition->setChecked(false); // When initialization is failed...
	}
	else // Stop Data Acquisition
	{
		// Stop Thread Process
		m_pDataAcquisition->StopAcquisition();
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
		m_pMainWnd->m_pDeviceControlTab->stopNirfAcquisition();
#endif
#endif
		pStreamTab->m_pThreadDeinterleave->stopThreading();
		pStreamTab->m_pThreadCh1Process->stopThreading();
		pStreamTab->m_pThreadCh2Process->stopThreading();
		pStreamTab->m_pThreadVisualization->stopThreading();

		m_pToggleButton_Acquisition->setText("Start &Acquisition");
		m_pToggleButton_Recording->setDisabled(true);
		m_pPushButton_DigitizerSetup->setEnabled(true);
	}
}

void QOperationTab::operateDataRecording(bool toggled)
{
	if (toggled) // Start Data Recording
	{
		if (m_pMemoryBuffer->startRecording())
		{
			m_pToggleButton_Recording->setText("Stop &Recording");
			m_pToggleButton_Acquisition->setDisabled(true);
			m_pToggleButton_Saving->setDisabled(true);
		}
		else
		{
			m_pToggleButton_Recording->setChecked(false);

			if (m_pMemoryBuffer->m_bIsSaved) // When select "discard"
				m_pToggleButton_Saving->setDisabled(true);
		}
	}
	else // Stop DataRecording
	{
		m_pMemoryBuffer->stopRecording();

		m_pToggleButton_Recording->setText("Start &Recording");
		m_pToggleButton_Acquisition->setEnabled(true);
		m_pToggleButton_Saving->setEnabled(m_pMemoryBuffer->m_nRecordedFrames != 0);
		m_pMainWnd->m_pResultTab->getRadioInBuffer()->setEnabled(m_pMemoryBuffer->m_nRecordedFrames != 0);

		if (m_pMemoryBuffer->m_nRecordedFrames > 1)
			m_pProgressBar->setRange(0, m_pMemoryBuffer->m_nRecordedFrames - 1);
		else
			m_pProgressBar->setRange(0, 1);
		m_pProgressBar->setValue(0);
	}
}

void QOperationTab::operateDataSaving(bool toggled)
{
	if (toggled)
	{
		if (m_pMemoryBuffer->startSaving())
		{
			m_pToggleButton_Saving->setText("Saving...");
			m_pToggleButton_Recording->setDisabled(true);
			m_pToggleButton_Saving->setDisabled(true);
			m_pProgressBar->setFormat("Writing recorded data... %p%");
		}
		else
			m_pToggleButton_Saving->setChecked(false);
	}
}

void QOperationTab::createDigitizerSetupDlg()
{
	if (m_pDigitizerSetupDlg == nullptr)
	{
		m_pDigitizerSetupDlg = new DigitizerSetupDlg(this);
		connect(m_pDigitizerSetupDlg, SIGNAL(finished(int)), this, SLOT(deleteDigitizerSetupDlg()));
		m_pDigitizerSetupDlg->show();
	}
	m_pDigitizerSetupDlg->raise();
	m_pDigitizerSetupDlg->activateWindow();
}

void QOperationTab::deleteDigitizerSetupDlg()
{
	m_pDigitizerSetupDlg->deleteLater();
	m_pDigitizerSetupDlg = nullptr;
}

void QOperationTab::setAcqRecEnable()
{
	m_pToggleButton_Acquisition->setEnabled(true); 
	m_pToggleButton_Recording->setEnabled(true);
}


void QOperationTab::setSaveButtonDefault(bool error)
{
	m_pProgressBar->setFormat("");
	m_pToggleButton_Saving->setText("&Save Recorded Data");
	m_pToggleButton_Saving->setChecked(false);
	m_pToggleButton_Saving->setDisabled(!error);
	if (m_pToggleButton_Acquisition->isChecked())
		m_pToggleButton_Recording->setEnabled(true);
}
