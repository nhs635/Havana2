                                                                                                                                        
#include "MainWindow.h"
#include "ui_MainWindow.h"

#include <Havana2/Configuration.h>

#include <Havana2/QOperationTab.h>
#include <Havana2/QDeviceControlTab.h>
#include <Havana2/QStreamTab.h>
#include <Havana2/QResultTab.h>

#include <Havana2/Dialog/DigitizerSetupDlg.h>
#include <Havana2/Dialog/OctCalibDlg.h>
#include <Havana2/Dialog/LongitudinalViewDlg.h>
#ifdef OCT_FLIM
#include <Havana2/Dialog/FlimCalibDlg.h>
#endif
#include <Havana2/Dialog/SaveResultDlg.h>
#ifdef OCT_FLIM
#include <Havana2/Dialog/PulseReviewDlg.h>
#endif
#ifdef OCT_NIRF
#include <Havana2/Dialog/NirfEmissionProfileDlg.h>
#include <Havana2/Dialog/NirfDistCompDlg.h>
#ifdef TWO_CHANNEL_NIRF
#include <Havana2/Dialog/NirfCrossTalkCompDlg.h>
#endif
#endif

#ifndef CUDA_ENABLED
#include <DataProcess/OCTProcess/OCTProcess.h>
#else
#include <CUDA/CudaOCTProcess.cuh>
#endif
#ifdef OCT_FLIM
#include <DataProcess/FLIMProcess/FLIMProcess.h>
#endif


#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
using namespace tbb;


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	
    // Initialize user interface
	QString windowTitle("Havana2 [%1] v%2");
#ifdef OCT_FLIM
	setWindowTitle(windowTitle.arg("OCT-FLIM").arg(VERSION));
#elif defined (STANDALONE_OCT) 	
#ifdef ECG_TRIGGERING
	setWindowTitle(windowTitle.arg("ECG-Triggered UltraHigh-Speed OCT").arg(VERSION));
#else
#ifdef OCT_NIRF
	setWindowTitle(windowTitle.arg("OCT-NIRF").arg(VERSION));
#else
	setWindowTitle(windowTitle.arg("Standalone OCT").arg(VERSION));
#endif
#endif
#endif	
	// Create configuration object
	m_pConfiguration = new Configuration;
	m_pConfiguration->getConfigFile("Havana2.ini");

	//m_pConfiguration->nScans = N_SCANS;
	m_pConfiguration->fnScans = m_pConfiguration->nScans * 4;
	m_pConfiguration->nScansFFT = NEAR_2_POWER((double)m_pConfiguration->nScans);
	m_pConfiguration->n2ScansFFT = m_pConfiguration->nScansFFT / 2;
	m_pConfiguration->nFrameSize = m_pConfiguration->nChannels * m_pConfiguration->nScans * m_pConfiguration->nAlines;

	// Set timer for renew configuration 
	m_pTimer = new QTimer(this);
	m_pTimer->start(5 * 60 * 1000); // renew per 5 min
	m_pTimerSync = new QTimer(this);
	m_pTimerSync->start(1000); // renew per 1 sec

    // Create tabs objects
    m_pOperationTab = new QOperationTab(this);
	m_pDeviceControlTab = new QDeviceControlTab(this);
    m_pStreamTab = new QStreamTab(this);
	m_pResultTab = new QResultTab(this);

    // Create group boxes and tab widgets
    m_pGroupBox_OperationTab = new QGroupBox("Operation Tab");
    m_pGroupBox_OperationTab->setLayout(m_pOperationTab->getLayout());
    m_pGroupBox_OperationTab->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);

    m_pGroupBox_DeviceControlTab = new QGroupBox("Device Control Tab");
    m_pGroupBox_DeviceControlTab->setLayout(m_pDeviceControlTab->getLayout());
    m_pGroupBox_DeviceControlTab->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

    m_pTabWidget = new QTabWidget(this);
    m_pTabWidget->addTab(m_pStreamTab, tr("&Data Streaming"));
    m_pTabWidget->addTab(m_pResultTab, tr("&Post Processing"));
	
	// Create status bar
	QLabel *pStatusLabel_Temp1 = new QLabel(this);
	m_pStatusLabel_ImagePos = new QLabel(QString("(%1, %2)").arg(0000, 4).arg(0000, 4), this);
	
	size_t di_bfn = m_pStreamTab->getDeinterleavingBufferQueueSize();
	size_t p1_bfn = m_pStreamTab->getCh1ProcessingBufferQueueSize();
	size_t p2_bfn = m_pStreamTab->getCh2ProcessingBufferQueueSize();
	size_t v1_bfn = m_pStreamTab->getCh1VisualizationBufferQueueSize();
	size_t v2_bfn = m_pStreamTab->getCh2VisualizationBufferQueueSize();
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
	size_t vn_bfn = m_pStreamTab->getNirfVisualizationBufferQueueSize();
	m_pStatusLabel_SyncStatus = new QLabel(QString("DI bufn: %1 / P1 bufn: %2 / P2 bufn: %3 / V1 bufn: %4 / V2 bufn: %5 / VN bufn: %6")
		.arg(di_bfn, 3).arg(p1_bfn, 3).arg(p2_bfn, 3).arg(v1_bfn, 3).arg(v2_bfn, 3).arg(vn_bfn, 3), this);
#else
	m_pStatusLabel_SyncStatus = new QLabel(QString("DI bufn: %1 / P1 bufn: %2 / P2 bufn: %3 / V1 bufn: %4 / V2 bufn: %5")
		.arg(di_bfn, 3).arg(p1_bfn, 3).arg(p2_bfn, 3).arg(v1_bfn, 3).arg(v2_bfn, 3), this);
#endif
#else
	m_pStatusLabel_SyncStatus = new QLabel(QString("DI bufn: %1 / P1 bufn: %2 / P2 bufn: %3 / V1 bufn: %4 / V2 bufn: %5")
		.arg(di_bfn, 3).arg(p1_bfn, 3).arg(p2_bfn, 3).arg(v1_bfn, 3).arg(v2_bfn, 3), this);
#endif

	pStatusLabel_Temp1->setFrameStyle(QFrame::Panel | QFrame::Sunken);
	m_pStatusLabel_ImagePos->setFrameStyle(QFrame::Panel | QFrame::Sunken);
	m_pStatusLabel_SyncStatus->setFrameStyle(QFrame::Panel | QFrame::Sunken);

	// then add the widget to the status bar
	statusBar()->addPermanentWidget(pStatusLabel_Temp1, 6);
	statusBar()->addPermanentWidget(m_pStatusLabel_ImagePos, 1);
	statusBar()->addPermanentWidget(m_pStatusLabel_SyncStatus, 2);

    // Set layout
    m_pGridLayout = new QGridLayout;

    m_pGridLayout->addWidget(m_pGroupBox_OperationTab, 0, 0);
    m_pGridLayout->addWidget(m_pGroupBox_DeviceControlTab, 1, 0);
    m_pGridLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding), 2, 0);
    m_pGridLayout->addWidget(m_pTabWidget, 0, 1, 3, 1);

    ui->centralWidget->setLayout(m_pGridLayout);

	// Connect signal and slot
	connect(m_pTimer, SIGNAL(timeout()), this, SLOT(onTimer()));
	connect(m_pTimerSync, SIGNAL(timeout()), this, SLOT(onTimerSync()));
	connect(m_pTabWidget, SIGNAL(currentChanged(int)), this, SLOT(changedTab(int)));
	
	/// Connect all devices
	///m_pDeviceControlTab->initiateAllDevices();
}

MainWindow::~MainWindow()
{
	m_pTimer->stop();
	m_pTimerSync->stop();

	if (m_pConfiguration)
	{
		m_pConfiguration->setConfigFile("Havana2.ini");
		delete m_pConfiguration;
	}
	
    delete ui;
}

void MainWindow::closeEvent(QCloseEvent *e)
{	
	if (m_pOperationTab->isSavingButtonToggled())
	{
		e->ignore();
		QMessageBox::critical(this, "Exit", "Please wait until the saving process is finished.");

		return;
	}
	if (m_pOperationTab->isAcquisitionButtonToggled())
		m_pOperationTab->getAcquisitionButton()->setChecked(false);
	
	m_pDeviceControlTab->terminateAllDevices();

	if (m_pOperationTab->getDigitizerSetupDlg())
		m_pOperationTab->getDigitizerSetupDlg()->close();
	if (m_pStreamTab->getOctCalibDlg())
		m_pStreamTab->getOctCalibDlg()->close();
#ifdef OCT_FLIM
	if (m_pStreamTab->getFlimCalibDlg())
		m_pStreamTab->getFlimCalibDlg()->close();
	if (m_pResultTab->getPulseReviewDlg())
		m_pResultTab->getPulseReviewDlg()->close();
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	if (m_pStreamTab->getNirfEmissionProfileDlg())
		m_pStreamTab->getNirfEmissionProfileDlg()->close();
#endif
#endif
	if (m_pResultTab->getSaveResultDlg())
		m_pResultTab->getSaveResultDlg()->close();
	if (m_pResultTab->getLongitudinalViewDlg())
		m_pResultTab->getLongitudinalViewDlg()->close();
#ifdef OCT_NIRF
    if (m_pResultTab->getNirfEmissionProfileDlg())
        m_pResultTab->getNirfEmissionProfileDlg()->close();
	if (m_pResultTab->getNirfDistCompDlg())
		m_pResultTab->getNirfDistCompDlg()->close();
#ifdef TWO_CHANNEL_NIRF
	if (m_pResultTab->getNirfCrossTalkCompDlg())
		m_pResultTab->getNirfCrossTalkCompDlg()->close();
#endif
#endif
	
	e->accept();
}


void MainWindow::onTimer()
{
	printf("Renewed configuration data!\n");
#ifdef OCT_FLIM
	m_pStreamTab->m_pFLIM->saveMaskData();
	m_pStreamTab->m_pOCT->saveCalibration();
#elif defined (STANDALONE_OCT)
#ifndef K_CLOCKING
	m_pStreamTab->m_pOCT1->saveCalibration();
#endif
#endif
	m_pConfiguration->setConfigFile("Havana2.ini");
}

void MainWindow::onTimerSync()
{
	size_t di_bfn = m_pStreamTab->getDeinterleavingBufferQueueSize();
	size_t p1_bfn = m_pStreamTab->getCh1ProcessingBufferQueueSize();
	size_t p2_bfn = m_pStreamTab->getCh2ProcessingBufferQueueSize();
	size_t v1_bfn = m_pStreamTab->getCh1VisualizationBufferQueueSize();
	size_t v2_bfn = m_pStreamTab->getCh2VisualizationBufferQueueSize();
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
	size_t vn_bfn = m_pStreamTab->getNirfVisualizationBufferQueueSize();
	m_pStatusLabel_SyncStatus->setText(QString("DI bufn: %1 / P1 bufn: %2 / P2 bufn: %3 / V1 bufn: %4 / V2 bufn: %5 / VN bufn: %6")
		.arg(di_bfn, 3).arg(p1_bfn, 3).arg(p2_bfn, 3).arg(v1_bfn, 3).arg(v2_bfn, 3).arg(vn_bfn, 3));
#else
	m_pStatusLabel_SyncStatus->setText(QString("DI bufn: %1 / P1 bufn: %2 / P2 bufn: %3 / V1 bufn: %4 / V2 bufn: %5")
		.arg(di_bfn, 3).arg(p1_bfn, 3).arg(p2_bfn, 3).arg(v1_bfn, 3).arg(v2_bfn, 3));
#endif
#else
	m_pStatusLabel_SyncStatus->setText(QString("DI bufn: %1 / P1 bufn: %2 / P2 bufn: %3 / V1 bufn: %4 / V2 bufn: %5")
		.arg(di_bfn, 3).arg(p1_bfn, 3).arg(p2_bfn, 3).arg(v1_bfn, 3).arg(v2_bfn, 3));
#endif
}

void MainWindow::changedTab(int index)
{
	m_pOperationTab->changedTab(index == 1);

	if (index == 1)
	{
		m_pResultTab->setWidgetsText();

		if (m_pOperationTab->getDigitizerSetupDlg())
			m_pOperationTab->getDigitizerSetupDlg()->close();
		if (m_pStreamTab->getOctCalibDlg())
			m_pStreamTab->getOctCalibDlg()->close();
#ifdef OCT_FLIM
		if (m_pStreamTab->getFlimCalibDlg())
			m_pStreamTab->getFlimCalibDlg()->close();

		if (m_pDeviceControlTab->getEnablePmtGainControl()->isChecked())
			m_pDeviceControlTab->getEnablePmtGainControl()->setChecked(false);
		if (m_pDeviceControlTab->getEnableFlimLaserSyncControl()->isChecked())
			if (!m_pDeviceControlTab->getFlimAsyncMode()->isChecked())
				m_pDeviceControlTab->getEnableFlimLaserSyncControl()->setChecked(false);
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
		if (m_pStreamTab->getNirfEmissionProfileDlg())
			m_pStreamTab->getNirfEmissionProfileDlg()->close();
#ifdef PROGRAMMATIC_GAIN_CONTROL
		if (m_pDeviceControlTab->getEnablePmtGainControl()->isChecked())
			m_pDeviceControlTab->getEnablePmtGainControl()->setChecked(false);
#endif
#endif
#endif

#ifdef GALVANO_MIRROR
		if (m_pDeviceControlTab->getEnableGalvanoMirrorControl()->isChecked())
			m_pDeviceControlTab->getEnableGalvanoMirrorControl()->setChecked(false);

		//m_pDeviceControlTab->setScrollBarRange(m_pResultTab->m_pCirc ? m_pResultTab->m_pCirc->alines : m_pConfiguration->nAlines);
		m_pDeviceControlTab->setScrollBarValue(m_pDeviceControlTab->getScrollBarValue());
		m_pDeviceControlTab->setScrollBarEnabled(true);
#endif
	}
	else
	{
		m_pStreamTab->setWidgetsText();

		if (m_pResultTab->getSaveResultDlg())
			m_pResultTab->getSaveResultDlg()->close();
		if (m_pResultTab->getLongitudinalViewDlg())
			m_pResultTab->getLongitudinalViewDlg()->close();
#ifdef OCT_FLIM
		if (m_pResultTab->getPulseReviewDlg())
			m_pResultTab->getPulseReviewDlg()->close();
#endif
#ifdef OCT_NIRF
        if (m_pResultTab->getNirfEmissionProfileDlg())
            m_pResultTab->getNirfEmissionProfileDlg()->close();
        if (m_pResultTab->getNirfDistCompDlg())
            m_pResultTab->getNirfDistCompDlg()->close();
#ifdef TWO_CHANNEL_NIRF
		if (m_pResultTab->getNirfCrossTalkCompDlg())
			m_pResultTab->getNirfCrossTalkCompDlg()->close();
#endif
#endif

#ifdef GALVANO_MIRROR
		if (!m_pDeviceControlTab->getEnableGalvanoMirrorControl()->isChecked())
			m_pDeviceControlTab->setScrollBarEnabled(false);
		m_pDeviceControlTab->setScrollBarRange(m_pConfiguration->nAlines);
		m_pDeviceControlTab->setScrollBarValue(m_pDeviceControlTab->getScrollBarValue());
#endif
	}
}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
