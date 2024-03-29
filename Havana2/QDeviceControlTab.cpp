
#include "QDeviceControlTab.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QOperationTab.h>
#include <Havana2/QStreamTab.h>
#include <Havana2/QResultTab.h>

#ifdef AXSUN_OCT_LASER
#include <DeviceControl/AxsunControl/AxsunControl.h>
#endif
#ifdef ECG_TRIGGERING
#include <Havana2/Viewer/QEcgScope.h>
#if NI_ENABLE
#include <DeviceControl/ECGMonitoring/EcgMonitoringTrigger.h>
#include <DeviceControl/ECGMonitoring/EcgMonitoring.h>
#include <DeviceControl/ECGMonitoring/Voltage800RPS.h>
#endif
#endif
#ifdef OCT_FLIM
#if NI_ENABLE
#include <DeviceControl/FLIMControl/PmtGainControl.h>
#include <DeviceControl/FLIMControl/SyncFLIM.h>
#endif
#include <DeviceControl/FLIMControl/ElforlightLaser.h>
#endif
#ifdef OCT_NIRF
#if NI_ENABLE
#ifndef ALAZAR_NIRF_ACQUISITION
#include <DeviceControl/NirfEmission/NirfEmissionTrigger.h>
#include <DeviceControl/NirfEmission/NirfEmission.h>
#endif
#ifdef TWO_CHANNEL_NIRF
#include <DeviceControl/NirfModulation/NirfModulation.h>
#endif
#ifdef PROGRAMMATIC_GAIN_CONTROL
#include <DeviceControl/FLIMControl/PmtGainControl.h>
#endif
#include <Havana2/Dialog/NirfEmissionProfileDlg.h>
#endif
#endif
#ifdef GALVANO_MIRROR
#if NI_ENABLE
#include <DeviceControl/GalvoScan/GalvoScan.h>
#endif
#endif
#ifdef PULLBACK_DEVICE
#ifndef ZABER_NEW_STAGE
#include <DeviceControl/ZaberStage/ZaberStage.h>
#else
#include <DeviceControl/ZaberStage/ZaberStage2.h>
#endif
#include <DeviceControl/FaulhaberMotor/FaulhaberMotor.h>
#endif

#include <iostream>
#include <deque>
#include <thread>
#include <chrono>


QDeviceControlTab::QDeviceControlTab(QWidget *parent) :
    QDialog(parent)
#ifdef PULLBACK_DEVICE
	,m_pFaulhaberMotor(nullptr), m_pZaberMonitorTimer(nullptr)
#endif
{
	// Set main window objects
	m_pMainWnd = (MainWindow*)parent;
	m_pConfig = m_pMainWnd->m_pConfiguration;
	m_pOperationTab = m_pMainWnd->m_pOperationTab;


    // Create layout
    m_pVBoxLayout = new QVBoxLayout;
    m_pVBoxLayout->setSpacing(0);

#ifdef OCT_FLIM
	m_pVBoxLayout_FlimControl = new QVBoxLayout;
	m_pVBoxLayout_FlimControl->setSpacing(5);
	m_pGroupBox_FlimControl = new QGroupBox;
#endif

#ifdef AXSUN_OCT_LASER
    createAxsunOCTLaserControl();
#endif
#ifdef ECG_TRIGGERING
	createEcgModuleControl();
#endif
#ifdef OCT_FLIM
	createPmtGainControl();
	createFlimLaserSyncControl();
	createFlimLaserPowerControl();
#endif
#ifdef OCT_NIRF
	createNirfAcquisitionControl();
#endif
#ifdef GALVANO_MIRROR
    createGalvanoMirrorControl();
#endif
#ifdef PULLBACK_DEVICE
    createZaberStageControl();
    createFaulhaberMotorControl();
#endif
	
    // Set layout
    setLayout(m_pVBoxLayout);	
}

QDeviceControlTab::~QDeviceControlTab()
{
}

void QDeviceControlTab::closeEvent(QCloseEvent* e)
{
	terminateAllDevices();
    e->accept();
}


#ifdef AXSUN_OCT_LASER
void QDeviceControlTab::createAxsunOCTLaserControl()
{
    // Create widgets for Axsun OCT laser control
    QGroupBox *pGroupBox_AxsunOCTLaserControl = new QGroupBox;
    QGridLayout *pGridLayout_AxsunOCTLaserControl = new QGridLayout;
    pGridLayout_AxsunOCTLaserControl->setSpacing(3);

    m_pCheckBox_AxsunOCTLaserControl = new QCheckBox(pGroupBox_AxsunOCTLaserControl);
    m_pCheckBox_AxsunOCTLaserControl->setText("Enable Axsun OCT Laser Control");
	m_pCheckBox_AxsunOCTLaserControl->setFixedWidth(180);
	
	// Create widgets for Axsun OCT laser emission turning on/off
	m_pToggleButton_OCTLaserSource = new QPushButton(pGroupBox_AxsunOCTLaserControl);
	m_pToggleButton_OCTLaserSource->setCheckable(true);
	m_pToggleButton_OCTLaserSource->setFixedWidth(40);
	m_pToggleButton_OCTLaserSource->setText("On");
	m_pToggleButton_OCTLaserSource->setDisabled(true);

	m_pLabel_OCTLaserSource = new QLabel("OCT Laser Emission ", pGroupBox_AxsunOCTLaserControl);
	m_pLabel_OCTLaserSource->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
	m_pLabel_OCTLaserSource->setBuddy(m_pToggleButton_OCTLaserSource);
	m_pLabel_OCTLaserSource->setDisabled(true);

#ifdef AXSUN_VDL_K_CLOCK_DELAY
	m_pSpinBox_VDLLength = new QMySpinBox(this);
	m_pSpinBox_VDLLength->setFixedWidth(55);
	m_pSpinBox_VDLLength->setRange(0.00, 15.00);
	m_pSpinBox_VDLLength->setSingleStep(0.05);
	m_pSpinBox_VDLLength->setValue(m_pConfig->axsunVDLLength);
	m_pSpinBox_VDLLength->setDecimals(2);
	m_pSpinBox_VDLLength->setAlignment(Qt::AlignCenter);
	m_pSpinBox_VDLLength->setDisabled(true);

	m_pLabel_VDLLength = new QLabel("VDL (mm) ", this);
	m_pLabel_VDLLength->setBuddy(m_pSpinBox_VDLLength);
	m_pLabel_VDLLength->setDisabled(true);

	m_pPushButton_VDLHome = new QPushButton(this);
	m_pPushButton_VDLHome->setFixedWidth(40);
	m_pPushButton_VDLHome->setText("Home");
	m_pPushButton_VDLHome->setDisabled(true);

	m_pSpinBox_kClockDelay = new QSpinBox(this);
	m_pSpinBox_kClockDelay->setFixedWidth(40);
	m_pSpinBox_kClockDelay->setRange(0, 63);
	m_pSpinBox_kClockDelay->setSingleStep(1);
	m_pSpinBox_kClockDelay->setValue(m_pConfig->axsunkClockDelay);	
	m_pSpinBox_kClockDelay->setAlignment(Qt::AlignCenter);
	m_pSpinBox_kClockDelay->setDisabled(true);

	m_pLabel_kClockDelay = new QLabel("k Clock Delay ", this);
	m_pLabel_kClockDelay->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
	m_pLabel_kClockDelay->setBuddy(m_pSpinBox_kClockDelay);
	m_pLabel_kClockDelay->setDisabled(true);
#endif


    pGridLayout_AxsunOCTLaserControl->addWidget(m_pCheckBox_AxsunOCTLaserControl, 0, 0, 1, 4);
	pGridLayout_AxsunOCTLaserControl->addWidget(m_pLabel_OCTLaserSource, 1, 1, 1, 2);
	pGridLayout_AxsunOCTLaserControl->addWidget(m_pToggleButton_OCTLaserSource, 1, 3);
#ifdef AXSUN_VDL_K_CLOCK_DELAY
	pGridLayout_AxsunOCTLaserControl->addWidget(m_pLabel_VDLLength, 2, 1);
	pGridLayout_AxsunOCTLaserControl->addWidget(m_pSpinBox_VDLLength, 2, 2);
	pGridLayout_AxsunOCTLaserControl->addWidget(m_pPushButton_VDLHome, 2, 3);
	pGridLayout_AxsunOCTLaserControl->addWidget(m_pLabel_kClockDelay, 3, 1, 1, 2);
	pGridLayout_AxsunOCTLaserControl->addWidget(m_pSpinBox_kClockDelay, 3, 3);
#endif


    pGroupBox_AxsunOCTLaserControl->setLayout(pGridLayout_AxsunOCTLaserControl);
    pGroupBox_AxsunOCTLaserControl->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");
    m_pVBoxLayout->addWidget(pGroupBox_AxsunOCTLaserControl);
		
    // Connect signal and slot
    connect(m_pCheckBox_AxsunOCTLaserControl, SIGNAL(toggled(bool)), this, SLOT(enableAxsunOCTLaserControl(bool)));
	connect(m_pToggleButton_OCTLaserSource, SIGNAL(toggled(bool)), this, SLOT(setLightSource(bool)));
#ifdef AXSUN_VDL_K_CLOCK_DELAY
	connect(m_pSpinBox_VDLLength, SIGNAL(valueChanged(double)), this, SLOT(setVDLLength(double)));
	connect(m_pPushButton_VDLHome, SIGNAL(clicked(bool)), this, SLOT(setVDLHome()));
	connect(m_pSpinBox_kClockDelay, SIGNAL(valueChanged(int)), this, SLOT(setkClockDelay(int)));
#endif
}
#endif

#ifdef ECG_TRIGGERING
void QDeviceControlTab::createEcgModuleControl()
{
	// Create widgets for ECG module control
	QGroupBox *pGroupBox_EcgModuleControl = new QGroupBox;
	QVBoxLayout *pVBoxLayout_EcgModuleControl = new QVBoxLayout;
	pVBoxLayout_EcgModuleControl->setSpacing(5);

	// Create widgets for ECG triggering
	m_pCheckBox_EcgModuleControl = new QCheckBox(pGroupBox_EcgModuleControl);
	m_pCheckBox_EcgModuleControl->setText("Enable ECG Module Control");	
	m_pCheckBox_EcgModuleControl->setFixedWidth(200);

	m_pLabel_EcgBitPerMinute = new QLabel("Heart Rate: 000.0 bpm", pGroupBox_EcgModuleControl);
	m_pLabel_EcgBitPerMinute->setFixedWidth(120);
	m_pLabel_EcgBitPerMinute->setAlignment(Qt::AlignRight);
	m_pLabel_EcgBitPerMinute->setEnabled(false);

	m_pToggledButton_EcgTriggering = new QPushButton(pGroupBox_EcgModuleControl);
	m_pToggledButton_EcgTriggering->setCheckable(true);
	m_pToggledButton_EcgTriggering->setText("ECG Triggering On");
	m_pToggledButton_EcgTriggering->setEnabled(false);

	m_pLabel_EcgDelayRate = new QLabel("Delay Rate  ", pGroupBox_EcgModuleControl);
	m_pLabel_EcgDelayRate->setEnabled(false);

	m_pDoubleSpinBox_EcgDelayRate = new QDoubleSpinBox(pGroupBox_EcgModuleControl);
	m_pDoubleSpinBox_EcgDelayRate->setFixedWidth(50);
	m_pDoubleSpinBox_EcgDelayRate->setRange(0.0, 1.0);
	m_pDoubleSpinBox_EcgDelayRate->setSingleStep(0.01);
	m_pDoubleSpinBox_EcgDelayRate->setValue((double)m_pConfig->ecgDelayRate);
	m_pDoubleSpinBox_EcgDelayRate->setDecimals(2);
	m_pDoubleSpinBox_EcgDelayRate->setAlignment(Qt::AlignCenter);
	m_pDoubleSpinBox_EcgDelayRate->setDisabled(true);
	
	m_pEcgScope = new QEcgScope({ 0, N_VIS_SAMPS_ECG }, { -ECG_VOLTAGE, ECG_VOLTAGE }, 2, 2, 0.001, 1, 0, 0, "sec", "V");
	m_pEcgScope->setFixedHeight(120);
	m_pEcgScope->setEnabled(false);

	// Create widgets for 800 rps Motor Control
	m_pCheckBox_Voltage800Rps = new QCheckBox(pGroupBox_EcgModuleControl);
	m_pCheckBox_Voltage800Rps->setText("Enable Voltage Control for 800 rps");
	
	m_pToggledButton_Voltage800Rps = new QPushButton(pGroupBox_EcgModuleControl);
	m_pToggledButton_Voltage800Rps->setCheckable(true);
	m_pToggledButton_Voltage800Rps->setText("Voltage On");
	m_pToggledButton_Voltage800Rps->setDisabled(true);

	m_pDoubleSpinBox_Voltage800Rps = new QDoubleSpinBox(pGroupBox_EcgModuleControl);
	m_pDoubleSpinBox_Voltage800Rps->setFixedWidth(50);
	m_pDoubleSpinBox_Voltage800Rps->setRange(0.0, 10.0);
	m_pDoubleSpinBox_Voltage800Rps->setSingleStep(0.5);
	m_pDoubleSpinBox_Voltage800Rps->setValue(0.0);
	m_pDoubleSpinBox_Voltage800Rps->setDecimals(1);
	m_pDoubleSpinBox_Voltage800Rps->setAlignment(Qt::AlignCenter);
	m_pDoubleSpinBox_Voltage800Rps->setDisabled(true);

	m_pLabel_Voltage800Rps = new QLabel("V", pGroupBox_EcgModuleControl);
	m_pLabel_Voltage800Rps->setDisabled(true);

	// Set Layout	
	QGridLayout *pGridLayout_Ecg = new QGridLayout;
	pGridLayout_Ecg->setSpacing(3);

	pGridLayout_Ecg->addWidget(m_pCheckBox_EcgModuleControl, 0, 0, 1, 3);

	QVBoxLayout *pVBoxLayout_Delay = new QVBoxLayout;
	pVBoxLayout_Delay->setSpacing(0);
	pVBoxLayout_Delay->addWidget(m_pLabel_EcgDelayRate);
	pVBoxLayout_Delay->addWidget(m_pDoubleSpinBox_EcgDelayRate);

	QVBoxLayout *pVBoxLayout_Triggering = new QVBoxLayout;
	pVBoxLayout_Triggering->setSpacing(3);
	pVBoxLayout_Triggering->addWidget(m_pToggledButton_EcgTriggering);
	pVBoxLayout_Triggering->addWidget(m_pLabel_EcgBitPerMinute);

	pGridLayout_Ecg->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0, 2, 1);
	pGridLayout_Ecg->addItem(pVBoxLayout_Delay, 1, 1, 2, 1);
	pGridLayout_Ecg->addItem(pVBoxLayout_Triggering, 1, 2, 2, 1);
	
	pGridLayout_Ecg->addWidget(m_pEcgScope, 3, 0, 1, 3);
		
	QGridLayout *pGridLayout_Voltage800Rps = new QGridLayout;
	pGridLayout_Voltage800Rps->setSpacing(3);

	pGridLayout_Voltage800Rps->addWidget(m_pCheckBox_Voltage800Rps, 0, 0, 1, 4);

	pGridLayout_Voltage800Rps->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
	pGridLayout_Voltage800Rps->addWidget(m_pToggledButton_Voltage800Rps, 1, 1);
	pGridLayout_Voltage800Rps->addWidget(m_pDoubleSpinBox_Voltage800Rps, 1, 2);
	pGridLayout_Voltage800Rps->addWidget(m_pLabel_Voltage800Rps, 1, 3);

	pVBoxLayout_EcgModuleControl->addItem(pGridLayout_Ecg);
	pVBoxLayout_EcgModuleControl->addItem(pGridLayout_Voltage800Rps);

	pGroupBox_EcgModuleControl->setLayout(pVBoxLayout_EcgModuleControl);
	pGroupBox_EcgModuleControl->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");
	m_pVBoxLayout->addWidget(pGroupBox_EcgModuleControl);

	// Connect signal and slot
	connect(m_pCheckBox_EcgModuleControl, SIGNAL(toggled(bool)), this, SLOT(enableEcgModuleControl(bool)));
	connect(m_pToggledButton_EcgTriggering, SIGNAL(toggled(bool)), this, SLOT(toggleEcgTriggerButton(bool)));
	connect(m_pDoubleSpinBox_EcgDelayRate, SIGNAL(valueChanged(double)), this, SLOT(changeEcgDelayRate(double)));
	connect(m_pCheckBox_Voltage800Rps, SIGNAL(toggled(bool)), this, SLOT(enable800rpsVoltControl(bool)));
	connect(m_pToggledButton_Voltage800Rps, SIGNAL(toggled(bool)), this, SLOT(toggle800rpsVoltButton(bool)));
	connect(m_pDoubleSpinBox_Voltage800Rps, SIGNAL(valueChanged(double)), this, SLOT(set800rpsVoltage(double)));
}
#endif

#ifdef OCT_FLIM
void QDeviceControlTab::createPmtGainControl()
{
    // Create widgets for PMT gain control
    QHBoxLayout *pHBoxLayout_PmtGainControl = new QHBoxLayout;
    pHBoxLayout_PmtGainControl->setSpacing(3);

    m_pCheckBox_PmtGainControl = new QCheckBox(m_pGroupBox_FlimControl);
    m_pCheckBox_PmtGainControl->setText("Enable PMT Gain Control");
	m_pCheckBox_PmtGainControl->setFixedWidth(140);

    m_pLineEdit_PmtGainVoltage = new QLineEdit(m_pGroupBox_FlimControl);
    m_pLineEdit_PmtGainVoltage->setFixedWidth(35);
    m_pLineEdit_PmtGainVoltage->setText(QString::number(m_pConfig->pmtGainVoltage, 'f', 2));
	m_pLineEdit_PmtGainVoltage->setAlignment(Qt::AlignCenter);
    m_pLineEdit_PmtGainVoltage->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	m_pLineEdit_PmtGainVoltage->setEnabled(true);

    m_pLabel_PmtGainVoltage = new QLabel("V", m_pGroupBox_FlimControl);
    m_pLabel_PmtGainVoltage->setBuddy(m_pLineEdit_PmtGainVoltage);
	m_pLabel_PmtGainVoltage->setEnabled(true);

    pHBoxLayout_PmtGainControl->addWidget(m_pCheckBox_PmtGainControl);
    pHBoxLayout_PmtGainControl->addWidget(m_pLineEdit_PmtGainVoltage);
    pHBoxLayout_PmtGainControl->addWidget(m_pLabel_PmtGainVoltage);

	m_pVBoxLayout_FlimControl->addItem(pHBoxLayout_PmtGainControl);

	// Connect signal and slot
	connect(m_pCheckBox_PmtGainControl, SIGNAL(toggled(bool)), this, SLOT(enablePmtGainControl(bool)));
	connect(m_pLineEdit_PmtGainVoltage, SIGNAL(textChanged(const QString &)), this, SLOT(changePmtGainVoltage(const QString &)));
}

void QDeviceControlTab::createFlimLaserSyncControl()
{
    // Create widgets for FLIM laser control
    QGridLayout *pGridLayout_FlimLaserSyncControl = new QGridLayout;
	pGridLayout_FlimLaserSyncControl->setSpacing(3);

	m_pCheckBox_FlimLaserSyncControl = new QCheckBox(m_pGroupBox_FlimControl);
	m_pCheckBox_FlimLaserSyncControl->setText("Enable FLIM Laser Sync Control");

    m_pCheckBox_AsyncMode = new QCheckBox(m_pGroupBox_FlimControl);
	m_pCheckBox_AsyncMode->setText("Async Mode");

    m_pLineEdit_RepetitionRate = new QLineEdit(m_pGroupBox_FlimControl);
    m_pLineEdit_RepetitionRate->setFixedWidth(25);
    m_pLineEdit_RepetitionRate->setText("30");
	m_pLineEdit_RepetitionRate->setAlignment(Qt::AlignCenter);
    m_pLineEdit_RepetitionRate->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	m_pLineEdit_RepetitionRate->setDisabled(true);

    m_pLabel_Hertz = new QLabel("kHz", m_pGroupBox_FlimControl);
    m_pLabel_Hertz->setBuddy(m_pLineEdit_RepetitionRate);
	m_pLabel_Hertz->setDisabled(true);

	pGridLayout_FlimLaserSyncControl->addWidget(m_pCheckBox_FlimLaserSyncControl, 0, 0, 1, 4);
	pGridLayout_FlimLaserSyncControl->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
	pGridLayout_FlimLaserSyncControl->addWidget(m_pCheckBox_AsyncMode, 1, 1);
	pGridLayout_FlimLaserSyncControl->addWidget(m_pLineEdit_RepetitionRate, 1, 2);
	pGridLayout_FlimLaserSyncControl->addWidget(m_pLabel_Hertz, 1, 3);

	m_pVBoxLayout_FlimControl->addItem(pGridLayout_FlimLaserSyncControl);

	// Connect signal and slot
	connect(m_pCheckBox_FlimLaserSyncControl, SIGNAL(toggled(bool)), this, SLOT(enableFlimLaserSyncControl(bool)));
	connect(m_pCheckBox_AsyncMode, SIGNAL(toggled(bool)), this, SLOT(enableAsyncMode(bool)));
}

void QDeviceControlTab::createFlimLaserPowerControl()
{
    // Create widgets for FLIM laser power control
    QGridLayout *pGridLayout_FlimLaserPowerControl = new QGridLayout;
    pGridLayout_FlimLaserPowerControl->setSpacing(3);

    m_pCheckBox_FlimLaserPowerControl = new QCheckBox(m_pGroupBox_FlimControl);
    m_pCheckBox_FlimLaserPowerControl->setText("Enable FLIM Laser Power Control");

    m_pPushButton_IncreasePower = new QPushButton(m_pGroupBox_FlimControl);
    m_pPushButton_IncreasePower->setText("Increase");
    m_pPushButton_IncreasePower->setDisabled(true);

    m_pPushButton_DecreasePower = new QPushButton(m_pGroupBox_FlimControl);
    m_pPushButton_DecreasePower->setText("Decrease");
    m_pPushButton_DecreasePower->setDisabled(true);

    pGridLayout_FlimLaserPowerControl->addWidget(m_pCheckBox_FlimLaserPowerControl, 0, 0, 1, 3);
    pGridLayout_FlimLaserPowerControl->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
    pGridLayout_FlimLaserPowerControl->addWidget(m_pPushButton_IncreasePower, 1, 1);
    pGridLayout_FlimLaserPowerControl->addWidget(m_pPushButton_DecreasePower, 1, 2);
	
	m_pVBoxLayout_FlimControl->addItem(pGridLayout_FlimLaserPowerControl);

	m_pGroupBox_FlimControl->setLayout(m_pVBoxLayout_FlimControl);
	m_pGroupBox_FlimControl->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");
	m_pGroupBox_FlimControl->resize(m_pVBoxLayout_FlimControl->minimumSize());
    m_pVBoxLayout->addWidget(m_pGroupBox_FlimControl);

	// Connect signal and slot
	connect(m_pCheckBox_FlimLaserPowerControl, SIGNAL(toggled(bool)), this, SLOT(enableFlimLaserPowerControl(bool)));
	connect(m_pPushButton_IncreasePower, SIGNAL(clicked(bool)), this, SLOT(increaseLaserPower()));
	connect(m_pPushButton_DecreasePower, SIGNAL(clicked(bool)), this, SLOT(decreaseLaserPower()));
}
#endif

#ifdef OCT_NIRF
void QDeviceControlTab::createNirfAcquisitionControl()
{
	// Create widgets for nirf emission acquisition control
	QGroupBox *pGroupBox_NirfAcquisitionControl = new QGroupBox;
	QGridLayout *pGridLayout_NirfAcquisitionControl = new QGridLayout;
	pGridLayout_NirfAcquisitionControl->setSpacing(3);

	m_pCheckBox_NirfAcquisitionControl = new QCheckBox(pGroupBox_NirfAcquisitionControl);
	m_pCheckBox_NirfAcquisitionControl->setText("Enable NIRF Acquisition Control");
	m_pCheckBox_NirfAcquisitionControl->setDisabled(true);
	
#ifdef PROGRAMMATIC_GAIN_CONTROL
	// Create widgets for PMT gain control
	QHBoxLayout *pHBoxLayout_PmtGainControl = new QHBoxLayout;
	pHBoxLayout_PmtGainControl->setSpacing(3);

	m_pCheckBox_PmtGainControl = new QCheckBox(pGroupBox_NirfAcquisitionControl);
	m_pCheckBox_PmtGainControl->setText("Enable PMT Gain Control");
	m_pCheckBox_PmtGainControl->setFixedWidth(140);

	m_pLabel_PmtGainVoltage = new QLabel("V", pGroupBox_NirfAcquisitionControl);
	m_pLabel_PmtGainVoltage->setEnabled(true);

#ifndef TWO_CHANNEL_NIRF
	m_pLineEdit_PmtGainVoltage = new QLineEdit(pGroupBox_NirfAcquisitionControl);
	m_pLineEdit_PmtGainVoltage->setFixedWidth(35);
	m_pLineEdit_PmtGainVoltage->setText(QString::number(m_pConfig->pmtGainVoltage, 'f', 3));
	m_pLineEdit_PmtGainVoltage->setAlignment(Qt::AlignCenter);
	m_pLineEdit_PmtGainVoltage->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	m_pLineEdit_PmtGainVoltage->setEnabled(true);
		
	pHBoxLayout_PmtGainControl->addWidget(m_pCheckBox_PmtGainControl);
	pHBoxLayout_PmtGainControl->addWidget(m_pLineEdit_PmtGainVoltage);
	pHBoxLayout_PmtGainControl->addWidget(m_pLabel_PmtGainVoltage);
#else
	m_pLineEdit_PmtGainVoltage[0] = new QLineEdit(pGroupBox_NirfAcquisitionControl);
	m_pLineEdit_PmtGainVoltage[0]->setFixedWidth(35);
	m_pLineEdit_PmtGainVoltage[0]->setText(QString::number(m_pConfig->pmtGainVoltage[0], 'f', 3));
	m_pLineEdit_PmtGainVoltage[0]->setAlignment(Qt::AlignCenter);
	m_pLineEdit_PmtGainVoltage[0]->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	m_pLineEdit_PmtGainVoltage[0]->setEnabled(true);
	
	m_pLineEdit_PmtGainVoltage[1] = new QLineEdit(pGroupBox_NirfAcquisitionControl);
	m_pLineEdit_PmtGainVoltage[1]->setFixedWidth(35);
	m_pLineEdit_PmtGainVoltage[1]->setText(QString::number(m_pConfig->pmtGainVoltage[1], 'f', 3));
	m_pLineEdit_PmtGainVoltage[1]->setAlignment(Qt::AlignCenter);
	m_pLineEdit_PmtGainVoltage[1]->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	m_pLineEdit_PmtGainVoltage[1]->setEnabled(true);
	
	pHBoxLayout_PmtGainControl->addWidget(m_pCheckBox_PmtGainControl);
	pHBoxLayout_PmtGainControl->addItem(new QSpacerItem(0, 0, QSizePolicy::MinimumExpanding, QSizePolicy::Fixed));

	QHBoxLayout *pHBoxLayout_PmtGainControl1 = new QHBoxLayout;
	pHBoxLayout_PmtGainControl1->setSpacing(3);

	pHBoxLayout_PmtGainControl1->addItem(new QSpacerItem(0, 0, QSizePolicy::MinimumExpanding, QSizePolicy::Fixed));
	pHBoxLayout_PmtGainControl1->addWidget(m_pLineEdit_PmtGainVoltage[0]);
	pHBoxLayout_PmtGainControl1->addWidget(m_pLineEdit_PmtGainVoltage[1]);
	pHBoxLayout_PmtGainControl1->addWidget(m_pLabel_PmtGainVoltage);
#endif
#endif

	pGridLayout_NirfAcquisitionControl->addWidget(m_pCheckBox_NirfAcquisitionControl, 0, 0);
#ifdef PROGRAMMATIC_GAIN_CONTROL
	pGridLayout_NirfAcquisitionControl->addItem(pHBoxLayout_PmtGainControl, 1, 0);
#ifdef TWO_CHANNEL_NIRF
	pGridLayout_NirfAcquisitionControl->addItem(pHBoxLayout_PmtGainControl1, 2, 0);
#endif
#endif

	pGroupBox_NirfAcquisitionControl->setLayout(pGridLayout_NirfAcquisitionControl);
	pGroupBox_NirfAcquisitionControl->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");
	m_pVBoxLayout->addWidget(pGroupBox_NirfAcquisitionControl);
	//m_pVBoxLayout_FlimControl->addItem(pHBoxLayout_PmtGainControl);

	// Connect signal and slot
	//connect(m_pCheckBox_NirfAcquisitionControl, SIGNAL(toggled(bool)), this, SLOT(enableNirfEmissionAcquisition(bool)));
#ifdef PROGRAMMATIC_GAIN_CONTROL
	connect(m_pCheckBox_PmtGainControl, SIGNAL(toggled(bool)), this, SLOT(enablePmtGainControl(bool)));
#ifndef TWO_CHANNEL_NIRF
	connect(m_pLineEdit_PmtGainVoltage, SIGNAL(textChanged(const QString &)), this, SLOT(changePmtGainVoltage(const QString &)));
#else
	connect(m_pLineEdit_PmtGainVoltage[0], SIGNAL(textChanged(const QString &)), this, SLOT(changePmtGainVoltage1(const QString &)));
	connect(m_pLineEdit_PmtGainVoltage[1], SIGNAL(textChanged(const QString &)), this, SLOT(changePmtGainVoltage2(const QString &)));
#endif
#endif
}
#endif

#ifdef GALVANO_MIRROR
void QDeviceControlTab::createGalvanoMirrorControl()
{
    // Create widgets for galvano mirror control
    QGroupBox *pGroupBox_GalvanoMirrorControl = new QGroupBox;
    QGridLayout *pGridLayout_GalvanoMirrorControl = new QGridLayout;
    pGridLayout_GalvanoMirrorControl->setSpacing(3);

    m_pCheckBox_GalvanoMirrorControl = new QCheckBox(pGroupBox_GalvanoMirrorControl);
    m_pCheckBox_GalvanoMirrorControl->setText("Enable Galvano Mirror Control");
	
	m_pToggleButton_ScanTriggering = new QPushButton(pGroupBox_GalvanoMirrorControl);
	m_pToggleButton_ScanTriggering->setCheckable(true);
	m_pToggleButton_ScanTriggering->setText("Scan Triggering Mode On");
	m_pToggleButton_ScanTriggering->setFixedWidth(150);

    m_pLineEdit_FastPeakToPeakVoltage = new QLineEdit(pGroupBox_GalvanoMirrorControl);
    m_pLineEdit_FastPeakToPeakVoltage->setFixedWidth(30);
    m_pLineEdit_FastPeakToPeakVoltage->setText(QString::number(m_pConfig->galvoFastScanVoltage, 'f', 1));
	m_pLineEdit_FastPeakToPeakVoltage->setAlignment(Qt::AlignCenter);
    m_pLineEdit_FastPeakToPeakVoltage->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

    m_pLineEdit_FastOffsetVoltage = new QLineEdit(pGroupBox_GalvanoMirrorControl);
    m_pLineEdit_FastOffsetVoltage->setFixedWidth(30);
    m_pLineEdit_FastOffsetVoltage->setText(QString::number(m_pConfig->galvoFastScanVoltageOffset, 'f', 1));
	m_pLineEdit_FastOffsetVoltage->setAlignment(Qt::AlignCenter);
    m_pLineEdit_FastOffsetVoltage->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

    m_pLabel_FastScanVoltage = new QLabel("Fast Scan Voltage ", pGroupBox_GalvanoMirrorControl);
    m_pLabel_FastScanPlusMinus = new QLabel("+", pGroupBox_GalvanoMirrorControl);
    m_pLabel_FastScanPlusMinus->setBuddy(m_pLineEdit_FastPeakToPeakVoltage);
    m_pLabel_FastGalvanoVoltage = new QLabel("V", pGroupBox_GalvanoMirrorControl);
    m_pLabel_FastGalvanoVoltage->setBuddy(m_pLineEdit_FastOffsetVoltage);

	m_pLineEdit_SlowPeakToPeakVoltage = new QLineEdit(pGroupBox_GalvanoMirrorControl);
	m_pLineEdit_SlowPeakToPeakVoltage->setFixedWidth(30);
	m_pLineEdit_SlowPeakToPeakVoltage->setText(QString::number(m_pConfig->galvoSlowScanVoltage, 'f', 1));
	m_pLineEdit_SlowPeakToPeakVoltage->setAlignment(Qt::AlignCenter);
	m_pLineEdit_SlowPeakToPeakVoltage->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	if (m_pConfig->galvoSlowScanVoltage == 0)
		m_pToggleButton_ScanTriggering->setDisabled(true);

	m_pLineEdit_SlowOffsetVoltage = new QLineEdit(pGroupBox_GalvanoMirrorControl);
	m_pLineEdit_SlowOffsetVoltage->setFixedWidth(30);
	m_pLineEdit_SlowOffsetVoltage->setText(QString::number(m_pConfig->galvoSlowScanVoltageOffset, 'f', 1));
	m_pLineEdit_SlowOffsetVoltage->setAlignment(Qt::AlignCenter);
	m_pLineEdit_SlowOffsetVoltage->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

	m_pLabel_SlowScanVoltage = new QLabel("Slow Scan Voltage ", pGroupBox_GalvanoMirrorControl);
	m_pLabel_SlowScanPlusMinus = new QLabel("+", pGroupBox_GalvanoMirrorControl);
	m_pLabel_SlowScanPlusMinus->setBuddy(m_pLineEdit_SlowPeakToPeakVoltage);
	m_pLabel_SlowGalvanoVoltage = new QLabel("V", pGroupBox_GalvanoMirrorControl);
	m_pLabel_SlowGalvanoVoltage->setBuddy(m_pLineEdit_SlowOffsetVoltage);

	m_pLineEdit_SlowScanIncrement = new QLineEdit(pGroupBox_GalvanoMirrorControl);
	m_pLineEdit_SlowScanIncrement->setFixedWidth(40);
	m_pLineEdit_SlowScanIncrement->setText(QString::number(m_pConfig->galvoSlowScanIncrement, 'f', 3)); 
	m_pLineEdit_SlowScanIncrement->setAlignment(Qt::AlignCenter);
	m_pLineEdit_SlowScanIncrement->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

	m_pLabel_SlowScanIncrement = new QLabel("Slow Scan Increment ", pGroupBox_GalvanoMirrorControl);
	m_pLabel_SlowScanIncrement->setBuddy(m_pLineEdit_SlowScanIncrement);
	m_pLabel_SlowScanIncrementVoltage = new QLabel("V", pGroupBox_GalvanoMirrorControl);
	m_pLabel_SlowScanIncrementVoltage->setBuddy(m_pLabel_SlowScanIncrement);

    m_pScrollBar_ScanAdjustment = new QScrollBar(pGroupBox_GalvanoMirrorControl);
    m_pScrollBar_ScanAdjustment->setOrientation(Qt::Horizontal);
	m_pScrollBar_ScanAdjustment->setRange(0, m_pConfig->nAlines - 1);
	m_pScrollBar_ScanAdjustment->setSingleStep(1);
	m_pScrollBar_ScanAdjustment->setPageStep(m_pScrollBar_ScanAdjustment->maximum() / 10);
	m_pScrollBar_ScanAdjustment->setFocusPolicy(Qt::StrongFocus);
	m_pScrollBar_ScanAdjustment->setDisabled(true);
	QString str; str.sprintf("Fast Scan Adjustment  %d / %d", 0, m_pConfig->nAlines - 1);
    m_pLabel_ScanAdjustment = new QLabel(str, pGroupBox_GalvanoMirrorControl);
    m_pLabel_ScanAdjustment->setBuddy(m_pScrollBar_ScanAdjustment);
	m_pLabel_ScanAdjustment->setDisabled(true);

#if !NI_ENABLE
    m_pCheckBox_GalvanoMirrorControl->setDisabled(true);
    m_pToggleButton_ScanTriggering->setDisabled(true);
    m_pLineEdit_FastPeakToPeakVoltage->setDisabled(true);
    m_pLineEdit_FastOffsetVoltage->setDisabled(true);
    m_pLabel_FastScanVoltage->setDisabled(true);
    m_pLabel_FastScanPlusMinus->setDisabled(true);
    m_pLabel_FastGalvanoVoltage->setDisabled(true);
    m_pLineEdit_SlowPeakToPeakVoltage->setDisabled(true);
    m_pLineEdit_SlowOffsetVoltage->setDisabled(true);
    m_pLabel_SlowScanVoltage->setDisabled(true);
    m_pLabel_SlowScanPlusMinus->setDisabled(true);
    m_pLabel_SlowGalvanoVoltage->setDisabled(true);
    m_pLineEdit_SlowScanIncrement->setDisabled(true);
    m_pLabel_SlowScanIncrement->setDisabled(true);
    m_pLabel_SlowScanIncrementVoltage->setDisabled(true);
#endif

    pGridLayout_GalvanoMirrorControl->addWidget(m_pCheckBox_GalvanoMirrorControl, 0, 0, 1, 6);

	QHBoxLayout *pHBoxLayout_ToggleButtons = new QHBoxLayout;
	pHBoxLayout_ToggleButtons->setSpacing(3);
	pHBoxLayout_ToggleButtons->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));
	pHBoxLayout_ToggleButtons->addWidget(m_pToggleButton_ScanTriggering);
	pGridLayout_GalvanoMirrorControl->addItem(pHBoxLayout_ToggleButtons, 1, 0, 1, 6);

    pGridLayout_GalvanoMirrorControl->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 2, 0);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_FastScanVoltage, 2, 1);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pLineEdit_FastPeakToPeakVoltage, 2, 2);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_FastScanPlusMinus, 2, 3);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pLineEdit_FastOffsetVoltage, 2, 4);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_FastGalvanoVoltage, 2, 5);
	pGridLayout_GalvanoMirrorControl->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 3, 0);
	pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_SlowScanVoltage, 3, 1);
	pGridLayout_GalvanoMirrorControl->addWidget(m_pLineEdit_SlowPeakToPeakVoltage, 3, 2);
	pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_SlowScanPlusMinus, 3, 3);
	pGridLayout_GalvanoMirrorControl->addWidget(m_pLineEdit_SlowOffsetVoltage, 3, 4);
	pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_SlowGalvanoVoltage, 3, 5);
	pGridLayout_GalvanoMirrorControl->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 4, 0);
	pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_SlowScanIncrement, 4, 1, 1, 2);
	pGridLayout_GalvanoMirrorControl->addWidget(m_pLineEdit_SlowScanIncrement, 4, 3, 1, 2);
	pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_SlowScanIncrementVoltage, 4, 5);
	
	pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_ScanAdjustment, 5, 0, 1, 6);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pScrollBar_ScanAdjustment, 6, 0, 1, 6);

    pGroupBox_GalvanoMirrorControl->setLayout(pGridLayout_GalvanoMirrorControl);
	pGroupBox_GalvanoMirrorControl->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");
    m_pVBoxLayout->addWidget(pGroupBox_GalvanoMirrorControl);

	// Connect signal and slot
	connect(m_pCheckBox_GalvanoMirrorControl, SIGNAL(toggled(bool)), this, SLOT(enableGalvanoMirror(bool)));
	connect(m_pToggleButton_ScanTriggering, SIGNAL(toggled(bool)), this, SLOT(triggering(bool)));
	connect(m_pLineEdit_FastPeakToPeakVoltage, SIGNAL(textChanged(const QString &)), this, SLOT(changeGalvoFastScanVoltage(const QString &)));
	connect(m_pLineEdit_FastOffsetVoltage, SIGNAL(textChanged(const QString &)), this, SLOT(changeGalvoFastScanVoltageOffset(const QString &)));
	connect(m_pLineEdit_SlowPeakToPeakVoltage, SIGNAL(textChanged(const QString &)), this, SLOT(changeGalvoSlowScanVoltage(const QString &)));
	connect(m_pLineEdit_SlowOffsetVoltage, SIGNAL(textChanged(const QString &)), this, SLOT(changeGalvoSlowScanVoltageOffset(const QString &)));
	connect(m_pLineEdit_SlowScanIncrement, SIGNAL(textChanged(const QString &)), this, SLOT(changeGalvoSlowScanIncrement(const QString &)));
	connect(m_pScrollBar_ScanAdjustment, SIGNAL(valueChanged(int)), this, SLOT(scanAdjusting(int)));
}
#endif

#ifdef PULLBACK_DEVICE
void QDeviceControlTab::createZaberStageControl()
{
    // Create widgets for Zaber stage control
    QGroupBox *pGroupBox_ZaberStageControl = new QGroupBox;
    QGridLayout *pGridLayout_ZaberStageControl = new QGridLayout;
    pGridLayout_ZaberStageControl->setSpacing(3);

    m_pCheckBox_ZaberStageControl = new QCheckBox(pGroupBox_ZaberStageControl);
    m_pCheckBox_ZaberStageControl->setText("Enable Zaber Stage Control");

    m_pPushButton_SetTargetSpeed = new QPushButton(pGroupBox_ZaberStageControl);
    m_pPushButton_SetTargetSpeed->setText("Target Speed");
	m_pPushButton_SetTargetSpeed->setDisabled(true);
    m_pPushButton_MoveAbsolute = new QPushButton(pGroupBox_ZaberStageControl);
    m_pPushButton_MoveAbsolute->setText("Move Absolute");
	m_pPushButton_MoveAbsolute->setDisabled(true);
    m_pPushButton_Home = new QPushButton(pGroupBox_ZaberStageControl);
    m_pPushButton_Home->setText("Home");
	m_pPushButton_Home->setFixedWidth(60);
	m_pPushButton_Home->setDisabled(true);
    m_pPushButton_Stop = new QPushButton(pGroupBox_ZaberStageControl);
    m_pPushButton_Stop->setText("Stop");
	m_pPushButton_Stop->setFixedWidth(60);
	m_pPushButton_Stop->setDisabled(true);

    m_pLineEdit_TargetSpeed = new QLineEdit(pGroupBox_ZaberStageControl);
    m_pLineEdit_TargetSpeed->setFixedWidth(25);
    m_pLineEdit_TargetSpeed->setText(QString::number(m_pConfig->zaberPullbackSpeed));
	m_pLineEdit_TargetSpeed->setAlignment(Qt::AlignCenter);
	m_pLineEdit_TargetSpeed->setDisabled(true);
    m_pLineEdit_TravelLength = new QLineEdit(pGroupBox_ZaberStageControl);
    m_pLineEdit_TravelLength->setFixedWidth(25);
    m_pLineEdit_TravelLength->setText(QString::number(m_pConfig->zaberPullbackLength));
	m_pLineEdit_TravelLength->setAlignment(Qt::AlignCenter);
	m_pLineEdit_TravelLength->setDisabled(true);

    m_pLabel_TargetSpeed = new QLabel("mm/s", pGroupBox_ZaberStageControl);
    m_pLabel_TargetSpeed->setBuddy(m_pLineEdit_TargetSpeed);
	m_pLabel_TargetSpeed->setDisabled(true);
    m_pLabel_TravelLength = new QLabel("mm", pGroupBox_ZaberStageControl);
    m_pLabel_TravelLength->setBuddy(m_pLineEdit_TravelLength);
	m_pLabel_TravelLength->setDisabled(true);

    pGridLayout_ZaberStageControl->addWidget(m_pCheckBox_ZaberStageControl, 0, 0, 1, 5);

    pGridLayout_ZaberStageControl->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
    pGridLayout_ZaberStageControl->addWidget(m_pPushButton_SetTargetSpeed, 1, 1, 1, 2);
    pGridLayout_ZaberStageControl->addWidget(m_pLineEdit_TargetSpeed, 1, 3);
    pGridLayout_ZaberStageControl->addWidget(m_pLabel_TargetSpeed, 1, 4);

    pGridLayout_ZaberStageControl->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 2, 0);
    pGridLayout_ZaberStageControl->addWidget(m_pPushButton_MoveAbsolute, 2, 1, 1, 2);
    pGridLayout_ZaberStageControl->addWidget(m_pLineEdit_TravelLength, 2, 3);
    pGridLayout_ZaberStageControl->addWidget(m_pLabel_TravelLength, 2, 4);

    pGridLayout_ZaberStageControl->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 3, 0);
    pGridLayout_ZaberStageControl->addWidget(m_pPushButton_Home, 3, 1);
    pGridLayout_ZaberStageControl->addWidget(m_pPushButton_Stop, 3, 2);

    pGroupBox_ZaberStageControl->setLayout(pGridLayout_ZaberStageControl);
	pGroupBox_ZaberStageControl->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");
    m_pVBoxLayout->addWidget(pGroupBox_ZaberStageControl);

	// Connect signal and slot
	connect(m_pCheckBox_ZaberStageControl, SIGNAL(toggled(bool)), this, SLOT(enableZaberStageControl(bool)));
	connect(m_pPushButton_MoveAbsolute, SIGNAL(clicked(bool)), this, SLOT(moveAbsolute()));
	connect(m_pLineEdit_TargetSpeed, SIGNAL(textChanged(const QString &)), this, SLOT(setTargetSpeed(const QString &)));
	connect(m_pLineEdit_TravelLength, SIGNAL(textChanged(const QString &)), this, SLOT(changeZaberPullbackLength(const QString &)));
	connect(m_pPushButton_Home, SIGNAL(clicked(bool)), this, SLOT(home()));
	connect(m_pPushButton_Stop, SIGNAL(clicked(bool)), this, SLOT(stop()));
}

void QDeviceControlTab::createFaulhaberMotorControl()
{
    // Create widgets for Faulhaber motor control
    QGroupBox *pGroupBox_FaulhaberMotorControl = new QGroupBox;
    QGridLayout *pGridLayout_FaulhaberMotorControl = new QGridLayout;
    pGridLayout_FaulhaberMotorControl->setSpacing(3);

    m_pCheckBox_FaulhaberMotorControl = new QCheckBox(pGroupBox_FaulhaberMotorControl);
    m_pCheckBox_FaulhaberMotorControl->setText("Enable Faulhaber Motor Control");

    m_pPushButton_Rotate = new QPushButton(pGroupBox_FaulhaberMotorControl);
	m_pPushButton_Rotate->setText("Rotate");
	m_pPushButton_Rotate->setFixedWidth(60);
	m_pPushButton_Rotate->setDisabled(true);

	m_pPushButton_RotateStop = new QPushButton(pGroupBox_FaulhaberMotorControl);
	m_pPushButton_RotateStop->setText("Stop");
	m_pPushButton_RotateStop->setFixedWidth(40);
	m_pPushButton_RotateStop->setDisabled(true);

    m_pLineEdit_RPM = new QLineEdit(pGroupBox_FaulhaberMotorControl);
    m_pLineEdit_RPM->setFixedWidth(40);
    m_pLineEdit_RPM->setText(QString::number(m_pConfig->faulhaberRpm));
	m_pLineEdit_RPM->setAlignment(Qt::AlignCenter);
    m_pLineEdit_RPM->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	m_pLineEdit_RPM->setDisabled(true);

    m_pLabel_RPM = new QLabel("RPM", pGroupBox_FaulhaberMotorControl);
    m_pLabel_RPM->setBuddy(m_pLineEdit_RPM);
	m_pLabel_RPM->setDisabled(true);

    pGridLayout_FaulhaberMotorControl->addWidget(m_pCheckBox_FaulhaberMotorControl, 0, 0, 1, 5);
    pGridLayout_FaulhaberMotorControl->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
    pGridLayout_FaulhaberMotorControl->addWidget(m_pPushButton_Rotate, 1, 1);
	pGridLayout_FaulhaberMotorControl->addWidget(m_pPushButton_RotateStop, 1, 2);
    pGridLayout_FaulhaberMotorControl->addWidget(m_pLineEdit_RPM, 1, 3);
    pGridLayout_FaulhaberMotorControl->addWidget(m_pLabel_RPM, 1, 4);

    pGroupBox_FaulhaberMotorControl->setLayout(pGridLayout_FaulhaberMotorControl);
	pGroupBox_FaulhaberMotorControl->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");
    m_pVBoxLayout->addWidget(pGroupBox_FaulhaberMotorControl);
	m_pVBoxLayout->addStretch(1);

	// Connect signal and slot
	connect(m_pCheckBox_FaulhaberMotorControl, SIGNAL(toggled(bool)), this, SLOT(enableFaulhaberMotorControl(bool)));
	connect(m_pPushButton_Rotate, SIGNAL(clicked(bool)), this, SLOT(rotate()));
	connect(m_pPushButton_RotateStop, SIGNAL(clicked(bool)), this, SLOT(rotateStop()));
	connect(m_pLineEdit_RPM, SIGNAL(textChanged(const QString &)), this, SLOT(changeFaulhaberRpm(const QString &)));
}
#endif

void QDeviceControlTab::initiateAllDevices()
{
#ifdef AXSUN_OCT_LASER
	//if (!m_pCheckBox_AxsunOCTLaserControl->isChecked()) m_pCheckBox_AxsunOCTLaserControl->setChecked(true);
#endif
//#ifdef ECG_TRIGGERING
//#if NI_ENABLE
//	if (!m_pCheckBox_EcgModuleControl->isChecked()) m_pCheckBox_EcgModuleControl->setChecked(true);
//	if (!m_pCheckBox_Voltage800Rps->isChecked()) m_pCheckBox_Voltage800Rps->setChecked(true);
//#endif
//#endif
//#ifdef OCT_FLIM
//#if NI_ENABLE
//	if (!m_pCheckBox_PmtGainControl->isChecked()) m_pCheckBox_PmtGainControl->setChecked(true);
//	if (!m_pCheckBox_FlimLaserSyncControl->isChecked()) m_pCheckBox_FlimLaserSyncControl->setChecked(true);
//#endif
//	if (!m_pCheckBox_FlimLaserPowerControl->isChecked()) m_pCheckBox_FlimLaserPowerControl->setChecked(true);
//#endif
//#ifdef OCT_NIRF
//#if NI_ENABLE
//	if (!m_pCheckBox_NirfAcquisitionControl->isChecked()) m_pCheckBox_NirfAcquisitionControl->setChecked(true);
//#ifdef PROGRAMMATIC_GAIN_CONTROL
//	if (!m_pCheckBox_PmtGainControl->isChecked()) m_pCheckBox_PmtGainControl->setChecked(true);
//#endif
//#endif
//#endif
#ifdef GALVANO_MIRROR
#if NI_ENABLE
	if (!m_pCheckBox_GalvanoMirrorControl->isChecked()) m_pCheckBox_GalvanoMirrorControl->setChecked(true);
#endif
#endif
#ifdef PULLBACK_DEVICE
	//if (!m_pCheckBox_ZaberStageControl->isChecked()) m_pCheckBox_ZaberStageControl->setChecked(true);
	//if (!m_pCheckBox_FaulhaberMotorControl->isChecked()) m_pCheckBox_FaulhaberMotorControl->setChecked(true);
#endif
}

void QDeviceControlTab::terminateAllDevices()
{
#ifdef AXSUN_OCT_LASER
	if (m_pCheckBox_AxsunOCTLaserControl->isChecked()) m_pCheckBox_AxsunOCTLaserControl->setChecked(false);
#endif
#ifdef ECG_TRIGGERING
#if NI_ENABLE
	if (m_pCheckBox_EcgModuleControl->isChecked()) m_pCheckBox_EcgModuleControl->setChecked(false);
	if (m_pCheckBox_Voltage800Rps->isChecked()) m_pCheckBox_Voltage800Rps->setChecked(false);
#endif
#endif
#ifdef OCT_FLIM
#if NI_ENABLE
	if (m_pCheckBox_PmtGainControl->isChecked()) m_pCheckBox_PmtGainControl->setChecked(false);
	if (m_pCheckBox_FlimLaserSyncControl->isChecked()) m_pCheckBox_FlimLaserSyncControl->setChecked(false);
#endif
	if (m_pCheckBox_FlimLaserPowerControl->isChecked()) m_pCheckBox_FlimLaserPowerControl->setChecked(false);
#endif
#ifdef OCT_NIRF
#if NI_ENABLE
	if (m_pCheckBox_NirfAcquisitionControl->isChecked()) m_pCheckBox_NirfAcquisitionControl->setChecked(false);
#ifdef PROGRAMMATIC_GAIN_CONTROL
	if (m_pCheckBox_PmtGainControl->isChecked()) m_pCheckBox_PmtGainControl->setChecked(false);
#endif
#endif
#endif
#ifdef GALVANO_MIRROR
#if NI_ENABLE
	if (m_pCheckBox_GalvanoMirrorControl->isChecked()) m_pCheckBox_GalvanoMirrorControl->setChecked(false);
#endif
#endif
#ifdef PULLBACK_DEVICE
	if (m_pCheckBox_ZaberStageControl->isChecked()) m_pCheckBox_ZaberStageControl->setChecked(false);
	if (m_pCheckBox_FaulhaberMotorControl->isChecked()) m_pCheckBox_FaulhaberMotorControl->setChecked(false);
#endif
}


#ifdef AXSUN_OCT_LASER
void QDeviceControlTab::enableAxsunOCTLaserControl(bool toggled)
{
    if (toggled)
    {
        // Set text
        m_pCheckBox_AxsunOCTLaserControl->setText("Disable Axsun OCT Laser Control");

        // Create Axsun OCT laser control objects
        m_pAxsunControl = new AxsunControl;
		m_pAxsunControl->SendStatusMessage += [&](const char* msg, bool is_fail)
		{
			printf("%s\n", msg);
			(void)is_fail;
		};

        // Connect the OCT laser
        if (!(m_pAxsunControl->initialize()))
        {
            m_pCheckBox_AxsunOCTLaserControl->setChecked(false);
            return;
        }
		m_pAxsunControl->setVDLLength(m_pConfig->axsunVDLLength);
		m_pAxsunControl->setClockDelay(m_pConfig->axsunkClockDelay);

        // Set enable true for Axsun OCT laser control widgets
		m_pLabel_OCTLaserSource->setEnabled(true);
		m_pToggleButton_OCTLaserSource->setEnabled(true);
		m_pToggleButton_OCTLaserSource->setStyleSheet("QPushButton { background-color:#ff0000; }");
#ifdef AXSUN_VDL_K_CLOCK_DELAY
		m_pLabel_VDLLength->setEnabled(true);
		m_pSpinBox_VDLLength->setEnabled(true);
		m_pPushButton_VDLHome->setEnabled(true);
		m_pLabel_kClockDelay->setEnabled(true);
		m_pSpinBox_kClockDelay->setEnabled(true);
#endif
    }
    else
    {
        // Set enable false for Axsun OCT laser control widgets
		if (m_pToggleButton_OCTLaserSource->isChecked()) m_pToggleButton_OCTLaserSource->setChecked(false);
		m_pToggleButton_OCTLaserSource->setStyleSheet("QPushButton { background-color:#353535; }");
		m_pToggleButton_OCTLaserSource->setDisabled(true);
		m_pLabel_OCTLaserSource->setDisabled(true);
#ifdef AXSUN_VDL_K_CLOCK_DELAY
		m_pLabel_VDLLength->setDisabled(true);
		m_pSpinBox_VDLLength->setDisabled(true);
		m_pPushButton_VDLHome->setDisabled(true);
		m_pLabel_kClockDelay->setDisabled(true);
		m_pSpinBox_kClockDelay->setDisabled(true);
#endif

        if (m_pAxsunControl)
        {
            // Delete Axsun OCT laser control objects
            delete m_pAxsunControl;
			m_pAxsunControl = nullptr;
        }

        // Set text
        m_pCheckBox_AxsunOCTLaserControl->setText("Enable Axsun OCT Laser Control");
    }
}

void QDeviceControlTab::setLightSource(bool toggled)
{
	if (m_pAxsunControl)
	{
		if (toggled)
		{
			// Set widgets
			m_pCheckBox_AxsunOCTLaserControl->setDisabled(true);

			// Set text
			m_pToggleButton_OCTLaserSource->setText("Off");
			m_pToggleButton_OCTLaserSource->setStyleSheet("QPushButton { background-color:#00ff00; }");

			// Start Axsun light source operation
			m_pAxsunControl->setLaserEmission(true);
		}
		else
		{	
			/// If acquisition is processing...
			///if (m_pOperationTab->isAcquisitionButtonToggled())
			///{
			///	QMessageBox MsgBox;
			///	MsgBox.setWindowTitle("Warning");
			///	MsgBox.setIcon(QMessageBox::Warning);
			///	MsgBox.setText("Re-turning the laser on does not guarantee the synchronized operation once you turn off the laser.\nWould you like to turn off the laser?");
			///	MsgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
			///	MsgBox.setDefaultButton(QMessageBox::No);

			///	int resp = MsgBox.exec();
			///	switch (resp)
			///	{
			///	case QMessageBox::Yes:
			///		m_pToggleButton_OCTLaserSource->setChecked(false);
			///		break;
			///	case QMessageBox::No:
			///		m_pToggleButton_OCTLaserSource->setChecked(true);
			///		return;
			///	default:
			///		m_pToggleButton_OCTLaserSource->setChecked(true);
			///		return;
			///	}
			///}
			
			// Stop Axsun light source operation
			if (m_pAxsunControl) m_pAxsunControl->setLaserEmission(false);

			// Set text
			m_pToggleButton_OCTLaserSource->setText("On");
			m_pToggleButton_OCTLaserSource->setStyleSheet("QPushButton { background-color:#ff0000; }");			
			
			// Set widgets
			m_pCheckBox_AxsunOCTLaserControl->setEnabled(true);
		}
	}
}

void QDeviceControlTab::turnOnOCTLaser(bool set)
{
	if (set)
	{
		if (!m_pCheckBox_AxsunOCTLaserControl->isChecked()) m_pCheckBox_AxsunOCTLaserControl->setChecked(true);
		m_pToggleButton_OCTLaserSource->setChecked(true);
	}
	else
	{
		m_pToggleButton_OCTLaserSource->setChecked(false);
	}
}

#ifdef AXSUN_VDL_K_CLOCK_DELAY
void QDeviceControlTab::setVDLLength(double length)
{
	if (m_pAxsunControl)
	{
		m_pConfig->axsunVDLLength = length;
		m_pAxsunControl->setVDLLength(length);
	}

	//m_pDeviceControl->setVDLLength(length);
	//if (m_pStreamTab)
	//	m_pStreamTab->getCalibScrollBar()->setValue(int(length * 100.0));
}

void QDeviceControlTab::setVDLHome()
{
	m_pAxsunControl->setVDLHome();
	m_pSpinBox_VDLLength->setValue(0.0);
}

void QDeviceControlTab::setVDLWidgets(bool enabled)
{
	//m_pLabel_VDLLength->setEnabled(enabled);
	//m_pSpinBox_VDLLength->setEnabled(enabled);
	//m_pPushButton_VDLHome->setEnabled(enabled);

	(void)enabled;
}

void QDeviceControlTab::setkClockDelay(int delay)
{
	if (m_pAxsunControl)
	{
		m_pConfig->axsunkClockDelay = delay;
		m_pAxsunControl->setClockDelay(delay);
	}

}
#endif
#endif

#ifdef ECG_TRIGGERING
#if NI_ENABLE
void QDeviceControlTab::setEcgRecording(bool set)
{
    if (set) std::deque<double>().swap(m_pEcgMonitoring->deque_record_ecg);
    m_pEcgMonitoring->isRecording = set;
}

double QDeviceControlTab::getEcgHeartInterval()
{
	return m_pEcgMonitoring->heart_interval;
}

std::deque<double>* QDeviceControlTab::getRecordedEcg()
{
	return &m_pEcgMonitoring->deque_record_ecg;
}
#endif        
#endif

#ifdef OCT_NIRF
bool QDeviceControlTab::initializeNiDaqAnalogInput()
{
#if NI_ENABLE
#ifndef ALAZAR_NIRF_ACQUISITION
	// Create NIRF emission acquisition objects
	///m_pNirfEmissionTrigger = new NirfEmissionTrigger;
	///m_pNirfEmissionTrigger->nAlines = m_pConfig->nAlines;

	m_pNirfEmission = new NirfEmission;
	m_pNirfEmission->nAlines = m_pConfig->nAlines;
	
	m_pMainWnd->m_pStreamTab->setNirfAcquisitionCallback();

	// Initializing
	if (!m_pNirfEmission->initialize()) // || !m_pNirfEmission->initialize()) Trigger
		return false;
#endif

#ifdef TWO_CHANNEL_NIRF
	// Create NIRF modulated excitation objects
	m_pNirfModulation = new NirfModulation;
	m_pNirfModulation->nCh = 2;

	if (!m_pNirfModulation->initialize())
		return false;
#endif

	// Apply PMT gain voltage
	m_pCheckBox_PmtGainControl->setChecked(true);
#endif

	return true;
}

bool QDeviceControlTab::startNiDaqAnalogInput()
{
#if NI_ENABLE
	// Open NIRF emission profile dialog
	m_pMainWnd->m_pStreamTab->makeNirfEmissionProfileDlg();

#ifndef ALAZAR_NIRF_ACQUISITION
	// Generate trigger pulse & Start NIRF acquisition
	if (m_pNirfEmission) m_pNirfEmission->start();
#endif

#ifdef TWO_CHANNEL_NIRF
	if (m_pNirfModulation) m_pNirfModulation->start();
#endif

	///Sleep(100);
	///if (m_pNirfEmissionTrigger) m_pNirfEmissionTrigger->start();
#endif

	return true;
}
#endif

#ifdef GALVANO_MIRROR
void QDeviceControlTab::setScrollBarValue(int pos)
{ 
	m_pScrollBar_ScanAdjustment->setValue(pos); 
}

void QDeviceControlTab::setScrollBarRange(int alines)
{ 
	QString str; str.sprintf("Fast Scan Adjustment  %d / %d", m_pScrollBar_ScanAdjustment->value(), alines - 1);
	m_pLabel_ScanAdjustment->setText(str);
	m_pScrollBar_ScanAdjustment->setRange(0, alines - 1);
}

void QDeviceControlTab::setScrollBarEnabled(bool enable)
{
	m_pLabel_ScanAdjustment->setEnabled(enable);
	m_pScrollBar_ScanAdjustment->setEnabled(enable); 
}
#endif


#ifdef ECG_TRIGGERING
void QDeviceControlTab::enableEcgModuleControl(bool toggled)
{
	if (toggled)
	{
#if NI_ENABLE
		// Set text
		m_pCheckBox_EcgModuleControl->setText("Disable ECG Module Control");
		m_pLabel_EcgBitPerMinute->setText(QString("Heart Rate: %1 bpm").arg(0.0, 4, 'f', 1));

		// Set enabled true for ECG module widgets
		m_pLabel_EcgBitPerMinute->setEnabled(true);
		m_pToggledButton_EcgTriggering->setEnabled(true);
		m_pLabel_EcgDelayRate->setEnabled(true);
		m_pDoubleSpinBox_EcgDelayRate->setEnabled(true);
		m_pEcgScope->setEnabled(true);

		// Create ECG monitoring objects
		m_pEcgMonitoringTrigger = new EcgMonitoringTrigger;
		m_pEcgMonitoring = new EcgMonitoring;

		// Initializing
		if (!m_pEcgMonitoringTrigger->initialize() || !m_pEcgMonitoring->initialize())
		{
			m_pCheckBox_EcgModuleControl->setChecked(false);
			return;
		}
		connect(this, SIGNAL(drawEcg(double, bool)), m_pEcgScope, SLOT(drawData(double, bool)));
		
		m_pEcgMonitoring->acquiredData += [&](double& data, bool& is_peak) {
			m_pConfig->ecgHeartRate = float(60.0 / m_pEcgMonitoring->heart_interval * 1000.0);
			emit drawEcg(data, is_peak);
		};
		m_pEcgMonitoring->startRecording += [&]() {
			std::unique_lock<std::mutex> mlock(m_mtxRpeakDetected);
			mlock.unlock();
			m_cvRpeakDetected.notify_one();
		};
		m_pEcgMonitoring->renewHeartRate += [&](double bpm) {
			m_pLabel_EcgBitPerMinute->setText(QString("Heart Rate: %1 bpm").arg(bpm, 4, 'f', 1));
		};

		// Generate 1 kHz trigger pulse & Start ECG monitoring
		m_pEcgMonitoringTrigger->start();
		m_pEcgMonitoring->start();
	}
	else
	{
		// ECG triggering off
		m_pToggledButton_EcgTriggering->setChecked(false);

		// Delete ECG monitoring objects		
		if (m_pEcgMonitoring)
		{
			m_pEcgMonitoring->stop();
			delete m_pEcgMonitoring;
		}
		if (m_pEcgMonitoringTrigger)
		{
			m_pEcgMonitoringTrigger->stop();
			delete m_pEcgMonitoringTrigger;
		}
		disconnect(this, SIGNAL(drawEcg(double, bool)), 0, 0);

		// Clear visualization buffer
		m_pEcgScope->clearDeque();

		// Set enabled false for ECG module widgets
		m_pLabel_EcgBitPerMinute->setEnabled(false);
		m_pToggledButton_EcgTriggering->setEnabled(false);
		m_pLabel_EcgDelayRate->setEnabled(false);
		m_pDoubleSpinBox_EcgDelayRate->setEnabled(false);
		m_pEcgScope->setEnabled(false);

		// Set text
		m_pCheckBox_EcgModuleControl->setText("Enable ECG Module Control");
#endif
	}
}

void QDeviceControlTab::toggleEcgTriggerButton(bool toggled)
{
	if (toggled)
#if NI_ENABLE
		m_pToggledButton_EcgTriggering->setText("ECG Triggering Off");
	else
		m_pToggledButton_EcgTriggering->setText("ECG Triggering On");
#endif
}

void QDeviceControlTab::changeEcgDelayRate(double d)
{
	m_pConfig->ecgDelayRate = (float)d;
}

void QDeviceControlTab::enable800rpsVoltControl(bool toggled)
{
	if (toggled)
	{
#if NI_ENABLE
		// Set text
		m_pCheckBox_Voltage800Rps->setText("Disable Voltage Control for 800 rps");

		// Set enable true for 800 rps voltage control widgets
		m_pToggledButton_Voltage800Rps->setEnabled(true);
	}
	else
	{
		// Voltage control off
		m_pToggledButton_Voltage800Rps->setChecked(false);

		// Set enable false for 800 rps voltage control widgets
		m_pToggledButton_Voltage800Rps->setEnabled(false);
		m_pDoubleSpinBox_Voltage800Rps->setEnabled(false);
		m_pLabel_Voltage800Rps->setEnabled(false);

		// Set text
		m_pCheckBox_Voltage800Rps->setText("Enable Voltage Control for 800 rps");
#endif
	}
}

void QDeviceControlTab::toggle800rpsVoltButton(bool toggled)
{
	if (toggled)
	{
#if NI_ENABLE
		// Set text
		m_pToggledButton_Voltage800Rps->setText("Voltage Off");

		// Create 800 rps voltage control objects
		m_pVoltage800Rps = new Voltage800RPS;

		// Initializing
		if (!m_pVoltage800Rps->initialize())
		{
			m_pToggledButton_Voltage800Rps->setChecked(false);
			return;
		}

		// Apply zero voltage
		m_pVoltage800Rps->apply(0);

		// Set enable true for 800 rps voltage control
		m_pDoubleSpinBox_Voltage800Rps->setEnabled(true);
		m_pLabel_Voltage800Rps->setEnabled(true);
	}
	else
	{
		// Set enable false for 800 rps voltage control
		m_pDoubleSpinBox_Voltage800Rps->setValue(0.0);
		m_pDoubleSpinBox_Voltage800Rps->setEnabled(false);
		m_pLabel_Voltage800Rps->setEnabled(false);

		// Delete 800 rps voltage control objects
		if (m_pVoltage800Rps)
		{
			m_pVoltage800Rps->apply(0);
			delete m_pVoltage800Rps;
		}

		// Set text
		m_pToggledButton_Voltage800Rps->setText("Voltage On");	
#endif
	}
}

void QDeviceControlTab::set800rpsVoltage(double voltage)
{	
#if NI_ENABLE
	// Generate voltage for 800 rps motor control
	m_pVoltage800Rps->apply(voltage);
#else
	(void)voltage;
#endif
}
#endif

#ifdef OCT_FLIM
void QDeviceControlTab::enablePmtGainControl(bool toggled)
{
	if (toggled)
	{
#if NI_ENABLE
		// Set text
		m_pCheckBox_PmtGainControl->setText("Disable PMT Gain Control");

		// Set enabled false for PMT gain control widgets
		m_pLineEdit_PmtGainVoltage->setEnabled(false);
		m_pLabel_PmtGainVoltage->setEnabled(false);

		// Create PMT gain control objects
		m_pPmtGainControl = new PmtGainControl;
		m_pPmtGainControl->voltage1 = m_pLineEdit_PmtGainVoltage->text().toDouble();
		if (m_pPmtGainControl->voltage1 > 1.0)
		{
			printf(">1.0V Gain cannot be assigned!\n");
			m_pCheckBox_PmtGainControl->setChecked(false);
			return;
		}

		// Initializing
		if (!m_pPmtGainControl->initialize())
		{
			m_pCheckBox_PmtGainControl->setChecked(false);
			return;
		}

		// Generate PMT gain voltage
		m_pPmtGainControl->start();
	}
	else
	{
		// Delete PMT gain control objects
		if (m_pPmtGainControl)
		{
			m_pPmtGainControl->stop();
			delete m_pPmtGainControl;
		}
		
		// Set enabled true for PMT gain control widgets
		m_pLineEdit_PmtGainVoltage->setEnabled(true);
		m_pLabel_PmtGainVoltage->setEnabled(true);

		// Set text
		m_pCheckBox_PmtGainControl->setText("Enable PMT Gain Control");
#endif
	}
}

void QDeviceControlTab::changePmtGainVoltage(const QString &str)
{
	m_pConfig->pmtGainVoltage = str.toFloat();
}


void QDeviceControlTab::enableFlimLaserSyncControl(bool toggled)
{
	if (toggled)
	{
#if NI_ENABLE
		// Set text
		m_pCheckBox_FlimLaserSyncControl->setText("Disable FLIM Laser Sync Control");

		// Set enabled false for FLIM laser sync control widgets
		m_pCheckBox_AsyncMode->setEnabled(false);
		m_pLineEdit_RepetitionRate->setEnabled(false);
		m_pLabel_Hertz->setEnabled(false);

		// Create FLIM laser sync control objects
		m_pSyncFlim = new SyncFLIM;
		m_pSyncFlim->sourceTerminal = (m_pCheckBox_AsyncMode->isChecked()) ? "20MHzTimeBase" : NI_FLIM_SYNC_SOURCE;
		m_pSyncFlim->slow = (m_pCheckBox_AsyncMode->isChecked()) ? 200000 / m_pLineEdit_RepetitionRate->text().toInt() : 4;

		// Initializing
		if (!m_pSyncFlim->initialize())
		{
			m_pCheckBox_FlimLaserSyncControl->setChecked(false);
			return;
		}

		// Generate FLIM laser sync
		m_pSyncFlim->start();

		// Find FLIM channel automatically
		//if (m_pOperationTab->isAcquisitionButtonToggled())
		//{
			std::thread find_flim_ch([&]() {

				printf("\nFinding FLIM channel automatically....\n");
				std::this_thread::sleep_for(std::chrono::seconds(1));

				int i = 0;
				for (i = 0; i < 4; i++)
				{
					float* pSrc = m_pMainWnd->m_pStreamTab->m_visPulse.raw_ptr() + i * m_pConfig->nScans + m_pConfig->preTrigSamps;

					int len = N_VIS_SAMPS_FLIM;
					int ind1, ind2;
					Ipp32f max, min;
					ippsMinMaxIndx_32f(pSrc, len, &min, &ind1, &max, &ind2);

					if (max > 35000.0)
					{
						m_pMainWnd->m_pStreamTab->getFlimCh()->setValue(i);
						printf("Active Channel: CH %d\n", i + 1);
						break;
					}
				}

				if (i == 4)
					printf("Fail to find FLIM channel...\n");

			});

			find_flim_ch.detach();
		//}
	}
	else
	{
		// Delete FLIM laser sync control objects
		if (m_pSyncFlim)
		{
			m_pSyncFlim->stop();
			delete m_pSyncFlim;
		}

		// Set enabled true for FLIM laser sync control widgets
		m_pCheckBox_AsyncMode->setEnabled(true);
		if (m_pCheckBox_AsyncMode->isChecked())
		{
			m_pLineEdit_RepetitionRate->setEnabled(true);
			m_pLabel_Hertz->setEnabled(true);
		}

		// Set text
		m_pCheckBox_FlimLaserSyncControl->setText("Enable FLIM Laser Sync Control");
#endif
	}
}

void QDeviceControlTab::enableAsyncMode(bool toggled)
{
	if (toggled)
	{
#if NI_ENABLE
		// Set enabled true for async mode widgets
		m_pLineEdit_RepetitionRate->setEnabled(true);
		m_pLabel_Hertz->setEnabled(true);
	}
	else
	{
		// Set enabled false for async mode widgets
		m_pLineEdit_RepetitionRate->setEnabled(false);
		m_pLabel_Hertz->setEnabled(false);
#endif
	}
}

void QDeviceControlTab::enableFlimLaserPowerControl(bool toggled)
{
	if (toggled)
	{
		// Set text
		m_pCheckBox_FlimLaserPowerControl->setText("Disable FLIM Laser Power Control");

		// Create FLIM laser power control objects
		m_pElforlightLaser = new ElforlightLaser;

		// Connect the laser
		if (!(m_pElforlightLaser->ConnectDevice()))
		{
			m_pCheckBox_FlimLaserPowerControl->setChecked(false);
			return;
		}

		// Set enabled true for FLIM laser power control widgets
		m_pPushButton_IncreasePower->setEnabled(true);
		m_pPushButton_DecreasePower->setEnabled(true);
	}
	else
	{
		// Set enabled false for FLIM laser power control widgets
		m_pPushButton_IncreasePower->setEnabled(false);
		m_pPushButton_DecreasePower->setEnabled(false);

		if (m_pElforlightLaser)
		{
			// Disconnect the laser
			m_pElforlightLaser->DisconnectDevice();

			// Delete FLIM laser power control objects
			delete m_pElforlightLaser;
		}
		
		// Set text
		m_pCheckBox_FlimLaserPowerControl->setText("Enable FLIM Laser Power Control");
	}
}

void QDeviceControlTab::increaseLaserPower()
{
	m_pElforlightLaser->IncreasePower();
}

void QDeviceControlTab::decreaseLaserPower()
{
	m_pElforlightLaser->DecreasePower();
}
#endif

#ifdef OCT_NIRF
void QDeviceControlTab::enableNirfEmissionAcquisition(bool toggled)
{
	if (toggled)
	{
#if NI_ENABLE
		// Set text
		m_pCheckBox_NirfAcquisitionControl->setText("Disable NIRF Acquisition Control");
		m_pCheckBox_NirfAcquisitionControl->setChecked(true);

		// Initializing
		if (!initializeNiDaqAnalogInput())
			return;
	}
	else
	{
		// Stop applying PMT gain voltage
		m_pCheckBox_PmtGainControl->setChecked(false);

		// Close NIRF emission profile dialog
		if (m_pMainWnd->m_pStreamTab->getNirfEmissionProfileDlg())
			m_pMainWnd->m_pStreamTab->getNirfEmissionProfileDlg()->close();
		memset(m_pMainWnd->m_pStreamTab->m_visNirf.raw_ptr(), 0, sizeof(double) * m_pMainWnd->m_pStreamTab->m_visNirf.length());

#ifndef ALAZAR_NIRF_ACQUISITION
		// Delete NIRF emission monitoring objects		
		if (m_pNirfEmission)
		{
			m_pNirfEmission->stop();
			delete m_pNirfEmission;
		}
		///if (m_pNirfEmissionTrigger)
		///{
		///	m_pNirfEmissionTrigger->stop();
		///	delete m_pNirfEmissionTrigger;
		///}
#endif

#ifdef TWO_CHANNEL_NIRF
		// Delete NIRF modulated excitation object
		if (m_pNirfModulation)
		{
			m_pNirfModulation->stop();
			delete m_pNirfModulation;
		}
#endif

		// Set text
		m_pCheckBox_NirfAcquisitionControl->setText("Enable NIRF Acquisition Control");
		m_pCheckBox_NirfAcquisitionControl->setChecked(false);
#endif
	}
}

#ifdef PROGRAMMATIC_GAIN_CONTROL
void QDeviceControlTab::enablePmtGainControl(bool toggled)
{
	if (toggled)
	{
#if NI_ENABLE
		// Set text
		m_pCheckBox_PmtGainControl->setText("Disable PMT Gain Control");

		// Set enabled false for PMT gain control widgets
#ifndef TWO_CHANNEL_NIRF
		m_pLineEdit_PmtGainVoltage->setEnabled(false);
#else
		m_pLineEdit_PmtGainVoltage[0]->setEnabled(false);
		m_pLineEdit_PmtGainVoltage[1]->setEnabled(false);
#endif
		m_pLabel_PmtGainVoltage->setEnabled(false);

		// Create PMT gain control objects
		m_pPmtGainControl = new PmtGainControl;
#ifndef TWO_CHANNEL_NIRF
		m_pPmtGainControl->voltage1 = m_pLineEdit_PmtGainVoltage->text().toDouble();
#else
		m_pPmtGainControl->voltage1 = m_pLineEdit_PmtGainVoltage[0]->text().toDouble();
		m_pPmtGainControl->voltage2 = m_pLineEdit_PmtGainVoltage[1]->text().toDouble();
#endif

#ifndef TWO_CHANNEL_NIRF
		if (m_pPmtGainControl->voltage1 > 1.0)
#else
		if ((m_pPmtGainControl->voltage1 > 1.0) || (m_pPmtGainControl->voltage2 > 1.0))
#endif
		{
			printf(">1.0V Gain cannot be assigned!\n");
			m_pCheckBox_PmtGainControl->setChecked(false);
			return;
		}

		// Initializing
		if (!m_pPmtGainControl->initialize())
		{
			m_pCheckBox_PmtGainControl->setChecked(false);
			return;
		}

		// Generate PMT gain voltage
		m_pPmtGainControl->start();
	}
	else
	{
		// Delete PMT gain control objects
		if (m_pPmtGainControl)
		{
			m_pPmtGainControl->stop();
			delete m_pPmtGainControl;
		}

		// Set enabled true for PMT gain control widgets
#ifndef TWO_CHANNEL_NIRF
		m_pLineEdit_PmtGainVoltage->setEnabled(true);
#else
		m_pLineEdit_PmtGainVoltage[0]->setEnabled(true);
		m_pLineEdit_PmtGainVoltage[1]->setEnabled(true);
#endif
		m_pLabel_PmtGainVoltage->setEnabled(true);

		// Set text
		m_pCheckBox_PmtGainControl->setText("Enable PMT Gain Control");
#endif
	}
}

#ifndef TWO_CHANNEL_NIRF
void QDeviceControlTab::changePmtGainVoltage(const QString &str)
{
	m_pConfig->pmtGainVoltage = str.toFloat();
}
#else
void QDeviceControlTab::changePmtGainVoltage1(const QString &str)
{
	m_pConfig->pmtGainVoltage[0] = str.toFloat();
}

void QDeviceControlTab::changePmtGainVoltage2(const QString &str)
{
	m_pConfig->pmtGainVoltage[1] = str.toFloat();
}
#endif
#endif
#endif

#ifdef GALVANO_MIRROR
void QDeviceControlTab::enableGalvanoMirror(bool toggled)
{
	if (toggled)
	{
#if NI_ENABLE
		// Set text
		m_pCheckBox_GalvanoMirrorControl->setText("Disable Galvano Mirror Control");

		// Set enabled false for Galvano mirror control widgets	
		m_pToggleButton_ScanTriggering->setEnabled(false);

		m_pLabel_FastScanVoltage->setEnabled(false);
		m_pLineEdit_FastPeakToPeakVoltage->setEnabled(false);
		m_pLabel_FastScanPlusMinus->setEnabled(false);
		m_pLineEdit_FastOffsetVoltage->setEnabled(false);
		m_pLabel_FastGalvanoVoltage->setEnabled(false);

		m_pLabel_SlowScanVoltage->setEnabled(false);
		m_pLineEdit_SlowPeakToPeakVoltage->setEnabled(false);
		m_pLabel_SlowScanPlusMinus->setEnabled(false);
		m_pLineEdit_SlowOffsetVoltage->setEnabled(false);
		m_pLabel_SlowGalvanoVoltage->setEnabled(false);

		m_pLabel_SlowScanIncrement->setEnabled(false);
		m_pLineEdit_SlowScanIncrement->setEnabled(false);
		m_pLabel_SlowScanIncrementVoltage->setEnabled(false);

		m_pScrollBar_ScanAdjustment->setEnabled(true);
		m_pLabel_ScanAdjustment->setEnabled(true);
		
		// Create Galvano mirror control objects
		m_pGalvoScan = new GalvoScan;
		m_pGalvoScan->nAlines = m_pConfig->nAlines;
		m_pGalvoScan->pp_voltage_fast = m_pLineEdit_FastPeakToPeakVoltage->text().toDouble();
		m_pGalvoScan->offset_fast = m_pLineEdit_FastOffsetVoltage->text().toDouble();
		m_pGalvoScan->pp_voltage_slow = m_pLineEdit_SlowPeakToPeakVoltage->text().toDouble();
		m_pGalvoScan->offset_slow = m_pLineEdit_SlowOffsetVoltage->text().toDouble();
		m_pGalvoScan->step_slow = m_pLineEdit_SlowScanIncrement->text().toDouble();
		if (m_pGalvoScan->step_slow < 0.00001)
		{
			m_pCheckBox_GalvanoMirrorControl->setChecked(false);
			return;
		}
		
		// Start & Stop function definition
		m_pGalvoScan->stopScan += [&]() {
			m_pCheckBox_GalvanoMirrorControl->setChecked(false);
		};
		m_pGalvoScan->startScan += [&]() {
			//printf("scan start");
			//m_pGalvoScan->start();
		};

		// Initializing
		if (!m_pGalvoScan->initialize())
		{
			m_pCheckBox_GalvanoMirrorControl->setChecked(false);
			return;
		}

		if (!m_pToggleButton_ScanTriggering->isChecked())
		{
			// Start Scanning
			m_pGalvoScan->start();
		}
	}
	else
	{
		// Delete Galvano mirror control objects
		if (m_pGalvoScan)
		{
			m_pGalvoScan->stop();
			delete m_pGalvoScan;
		}

		// Set enabled false for Galvano mirror control widgets	
		if (m_pConfig->galvoSlowScanVoltage != 0)
			m_pToggleButton_ScanTriggering->setEnabled(true);

		m_pLabel_FastScanVoltage->setEnabled(true);
		m_pLineEdit_FastPeakToPeakVoltage->setEnabled(true);
		m_pLabel_FastScanPlusMinus->setEnabled(true);
		m_pLineEdit_FastOffsetVoltage->setEnabled(true);
		m_pLabel_FastGalvanoVoltage->setEnabled(true);

		m_pLabel_SlowScanVoltage->setEnabled(true);
		m_pLineEdit_SlowPeakToPeakVoltage->setEnabled(true);
		m_pLabel_SlowScanPlusMinus->setEnabled(true);
		m_pLineEdit_SlowOffsetVoltage->setEnabled(true);
		m_pLabel_SlowGalvanoVoltage->setEnabled(true);

		m_pLabel_SlowScanIncrement->setEnabled(true);
		m_pLineEdit_SlowScanIncrement->setEnabled(true);
		m_pLabel_SlowScanIncrementVoltage->setEnabled(true);
		
		m_pScrollBar_ScanAdjustment->setEnabled(false);
		m_pScrollBar_ScanAdjustment->setValue(0);
		m_pLabel_ScanAdjustment->setEnabled(false);
		QString str; str.sprintf("Fast Scan Adjustment  %d / %d", 0, m_pConfig->nAlines - 1);
		m_pLabel_ScanAdjustment->setText(str);

		// Set text
		m_pCheckBox_GalvanoMirrorControl->setText("Enable Galvano Mirror Control");
#endif
	}
}

void QDeviceControlTab::triggering(bool checked)
{
	if (checked)
	{
		// Set text
		m_pToggleButton_ScanTriggering->setText("Scan Triggering Mode Off");		
	}
	else
	{
		// Set text
		m_pToggleButton_ScanTriggering->setText("Scan Triggering Mode On");
	}
}

void QDeviceControlTab::changeGalvoFastScanVoltage(const QString &str)
{
	m_pConfig->galvoFastScanVoltage = str.toFloat();
}

void QDeviceControlTab::changeGalvoFastScanVoltageOffset(const QString &str)
{
	m_pConfig->galvoFastScanVoltageOffset = str.toFloat();
}

void QDeviceControlTab::changeGalvoSlowScanVoltage(const QString &str)
{
	m_pConfig->galvoSlowScanVoltage = str.toFloat();

	if (m_pConfig->galvoSlowScanVoltage == 0.0)
	{
		m_pToggleButton_ScanTriggering->setChecked(false);
		m_pToggleButton_ScanTriggering->setEnabled(false);
	}
	else
		m_pToggleButton_ScanTriggering->setEnabled(true);

	int nIter = int(ceil(m_pConfig->galvoSlowScanVoltage / m_pConfig->galvoSlowScanIncrement)) + 1;
	printf("Number of fast scan: %d (Estimated scan time: %.2f sec)\n", nIter, double(nIter) / (200'000.0 / m_pConfig->nAlines));
}

void QDeviceControlTab::changeGalvoSlowScanVoltageOffset(const QString &str)
{
	m_pConfig->galvoSlowScanVoltageOffset = str.toFloat();
}

void QDeviceControlTab::changeGalvoSlowScanIncrement(const QString &str)
{
	m_pConfig->galvoSlowScanIncrement = str.toFloat();
	int nIter = int(ceil(m_pConfig->galvoSlowScanVoltage / m_pConfig->galvoSlowScanIncrement)) + 1;
	printf("Number of fast scan: %d (Estimated scan time: %.2f sec)\n", nIter, double(nIter) / (200'000.0 / m_pConfig->nAlines));
}

void QDeviceControlTab::scanAdjusting(int horizontalShift)
{
	m_pConfig->galvoHorizontalShift = horizontalShift;

	QString str;
	if (!m_pMainWnd->m_pTabWidget->currentIndex()) // Streaming Tab
	{
		str.sprintf("Fast Scan Adjustment  %d / %d", horizontalShift, m_pConfig->nAlines - 1);
		m_pMainWnd->m_pStreamTab->invalidate();
	}
	else // Result Tab
	{
		//str.sprintf("Fast Scan Adjustment  %d / %d", horizontalShift,
			//(m_pMainWnd->m_pResultTab->m_pCirc ? m_pMainWnd->m_pResultTab->m_pCirc->alines : m_pConfig->nAlines) - 1);
		m_pMainWnd->m_pResultTab->invalidate();
	}

	m_pLabel_ScanAdjustment->setText(str);
}
#endif

#ifdef PULLBACK_DEVICE
void QDeviceControlTab::enableZaberStageControl(bool toggled)
{
	if (toggled)
	{
		// Set text
		m_pCheckBox_ZaberStageControl->setText("Disable Zaber Stage Control");

		// Create Zaber stage control objects
#ifndef DOTTER_STAGE
#ifndef ZABER_NEW_STAGE
		m_pZaberStage = new ZaberStage;

		// Connect stage
		if (!(m_pZaberStage->ConnectStage()))
		{
			m_pCheckBox_ZaberStageControl->setChecked(false);
			return;
		}

		// Set target speed first
		m_pZaberStage->SetTargetSpeed(m_pLineEdit_TargetSpeed->text().toDouble());
#else
		m_pZaberStage = new ZaberStage2;

		// Connect stage
		if (!(m_pZaberStage->ConnectDevice()))
		{
			m_pCheckBox_ZaberStageControl->setChecked(false);
			return;
		}

		m_pZaberMonitorTimer = new QTimer(this);
		connect(m_pZaberMonitorTimer, &QTimer::timeout, [&]() 
		{ 
			if (m_pZaberStage->getIsMoving())
				m_pZaberStage->GetPos(1); 
		});
		m_pZaberMonitorTimer->start(100);

		// Set target speed first
		m_pZaberStage->SetTargetSpeed(1, m_pLineEdit_TargetSpeed->text().toDouble());
#endif
#else
		if (!m_pFaulhaberMotor)
		{
			// Create Faulhaber motor control objects (as a surrogate of stage object)
			m_pFaulhaberMotor = new FaulhaberMotor;

			// Connect the motor
			if (!(m_pFaulhaberMotor->ConnectDevice()))
			{
				m_pCheckBox_FaulhaberMotorControl->setChecked(false);
				return;
			}
		}

		// Set target speed first
		home();
		m_pFaulhaberMotor->SetProfileVelocity(2, m_pLineEdit_TargetSpeed->text().toInt());
#endif

		// Set enable true for Zaber stage control widgets
		m_pPushButton_MoveAbsolute->setEnabled(true);
		m_pPushButton_SetTargetSpeed->setEnabled(true);
		m_pPushButton_Home->setEnabled(true);
		m_pPushButton_Stop->setEnabled(true);
		m_pLineEdit_TravelLength->setEnabled(true);
		m_pLineEdit_TargetSpeed->setEnabled(true);
		m_pLabel_TravelLength->setEnabled(true);
		m_pLabel_TargetSpeed->setEnabled(true);
	}
	else
	{
		// Set enable false for Zaber stage control widgets
		m_pPushButton_MoveAbsolute->setEnabled(false);
		m_pPushButton_SetTargetSpeed->setEnabled(false);
		m_pPushButton_Home->setEnabled(false);
		m_pPushButton_Stop->setEnabled(false);
		m_pLineEdit_TravelLength->setEnabled(false);
		m_pLineEdit_TargetSpeed->setEnabled(false);
		m_pLabel_TravelLength->setEnabled(false);
		m_pLabel_TargetSpeed->setEnabled(false);

#ifndef DOTTER_STAGE
		if (m_pZaberStage)
		{
#ifndef ZABER_NEW_STAGE
			// Stop Wait Thread
			m_pZaberStage->StopWaitThread();

			// Disconnect the Stage
			m_pZaberStage->DisconnectStage();
#else
			// Disconnect the Stage
			m_pZaberStage->DisconnectDevice();

			if (m_pZaberMonitorTimer)
			{
				m_pZaberMonitorTimer->stop();
				delete m_pZaberMonitorTimer;
			}
#endif

			// Delete Zaber stage control objects
			delete m_pZaberStage;
		}
#else
		if (!m_pCheckBox_FaulhaberMotorControl->isChecked())
		{
			if (m_pFaulhaberMotor)
			{
				// Disconnect the motor
				m_pFaulhaberMotor->DisconnectDevice();

				// Delete Faulhaber motor control objects
				delete m_pFaulhaberMotor;
				m_pFaulhaberMotor = nullptr;
			}
		}
#endif

		// Set text
		m_pCheckBox_ZaberStageControl->setText("Enable Zaber Stage Control");
	}
}

void QDeviceControlTab::moveAbsolute()
{
#ifndef DOTTER_STAGE
#ifndef ZABER_NEW_STAGE
	m_pZaberStage->MoveAbsoulte(m_pLineEdit_TravelLength->text().toDouble());
#else
	m_pZaberStage->MoveAbsolute(1, m_pLineEdit_TravelLength->text().toDouble());
#endif
#else
	int travel_length = int(-m_pLineEdit_TravelLength->text().toDouble() / 10.0 * 4096.0);
	m_pFaulhaberMotor->SetTargetPosition(2, travel_length);
#endif
}

void QDeviceControlTab::setTargetSpeed(const QString & str)
{
#ifndef DOTTER_STAGE
#ifndef ZABER_NEW_STAGE
	m_pZaberStage->SetTargetSpeed(str.toDouble());
#else
	m_pZaberStage->SetTargetSpeed(1, str.toDouble());
#endif	
#else
	m_pFaulhaberMotor->SetProfileVelocity(2, str.toInt());
#endif
	m_pConfig->zaberPullbackSpeed = str.toInt();
}

void QDeviceControlTab::changeZaberPullbackLength(const QString &str)
{
	m_pConfig->zaberPullbackLength = str.toInt();
}

void QDeviceControlTab::home()
{
#ifndef DOTTER_STAGE
#ifndef ZABER_NEW_STAGE
	m_pZaberStage->Home();
#else
	m_pZaberStage->Home(1);
#endif
#else
	m_pFaulhaberMotor->SetTargetPosition(2, 40960);	
#endif
}

void QDeviceControlTab::stop()
{
#ifndef DOTTER_STAGE
#ifndef ZABER_NEW_STAGE
	m_pZaberStage->Stop();
#else
	m_pZaberStage->Stop(1);
	m_pZaberStage->GetPos(1);
#endif
#else
	m_pFaulhaberMotor->DisableMotor(2);
#endif
}


void QDeviceControlTab::enableFaulhaberMotorControl(bool toggled)
{
	if (toggled)
	{
		// Set text
		m_pCheckBox_FaulhaberMotorControl->setText("Disable Faulhaber Motor Control");

		// Create Faulhaber motor control objects
		if (!m_pFaulhaberMotor)
		{
			m_pFaulhaberMotor = new FaulhaberMotor;

			// Connect the motor
			if (!(m_pFaulhaberMotor->ConnectDevice()))
			{
				m_pCheckBox_FaulhaberMotorControl->setChecked(false);
				return;
			}
		}
		
		// Set enable true for Faulhaber motor control widgets
		m_pPushButton_Rotate->setEnabled(true);
		m_pPushButton_RotateStop->setEnabled(true);
		m_pLineEdit_RPM->setEnabled(true);
		m_pLabel_RPM->setEnabled(true);
	}
	else
	{
		// Set enable false for Faulhaber motor control widgets
		m_pPushButton_Rotate->setEnabled(false);
		m_pPushButton_RotateStop->setEnabled(false);
		m_pLineEdit_RPM->setEnabled(false);
		m_pLabel_RPM->setEnabled(false);
		
#ifdef DOTTER_STAGE
		if (!m_pCheckBox_ZaberStageControl->isChecked())
		{
#endif
			if (m_pFaulhaberMotor)
			{
				// Disconnect the motor
				m_pFaulhaberMotor->DisconnectDevice();

				// Delete Faulhaber motor control objects
				delete m_pFaulhaberMotor;
				m_pFaulhaberMotor = nullptr;
			}
#ifdef DOTTER_STAGE
		}
#endif

		// Set text
		m_pCheckBox_FaulhaberMotorControl->setText("Enable Faulhaber Motor Control");
	}
}

void QDeviceControlTab::rotate()
{
	m_pFaulhaberMotor->RotateMotor(1, m_pLineEdit_RPM->text().toInt());
}

void QDeviceControlTab::rotateStop()
{
	m_pFaulhaberMotor->StopMotor(1);
}

void QDeviceControlTab::changeFaulhaberRpm(const QString &str)
{
	m_pConfig->faulhaberRpm = str.toInt();
}
#endif
