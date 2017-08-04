
#include "QDeviceControlTab.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QStreamTab.h>

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
#ifdef GALVANO_MIRROR
#if NI_ENABLE
#include <DeviceControl/GalvoScan/GalvoScan.h>
#endif
#endif
#ifdef PULLBACK_DEVICE
#include <DeviceControl/ZaberStage/ZaberStage.h>
#include <DeviceControl/FaulhaberMotor/FaulhaberMotor.h>
#endif

#include <iostream>
#include <deque>
#include <thread>
#include <chrono>


QDeviceControlTab::QDeviceControlTab(QWidget *parent) :
    QDialog(parent)
{
	// Set main window objects
	m_pMainWnd = (MainWindow*)parent;
	m_pConfig = m_pMainWnd->m_pConfiguration;
	m_pStreamTab = m_pMainWnd->m_pStreamTab;


    // Create layout
    m_pVBoxLayout = new QVBoxLayout;
    m_pVBoxLayout->setSpacing(0);

#ifdef OCT_FLIM
	m_pVBoxLayout_FlimControl = new QVBoxLayout;
	m_pVBoxLayout_FlimControl->setSpacing(5);
	m_pGroupBox_FlimControl = new QGroupBox;
#endif

#ifdef ECG_TRIGGERING
	createEcgModuleControl();
#endif
#ifdef OCT_FLIM
	createPmtGainControl();
	createFlimLaserSyncControl();
	createFlimLaserPowerControl();
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
#ifdef GALVANO_MIRROR
#if NI_ENABLE
	if (m_pCheckBox_GalvanoMirrorControl->isChecked()) m_pCheckBox_GalvanoMirrorControl->setChecked(false);
#endif
#endif
#ifdef PULLBACK_DEVICE
	if (m_pCheckBox_ZaberStageControl->isChecked()) m_pCheckBox_ZaberStageControl->setChecked(false);
	if (m_pCheckBox_FaulhaberMotorControl->isChecked()) m_pCheckBox_FaulhaberMotorControl->setChecked(false);
#endif

	e->accept();
}


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
	
	m_pEcgScope = new QEcgScope({ 0, N_VIS_SAMPS_ECG }, { -1, 1 }, 2, 2, 0.001, 1, 0, 0, "sec", "V");
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
	m_pGroupBox_FlimControl->resize(m_pVBoxLayout_FlimControl->minimumSize());
    m_pVBoxLayout->addWidget(m_pGroupBox_FlimControl);

	// Connect signal and slot
	connect(m_pCheckBox_FlimLaserPowerControl, SIGNAL(toggled(bool)), this, SLOT(enableFlimLaserPowerControl(bool)));
	connect(m_pPushButton_IncreasePower, SIGNAL(clicked(bool)), this, SLOT(increaseLaserPower()));
	connect(m_pPushButton_DecreasePower, SIGNAL(clicked(bool)), this, SLOT(decreaseLaserPower()));
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

    m_pLineEdit_PeakToPeakVoltage = new QLineEdit(pGroupBox_GalvanoMirrorControl);
    m_pLineEdit_PeakToPeakVoltage->setFixedWidth(30);
    m_pLineEdit_PeakToPeakVoltage->setText(QString::number(m_pConfig->galvoScanVoltage, 'f', 1));
	m_pLineEdit_PeakToPeakVoltage->setAlignment(Qt::AlignCenter);
    m_pLineEdit_PeakToPeakVoltage->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

    m_pLineEdit_OffsetVoltage = new QLineEdit(pGroupBox_GalvanoMirrorControl);
    m_pLineEdit_OffsetVoltage->setFixedWidth(30);
    m_pLineEdit_OffsetVoltage->setText(QString::number(m_pConfig->galvoScanVoltageOffset, 'f', 1));
	m_pLineEdit_OffsetVoltage->setAlignment(Qt::AlignCenter);
    m_pLineEdit_OffsetVoltage->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

    m_pLabel_ScanVoltage = new QLabel("Scan Voltage  ", pGroupBox_GalvanoMirrorControl);
    m_pLabel_ScanPlusMinus = new QLabel("+", pGroupBox_GalvanoMirrorControl);
    m_pLabel_ScanPlusMinus->setBuddy(m_pLineEdit_PeakToPeakVoltage);
    m_pLabel_GalvanoVoltage = new QLabel("V", pGroupBox_GalvanoMirrorControl);
    m_pLabel_GalvanoVoltage->setBuddy(m_pLineEdit_OffsetVoltage);

    m_pScrollBar_ScanAdjustment = new QScrollBar(pGroupBox_GalvanoMirrorControl);
    m_pScrollBar_ScanAdjustment->setOrientation(Qt::Horizontal);
	m_pScrollBar_ScanAdjustment->setRange(0, m_pConfig->nAlines - 1);
	m_pScrollBar_ScanAdjustment->setSingleStep(1);
	m_pScrollBar_ScanAdjustment->setPageStep(m_pScrollBar_ScanAdjustment->maximum() / 10);
    m_pLabel_ScanAdjustment = new QLabel("Scan Adjustment", pGroupBox_GalvanoMirrorControl);
    m_pLabel_ScanAdjustment->setBuddy(m_pScrollBar_ScanAdjustment);

    pGridLayout_GalvanoMirrorControl->addWidget(m_pCheckBox_GalvanoMirrorControl, 0, 0, 1, 6);
    pGridLayout_GalvanoMirrorControl->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_ScanVoltage, 1, 1);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pLineEdit_PeakToPeakVoltage, 1, 2);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_ScanPlusMinus, 1, 3);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pLineEdit_OffsetVoltage, 1, 4);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_GalvanoVoltage, 1, 5);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pLabel_ScanAdjustment, 2, 0, 1, 6);
    pGridLayout_GalvanoMirrorControl->addWidget(m_pScrollBar_ScanAdjustment, 3, 0, 1, 6);

    pGroupBox_GalvanoMirrorControl->setLayout(pGridLayout_GalvanoMirrorControl);
    m_pVBoxLayout->addWidget(pGroupBox_GalvanoMirrorControl);

	// Connect signal and slot
	connect(m_pCheckBox_GalvanoMirrorControl, SIGNAL(toggled(bool)), this, SLOT(enableGalvanoMirror(bool)));
	connect(m_pScrollBar_ScanAdjustment, SIGNAL(valueChanged(int)), this, SLOT(scanAdjusting(int)));
	connect(m_pLineEdit_PeakToPeakVoltage, SIGNAL(textChanged(const QString &)), this, SLOT(changeGalvoScanVoltage(const QString &)));
	connect(m_pLineEdit_OffsetVoltage, SIGNAL(textChanged(const QString &)), this, SLOT(changeGalvoScanVoltageOffset(const QString &)));
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

    m_pToggleButton_Rotate = new QPushButton(pGroupBox_FaulhaberMotorControl);
    m_pToggleButton_Rotate->setText("Rotate");
    m_pToggleButton_Rotate->setCheckable(pGroupBox_FaulhaberMotorControl);
	m_pToggleButton_Rotate->setDisabled(true);

    m_pLineEdit_RPM = new QLineEdit(pGroupBox_FaulhaberMotorControl);
    m_pLineEdit_RPM->setFixedWidth(40);
    m_pLineEdit_RPM->setText(QString::number(m_pConfig->faulhaberRpm));
	m_pLineEdit_RPM->setAlignment(Qt::AlignCenter);
    m_pLineEdit_RPM->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	m_pLineEdit_RPM->setDisabled(true);

    m_pLabel_RPM = new QLabel("RPM", pGroupBox_FaulhaberMotorControl);
    m_pLabel_RPM->setBuddy(m_pLineEdit_RPM);
	m_pLabel_RPM->setDisabled(true);

    pGridLayout_FaulhaberMotorControl->addWidget(m_pCheckBox_FaulhaberMotorControl, 0, 0, 1, 4);
    pGridLayout_FaulhaberMotorControl->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
    pGridLayout_FaulhaberMotorControl->addWidget(m_pToggleButton_Rotate, 1, 1);
    pGridLayout_FaulhaberMotorControl->addWidget(m_pLineEdit_RPM, 1, 2);
    pGridLayout_FaulhaberMotorControl->addWidget(m_pLabel_RPM, 1, 3);

    pGroupBox_FaulhaberMotorControl->setLayout(pGridLayout_FaulhaberMotorControl);
    m_pVBoxLayout->addWidget(pGroupBox_FaulhaberMotorControl);
	m_pVBoxLayout->addStretch(1);

	// Connect signal and slot
	connect(m_pCheckBox_FaulhaberMotorControl, SIGNAL(toggled(bool)), this, SLOT(enableFaulhaberMotorControl(bool)));
	connect(m_pToggleButton_Rotate, SIGNAL(toggled(bool)), this, SLOT(rotate(bool)));
	connect(m_pLineEdit_RPM, SIGNAL(textChanged(const QString &)), this, SLOT(changeFaulhaberRpm(const QString &)));
}
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
		
		m_pEcgMonitoring->acquiredData += [&](double& data) {
			m_pEcgScope->drawData(data);
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
		if (m_pEcgMonitoringTrigger)
		{
			m_pEcgMonitoringTrigger->stop();
			delete m_pEcgMonitoringTrigger;
		}
		if (m_pEcgMonitoring)
		{
			m_pEcgMonitoring->stop();
			delete m_pEcgMonitoring;
		}

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
		m_pPmtGainControl->voltage = m_pLineEdit_PmtGainVoltage->text().toDouble();
		if (m_pPmtGainControl->voltage > 1.0)
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
		std::thread find_flim_ch([&]() {

			printf("\nFinding FLIM channel automatically....\n");
			std::this_thread::sleep_for(std::chrono::seconds(1));

			int i = 0;
			for (i = 0; i < 4; i++)
			{
				float* pSrc = m_pStreamTab->m_visPulse.raw_ptr() + i * m_pConfig->nScans + m_pConfig->preTrigSamps;

				int len = N_VIS_SAMPS_FLIM;
				int ind1, ind2;
				Ipp32f max, min;
				ippsMinMaxIndx_32f(pSrc, len, &min, &ind1, &max, &ind2);

				if (max > 35000.0)
				{
					m_pStreamTab->getFlimCh()->setValue(i);
					printf("Active Channel: CH %d\n", i + 1);
					break;
				}
			}

			if (i == 4)
				printf("Fail to find FLIM channel...\n");

		});
		find_flim_ch.detach();
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

#ifdef GALVANO_MIRROR
void QDeviceControlTab::enableGalvanoMirror(bool toggled)
{
	if (toggled)
	{
#if NI_ENABLE
		// Set text
		m_pCheckBox_GalvanoMirrorControl->setText("Disable Galvano Mirror Control");

		// Set enabled false for Galvano mirror control widgets	
		m_pLabel_ScanVoltage->setEnabled(false);
		m_pLineEdit_PeakToPeakVoltage->setEnabled(false);
		m_pLabel_ScanPlusMinus->setEnabled(false);
		m_pLineEdit_OffsetVoltage->setEnabled(false);
		m_pLabel_GalvanoVoltage->setEnabled(false);

		// Create Galvano mirror control objects
		m_pGalvoScan = new GalvoScan;
		m_pGalvoScan->nAlines = m_pConfig->nAlines;
		m_pGalvoScan->pp_voltage = m_pLineEdit_PeakToPeakVoltage->text().toDouble();
		m_pGalvoScan->offset = m_pLineEdit_OffsetVoltage->text().toDouble();

		// Initializing
		if (!m_pGalvoScan->initialize())
		{
			m_pCheckBox_GalvanoMirrorControl->setChecked(false);
			return;
		}

		// Start scanning with Galvano mirror
		m_pGalvoScan->start();
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
		m_pLabel_ScanVoltage->setEnabled(true);
		m_pLineEdit_PeakToPeakVoltage->setEnabled(true);
		m_pLabel_ScanPlusMinus->setEnabled(true);
		m_pLineEdit_OffsetVoltage->setEnabled(true);
		m_pLabel_GalvanoVoltage->setEnabled(true);

		// Set text
		m_pCheckBox_GalvanoMirrorControl->setText("Enable Galvano Mirror Control");
#endif
	}
}

void QDeviceControlTab::scanAdjusting(int)
{
	m_pStreamTab->invalidate();
}

void QDeviceControlTab::changeGalvoScanVoltage(const QString &str)
{
	m_pConfig->galvoScanVoltage = str.toFloat();
}

void QDeviceControlTab::changeGalvoScanVoltageOffset(const QString &str)
{
	m_pConfig->galvoScanVoltageOffset = str.toFloat();
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
		m_pZaberStage = new ZaberStage;
		
		// Connect stage
		if (!(m_pZaberStage->ConnectStage()))
		{
			m_pCheckBox_ZaberStageControl->setChecked(false);
			return;
		}

		// Set target speed first
		m_pZaberStage->SetTargetSpeed(m_pLineEdit_TargetSpeed->text().toDouble());

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

		if (m_pZaberStage)
		{
			// Stop Wait Thread
			m_pZaberStage->StopWaitThread();

			// Disconnect the Stage
			m_pZaberStage->DisconnectStage();

			// Delete Zaber stage control objects
			delete m_pZaberStage;
		}

		// Set text
		m_pCheckBox_ZaberStageControl->setText("Enable Zaber Stage Control");
	}
}

void QDeviceControlTab::moveAbsolute()
{
	m_pZaberStage->MoveAbsoulte(m_pLineEdit_TravelLength->text().toDouble());
}

void QDeviceControlTab::setTargetSpeed(const QString & str)
{
	m_pZaberStage->SetTargetSpeed(str.toDouble());
	m_pConfig->zaberPullbackSpeed = str.toInt();
}

void QDeviceControlTab::changeZaberPullbackLength(const QString &str)
{
	m_pConfig->zaberPullbackLength = str.toInt();
}

void QDeviceControlTab::home()
{
	m_pZaberStage->Home();
}

void QDeviceControlTab::stop()
{
	m_pZaberStage->Stop();
}


void QDeviceControlTab::enableFaulhaberMotorControl(bool toggled)
{
	if (toggled)
	{
		// Set text
		m_pCheckBox_FaulhaberMotorControl->setText("Disable Faulhaber Motor Control");

		// Create Faulhaber motor control objects
		m_pFaulhaberMotor = new FaulhaberMotor;

		// Connect the motor
		if (!(m_pFaulhaberMotor->ConnectDevice()))
		{
			m_pCheckBox_FaulhaberMotorControl->setChecked(false);
			return;
		}
		
		// Set enable true for Faulhaber motor control widgets
		m_pToggleButton_Rotate->setEnabled(true);
		m_pLineEdit_RPM->setEnabled(true);
		m_pLabel_RPM->setEnabled(true);
	}
	else
	{
		// Set enable false for Faulhaber motor control widgets
		m_pToggleButton_Rotate->setChecked(false);
		m_pToggleButton_Rotate->setEnabled(false);
		m_pLineEdit_RPM->setEnabled(false);
		m_pLabel_RPM->setEnabled(false);
		
		if (m_pFaulhaberMotor)
		{
			// Disconnect the motor
			m_pFaulhaberMotor->DisconnectDevice();

			// Delete Faulhaber motor control objects
			delete m_pFaulhaberMotor;
		}

		// Set text
		m_pCheckBox_FaulhaberMotorControl->setText("Enable Faulhaber Motor Control");
	}
}

void QDeviceControlTab::rotate(bool toggled)
{
	if (toggled)
	{
		m_pFaulhaberMotor->RotateMotor(m_pLineEdit_RPM->text().toInt());
		m_pLineEdit_RPM->setEnabled(false);
		m_pToggleButton_Rotate->setText("Stop");
	}
	else
	{
		m_pFaulhaberMotor->StopMotor();
		m_pLineEdit_RPM->setEnabled(true);
		m_pToggleButton_Rotate->setText("Rotate");
	}
}

void QDeviceControlTab::changeFaulhaberRpm(const QString &str)
{
	m_pConfig->faulhaberRpm = str.toInt();
}
#endif
