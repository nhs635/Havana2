#ifndef QDEVICECONTROLTAB_H
#define QDEVICECONTROLTAB_H

#include <QDialog>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>

#include <iostream>
#include <deque>
#include <mutex>
#include <condition_variable>

class MainWindow;
class QOperationTab;
class QStreamTab;
class QResultTab;

#ifdef AXSUN_OCT_LASER
class AxsunControl;
#endif
#ifdef ECG_TRIGGERING
#if NI_ENABLE
class QEcgScope;

class EcgMonitoringTrigger;
class EcgMonitoring;
class Voltage800RPS;
#endif
#endif
#ifdef OCT_FLIM
#if NI_ENABLE
class PmtGainControl;
class SyncFLIM;
#endif
class ElforlightLaser;
#endif
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
#ifndef ALAZAR_NIRF_ACQUISITION
class NirfEmissionTrigger;
class NirfEmission;
#endif
#ifdef TWO_CHANNEL_NIRF
class NirfModulation;
#endif
#ifdef PROGRAMMATIC_GAIN_CONTROL
class PmtGainControl;
#endif
#endif
#endif
#ifdef GALVANO_MIRROR
#if NI_ENABLE
class GalvoScan;
#endif
#endif
#ifdef PULLBACK_DEVICE
#ifndef ZABER_NEW_STAGE
class ZaberStage;
#else
class ZaberStage2;
#endif
class FaulhaberMotor;
#endif


class QDeviceControlTab : public QDialog
{
    Q_OBJECT

// Constructer & Destructer /////////////////////////////
public:
    explicit QDeviceControlTab(QWidget *parent = 0);
    virtual ~QDeviceControlTab();

// Methods //////////////////////////////////////////////
protected:
	void closeEvent(QCloseEvent*);

public: ////////////////////////////////////////////////////////////////////////////////////////////////
    inline QVBoxLayout* getLayout() const { return m_pVBoxLayout; }
#ifdef AXSUN_OCT_LASER
	inline QCheckBox* getEnableAxsunOCTLaserControl() const { return m_pCheckBox_AxsunOCTLaserControl; }
#endif
#ifdef OCT_FLIM
	inline QCheckBox* getEnablePmtGainControl() const { return m_pCheckBox_PmtGainControl; }
	inline QCheckBox* getEnableFlimLaserSyncControl() const { return m_pCheckBox_FlimLaserSyncControl; }
	inline QCheckBox* getFlimAsyncMode() const { return m_pCheckBox_AsyncMode; }
#endif
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
#ifndef ALAZAR_NIRF_ACQUISITION
	inline NirfEmission* getNirfEmission() const { return m_pNirfEmission; }
#endif
#ifdef PROGRAMMATIC_GAIN_CONTROL
	inline QCheckBox* getEnablePmtGainControl() const { return m_pCheckBox_PmtGainControl; }
#endif
	inline void startNirfAcquisition() { enableNirfEmissionAcquisition(true); }
	inline void stopNirfAcquisition() { enableNirfEmissionAcquisition(false); }
#endif
#endif
#ifdef GALVANO_MIRROR
	inline QCheckBox* getEnableGalvanoMirrorControl() const { return m_pCheckBox_GalvanoMirrorControl; }
#endif
#ifdef PULLBACK_DEVICE
#ifndef ZABER_NEW_STAGE
	inline ZaberStage* getZaberStage() const { return m_pZaberStage; }
#else
	inline ZaberStage2* getZaberStage() const { return m_pZaberStage; }
#endif
	inline FaulhaberMotor* getFaulhaberMotor() const { return m_pFaulhaberMotor; }
#endif

private: ////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef AXSUN_OCT_LASER
    void createAxsunOCTLaserControl();
#endif
#ifdef ECG_TRIGGERING
	void createEcgModuleControl();
#endif
#ifdef OCT_FLIM
	void createPmtGainControl();
	void createFlimLaserSyncControl();
	void createFlimLaserPowerControl();
#endif
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
	void createNirfAcquisitionControl();
#endif
#endif
#ifdef GALVANO_MIRROR
    void createGalvanoMirrorControl();
#endif
#ifdef PULLBACK_DEVICE
    void createZaberStageControl();
    void createFaulhaberMotorControl();
#endif

public: ////////////////////////////////////////////////////////////////////////////////////////////////
	// Initiating & Terminating..
	void initiateAllDevices();
	void terminateAllDevices();

#ifdef AXSUN_OCT_LASER
    // Axsun OCT Laser Control
	void turnOnOCTLaser(bool set);
#endif
#ifdef ECG_TRIGGERING
#if NI_ENABLE
	// ECG Module Control
	bool isEcgModuleEnabled() { return m_pCheckBox_EcgModuleControl->isChecked(); }
	bool isEcgTriggered() { return m_pToggledButton_EcgTriggering->isChecked(); }
	double getEcgDelayRate() { return m_pDoubleSpinBox_EcgDelayRate->value(); }
	void setEcgRecording(bool set);
	double getEcgHeartInterval();
	std::deque<double>* getRecordedEcg();
#endif	
#endif
#ifdef OCT_NIRF
    // NIRF Analog Input Control
	bool initializeNiDaqAnalogInput();
	bool startNiDaqAnalogInput();
#endif
#ifdef GALVANO_MIRROR
    // Galvano Stage Control
	int getScrollBarValue() { return m_pScrollBar_ScanAdjustment->value(); }
	void setScrollBarValue(int pos); 
	void setScrollBarRange(int alines); 
	void setScrollBarEnabled(bool enable); 
#endif
#ifdef PULLBACK_DEVICE
	// Zaber Stage Control
	bool isZaberStageEnabled() { return m_pCheckBox_ZaberStageControl->isChecked(); }
	void pullback() { moveAbsolute(); }
#endif

signals: ////////////////////////////////////////////////////////////////////////////////////////////////
	void drawEcg(double, bool);

private slots: //////////////////////////////////////////////////////////////////////////////////////////
#ifdef AXSUN_OCT_LASER
    // Axsun OCT Laser Control
    void enableAxsunOCTLaserControl(bool);
	void setLightSource(bool);
#endif
#ifdef ECG_TRIGGERING
	// ECG Module Control
	void enableEcgModuleControl(bool);
	void toggleEcgTriggerButton(bool);
	void changeEcgDelayRate(double);
	void enable800rpsVoltControl(bool);
	void toggle800rpsVoltButton(bool);
	void set800rpsVoltage(double);
#endif
#ifdef OCT_FLIM
	// PMT Gain Control
	void enablePmtGainControl(bool);
	void changePmtGainVoltage(const QString &);

	// FLIM Laser Synchronization Control
	void enableFlimLaserSyncControl(bool);
	void enableAsyncMode(bool);

	// FLIM Laser Power Control
	void enableFlimLaserPowerControl(bool);
	void increaseLaserPower();
	void decreaseLaserPower();
#endif
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
	// NIRF Emission Acquisition
	void enableNirfEmissionAcquisition(bool);
#ifdef PROGRAMMATIC_GAIN_CONTROL
	// PMT Gain Control
	void enablePmtGainControl(bool);
#ifndef TWO_CHANNEL_NIRF
	void changePmtGainVoltage(const QString &);
#else
	void changePmtGainVoltage1(const QString &);
	void changePmtGainVoltage2(const QString &);
#endif
#endif
#endif
#endif
#ifdef GALVANO_MIRROR
	// Galvano Mirror
	void enableGalvanoMirror(bool);
	void triggering(bool);
	void changeGalvoFastScanVoltage(const QString &);
	void changeGalvoFastScanVoltageOffset(const QString &);
	void changeGalvoSlowScanVoltage(const QString &);
	void changeGalvoSlowScanVoltageOffset(const QString &);
	void changeGalvoSlowScanIncrement(const QString &);
	void scanAdjusting(int);
#endif
#ifdef PULLBACK_DEVICE
	// Zaber Stage Control
	void enableZaberStageControl(bool);
	void moveAbsolute();
	void setTargetSpeed(const QString &);
	void changeZaberPullbackLength(const QString &);
	void home();
	void stop();

	// Faulhaber Motor Control
	void enableFaulhaberMotorControl(bool);
	void rotate();
	void rotateStop();
	void changeFaulhaberRpm(const QString &);
#endif

// Variables ////////////////////////////////////////////
private: ////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef AXSUN_OCT_LASER
    // Axsun OCT Laser Control
    AxsunControl* m_pAxsunControl;
#endif
#ifdef ECG_TRIGGERING
#if NI_ENABLE
	// ECG Module Control
	EcgMonitoringTrigger* m_pEcgMonitoringTrigger;
	EcgMonitoring* m_pEcgMonitoring;
	// 800 RPS Motor Voltage Control
	Voltage800RPS* m_pVoltage800Rps;
#endif
#endif	
#ifdef OCT_FLIM
#if NI_ENABLE
	// PMT Gain Control
	PmtGainControl* m_pPmtGainControl;
	// FLIM Synchronization Control
	SyncFLIM* m_pSyncFlim;
#endif
	// Elforlight Laser Control
	ElforlightLaser* m_pElforlightLaser;
#endif
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
#ifndef ALAZAR_NIRF_ACQUISITION
	// NIRF Emission
	NirfEmissionTrigger* m_pNirfEmissionTrigger;
	NirfEmission* m_pNirfEmission;
#endif
#ifdef TWO_CHANNEL_NIRF
	NirfModulation* m_pNirfModulation;
#endif
#ifdef PROGRAMMATIC_GAIN_CONTROL
	// PMT Gain Control
	PmtGainControl* m_pPmtGainControl;
#endif
#endif
#endif
#ifdef GALVANO_MIRROR
#if NI_ENABLE
	GalvoScan* m_pGalvoScan;
#endif
#endif
#ifdef PULLBACK_DEVICE
	// Zaber Stage Control
#ifndef ZABER_NEW_STAGE
	ZaberStage* m_pZaberStage;
#else
	ZaberStage2* m_pZaberStage;
#endif
	// Faulhaber Motor Control
	FaulhaberMotor* m_pFaulhaberMotor;
#endif

public: ////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ECG_TRIGGERING
#if NI_ENABLE
	// ECG Module Control
	bool m_bRpeakNotDetected;
	std::mutex m_mtxRpeakDetected;
	std::condition_variable m_cvRpeakDetected;
#endif
#endif
	QTimer* m_pZaberMonitorTimer;

private: ////////////////////////////////////////////////////////////////////////////////////////////////
	MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
	QOperationTab* m_pOperationTab;


    // Layout
    QVBoxLayout *m_pVBoxLayout;

#ifdef AXSUN_OCT_LASER
    // Widgets for Axsun OCT Control
    QCheckBox *m_pCheckBox_AxsunOCTLaserControl;
	QLabel *m_pLabel_OCTLaserSource;
	QPushButton *m_pToggleButton_OCTLaserSource;

#endif
#ifdef ECG_TRIGGERING	
	// Widgets for ECG module control
	QCheckBox *m_pCheckBox_EcgModuleControl; 
	QPushButton *m_pToggledButton_EcgTriggering;
	QLabel *m_pLabel_EcgBitPerMinute;	
	QEcgScope *m_pEcgScope;
	QLabel *m_pLabel_EcgDelayRate;
	QDoubleSpinBox *m_pDoubleSpinBox_EcgDelayRate;

	QCheckBox *m_pCheckBox_Voltage800Rps;
	QPushButton *m_pToggledButton_Voltage800Rps;
	QDoubleSpinBox *m_pDoubleSpinBox_Voltage800Rps;
	QLabel *m_pLabel_Voltage800Rps;
#endif
#ifdef OCT_FLIM
	// FLIM Layout
	QVBoxLayout *m_pVBoxLayout_FlimControl;
	QGroupBox *m_pGroupBox_FlimControl;

	// Widgets for FLIM control	// Gain control
	QCheckBox *m_pCheckBox_PmtGainControl;
	QLineEdit *m_pLineEdit_PmtGainVoltage;
	QLabel *m_pLabel_PmtGainVoltage;

	// Widgets for FLIM control // Laser sync control
	QCheckBox *m_pCheckBox_FlimLaserSyncControl;
	QCheckBox *m_pCheckBox_AsyncMode;
	QLineEdit *m_pLineEdit_RepetitionRate;
	QLabel *m_pLabel_Hertz;

	// Widgets for FLIM control // Laser power control
	QCheckBox *m_pCheckBox_FlimLaserPowerControl;
	QPushButton *m_pPushButton_IncreasePower;
	QPushButton *m_pPushButton_DecreasePower;
#endif
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
	// Widgets for NIRF acquisition control
	QCheckBox *m_pCheckBox_NirfAcquisitionControl;
#ifdef PROGRAMMATIC_GAIN_CONTROL
	// Widgets for FLIM control	// Gain control
	QCheckBox *m_pCheckBox_PmtGainControl;
#ifndef TWO_CHANNEL_NIRF
	QLineEdit *m_pLineEdit_PmtGainVoltage;
#else
	QLineEdit *m_pLineEdit_PmtGainVoltage[2];
#endif
	QLabel *m_pLabel_PmtGainVoltage;
#endif
#endif
#endif
#ifdef GALVANO_MIRROR
    // Widgets for galvano mirror control
    QCheckBox *m_pCheckBox_GalvanoMirrorControl;
	QPushButton *m_pToggleButton_ScanTriggering;
    QLineEdit *m_pLineEdit_FastPeakToPeakVoltage;
    QLineEdit *m_pLineEdit_FastOffsetVoltage;
	QLineEdit *m_pLineEdit_SlowPeakToPeakVoltage;
	QLineEdit *m_pLineEdit_SlowOffsetVoltage;
	QLineEdit *m_pLineEdit_SlowScanIncrement;
    QLabel *m_pLabel_FastScanVoltage;
    QLabel *m_pLabel_FastScanPlusMinus;
    QLabel *m_pLabel_FastGalvanoVoltage;
	QLabel *m_pLabel_SlowScanVoltage;
	QLabel *m_pLabel_SlowScanPlusMinus;
	QLabel *m_pLabel_SlowGalvanoVoltage;
	QLabel *m_pLabel_SlowScanIncrement;
	QLabel *m_pLabel_SlowScanIncrementVoltage;
	QScrollBar *m_pScrollBar_ScanAdjustment;
    QLabel *m_pLabel_ScanAdjustment;
#endif
#ifdef PULLBACK_DEVICE
    // Widgets for Zaber stage control
    QCheckBox *m_pCheckBox_ZaberStageControl;
    QPushButton *m_pPushButton_SetTargetSpeed;
    QPushButton *m_pPushButton_MoveAbsolute;
    QPushButton *m_pPushButton_Home;
    QPushButton *m_pPushButton_Stop;
    QLineEdit *m_pLineEdit_TargetSpeed;
    QLineEdit *m_pLineEdit_TravelLength;
    QLabel *m_pLabel_TargetSpeed;
    QLabel *m_pLabel_TravelLength;

    // Widgets for Faulhaber motor control
    QCheckBox *m_pCheckBox_FaulhaberMotorControl;
    QPushButton *m_pPushButton_Rotate;
	QPushButton *m_pPushButton_RotateStop;
    QLineEdit *m_pLineEdit_RPM;
    QLabel *m_pLabel_RPM;
#endif
};

#endif // QDEVICECONTROLTAB_H
