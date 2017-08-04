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
class QStreamTab;

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
#ifdef GALVANO_MIRROR
#if NI_ENABLE
class GalvoScan;
#endif
#endif
#ifdef PULLBACK_DEVICE
class ZaberStage;
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

private: ////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ECG_TRIGGERING
	void createEcgModuleControl();
#endif
#ifdef OCT_FLIM
	void createPmtGainControl();
	void createFlimLaserSyncControl();
	void createFlimLaserPowerControl();
#endif
#ifdef GALVANO_MIRROR
    void createGalvanoMirrorControl();
#endif
#ifdef PULLBACK_DEVICE
    void createZaberStageControl();
    void createFaulhaberMotorControl();
#endif

public: ////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ECG_TRIGGERING
#if NI_ENABLE
	// ECG Module Control
	bool isEcgTriggered() { return m_pToggledButton_EcgTriggering->isChecked(); }
<<<<<<< HEAD
	void setEcgRecording(bool set);
=======
	void setEcgRecording(bool set);	
>>>>>>> 258c41f81233548b2e400ccddc2abf811cd663a4
	std::deque<double>* getRecordedEcg();
#endif	
#endif
#ifdef GALVANO_MIRROR
	// ?
#endif
#ifdef PULLBACK_DEVICE
	// Zaber Stage Control
	bool isZaberStageEnabled() { return m_pCheckBox_ZaberStageControl->isChecked(); }
	void pullback() { moveAbsolute(); }
#endif

private slots: //////////////////////////////////////////////////////////////////////////////////////////
#ifdef ECG_TRIGGERING
	// ECG Module Control
	void enableEcgModuleControl(bool);
	void toggleEcgTriggerButton(bool);
	void enable800rpsVoltControl(bool);
	void toggle800rpsVoltButton(bool);
	void set800rpsVoltage(double);
#endif
#ifdef OCT_FLIM
	// PMT Gain Control
	void enablePmtGainControl(bool);

	// FLIM Laser Synchronization Control
	void enableFlimLaserSyncControl(bool);
	void enableAsyncMode(bool);

	// FLIM Laser Power Control
	void enableFlimLaserPowerControl(bool);
	void increaseLaserPower();
	void decreaseLaserPower();
#endif
#ifdef GALVANO_MIRROR

#endif
#ifdef PULLBACK_DEVICE
	// Zaber Stage Control
	void enableZaberStageControl(bool);
	void moveAbsolute();
	void setTargetSpeed(const QString &);
	void home();
	void stop();

	// Faulhaber Motor Control
	void enableFaulhaberMotorControl(bool);
	void rotate(bool);
#endif

// Variables ////////////////////////////////////////////
private: ////////////////////////////////////////////////////////////////////////////////////////////////
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
#ifdef GALVANO_MIRROR
#if NI_ENABLE
	GalvoScan* m_pGalvoScan;
#endif
#endif
#ifdef PULLBACK_DEVICE
	// Zaber Stage Control
	ZaberStage* m_pZaberStage;
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
	
private: ////////////////////////////////////////////////////////////////////////////////////////////////
	MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
	QStreamTab* m_pStreamTab;


    // Layout
    QVBoxLayout *m_pVBoxLayout;

#ifdef ECG_TRIGGERING	
	// Widgets for ECG module control
	QCheckBox *m_pCheckBox_EcgModuleControl; 
	QPushButton *m_pToggledButton_EcgTriggering;
	QLabel *m_pLabel_EcgBitPerMinute;	
	QEcgScope *m_pEcgScope;

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
#ifdef GALVANO_MIRROR
    // Widgets for galvano mirror control
    QCheckBox *m_pCheckBox_GalvanoMirrorControl;    
    QLineEdit *m_pLineEdit_PeakToPeakVoltage;
    QLineEdit *m_pLineEdit_OffsetVoltage;
    QLabel *m_pLabel_ScanVoltage;
    QLabel *m_pLabel_ScanPlusMinus;
    QLabel *m_pLabel_GalvanoVoltage;
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
    QPushButton *m_pToggleButton_Rotate;
    QLineEdit *m_pLineEdit_RPM;
    QLabel *m_pLabel_RPM;
#endif
};

#endif // QDEVICECONTROLTAB_H
