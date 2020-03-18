#ifndef DIGITIZERSETUPDLG_H
#define DIGITIZERSETUPDLG_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>

#include <Common/array.h>

class MainWindow;
class QOperationTab;
class QStreamTab;
class QResultTab;
class DataAcquisition;
class MemoryBuffer;


class DigitizerSetupDlg : public QDialog
{
    Q_OBJECT

// Constructer & Destructer /////////////////////////////
public:
    explicit DigitizerSetupDlg(QWidget *parent = 0);
    virtual ~DigitizerSetupDlg();
		
// Methods //////////////////////////////////////////////
private:
	void keyPressEvent(QKeyEvent *);

private slots:
	void changeVoltageRangeCh1(int);
#ifdef OCT_FLIM
	void changeVoltageRangeCh2(int);
#endif
	void changePreTrigger(const QString &);
    void changeTriggerDelay(const QString &);
	void changeNchannels(const QString &);
	void changeNscans(const QString &);
	void changeNalines(const QString &);

	void changeBootTimeBufIdx(int);
	void getBootTimeBufCfg();
	void setBootTimeBufCfg();

// Variables ////////////////////////////////////////////
private:
    MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
	DataAcquisition* m_pDataAcq;
	MemoryBuffer* m_pMemBuff;
	QOperationTab* m_pOperationTab;
	QStreamTab* m_pStreamTab;
	QResultTab* m_pResultTab;

private:
	QLineEdit *m_pLineEdit_SamplingRate;
	QComboBox *m_pComboBox_VoltageRangeCh1;
#ifdef OCT_FLIM
	QComboBox *m_pComboBox_VoltageRangeCh2;
#endif
#if PX14_ENABLE
	QLineEdit *m_pLineEdit_PreTrigger;
#elif ALAZAR_ENABLE
    QLineEdit *m_pLineEdit_TriggerDelay;
#endif

	QLineEdit *m_pLineEdit_nChannels;
	QLineEdit *m_pLineEdit_nScans;
	QLineEdit *m_pLineEdit_nAlines;

#if PX14_ENABLE
	QLabel *m_pLabel_BootTimeBufTitle[2];
	QRadioButton *m_pRadioButton_BootTimeBufIdx[4];
	QButtonGroup *m_pButtonGroup_IndexSelection;
	QLineEdit *m_pLineEdit_BootTimeBufSamps[4];
	QLabel *m_pLabel_BootTimeBufInstruction;

	QPushButton *m_pPushButton_BootTimeBufSet;
	QPushButton *m_pPushButton_BootTimeBufGet;
#endif
};

#endif // DIGITIZERSETUPDLG_H
