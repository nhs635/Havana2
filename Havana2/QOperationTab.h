#ifndef QOPERATIONTAB_H
#define QOPERATIONTAB_H

#include <QDialog>
#include <QtWidgets>
#include <QtCore>

class MainWindow;
class Configuration;

class DigitizerSetupDlg;
class DataAcquisition;
class MemoryBuffer;


class QOperationTab : public QDialog
{
    Q_OBJECT

// Constructer & Destructer /////////////////////////////
public:
    explicit QOperationTab(QWidget *parent = 0);
    virtual ~QOperationTab();

// Methods //////////////////////////////////////////////
public:
	inline QVBoxLayout* getLayout() const { return m_pVBoxLayout; }
	inline MainWindow* getMainWnd() const { return m_pMainWnd; }
	inline DigitizerSetupDlg* getDigitizerSetupDlg() const { return m_pDigitizerSetupDlg; }
	bool isAcquisitionButtonToggled() { return m_pToggleButton_Acquisition->isChecked(); }	
	bool isSavingButtonToggled() { return m_pToggleButton_Saving->isChecked(); }
	
public:
	void changedTab(bool change);

private slots:
	// Slots for widgets
	void operateDataAcquisition(bool toggled);
	void operateDataRecording(bool toggled);
	void operateDataSaving(bool toggled);
	void createDigitizerSetupDlg();
	void deleteDigitizerSetupDlg();

public slots :
	void setAcqRecEnable();
	void setSaveButtonDefault(bool error);
	void setRecordingButton(bool toggled) { m_pToggleButton_Recording->setChecked(toggled); }

// Variables ////////////////////////////////////////////
private:	
	MainWindow* m_pMainWnd;
	Configuration* m_pConfig;

	// Digitizer setup dialog
	DigitizerSetupDlg *m_pDigitizerSetupDlg;

public:
	// Data acquisition and memory operation object
	DataAcquisition* m_pDataAcquisition;
	MemoryBuffer* m_pMemoryBuffer;

private:
	// Layout
	QVBoxLayout *m_pVBoxLayout;

	// Operation buttons
	QPushButton *m_pToggleButton_Acquisition;
	QPushButton *m_pToggleButton_Recording;
	QPushButton *m_pToggleButton_Saving;
	QPushButton *m_pPushButton_DigitizerSetup;
	QProgressBar *m_pProgressBar;
};

#endif // QOPERATIONTAB_H
