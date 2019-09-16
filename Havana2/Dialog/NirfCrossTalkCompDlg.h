#ifndef NIRFCROSSTALKCOMPDLG_H
#define NIRFCROSSTALKCOMPDLG_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>
#include <Havana2/Viewer/QScope.h>

#include <Common/array.h>
#include <Common/callback.h>

#define RATIO_BASED		-2
#define SPECTRUM_BASED	-3

class MainWindow;
class QResultTab;


class NirfCrossTalkCompDlg : public QDialog
{
    Q_OBJECT

#ifdef OCT_NIRF
#ifdef TWO_CHANNEL_NIRF
// Constructer & Destructer /////////////////////////////
public:
    explicit NirfCrossTalkCompDlg(QWidget *parent = 0);
    virtual ~NirfCrossTalkCompDlg();

// Methods //////////////////////////////////////////////
private:
	void keyPressEvent(QKeyEvent *e);    

private:
	void loadNirfBackground();

private slots:
	void changeCh1NirfBackground(const QString &);
	void changeCh2NirfBackground(const QString &);
	void changeCh1GainVoltage(const QString &);
	void changeCh2GainVoltage(const QString &);
	void changeCompensationMode(int id);
	void changeRatio(const QString &);
	void setRatio(bool);

	
// Variables ////////////////////////////////////////////
private:	
    MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
	Configuration* m_pConfigTemp;
    QResultTab* m_pResultTab;
	
public:
	int compensationMode;
	float nirfBg1, nirfBg2;
	float gainValue1, gainValue2;
	float ratio;

private:
	// Common Parameter
	QLabel* m_pLabel_Ch1_Bg;
	QLineEdit* m_pLineEdit_Ch1_Bg;

	QLabel* m_pLabel_Ch2_Bg;
	QLineEdit* m_pLineEdit_Ch2_Bg;

	QLabel* m_pLabel_Ch1_GainVoltage;
	QLineEdit* m_pLineEdit_Ch1_GainVoltage;
	QLabel* m_pLabel_Ch1_GainValue;

	QLabel* m_pLabel_Ch2_GainVoltage;
	QLineEdit* m_pLineEdit_Ch2_GainVoltage;
	QLabel* m_pLabel_Ch2_GainValue;
	
	// Compensation Mode Selection
	QButtonGroup* m_pButtonGroup_CompensationMode;
	QRadioButton* m_pRadioButton_RatioBased;
	QRadioButton* m_pRadioButton_SpectrumBased;

	// Ratio-based Compensation
	QLabel* m_pLabel_Ratio;
	QLineEdit* m_pLineEdit_Ratio;
	QPushButton* m_pToggleButton_RatioSetting;

	// Spectrum-based Compensation
	QScope* m_pScope_Spectrum;


#endif
#endif
};

#endif // NIRFCROSSTALKCOMPDLG_H
