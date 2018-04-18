#ifndef FLIMCALIBDLG_H
#define FLIMCALIBDLG_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>
#include <Havana2/Viewer/QScope.h>
#include <Havana2/Viewer/QImageView.h>
#include <Havana2/Dialog/OctIntensityHistDlg.h>

#include <Common/array.h>
#include <Common/callback.h>

#include <ipps.h>
#include <ippi.h>

class MainWindow;
class QStreamTab;
class FLIMProcess;

class QMySpinBox : public QDoubleSpinBox
{
public:
	explicit QMySpinBox(QWidget *parent = 0) : QDoubleSpinBox(parent)
	{
		lineEdit()->setReadOnly(true);
	}
	virtual ~QMySpinBox() {};
};


class FlimCalibDlg : public QDialog
{
    Q_OBJECT

#ifdef OCT_FLIM
// Constructer & Destructer /////////////////////////////
public:
    explicit FlimCalibDlg(QWidget *parent = 0);
    virtual ~FlimCalibDlg();

// Methods //////////////////////////////////////////////
private:
	void keyPressEvent(QKeyEvent *e);
	
private:
	void createPulseView();
	void createCalibWidgets();
	void createHistogram();

private: // callback
	

public slots : // widgets
	void drawRoiPulse(FLIMProcess*, int);

	void showWindow(bool);
	void showMeanDelay(bool);
	void splineView(bool);

	void showMask(bool);
	void modifyMask(bool);
	void addMask();
	void removeMask();

	void captureBackground();
	void captureBackground(const QString &);

	void resetChStart0(double);
	void resetChStart1(double);
	void resetChStart2(double);
	void resetChStart3(double);
	void resetDelayTimeOffset();

signals:
	void plotRoiPulse(FLIMProcess*, int);

// Variables ////////////////////////////////////////////
private:	
    MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
    QStreamTab* m_pStreamTab;
#ifdef OCT_FLIM
	FLIMProcess* m_pFLIM;
#endif

private:
	int* m_pStart;
	int* m_pEnd;

private:
	Histogram* m_pHistogramIntensity;
	Histogram* m_pHistogramLifetime;

private:
	// Layout
	QVBoxLayout *m_pVBoxLayout;

	// Widgets for pulse view
	QScope *m_pScope_PulseView;
	
	QCheckBox *m_pCheckBox_ShowWindow;
	QCheckBox *m_pCheckBox_ShowMeanDelay;
	QCheckBox *m_pCheckBox_SplineView;
	QCheckBox *m_pCheckBox_ShowMask;
	QPushButton *m_pPushButton_ModifyMask;
	QPushButton *m_pPushButton_AddMask;
	QPushButton *m_pPushButton_RemoveMask;

	// Widgets for FLIM calibration widgets
	QPushButton *m_pPushButton_CaptureBackground;
	QLineEdit *m_pLineEdit_Background;

	QLabel *m_pLabel_ChStart;
	QLabel *m_pLabel_DelayTimeOffset;
	QLabel *m_pLabel_Ch[4];
	QMySpinBox *m_pSpinBox_ChStart[4];
	QLineEdit *m_pLineEdit_DelayTimeOffset[3];
	QLabel *m_pLabel_NanoSec[2];

	// Widgets for histogram
	QLabel *m_pLabel_FluIntensity;
	QRenderArea *m_pRenderArea_FluIntensity;
	QImageView *m_pColorbar_FluIntensity;
	QLabel *m_pLabel_FluIntensityMin;
	QLabel *m_pLabel_FluIntensityMax;
	QLabel *m_pLabel_FluIntensityMean;
	QLabel *m_pLabel_FluIntensityStd;

	QLabel *m_pLabel_FluLifetime;
	QRenderArea *m_pRenderArea_FluLifetime;
	QImageView *m_pColorbar_FluLifetime;
	QLabel *m_pLabel_FluLifetimeMin;
	QLabel *m_pLabel_FluLifetimeMax;
	QLabel *m_pLabel_FluLifetimeMean;
	QLabel *m_pLabel_FluLifetimeStd;
#endif
};

#endif // FLIMCALIBDLG_H
