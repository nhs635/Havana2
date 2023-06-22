#ifndef FLIMCALIBDLG_H
#define FLIMCALIBDLG_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>
#include <Havana2/Viewer/QScope.h>
#include <Havana2/Viewer/QImageView.h>

#include <Common/array.h>
#include <Common/callback.h>

#include <ipps.h>
#include <ippi.h>
#include <ippvm.h>

class MainWindow;
class QStreamTab;
class FLIMProcess;

class QMySpinBox1 : public QDoubleSpinBox
{
public:
	explicit QMySpinBox1(QWidget *parent = 0) : QDoubleSpinBox(parent)
	{
		lineEdit()->setReadOnly(true);
	}
	virtual ~QMySpinBox1() {};
};

struct Histogram
{
public:
	Histogram() : pHistObj(nullptr), pBuffer(nullptr), pLevels(nullptr), lowerLevel(0), upperLevel(0)
	{
	}

	Histogram(int _nBins, int _length) : pHistObj(nullptr), pBuffer(nullptr), pLevels(nullptr), lowerLevel(0), upperLevel(0)
	{
		initialize(_nBins, _length);
	}

	~Histogram()
	{
		if (pHistObj) ippsFree(pHistObj);
		if (pBuffer) ippsFree(pBuffer);
		if (pLevels) ippsFree(pLevels);
		if (pHistTemp1) ippsFree(pHistTemp1);
		if (pHistTemp2) ippsFree(pHistTemp2);
	}

public:
	void operator() (const Ipp32f* pSrc, Ipp32f* pHist, Ipp32f _lowerLevel, Ipp32f _upperLevel, bool logScale = false)
	{
		if ((lowerLevel != _lowerLevel) || (upperLevel != _upperLevel))
		{
			// set vars
			lowerLevel = _lowerLevel;
			upperLevel = _upperLevel;

			// initialize spec
			ippiHistogramUniformInit(ipp32f, &_lowerLevel, &_upperLevel, &nLevels, 1, pHistObj);

			// check levels of bins
			ippiHistogramGetLevels(pHistObj, &pLevels);
		}

		// calculate histogram
		ippiHistogram_32f_C1R(pSrc, roiSize.width * sizeof(Ipp32f), roiSize, pHistTemp1, pHistObj, pBuffer);
		if (!logScale)
			ippsConvert_32s32f((Ipp32s*)pHistTemp1, pHist, nBins);
		else
		{
			ippsConvert_32s32f((Ipp32s*)pHistTemp1, pHistTemp2, nBins);
			ippsLog10_32f_A11(pHistTemp2, pHist, nBins);
		}
	}

public:
	void initialize(int _nBins, int _length)
	{
		// init vars
		roiSize = { _length, 1 };
		nBins = _nBins; nLevels = nBins + 1;
		pLevels = ippsMalloc_32f(nLevels);

		// get sizes for spec and buffer
		ippiHistogramGetBufferSize(ipp32f, roiSize, &nLevels, 1/*nChan*/, 1/*uniform*/, &sizeHistObj, &sizeBuffer);

		pHistObj = (IppiHistogramSpec*)ippsMalloc_8u(sizeHistObj);
		pBuffer = (Ipp8u*)ippsMalloc_8u(sizeBuffer);

		pHistTemp1 = ippsMalloc_32u(nBins);
		pHistTemp2 = ippsMalloc_32f(nBins);
	}

private:
	IppiSize roiSize;
	int nBins, nLevels;
	int sizeHistObj, sizeBuffer;
	Ipp32f lowerLevel, upperLevel;
	IppiHistogramSpec* pHistObj;
	Ipp8u* pBuffer;
	Ipp32f* pLevels;
	Ipp32u* pHistTemp1;
	Ipp32f* pHistTemp2;
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
	QMySpinBox1 *m_pSpinBox_ChStart[4];
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
