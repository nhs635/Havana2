#ifndef OCTINTENSITYHISTDLG_H
#define OCTINTENSITYHISTDLG_H

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
class QResultTab;

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


class OctIntensityHistDlg : public QDialog
{
    Q_OBJECT

// Constructer & Destructer /////////////////////////////
public:
    explicit OctIntensityHistDlg(bool isStreamTab = true, QWidget *parent = 0);
    virtual ~OctIntensityHistDlg();

// Methods //////////////////////////////////////////////
private:
	void keyPressEvent(QKeyEvent *e);
	
private:
	void createHistogram(int _nAlines);

private: // callback
	

public slots : // widgets
	void drawHistogram(float* pImg);

signals:
	void plotHistogram(float*);

// Variables ////////////////////////////////////////////
private:	
    MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
    QStreamTab* m_pStreamTab;
	QResultTab* m_pResultTab;
	
private:
	Histogram* m_pHistogram;

private:
	// Layout
	QVBoxLayout *m_pVBoxLayout;

	// Widgets for histogram
	QLabel *m_pLabel_Title;
	QRenderArea *m_pRenderArea;
	QImageView *m_pColorbar;
	QLabel *m_pLabel_Min;
	QLabel *m_pLabel_Max;
};

#endif // OCTINTENSITYHISTDLG_H
