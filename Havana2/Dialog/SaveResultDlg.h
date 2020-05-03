#ifndef SAVERESULTDLG_H
#define SAVERESULTDLG_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>

#include <iostream>
#include <vector>

#include <Common/array.h>
#include <Common/circularize.h>
#include <Common/medfilt.h>
#ifdef CUDA_ENABLED
#include <CUDA/CudaCircularize.cuh>
#endif
#include <Common/Queue.h>
#include <Common/ImageObject.h>
#include <Common/basic_functions.h>

class MainWindow;
class QResultTab;

using ImgObjVector = std::vector<ImageObject*>; 

struct CrossSectionCheckList
{
	bool bRect, bCirc, bLongi;
	bool bRectResize, bCircResize, bLongiResize;
	int nRectWidth, nRectHeight;
	int nCircDiameter;
	int nLongiWidth, nLongiHeight;
#ifdef OCT_FLIM
	bool bCh[3];
	bool bMulti;
#endif
#ifdef OCT_NIRF
    bool bNirf;
	bool bNirfRingOnly;
#endif
};

struct EnFaceCheckList
{
	bool bRaw;
	bool bScaled;
#ifdef OCT_FLIM
	bool bCh[3];
#endif
#ifdef OCT_NIRF
    bool bNirf;
#endif
	bool bOctProj;
};


class SaveResultDlg : public QDialog
{
    Q_OBJECT

// Constructer & Destructer /////////////////////////////
public:
    explicit SaveResultDlg(QWidget *parent = 0);
    virtual ~SaveResultDlg();
		
// Methods //////////////////////////////////////////////
private:
	void closeEvent(QCloseEvent *e);
	void keyPressEvent(QKeyEvent *);
	
public:
	void setCircRadius(int circ_radius);

private slots:
	void saveCrossSections();
	void saveEnFaceMaps();
    void setRange();
	void enableRectResize(bool);
	void enableCircResize(bool);
	void enableLongiResize(bool);
	void setWidgetsEnabled(bool);

signals:
	void setWidgets(bool);
	void savedSingleFrame(int);

private:
#ifdef OCT_FLIM
	void scaling(std::vector<np::FloatArray2>& vectorOctImage,
		std::vector<np::Uint8Array2>& intensityMap, std::vector<np::Uint8Array2>& lifetimeMap,
		CrossSectionCheckList checkList);
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
	void scaling(std::vector<np::FloatArray2>& vectorOctImage);
#else
#ifndef TWO_CHANNEL_NIRF
    void scaling(std::vector<np::FloatArray2>& vectorOctImage,
        np::Uint8Array2& nirfMap, CrossSectionCheckList checkList);
#else
	void scaling(std::vector<np::FloatArray2>& vectorOctImage,
		np::Uint8Array2& nirfMap1, np::Uint8Array2& nirfMap2, CrossSectionCheckList checkList);
#endif
#endif
#endif
	void converting(CrossSectionCheckList checkList);
	void rectWriting(CrossSectionCheckList checkList);
	void circularizing(CrossSectionCheckList checkList);
	void circWriting(CrossSectionCheckList checkList);
	
// Variables ////////////////////////////////////////////
private:
    MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
	QResultTab* m_pResultTab;

private:
	int m_nSavedFrames;

public:
	Qt::TransformationMode m_defaultTransformation;

private: // for threading operation
#if defined(OCT_FLIM) || defined(OCT_NIRF)
	Queue<ImgObjVector*> m_syncQueueConverting;
#endif
	Queue<ImgObjVector*> m_syncQueueRectWriting;
	Queue<ImgObjVector*> m_syncQueueLongitudinal;
	Queue<ImgObjVector*> m_syncQueueCircularizing;
	Queue<ImgObjVector*> m_syncQueueCircWriting;

private:
	// Save Cross-sections
	QPushButton* m_pPushButton_SaveCrossSections;
	QCheckBox* m_pCheckBox_RectImage;
	QCheckBox* m_pCheckBox_ResizeRectImage;
	QLineEdit* m_pLineEdit_RectWidth;
	QLineEdit* m_pLineEdit_RectHeight;
	QCheckBox* m_pCheckBox_CircImage;
	QCheckBox* m_pCheckBox_ResizeCircImage;
	QLineEdit* m_pLineEdit_CircDiameter;
	QCheckBox* m_pCheckBox_LongiImage;
	QCheckBox* m_pCheckBox_ResizeLongiImage;
	QLineEdit* m_pLineEdit_LongiWidth;
	QLineEdit* m_pLineEdit_LongiHeight;
#ifdef OCT_FLIM
	QCheckBox* m_pCheckBox_CrossSectionCh1;
	QCheckBox* m_pCheckBox_CrossSectionCh2;
	QCheckBox* m_pCheckBox_CrossSectionCh3;
	QCheckBox* m_pCheckBox_Multichannel;
#endif
#ifdef OCT_NIRF
    QCheckBox* m_pCheckBox_CrossSectionNirf;
	QCheckBox* m_pCheckBox_NirfRingOnly;
#endif

    // Set Range
    QLabel* m_pLabel_Range;
    QLineEdit* m_pLineEdit_RangeStart;
    QLineEdit* m_pLineEdit_RangeEnd;

	// Save En Face Maps
	QPushButton* m_pPushButton_SaveEnFaceMaps;
	QCheckBox* m_pCheckBox_RawData;
	QCheckBox* m_pCheckBox_ScaledImage;
#ifdef OCT_FLIM
	QCheckBox* m_pCheckBox_EnFaceCh1;
	QCheckBox* m_pCheckBox_EnFaceCh2;
	QCheckBox* m_pCheckBox_EnFaceCh3;
#endif
#ifdef OCT_NIRF
    QCheckBox* m_pCheckBox_EnFaceNirf;
#endif
	QCheckBox* m_pCheckBox_OctMaxProjection;
};

#endif // SAVERESULTDLG_H
