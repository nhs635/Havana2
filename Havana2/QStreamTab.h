#ifndef QSTREAMTAB_H
#define QSTREAMTAB_H

#include <QDialog>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>

#include <Common/array.h>
#include <Common/circularize.h>
#include <Common/medfilt.h>
#include <Common/SyncObject.h>
#include <Common/ImageObject.h>
#include <Common/basic_functions.h>

class MainWindow;
class QOperationTab;
class DataAcquisition;
class MemoryBuffer;

class QScope;
#ifdef STANDALONE_OCT
class QScope2;
#endif
class QImageView;

class OCTProcess;
#ifdef OCT_FLIM
class FLIMProcess;
#endif
class ThreadManager;

class OctCalibDlg;
class OctIntensityHistDlg;
#ifdef OCT_FLIM
class FlimCalibDlg;
#endif
#ifdef OCT_NIRF
class NirfEmissionProfileDlg;
#endif


class QStreamTab : public QDialog
{
    Q_OBJECT

// Constructer & Destructer /////////////////////////////
public:
    explicit QStreamTab(QWidget *parent = 0);
	virtual ~QStreamTab();
	
// Methods //////////////////////////////////////////////
protected:
	void keyPressEvent(QKeyEvent *);

public:
	inline MainWindow* getMainWnd() const { return m_pMainWnd; }
	inline OctCalibDlg* getOctCalibDlg() const { return m_pOctCalibDlg; }
	inline OctIntensityHistDlg* getOctIntensityHistDlg() const { return m_pOctIntensityHistDlg; }
#ifdef OCT_FLIM
	inline FlimCalibDlg* getFlimCalibDlg() const { return m_pFlimCalibDlg; }
	inline QScope* getPulseScope() const { return m_pScope_FlimPulse; }
	inline QSpinBox* getFlimCh() const { return m_pSpinBox_DataChannel; }
	inline int getCurrentEmCh() const { return m_pComboBox_EmissionChannel->currentIndex(); }
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	inline void makeNirfEmissionProfileDlg() { createNirfEmissionProfileDlg(); }
    inline NirfEmissionProfileDlg* getNirfEmissionProfileDlg() const { return m_pNirfEmissionProfileDlg; }
#endif
#endif
	inline int getCurrentAline() const { return m_pSlider_SelectAline->value(); }
	inline float getOctMaxDb() { return m_pLineEdit_OctDbMax->text().toFloat(); }
	inline float getOctMinDb() { return m_pLineEdit_OctDbMin->text().toFloat(); }
	inline void updateFrigneBg() { changeFringeBg(m_pCheckBox_ShowBgRemovedSignal->isChecked()); }
#ifdef GALVANO_MIRROR
#ifdef OCT_FLIM
	inline void invalidate() { visualizeImage(m_visImage.raw_ptr(), m_visIntensity.raw_ptr(), m_visMeanDelay.raw_ptr(), m_visLifetime.raw_ptr()); }
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	inline void invalidate() { visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr(), nullptr); }
#else
	inline void invalidate() { visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr()); }
#endif
#endif
#endif
	inline size_t getDeinterleavingBufferQueueSize() const { return m_syncDeinterleaving.queue_buffer.size(); }
	inline size_t getCh1ProcessingBufferQueueSize() const { return m_syncCh1Processing.queue_buffer.size(); }
	inline size_t getCh2ProcessingBufferQueueSize() const { return m_syncCh2Processing.queue_buffer.size(); }
	inline size_t getCh1VisualizationBufferQueueSize() const { return m_syncCh1Visualization.queue_buffer.size(); }
	inline size_t getCh2VisualizationBufferQueueSize() const { return m_syncCh2Visualization.queue_buffer.size(); }
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
	inline size_t getNirfVisualizationBufferQueueSize() const { return m_syncNirfVisualization.queue_buffer.size(); }
#endif
#endif
	void setWidgetsText();

private:
#ifdef OCT_FLIM
    void createFlimVisualizationOptionTab();
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	void createNirfVisualizationOptionTab();
#endif
#endif
    void createOctVisualizationOptionTab();
		
private:
	// Set thread callback objects
	void setDataAcquisitionCallback();
	void setDeinterleavingCallback();
	void setCh1ProcessingCallback();
	void setCh2ProcessingCallback();
	void setVisualizationCallback();

public:
	// Set NIRF thread callback objects
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
	void setNirfAcquisitionCallback();
#endif
#endif

public: 
	void setCh1ScopeVoltRange(int idx);
#ifdef OCT_FLIM
	void setCh2ScopeVoltRange(int idx);
#endif
	void resetObjectsForAline(int nAlines);
#ifdef OCT_FLIM
	void visualizeImage(float* res1, float* res2, float* res3, float* res4); // OCT-FLIM
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
	void visualizeImage(float* res1, float* res2); // Standalone OCT
#else
	void visualizeImage(float* res1, float* res2, double* res3); // OCT-NIRF
#endif
#endif

private slots:
#ifdef OCT_FLIM
	void constructRgbImage(ImageObject*, ImageObject*, ImageObject*, ImageObject*);
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	void constructRgbImage(ImageObject*, ImageObject*, ImageObject*);
#endif
#endif
	void updateAlinePos(int);
#ifdef OCT_FLIM
	void changeFlimCh(int);
	void changeLifetimeColorTable(int);
	void adjustFlimContrast();
	void createFlimCalibDlg();
	void deleteFlimCalibDlg();
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	void adjustNirfContrast();
	void createNirfEmissionProfileDlg();
	void deleteNirfEmissionProfileDlg();
#endif
#endif
	void changeVisImage(bool);
	void changeFringeBg(bool);
	void checkCircCenter(const QString &);
	void changeOctColorTable(int);
	void adjustOctContrast();	
	void createOctCalibDlg();
	void deleteOctCalibDlg();
	void createOctIntensityHistDlg();
	void deleteOctIntensityHistDlg();

signals:
#ifdef OCT_FLIM
	void plotPulse(const float*);
	void plotFringe(const float*);
	void plotAline(const float*);
	void makeRgb(ImageObject*, ImageObject*, ImageObject*, ImageObject*);
#elif defined (STANDALONE_OCT)
	void plotFringe(const float*, const float*);
	void plotAline(const float*, const float*);
#ifndef OCT_NIRF
	void paintRectImage(uint8_t*);
	void paintCircImage(uint8_t*);
#else
	void makeRgb(ImageObject*, ImageObject*, ImageObject*);
#endif
#endif

// Variables ////////////////////////////////////////////
private:
    MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
	QOperationTab* m_pOperationTab;

	DataAcquisition* m_pDataAcq;
	MemoryBuffer* m_pMemBuff;

public:
	// Data process objects
#ifdef OCT_FLIM
	OCTProcess* m_pOCT;
	FLIMProcess* m_pFLIM;
#elif defined (STANDALONE_OCT)
	OCTProcess* m_pOCT1;
	OCTProcess* m_pOCT2;
#endif

public:
	// Thread manager objects
	ThreadManager* m_pThreadDeinterleave;
	ThreadManager* m_pThreadCh1Process;
	ThreadManager* m_pThreadCh2Process;
	ThreadManager* m_pThreadVisualization;

private:
	// Thread synchronization objects
	SyncObject<uint16_t> m_syncDeinterleaving;
	SyncObject<uint16_t> m_syncCh1Processing;
	SyncObject<uint16_t> m_syncCh2Processing;
	SyncObject<float> m_syncCh1Visualization;
	SyncObject<float> m_syncCh2Visualization;
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
	SyncObject<double> m_syncNirfVisualization;
#endif
#endif

public:
	// Visualization buffers
#ifdef OCT_FLIM
	np::FloatArray2 m_visFringe;
	np::FloatArray2 m_visFringeBg;
	np::FloatArray2 m_visFringeRm;
	np::FloatArray2 m_visImage;

	np::FloatArray2 m_visPulse;
	np::FloatArray2 m_visIntensity;
	np::FloatArray2 m_visMeanDelay;
	np::FloatArray2 m_visLifetime;
#elif defined (STANDALONE_OCT)
	np::FloatArray2 m_visFringe1;
	np::FloatArray2 m_visFringeBg1;
	np::FloatArray2 m_visFringe1Rm;
	np::FloatArray2 m_visImage1;

	np::FloatArray2 m_visFringe2;
	np::FloatArray2 m_visFringeBg2;
	np::FloatArray2 m_visFringe2Rm;
	np::FloatArray2 m_visImage2;

#ifdef OCT_NIRF
	np::DoubleArray2 m_visNirf;
#endif
#endif

	ImageObject *m_pImgObjRectImage;
	ImageObject *m_pImgObjCircImage;
#ifdef OCT_FLIM
	ImageObject *m_pImgObjIntensity;
	ImageObject *m_pImgObjLifetime;
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	ImageObject *m_pImgObjNirf;
#endif
#endif
	circularize* m_pCirc;
	medfilt* m_pMedfilt;


private:
    // Layout
    QHBoxLayout *m_pHBoxLayout;

    // Graph viewer widgets
#ifdef OCT_FLIM
    QScope *m_pScope_FlimPulse;
    QScope *m_pScope_OctFringe;
    QScope *m_pScope_OctDepthProfile;
#elif defined (STANDALONE_OCT)
	QScope2 *m_pScope_OctFringe;
	QScope2 *m_pScope_OctDepthProfile;
#endif
    QSlider *m_pSlider_SelectAline;
    QLabel *m_pLabel_SelectAline;

    // Image viewer widgets
    QImageView *m_pImageView_RectImage;
    QImageView *m_pImageView_CircImage;

#ifdef OCT_FLIM
    // FLIM visualization option widgets
	QGroupBox *m_pGroupBox_FlimVisualization;

	QPushButton *m_pPushButton_FlimCalibration;
    FlimCalibDlg *m_pFlimCalibDlg;

    QLabel *m_pLabel_DataChannel;
    QSpinBox *m_pSpinBox_DataChannel;

    QLabel *m_pLabel_EmissionChannel;
    QComboBox *m_pComboBox_EmissionChannel;

	QLabel *m_pLabel_LifetimeColorTable;
	QComboBox *m_pComboBox_LifetimeColorTable;

    QLabel *m_pLabel_NormIntensity;
    QLabel *m_pLabel_Lifetime;
    QLineEdit *m_pLineEdit_IntensityMax;
    QLineEdit *m_pLineEdit_IntensityMin;
    QLineEdit *m_pLineEdit_LifetimeMax;
    QLineEdit *m_pLineEdit_LifetimeMin;
    QImageView *m_pImageView_IntensityColorbar;
    QImageView *m_pImageView_LifetimeColorbar;
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	// NIRF visualization option widgets
	QGroupBox *m_pGroupBox_NirfVisualization;

	QPushButton *m_pPushButton_NirfEmissionProfile;
    NirfEmissionProfileDlg *m_pNirfEmissionProfileDlg;

	QLabel *m_pLabel_NirfEmission;
	QLineEdit *m_pLineEdit_NirfEmissionMax;
	QLineEdit *m_pLineEdit_NirfEmissionMin;
	QImageView *m_pImageView_NirfEmissionColorbar;
#endif
#endif

    // OCT visualization option widgets
	QGroupBox *m_pGroupBox_OctVisualization;

	QPushButton *m_pPushButton_OctCalibration;
    OctCalibDlg *m_pOctCalibDlg;

	QPushButton *m_pPushButton_OctIntensityHistogram;
	OctIntensityHistDlg *m_pOctIntensityHistDlg;

    QCheckBox *m_pCheckBox_CircularizeImage;
    QCheckBox *m_pCheckBox_ShowBgRemovedSignal;

    QLabel *m_pLabel_CircCenter;
    QLineEdit *m_pLineEdit_CircCenter;

    QLabel *m_pLabel_OctColorTable;
    QComboBox *m_pComboBox_OctColorTable;

    QLabel *m_pLabel_OctDb;
    QLineEdit *m_pLineEdit_OctDbMax;
    QLineEdit *m_pLineEdit_OctDbMin;
    QImageView *m_pImageView_OctDbColorbar;
};

#endif // QSTREAMTAB_H
