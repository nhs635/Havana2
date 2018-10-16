#ifndef QRESULTTAB_H
#define QRESULTTAB_H

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
#ifdef GALVANO_MIRROR
class QDeviceControlTab;
#endif
class MemoryBuffer;

class QImageView;

class SaveResultDlg;
class OctIntensityHistDlg;
#ifdef OCT_FLIM
class PulseReviewDlg;
#endif
#ifdef OCT_NIRF
class NirfEmissionProfileDlg;
class NirfDistCompDlg;
#endif

class OCTProcess;
#ifdef OCT_FLIM
class FLIMProcess;
#endif


class QResultTab : public QDialog
{
    Q_OBJECT

// Constructer & Destructer /////////////////////////////
public:
    explicit QResultTab(QWidget *parent = 0);
	virtual ~QResultTab();

// Methods //////////////////////////////////////////////
protected:
	void keyPressEvent(QKeyEvent *);

public:
	inline MainWindow* getMainWnd() const { return m_pMainWnd; }
	inline SaveResultDlg* getSaveResultDlg() const { return m_pSaveResultDlg; }
	inline OctIntensityHistDlg* getOctIntensityHistDlg() const { return m_pOctIntensityHistDlg; }
#ifdef OCT_FLIM
	inline PulseReviewDlg* getPulseReviewDlg() const { return m_pPulseReviewDlg; }
#endif
#ifdef OCT_NIRF

    inline NirfEmissionProfileDlg* getNirfEmissionProfileDlg() const { return m_pNirfEmissionProfileDlg; }
	inline NirfDistCompDlg* getNirfDistCompDlg() const { return m_pNirfDistCompDlg; }
#endif
	inline QRadioButton* getRadioInBuffer() const { return m_pRadioButton_InBuffer; }
	inline QProgressBar* getProgressBar() const { return m_pProgressBar_PostProcessing; }
	inline QImageView* getRectImageView() const { return m_pImageView_RectImage; }
	inline QImageView* getCircImageView() const { return m_pImageView_CircImage; }
	inline int getCurrentFrame() const { return m_pSlider_SelectFrame->value(); }
	inline int getCurrentOctColorTable() const { return m_pComboBox_OctColorTable->currentIndex(); }
#ifdef OCT_FLIM
	inline int getCurrentLifetimeColorTable() const { return m_pComboBox_LifetimeColorTable->currentIndex(); }
#endif
#ifdef OCT_NIRF
    inline int getCurrentNirfOffset() const { return m_pScrollBar_NirfOffset->value(); }
#endif
	inline void setUserDefinedAlines(int nAlines) { m_pLineEdit_UserDefinedAlines->setText(QString::number(nAlines)); }
	inline void invalidate() { visualizeEnFaceMap(true); visualizeImage(getCurrentFrame()); }
	void setWidgetsText();

private:
    void createDataLoadingWritingTab();
    void createVisualizationOptionTab();
    void createEnFaceMapTab();
	
private slots: // widget operation
	void changeDataSelection(int id);
	void createSaveResultDlg();
	void deleteSaveResultDlg();
	void createOctIntensityHistDlg();
	void deleteOctIntensityHistDlg();
#ifdef OCT_FLIM
	void createPulseReviewDlg();
	void deletePulseReviewDlg();
#endif
#ifdef OCT_NIRF    
    void createNirfEmissionProfileDlg();
    void deleteNirfEmissionProfileDlg();
	void createNirfDistCompDlg();
	void deleteNirfDistCompDlg();
#endif
	void enableUserDefinedAlines(bool);
	void visualizeImage(int);
#ifdef OCT_FLIM
	void constructRgbImage(ImageObject*, ImageObject*, ImageObject*, ImageObject*);
#endif
#ifdef OCT_NIRF
	void constructRgbImage(ImageObject*, ImageObject*, ImageObject*);
#endif
	void visualizeEnFaceMap(bool scaling = true);
	void measureDistance(bool);
	void showGuideLine(bool);
	void changeVisImage(bool);
	void checkCircCenter(const QString &);
	void adjustOctContrast();
#ifdef OCT_FLIM
	void changeFlimCh(int);
	void enableHsvEnhancingMode(bool);
	void adjustFlimContrast();
	void changeLifetimeColorTable(int);
#endif
#ifdef OCT_NIRF
	void adjustNirfContrast();
	void adjustNirfOffset(int);
#endif

private slots: // processing
	void startProcessing();
	void setWidgetsEnabled(bool, Configuration*);

public slots: // for saving
	void setWidgetsEnabled(bool);
	
signals:
	void setWidgets(bool, Configuration*);
	void setWidgets(bool);

#ifdef OCT_FLIM
	void makeRgb(ImageObject*, ImageObject*, ImageObject*, ImageObject*);
#endif
#ifdef OCT_NIRF
	void makeRgb(ImageObject*, ImageObject*, ImageObject*);
#endif
	void paintRectImage(uint8_t*);
	void paintCircImage(uint8_t*);

	void paintOctProjection(uint8_t*);
#ifdef OCT_FLIM
	void paintIntensityMap(uint8_t*);
	void paintLifetimeMap(uint8_t*);
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	void paintNirfMap(uint8_t*);
#else
	void paintNirfMap1(uint8_t*);
	void paintNirfMap2(uint8_t*);
#endif
#endif
	void processedSingleFrame(int);

private:
	void inBufferDataProcessing();
	void externalDataProcessing();

	void setObjects(Configuration* pConfig);

	void loadingRawData(QFile* pFile, Configuration* pConfig);
	void deinterleaving(Configuration* pConfig);
	void deinterleavingInBuffer(Configuration* pConfig);
#ifdef OCT_FLIM
	void octProcessing(OCTProcess* pOCT, Configuration* pConfig);	
	void flimProcessing(FLIMProcess* pFLIM, Configuration* pConfig);
#elif defined (STANDALONE_OCT)
	void octProcessing1(OCTProcess* pOCT, Configuration* pConfig);
	void octProcessing2(OCTProcess* pOCT, Configuration* pConfig);
#ifdef DUAL_CHANNEL
	void imageMerge(Configuration* pConfig);
#endif
#endif

private:
	void getOctProjection(std::vector<np::FloatArray2>& vecImg, np::FloatArray2& octProj, int offset);

// Variables ////////////////////////////////////////////
private: // main pointer
	MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
#ifdef GALVANO_MIRROR
	QDeviceControlTab* m_pDeviceControlTab;
#endif
	MemoryBuffer* m_pMemBuff;

#ifdef OCT_FLIM
public:
	FLIMProcess* m_pFLIMpost;
#endif

private: // for threading operation
	SyncObject<uint16_t> m_syncDeinterleaving;
	SyncObject<uint16_t> m_syncCh1Processing;
	SyncObject<uint16_t> m_syncCh2Processing;
#ifdef STANDALONE_OCT
#ifdef DUAL_CHANNEL
	SyncObject<float> m_syncCh1Visualization;
	SyncObject<float> m_syncCh2Visualization;
#endif
#endif

public: // for visualization
	std::vector<np::FloatArray2> m_vectorOctImage;
	np::FloatArray2 m_octProjection;
	int m_polishedSurface;
#ifdef OCT_FLIM
	std::vector<np::FloatArray2> m_intensityMap; // (256 x N) x 3
	std::vector<np::FloatArray2> m_lifetimeMap; // (256 x N) x 3
	std::vector<np::FloatArray2> m_vectorPulseCrop;
	std::vector<np::FloatArray2> m_vectorPulseMask;
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	np::FloatArray m_nirfSignal;
	np::FloatArray2 m_nirfMap;
    np::FloatArray2 m_nirfMap0;
#else
	np::FloatArray m_nirfSignal1;
	np::FloatArray2 m_nirfMap1;
	np::FloatArray2 m_nirfMap1_0;
	np::FloatArray m_nirfSignal2;
	np::FloatArray2 m_nirfMap2;
	np::FloatArray2 m_nirfMap2_0;
#endif
	int m_nirfOffset;
#endif

private:
	ImageObject *m_pImgObjRectImage;
	ImageObject *m_pImgObjCircImage;
#ifdef OCT_FLIM
	ImageObject *m_pImgObjIntensity;
	ImageObject *m_pImgObjLifetime;
#endif
#ifdef OCT_NIRF
	ImageObject *m_pImgObjNirf;
#endif

	np::Uint8Array2 m_visOctProjection;
#ifdef OCT_FLIM
	ImageObject *m_pImgObjIntensityMap;
	ImageObject *m_pImgObjLifetimeMap;
	ImageObject *m_pImgObjHsvEnhancedMap;
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	ImageObject *m_pImgObjNirfMap;
#else
	ImageObject *m_pImgObjNirfMap1;
	ImageObject *m_pImgObjNirfMap2;
#endif
#endif	

public:
	circularize* m_pCirc;
	medfilt* m_pMedfiltRect;
#ifdef OCT_FLIM
	medfilt* m_pMedfiltIntensityMap;
	medfilt* m_pMedfiltLifetimeMap;
#endif
#ifdef OCT_NIRF
    medfilt* m_pMedfiltNirf;
#endif


public:
	QString m_path;


private:
    // Layout
    QHBoxLayout *m_pHBoxLayout;

    // Image viewer widgets
    QImageView *m_pImageView_RectImage;
    QImageView *m_pImageView_CircImage;
#ifdef OCT_FLIM
	PulseReviewDlg *m_pPulseReviewDlg;
#endif

    // Data loading & writing widgets
	QGroupBox *m_pGroupBox_DataLoadingWriting;

	QButtonGroup *m_pButtonGroup_DataSelection;
    QRadioButton *m_pRadioButton_InBuffer;
    QRadioButton *m_pRadioButton_External;

    QPushButton *m_pPushButton_StartProcessing;
    QPushButton *m_pPushButton_SaveResults;
	SaveResultDlg *m_pSaveResultDlg;

	QCheckBox *m_pCheckBox_UserDefinedAlines;
	QLineEdit *m_pLineEdit_UserDefinedAlines;

	QCheckBox *m_pCheckBox_SingleFrame;

	QLabel *m_pLabel_DiscomValue;
	QLineEdit *m_pLineEdit_DiscomValue;

	QProgressBar *m_pProgressBar_PostProcessing;

    // Visualization option widgets
	QGroupBox *m_pGroupBox_Visualization;

    QSlider *m_pSlider_SelectFrame;
    QLabel *m_pLabel_SelectFrame;
	
    QPushButton* m_pToggleButton_MeasureDistance;

#ifdef OCT_NIRF
	QPushButton *m_pPushButton_NirfDistanceCompensation;
	NirfDistCompDlg *m_pNirfDistCompDlg;
	NirfEmissionProfileDlg *m_pNirfEmissionProfileDlg;
#endif

	QPushButton *m_pPushButton_OctIntensityHistogram;
	OctIntensityHistDlg *m_pOctIntensityHistDlg;

    QCheckBox *m_pCheckBox_ShowGuideLine;
    QCheckBox *m_pCheckBox_CircularizeImage;

    QLabel *m_pLabel_CircCenter;
    QLineEdit *m_pLineEdit_CircCenter;

    QLabel *m_pLabel_OctColorTable;
    QComboBox *m_pComboBox_OctColorTable;

#ifdef OCT_NIRF
	QLabel *m_pLabel_NirfOffset;
	QLineEdit *m_pLineEdit_NirfOffset;
	QScrollBar *m_pScrollBar_NirfOffset;
#endif

#ifdef OCT_FLIM
    QLabel *m_pLabel_EmissionChannel;
    QComboBox *m_pComboBox_EmissionChannel;

    QCheckBox* m_pCheckBox_HsvEnhancedMap;

	QLabel *m_pLabel_LifetimeColorTable;
	QComboBox *m_pComboBox_LifetimeColorTable;
#endif

    // En face map tab widgets
	QGroupBox *m_pGroupBox_EnFace;

    QLabel *m_pLabel_OctProjection;
#ifdef OCT_FLIM
    QLabel *m_pLabel_IntensityMap;
    QLabel *m_pLabel_LifetimeMap;
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	QLabel *m_pLabel_NirfMap;
#else
	QLabel *m_pLabel_NirfMap1;
	QLabel *m_pLabel_NirfMap2;
#endif
#endif

    QImageView *m_pImageView_OctProjection;
#ifdef OCT_FLIM
    QImageView *m_pImageView_IntensityMap;
    QImageView *m_pImageView_LifetimeMap;
    QImageView *m_pImageView_HsvEnhancedMap;
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
    QImageView *m_pImageView_NirfMap;
#else
	QImageView *m_pImageView_NirfMap1;
	QImageView *m_pImageView_NirfMap2;
#endif
#endif

    QImageView *m_pImageView_ColorbarOctProjection;
#ifdef OCT_FLIM
    QImageView *m_pImageView_ColorbarIntensityMap;
    QImageView *m_pImageView_ColorbarLifetimeMap;
#endif
#ifdef OCT_NIRF
	QImageView *m_pImageView_ColorbarNirfMap;
#endif

    QLineEdit *m_pLineEdit_OctDbMax;
    QLineEdit *m_pLineEdit_OctDbMin;
#ifdef OCT_FLIM
    QLineEdit *m_pLineEdit_IntensityMax;
    QLineEdit *m_pLineEdit_IntensityMin;
    QLineEdit *m_pLineEdit_LifetimeMax;
    QLineEdit *m_pLineEdit_LifetimeMin;
#endif
#ifdef OCT_NIRF
	QLineEdit *m_pLineEdit_NirfMax;
	QLineEdit *m_pLineEdit_NirfMin;
#endif

};

#endif // QRESULTTAB_H
