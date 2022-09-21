
#include "QResultTab.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QOperationTab.h>
#include <Havana2/QStreamTab.h>
#ifdef GALVANO_MIRROR
#include <Havana2/QDeviceControlTab.h>
#endif
#include <Havana2/Viewer/QImageView.h>
#include <Havana2/Dialog/SaveResultDlg.h>
#include <Havana2/Dialog/OctIntensityHistDlg.h>
#include <Havana2/Dialog/LongitudinalViewDlg.h>
#include <Havana2/Dialog/SpectroOCTDlg.h>
#ifdef OCT_FLIM
#include <Havana2/Dialog/PulseReviewDlg.h>
#endif
#ifdef OCT_NIRF
#include <Havana2/Dialog/NirfEmissionProfileDlg.h>
#include <Havana2/Dialog/NirfDistCompDlg.h>
#ifdef TWO_CHANNEL_NIRF
#include <Havana2/Dialog/NirfCrossTalkCompDlg.h>
#endif
#endif

#include <MemoryBuffer/MemoryBuffer.h>

#ifndef CUDA_ENABLED
#include <DataProcess/OCTProcess/OCTProcess.h>
#else
#include <CUDA/CudaOCTProcess.cuh>
#endif
#ifdef OCT_FLIM
#include <DataProcess/FLIMProcess/FLIMProcess.h>
#endif

#include <ipps.h>
#include <ippi.h>
#include <ippcc.h>
#include <ippvm.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <iostream>
#include <thread>

#include <time.h>

#define IN_BUFFER_DATA -2
#define EXTERNAL_DATA  -3


QResultTab::QResultTab(QWidget *parent) :
    QDialog(parent), 
	m_pConfigTemp(nullptr),
	m_pImgObjRectImage(nullptr), m_pImgObjCircImage(nullptr), m_pCirc(nullptr), m_polishedSurface(0),
	m_pMedfiltRect(nullptr), m_pLumenDetection(nullptr),
	m_pSaveResultDlg(nullptr), m_pLongitudinalViewDlg(nullptr), m_pSpectroOCTDlg(nullptr)
#ifdef OCT_FLIM
	, m_pFLIMpost(nullptr), m_pPulseReviewDlg(nullptr)
	, m_pImgObjIntensity(nullptr), m_pImgObjLifetime(nullptr)
	, m_pImgObjIntensityMap(nullptr), m_pImgObjLifetimeMap(nullptr), m_pImgObjHsvEnhancedMap(nullptr)
	, m_pMedfiltIntensityMap(nullptr), m_pMedfiltLifetimeMap(nullptr), m_pAnn(nullptr)
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	, m_pImgObjNirf(nullptr), m_pImgObjNirfMap(nullptr)
#else
	, m_pImgObjNirf1(nullptr), m_pImgObjNirf2(nullptr)
	, m_pImgObjNirfMap1(nullptr), m_pImgObjNirfMap2(nullptr)
#endif	
	, m_nirfOffset(0)
	, m_pMedfiltNirf(nullptr)
    , m_pNirfEmissionProfileDlg(nullptr), m_pNirfDistCompDlg(nullptr)
#ifdef TWO_CHANNEL_NIRF
	, m_pNirfCrossTalkCompDlg(nullptr)
#endif
#endif
{
	// Set main window objects
	m_pMainWnd = (MainWindow*)parent;
	m_pConfig = m_pMainWnd->m_pConfiguration;
#ifdef GALVANO_MIRROR
	m_pDeviceControlTab = m_pMainWnd->m_pDeviceControlTab;
#endif
	m_pMemBuff = m_pMainWnd->m_pOperationTab->m_pMemoryBuffer;


    // Create layout
	QHBoxLayout* pHBoxLayout = new QHBoxLayout;
	pHBoxLayout->setSpacing(0);

    // Create image view
	QVBoxLayout* pVBoxLayout_ImageView = new QVBoxLayout;
	pVBoxLayout_ImageView->setSpacing(0);

#ifdef OCT_FLIM
    bool rgb_used = true;
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
	bool rgb_used = false;
#else
	bool rgb_used = true;
#endif
#endif
    m_pImageView_RectImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), m_pConfig->nAlines, m_pConfig->n2ScansFFT, m_pConfig->octDbGamma, rgb_used);
    m_pImageView_RectImage->setMinimumSize(600, 600);
	m_pImageView_RectImage->setDisabled(true);
	m_pImageView_RectImage->setMovedMouseCallback([&](QPoint& p) { m_pMainWnd->m_pStatusLabel_ImagePos->setText(QString("(%1, %2)").arg(p.x(), 4).arg(p.y(), 4)); });
#ifdef OCT_FLIM
	m_pImageView_RectImage->setDoubleClickedMouseCallback([&]() { createPulseReviewDlg(); });
#endif
	m_pImageView_RectImage->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
	m_pImageView_RectImage->getRender()->m_bCanBeMagnified = true;
	m_pImageView_RectImage->setVisible(false);

    m_pImageView_CircImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), 2 * m_pConfig->circRadius, 2 * m_pConfig->circRadius, m_pConfig->octDbGamma, rgb_used);
    m_pImageView_CircImage->setMinimumSize(600, 600);
	m_pImageView_CircImage->setDisabled(true);
	m_pImageView_CircImage->setMovedMouseCallback([&](QPoint& p) { m_pMainWnd->m_pStatusLabel_ImagePos->setText(QString("(%1, %2)").arg(p.x(), 4).arg(p.y(), 4)); });
#ifdef OCT_FLIM
	m_pImageView_CircImage->setDoubleClickedMouseCallback([&]() { createPulseReviewDlg(); });
#endif
	m_pImageView_CircImage->setSquare(true);
	m_pImageView_CircImage->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
	m_pImageView_CircImage->getRender()->m_bCanBeMagnified = true;
	m_pImageView_CircImage->setVisible(true);

	QLabel *pNullLabel = new QLabel("", this);
	pNullLabel->setMinimumWidth(600);
	pNullLabel->setFixedHeight(0);
	pNullLabel->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

	// Create image view buffers
	ColorTable temp_ctable;
	m_pImgObjRectImage = new ImageObject(m_pConfig->nAlines4, m_pConfig->n2ScansFFT, temp_ctable.m_colorTableVector.at(m_pConfig->octColorTable), m_pConfig->octDbGamma);
    m_pImgObjCircImage = new ImageObject(2 * m_pConfig->circRadius, 2 * m_pConfig->circRadius, temp_ctable.m_colorTableVector.at(m_pConfig->octColorTable), m_pConfig->octDbGamma);

    // Set layout for left panel
	pVBoxLayout_ImageView->addWidget(m_pImageView_RectImage);
	pVBoxLayout_ImageView->addWidget(m_pImageView_CircImage);
	pVBoxLayout_ImageView->addWidget(pNullLabel);

	pHBoxLayout->addItem(pVBoxLayout_ImageView);
	
    // Create data loading & writing tab
    createDataLoadingWritingTab();

    // Create visualization option tab
    createVisualizationOptionTab();

    // Create en face map tab
    createEnFaceMapTab();

    // Set layout for right panel
    QVBoxLayout *pVBoxLayout_RightPanel = new QVBoxLayout;
    pVBoxLayout_RightPanel->setSpacing(0);
	pVBoxLayout_RightPanel->setContentsMargins(0, 0, 0, 0);

    pVBoxLayout_RightPanel->addWidget(m_pGroupBox_DataLoadingWriting);
	pVBoxLayout_RightPanel->addWidget(m_pGroupBox_Visualization);
    pVBoxLayout_RightPanel->addWidget(m_pGroupBox_EnFace);

    pHBoxLayout->addItem(pVBoxLayout_RightPanel);

    this->setLayout(pHBoxLayout);
	
	// Connect signal and slot
	connect(this, SIGNAL(setWidgets(bool, Configuration*)), this, SLOT(setWidgetsEnabled(bool, Configuration*)));

#ifdef OCT_FLIM
	connect(this, SIGNAL(makeRgb(ImageObject*, ImageObject*, ImageObject*, ImageObject*)), 
		this, SLOT(constructRgbImage(ImageObject*, ImageObject*, ImageObject*, ImageObject*)));
	connect(this, SIGNAL(paintRectImage(uint8_t*)), m_pImageView_RectImage, SLOT(drawRgbImage(uint8_t*)));
	connect(this, SIGNAL(paintCircImage(uint8_t*)), m_pImageView_CircImage, SLOT(drawRgbImage(uint8_t*)));
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	connect(this, SIGNAL(makeRgb(ImageObject*, ImageObject*, ImageObject*)),
		this, SLOT(constructRgbImage(ImageObject*, ImageObject*, ImageObject*)));
#else
	connect(this, SIGNAL(makeRgb(ImageObject*, ImageObject*, ImageObject*, ImageObject*)),
		this, SLOT(constructRgbImage(ImageObject*, ImageObject*, ImageObject*, ImageObject*)));
#endif
	connect(this, SIGNAL(paintRectImage(uint8_t*)), m_pImageView_RectImage, SLOT(drawRgbImage(uint8_t*)));
	connect(this, SIGNAL(paintCircImage(uint8_t*)), m_pImageView_CircImage, SLOT(drawRgbImage(uint8_t*)));
#else
	connect(this, SIGNAL(paintRectImage(uint8_t*)), m_pImageView_RectImage, SLOT(drawImage(uint8_t*)));
	connect(this, SIGNAL(paintCircImage(uint8_t*)), m_pImageView_CircImage, SLOT(drawImage(uint8_t*)));
#endif
#endif
}

QResultTab::~QResultTab()
{
	if (m_pImgObjRectImage) delete m_pImgObjRectImage;
	if (m_pImgObjCircImage) delete m_pImgObjCircImage;
#ifdef OCT_FLIM
	if (m_pImgObjIntensity) delete m_pImgObjIntensity;
	if (m_pImgObjLifetime) delete m_pImgObjLifetime;
	if (m_pImgObjIntensityMap) delete m_pImgObjIntensityMap;
	if (m_pImgObjLifetimeMap) delete m_pImgObjLifetimeMap;
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	if (m_pImgObjNirf) delete m_pImgObjNirf;
	if (m_pImgObjNirfMap) delete m_pImgObjNirfMap;
#else
	if (m_pImgObjNirf1) delete m_pImgObjNirf1;
	if (m_pImgObjNirf2) delete m_pImgObjNirf2;
	if (m_pImgObjNirfMap1) delete m_pImgObjNirfMap1;
	if (m_pImgObjNirfMap2) delete m_pImgObjNirfMap2;
#endif
#endif
	if (m_pMedfiltRect) delete m_pMedfiltRect;
#ifdef OCT_FLIM
	if (m_pMedfiltIntensityMap) delete m_pMedfiltIntensityMap;
	if (m_pMedfiltLifetimeMap) delete m_pMedfiltLifetimeMap;
#endif
#ifdef OCT_NIRF
    if (m_pMedfiltNirf) delete m_pMedfiltNirf;
#endif
	if (m_pCirc) delete m_pCirc;	
}

void QResultTab::keyPressEvent(QKeyEvent *e)
{
	if (e->key() != Qt::Key_Escape)
		QDialog::keyPressEvent(e);
}

void QResultTab::setWidgetsText()
{
	if (!m_pConfigTemp)
		m_pLineEdit_DiscomValue->setText(QString::number(m_pConfig->octDiscomVal));
	m_pLineEdit_CircCenter->setText(QString::number(m_pConfig->circCenter));

	if (m_pConfigTemp)
	{
		m_pLineEdit_CircRadius->setText(QString::number(m_pConfigTemp->circRadius));
		
#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
		if (m_pConfig->ringThickness > m_pConfigTemp->circRadius - 2)
			m_pConfig->ringThickness = m_pConfigTemp->circRadius - 2;
		m_pLineEdit_RingThickness->setText(QString::number(m_pConfig->ringThickness));
#endif
	}
	else
	{
		if (m_pConfig->circCenter + m_pConfig->circRadius + 1 > m_pConfig->n2ScansFFT)
			m_pConfig->circRadius = m_pConfig->n2ScansFFT - m_pConfig->circCenter - 1;
		m_pLineEdit_CircRadius->setText(QString::number(m_pConfig->circRadius));

#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
		if (m_pConfig->ringThickness > m_pConfig->circRadius - 2)
			m_pConfig->ringThickness = m_pConfig->circRadius - 2;
		m_pLineEdit_RingThickness->setText(QString::number(m_pConfig->ringThickness));
#endif
	}

	m_pComboBox_OctColorTable->setCurrentIndex(m_pConfig->octColorTable);
	m_pLineEdit_OctDbMax->setText(QString::number(m_pConfig->octDbRange.max));
	m_pLineEdit_OctDbMin->setText(QString::number(m_pConfig->octDbRange.min));

#ifdef OCT_FLIM
	m_pComboBox_LifetimeColorTable->setCurrentIndex(m_pConfig->flimLifetimeColorTable);
	m_pLineEdit_IntensityMax->setText(QString::number(m_pConfig->flimIntensityRange.max, 'f', 1));
	m_pLineEdit_IntensityMin->setText(QString::number(m_pConfig->flimIntensityRange.min, 'f', 1));
	m_pLineEdit_LifetimeMax->setText(QString::number(m_pConfig->flimLifetimeRange.max, 'f', 1));
	m_pLineEdit_LifetimeMin->setText(QString::number(m_pConfig->flimLifetimeRange.min, 'f', 1));
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	m_pLineEdit_NirfMax->setText(QString::number(m_pConfig->nirfRange.max, 'f', 2));
	m_pLineEdit_NirfMin->setText(QString::number(m_pConfig->nirfRange.min, 'f', 2));
#else
	m_pLineEdit_NirfMax[0]->setText(QString::number(m_pConfig->nirfRange[0].max, 'f', 2));
	m_pLineEdit_NirfMin[0]->setText(QString::number(m_pConfig->nirfRange[0].min, 'f', 2));
	m_pLineEdit_NirfMax[1]->setText(QString::number(m_pConfig->nirfRange[1].max, 'f', 2));
	m_pLineEdit_NirfMin[1]->setText(QString::number(m_pConfig->nirfRange[1].min, 'f', 2));
#endif
#endif
#endif
}


void QResultTab::createDataLoadingWritingTab()
{
    // Create widgets for loading and writing tab
    m_pGroupBox_DataLoadingWriting = new QGroupBox;
	m_pGroupBox_DataLoadingWriting->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    QGridLayout *pGridLayout_DataLoadingWriting = new QGridLayout;
    pGridLayout_DataLoadingWriting->setSpacing(3);
	
    // Create radio buttons to chooose data for the processing
	m_pRadioButton_InBuffer = new QRadioButton(this);
    m_pRadioButton_InBuffer->setText("Use In-Buffer Data  ");
	m_pRadioButton_InBuffer->setDisabled(true);
    m_pRadioButton_External = new QRadioButton(this);
    m_pRadioButton_External->setText("Use External Data  ");

	m_pButtonGroup_DataSelection = new QButtonGroup(this);
	m_pButtonGroup_DataSelection->addButton(m_pRadioButton_InBuffer, IN_BUFFER_DATA);
	m_pButtonGroup_DataSelection->addButton(m_pRadioButton_External, EXTERNAL_DATA);
	m_pRadioButton_External->setChecked(true);

    // Create buttons for the processing
    m_pPushButton_StartProcessing = new QPushButton(this);
	m_pPushButton_StartProcessing->setText("Start Processing");
	m_pPushButton_StartProcessing->setFixedWidth(130);

    m_pPushButton_SaveResults = new QPushButton(this);
	m_pPushButton_SaveResults->setText("Save Results...");
	m_pPushButton_SaveResults->setFixedWidth(130);
	m_pPushButton_SaveResults->setDisabled(true);

	// Create widgets for user defined parameter 
	m_pCheckBox_UserDefinedAlines = new QCheckBox(this);
	m_pCheckBox_UserDefinedAlines->setText("User-Defined nAlines");

	m_pLineEdit_UserDefinedAlines = new QLineEdit(this);
	m_pLineEdit_UserDefinedAlines->setText(QString::number(m_pConfig->nAlines));
	m_pLineEdit_UserDefinedAlines->setFixedWidth(35);
	m_pLineEdit_UserDefinedAlines->setAlignment(Qt::AlignCenter);
	m_pLineEdit_UserDefinedAlines->setDisabled(true);
	
	m_pCheckBox_SingleFrame = new QCheckBox(this);
	m_pCheckBox_SingleFrame->setText("Single Frame Processing");

	m_pCheckBox_DiscomValue = new QCheckBox(this);
	m_pCheckBox_DiscomValue->setText("Discom Value");

	m_pLineEdit_DiscomValue = new QLineEdit(this);
	m_pLineEdit_DiscomValue->setText(QString::number(m_pConfig->octDiscomVal));
	m_pLineEdit_DiscomValue->setFixedWidth(30);
	m_pLineEdit_DiscomValue->setAlignment(Qt::AlignCenter);
	m_pLineEdit_DiscomValue->setDisabled(true);

	// Create progress bar
	m_pProgressBar_PostProcessing = new QProgressBar(this);
	m_pProgressBar_PostProcessing->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	m_pProgressBar_PostProcessing->setFormat("");
	m_pProgressBar_PostProcessing->setValue(0);
	
    // Set layout
	pGridLayout_DataLoadingWriting->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 0, 3, 1);

	pGridLayout_DataLoadingWriting->addWidget(m_pRadioButton_InBuffer, 0, 1);
	pGridLayout_DataLoadingWriting->addWidget(m_pRadioButton_External, 1, 1);

	pGridLayout_DataLoadingWriting->addWidget(m_pPushButton_StartProcessing, 0, 2);
	pGridLayout_DataLoadingWriting->addWidget(m_pPushButton_SaveResults, 1, 2);
	
	QHBoxLayout *pHBoxLayout_UserDefined = new QHBoxLayout;
	pHBoxLayout_UserDefined->setSpacing(3);

	pHBoxLayout_UserDefined->addWidget(m_pCheckBox_UserDefinedAlines);
	pHBoxLayout_UserDefined->addWidget(m_pLineEdit_UserDefinedAlines);
	pHBoxLayout_UserDefined->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));
	
	QHBoxLayout *pHBoxLayout_SingleFrameDiscomValue = new QHBoxLayout;
	pHBoxLayout_SingleFrameDiscomValue->setSpacing(3);

	pHBoxLayout_SingleFrameDiscomValue->addWidget(m_pCheckBox_SingleFrame);
	pHBoxLayout_SingleFrameDiscomValue->addWidget(m_pCheckBox_DiscomValue);
	pHBoxLayout_SingleFrameDiscomValue->addWidget(m_pLineEdit_DiscomValue);

	pGridLayout_DataLoadingWriting->addItem(pHBoxLayout_UserDefined, 2, 1, 1, 2);
	pGridLayout_DataLoadingWriting->addItem(pHBoxLayout_SingleFrameDiscomValue, 3, 1, 1, 2);

	pGridLayout_DataLoadingWriting->addWidget(m_pProgressBar_PostProcessing, 4, 1, 1, 2);

	pGridLayout_DataLoadingWriting->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 3, 3, 1);

    m_pGroupBox_DataLoadingWriting->setLayout(pGridLayout_DataLoadingWriting);
	m_pGroupBox_DataLoadingWriting->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");

	// Connect signal and slot
	connect(m_pButtonGroup_DataSelection, SIGNAL(buttonClicked(int)), this, SLOT(changeDataSelection(int)));
	connect(m_pPushButton_StartProcessing, SIGNAL(clicked(bool)), this, SLOT(startProcessing(void)));
	connect(m_pPushButton_SaveResults, SIGNAL(clicked(bool)), this, SLOT(createSaveResultDlg()));
	connect(m_pCheckBox_UserDefinedAlines, SIGNAL(toggled(bool)), this, SLOT(enableUserDefinedAlines(bool)));
	connect(m_pCheckBox_DiscomValue, SIGNAL(toggled(bool)), this, SLOT(enableDiscomValue(bool)));

	connect(this, SIGNAL(processedSingleFrame(int)), m_pProgressBar_PostProcessing, SLOT(setValue(int)));
}

void QResultTab::createVisualizationOptionTab()
{
    // Create widgets for visualization option tab
    m_pGroupBox_Visualization = new QGroupBox;
	m_pGroupBox_Visualization->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    QGridLayout *pGridLayout_Visualization = new QGridLayout;    
	pGridLayout_Visualization->setSpacing(3);

    // Create slider for exploring frames
    m_pSlider_SelectFrame = new QSlider(this);
    m_pSlider_SelectFrame->setOrientation(Qt::Horizontal);
    m_pSlider_SelectFrame->setValue(0);
	m_pSlider_SelectFrame->setDisabled(true);

    m_pLabel_SelectFrame = new QLabel(this);
    QString str; str.sprintf("Current Frame : %3d / %3d", 1, 1);
    m_pLabel_SelectFrame->setText(str);
	m_pLabel_SelectFrame->setFixedWidth(130);
    m_pLabel_SelectFrame->setBuddy(m_pSlider_SelectFrame);
	m_pLabel_SelectFrame->setDisabled(true);

    // Create push button for measuring distance
    m_pToggleButton_MeasureDistance = new QPushButton(this);
	m_pToggleButton_MeasureDistance->setCheckable(true);
    m_pToggleButton_MeasureDistance->setText("Measure Distance");
	m_pToggleButton_MeasureDistance->setDisabled(true);

#ifdef OCT_NIRF
	// Create push button for NIRF Distance Compensation
	m_pPushButton_NirfDistanceCompensation = new QPushButton(this);
	m_pPushButton_NirfDistanceCompensation->setText("Distance Compensation...");
    m_pPushButton_NirfDistanceCompensation->setDisabled(true);

#ifdef TWO_CHANNEL_NIRF
	// Create push button for 2Ch NIRF Cross Talk Correction
	m_pPushButton_NirfCrossTalkCompensation = new QPushButton(this);
	m_pPushButton_NirfCrossTalkCompensation->setText("Cross Talk Correction...");
	m_pPushButton_NirfCrossTalkCompensation->setDisabled(true);
	m_pPushButton_NirfCrossTalkCompensation->setVisible(false);
#endif
#endif
		
    // Create checkboxs for OCT operation
    m_pCheckBox_ShowGuideLine = new QCheckBox(this);
    m_pCheckBox_ShowGuideLine->setText("Show Guide Line");
	m_pCheckBox_ShowGuideLine->setDisabled(true);
    m_pCheckBox_CircularizeImage = new QCheckBox(this);
    m_pCheckBox_CircularizeImage->setText("Circularize Image");
	m_pCheckBox_CircularizeImage->setChecked(true);
	m_pCheckBox_CircularizeImage->setDisabled(true);

	// Create widegts for OCT longitudinal visualization
	m_pPushButton_LongitudinalView = new QPushButton(this);
	m_pPushButton_LongitudinalView->setText("Longitudinal View...");
	m_pPushButton_LongitudinalView->setDisabled(true);

	// Create widegts for Spectroscopic OCT processing
	m_pPushButton_SpectroscopicView = new QPushButton(this);
	m_pPushButton_SpectroscopicView->setText("Spectroscopic OCT...");
	m_pPushButton_SpectroscopicView->setDisabled(true);
	
    // Create widgets for OCT circularizing
	//m_pToggleButton_FindPolishedSurfaces = new QPushButton(this);
	//m_pToggleButton_FindPolishedSurfaces->setCheckable(true);
	//m_pToggleButton_FindPolishedSurfaces->setText("Find Polished Surfaces");
	//m_pToggleButton_FindPolishedSurfaces->setDisabled(true);
	m_pToggleButton_AutoContour = new QPushButton(this);
	m_pToggleButton_AutoContour->setCheckable(true);
	m_pToggleButton_AutoContour->setText("Auto Lumen Contour");
	m_pToggleButton_AutoContour->setDisabled(true);

    m_pLineEdit_CircCenter = new QLineEdit(this);
    m_pLineEdit_CircCenter->setFixedWidth(30);
    m_pLineEdit_CircCenter->setText(QString::number(m_pConfig->circCenter));
	m_pLineEdit_CircCenter->setAlignment(Qt::AlignCenter);
	m_pLineEdit_CircCenter->setDisabled(true);
    m_pLabel_CircCenter = new QLabel("Circ Center", this);
    m_pLabel_CircCenter->setBuddy(m_pLineEdit_CircCenter);
	m_pLabel_CircCenter->setDisabled(true);
	
	m_pLineEdit_CircRadius = new QLineEdit(this);
	m_pLineEdit_CircRadius->setFixedWidth(30);
	m_pLineEdit_CircRadius->setText(QString::number(m_pConfig->circRadius));
	m_pLineEdit_CircRadius->setAlignment(Qt::AlignCenter);
	m_pLineEdit_CircRadius->setDisabled(true);
	m_pLabel_CircRadius = new QLabel("Circ Radius", this);
	m_pLabel_CircRadius->setBuddy(m_pLineEdit_CircRadius);
	m_pLabel_CircRadius->setDisabled(true);

	m_pLineEdit_SheathRadius = new QLineEdit(this);
	m_pLineEdit_SheathRadius->setFixedWidth(30);
	m_pLineEdit_SheathRadius->setText(QString::number(m_pConfig->sheathRadius));
	m_pLineEdit_SheathRadius->setAlignment(Qt::AlignCenter);
	m_pLineEdit_SheathRadius->setDisabled(true);
	m_pLabel_SheathRadius = new QLabel("Sheath Radius  ", this);
	m_pLabel_SheathRadius->setBuddy(m_pLineEdit_SheathRadius);
	m_pLabel_SheathRadius->setDisabled(true);

#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
	m_pLineEdit_RingThickness = new QLineEdit(this);
	m_pLineEdit_RingThickness->setFixedWidth(30);
	m_pLineEdit_RingThickness->setText(QString::number(m_pConfig->ringThickness));
	m_pLineEdit_RingThickness->setAlignment(Qt::AlignCenter);
	m_pLineEdit_RingThickness->setDisabled(true);
	m_pLabel_RingThickness = new QLabel("Ring Thickness", this);
	m_pLabel_RingThickness->setBuddy(m_pLineEdit_RingThickness);
	m_pLabel_RingThickness->setDisabled(true);
#endif

    // Create widgets for OCT color table
    m_pComboBox_OctColorTable = new QComboBox(this);
    m_pComboBox_OctColorTable->addItem("gray");
    m_pComboBox_OctColorTable->addItem("invgray");
    m_pComboBox_OctColorTable->addItem("sepia");
	m_pComboBox_OctColorTable->setCurrentIndex(m_pConfig->octColorTable);
    m_pComboBox_OctColorTable->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	m_pComboBox_OctColorTable->setDisabled(true);
    m_pLabel_OctColorTable = new QLabel("OCT Colortable", this);
    m_pLabel_OctColorTable->setBuddy(m_pComboBox_OctColorTable);
	m_pLabel_OctColorTable->setDisabled(true);

#ifdef OCT_FLIM
    // Create widgets for FLIM emission control
    m_pComboBox_EmissionChannel = new QComboBox(this);
    m_pComboBox_EmissionChannel->addItem("Ch 1");
    m_pComboBox_EmissionChannel->addItem("Ch 2");
    m_pComboBox_EmissionChannel->addItem("Ch 3");
    m_pComboBox_EmissionChannel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed); 
	m_pComboBox_EmissionChannel->setCurrentIndex(1);
	m_pComboBox_EmissionChannel->setDisabled(true);
    m_pLabel_EmissionChannel = new QLabel("Em Channel ", this);
    m_pLabel_EmissionChannel->setBuddy(m_pComboBox_EmissionChannel);
	m_pLabel_EmissionChannel->setDisabled(true);

    m_pCheckBox_HsvEnhancedMap = new QCheckBox(this);
    m_pCheckBox_HsvEnhancedMap->setText("HSV Enhanced Map");
	m_pCheckBox_HsvEnhancedMap->setChecked(true);
	m_pCheckBox_HsvEnhancedMap->setDisabled(true);

	ColorTable temp_ctable;
	m_pComboBox_LifetimeColorTable = new QComboBox(this);
	for (int i = 0; i < temp_ctable.m_cNameVector.size(); i++)
		m_pComboBox_LifetimeColorTable->addItem(temp_ctable.m_cNameVector.at(i));
	m_pComboBox_LifetimeColorTable->setCurrentIndex(m_pConfig->flimLifetimeColorTable);
	m_pComboBox_LifetimeColorTable->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	m_pComboBox_LifetimeColorTable->setDisabled(true);
	m_pComboBox_LifetimeColorTable->setFixedSize(1, 1);
	m_pLabel_LifetimeColorTable = new QLabel("Lifetime Colortable ", this);
	m_pLabel_LifetimeColorTable->setBuddy(m_pComboBox_LifetimeColorTable);
	m_pLabel_LifetimeColorTable->setDisabled(true);
	m_pLabel_LifetimeColorTable->setFixedSize(1, 1);

	m_pCheckBox_IntensityRatio = new QCheckBox(this);
	m_pCheckBox_IntensityRatio->setText("Show Intensity Ratio");
	m_pCheckBox_IntensityRatio->setDisabled(true);

	m_pCheckBox_Classification = new QCheckBox(this);
	m_pCheckBox_Classification->setText("Classification");
	m_pCheckBox_Classification->setDisabled(true);
#endif
	
#ifdef OCT_NIRF
	// Create widgets for NIRF offset control
	m_pLineEdit_NirfOffset = new QLineEdit(this);
	m_pLineEdit_NirfOffset->setFixedWidth(35);
	m_pLineEdit_NirfOffset->setText(QString::number(0));
	m_pLineEdit_NirfOffset->setAlignment(Qt::AlignCenter);
	m_pLineEdit_NirfOffset->setDisabled(true);
	
	m_pLabel_NirfOffset = new QLabel("NIRF Offset  ", this);
	m_pLabel_NirfOffset->setBuddy(m_pLineEdit_NirfOffset);
	m_pLabel_NirfOffset->setDisabled(true);

	m_pScrollBar_NirfOffset = new QScrollBar(this);
	m_pScrollBar_NirfOffset->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	m_pScrollBar_NirfOffset->setFocusPolicy(Qt::StrongFocus);
	m_pScrollBar_NirfOffset->setOrientation(Qt::Horizontal);
	m_pScrollBar_NirfOffset->setRange(-m_pConfig->nAlines + 1, m_pConfig->nAlines - 1);
	m_pScrollBar_NirfOffset->setSingleStep(1);
	m_pScrollBar_NirfOffset->setPageStep(10);
	m_pScrollBar_NirfOffset->setValue(0);
	m_pScrollBar_NirfOffset->setDisabled(true);
#endif

	// Set layout
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 0);
	pGridLayout_Visualization->addWidget(m_pLabel_SelectFrame, 0, 1);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 2);
	pGridLayout_Visualization->addWidget(m_pToggleButton_MeasureDistance, 0, 3, 1, 2);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 5);

	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
	pGridLayout_Visualization->addWidget(m_pSlider_SelectFrame, 1, 1, 1, 4);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 5);

#ifdef OCT_NIRF
#ifdef TWO_CHANNEL_NIRF
	//pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 2, 0);
	//pGridLayout_Visualization->addWidget(m_pPushButton_NirfCrossTalkCompensation, 2, 1, 1, 2);
#endif
	pGridLayout_Visualization->addWidget(m_pPushButton_NirfDistanceCompensation, 2, 3, 1, 2);
#else	
	//pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 2, 0, 1, 3);
#endif
	pGridLayout_Visualization->addWidget(m_pPushButton_SpectroscopicView, 2, 1, 1, 2);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 2, 5);

	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 3, 0);
	//pGridLayout_Visualization->addWidget(m_pToggleButton_FindPolishedSurfaces, 3, 1, 1, 2);
	pGridLayout_Visualization->addWidget(m_pToggleButton_AutoContour, 3, 1, 1, 2);
	pGridLayout_Visualization->addWidget(m_pPushButton_LongitudinalView, 3, 3, 1, 2);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 3, 5);

#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 4, 0, 1, 3);
	pGridLayout_Visualization->addWidget(m_pLabel_RingThickness, 4, 3);
	pGridLayout_Visualization->addWidget(m_pLineEdit_RingThickness, 4, 4);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 4, 5);
#endif

	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 5, 0);

	QHBoxLayout *pHBoxLayout_SheathRadius = new QHBoxLayout;
	pHBoxLayout_SheathRadius->setSpacing(3);
	pHBoxLayout_SheathRadius->addWidget(m_pLabel_SheathRadius);
	pHBoxLayout_SheathRadius->addWidget(m_pLineEdit_SheathRadius);
	pHBoxLayout_SheathRadius->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));

	pGridLayout_Visualization->addItem(pHBoxLayout_SheathRadius, 5, 1);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 5, 2);
	pGridLayout_Visualization->addWidget(m_pLabel_CircCenter, 5, 3);
	pGridLayout_Visualization->addWidget(m_pLineEdit_CircCenter, 5, 4);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 5, 5);

	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 6, 0);
	pGridLayout_Visualization->addWidget(m_pCheckBox_CircularizeImage, 6, 1);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 6, 2);
	pGridLayout_Visualization->addWidget(m_pLabel_CircRadius, 6, 3);
	pGridLayout_Visualization->addWidget(m_pLineEdit_CircRadius, 6, 4);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 6, 5);

	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 7, 0);
	pGridLayout_Visualization->addWidget(m_pCheckBox_ShowGuideLine, 7, 1);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 7, 2);
	pGridLayout_Visualization->addWidget(m_pLabel_OctColorTable, 7, 3);
	pGridLayout_Visualization->addWidget(m_pComboBox_OctColorTable, 7, 4);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 7, 5);

#ifdef OCT_FLIM
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 8, 0);
	pGridLayout_Visualization->addWidget(m_pCheckBox_HsvEnhancedMap, 8, 1);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 8, 2);
	pGridLayout_Visualization->addWidget(m_pLabel_EmissionChannel, 8, 3);
	pGridLayout_Visualization->addWidget(m_pComboBox_EmissionChannel, 8, 4);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 8, 5);

	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 9, 0);
	pGridLayout_Visualization->addWidget(m_pCheckBox_Classification, 9, 1);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 9, 2);

	pGridLayout_Visualization->addWidget(m_pCheckBox_IntensityRatio, 9, 3, 1, 2);
	//pGridLayout_Visualization->addWidget(m_pLabel_LifetimeColorTable, 9, 3);
	//pGridLayout_Visualization->addWidget(m_pComboBox_LifetimeColorTable, 9, 4);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 9, 5);
#endif

#ifdef OCT_NIRF
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 8, 0);
	QHBoxLayout *pHBoxLayout_Nirf = new QHBoxLayout;
	pHBoxLayout_Nirf->setSpacing(3);
	pHBoxLayout_Nirf->addWidget(m_pLabel_NirfOffset);
	pHBoxLayout_Nirf->addWidget(m_pLineEdit_NirfOffset);
	pHBoxLayout_Nirf->addWidget(new QLabel("  ", this));
	pHBoxLayout_Nirf->addWidget(m_pScrollBar_NirfOffset);
	pGridLayout_Visualization->addItem(pHBoxLayout_Nirf, 8, 1, 1, 4);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 8, 5);
#endif

    m_pGroupBox_Visualization->setLayout(pGridLayout_Visualization);
	m_pGroupBox_Visualization->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");

	// Connect signal and slot
	connect(m_pSlider_SelectFrame, SIGNAL(valueChanged(int)), this, SLOT(visualizeImage(int)));
	connect(m_pToggleButton_MeasureDistance, SIGNAL(toggled(bool)), this, SLOT(measureDistance(bool)));
#ifdef OCT_NIRF
	connect(m_pPushButton_NirfDistanceCompensation, SIGNAL(clicked(bool)), this, SLOT(createNirfDistCompDlg()));
#ifdef TWO_CHANNEL_NIRF
	//connect(m_pPushButton_NirfCrossTalkCompensation, SIGNAL(clicked(bool)), this, SLOT(createNirfCrossTalkCompDlg()));
#endif
#endif
	connect(m_pPushButton_LongitudinalView, SIGNAL(clicked(bool)), this, SLOT(createLongitudinalViewDlg()));
	connect(m_pPushButton_SpectroscopicView, SIGNAL(clicked(bool)), this, SLOT(createSpectroOCTDlg()));
	connect(m_pCheckBox_ShowGuideLine, SIGNAL(toggled(bool)), this, SLOT(showGuideLine(bool)));
	//connect(m_pToggleButton_FindPolishedSurfaces, SIGNAL(toggled(bool)), this, SLOT(findPolishedSurface(bool)));
	connect(m_pToggleButton_AutoContour, SIGNAL(toggled(bool)), this, SLOT(autoContouring(bool)));
	connect(m_pCheckBox_CircularizeImage, SIGNAL(toggled(bool)), this, SLOT(changeVisImage(bool)));
	connect(m_pLineEdit_CircCenter, SIGNAL(textEdited(const QString &)), this, SLOT(checkCircCenter(const QString &)));
	connect(m_pLineEdit_CircRadius, SIGNAL(textEdited(const QString &)), this, SLOT(checkCircRadius(const QString &)));
	connect(m_pLineEdit_SheathRadius, SIGNAL(textEdited(const QString &)), this, SLOT(checkSheathRadius(const QString &)));
#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
	connect(m_pLineEdit_RingThickness, SIGNAL(textEdited(const QString &)), this, SLOT(checkRingThickness(const QString &)));
#endif
	connect(m_pComboBox_OctColorTable, SIGNAL(currentIndexChanged(int)), this, SLOT(adjustOctContrast()));
#ifdef OCT_FLIM
	connect(m_pCheckBox_HsvEnhancedMap, SIGNAL(toggled(bool)), this, SLOT(enableHsvEnhancingMode(bool)));
	connect(m_pComboBox_EmissionChannel, SIGNAL(currentIndexChanged(int)), this, SLOT(changeFlimCh(int)));	
	connect(m_pComboBox_LifetimeColorTable, SIGNAL(currentIndexChanged(int)), this, SLOT(changeLifetimeColorTable(int)));
	connect(m_pCheckBox_IntensityRatio, SIGNAL(toggled(bool)), this, SLOT(enableIntensityRatio(bool)));
	connect(m_pCheckBox_Classification, SIGNAL(toggled(bool)), this, SLOT(enableClassification(bool)));
#endif
#ifdef OCT_NIRF
	connect(m_pLineEdit_NirfOffset, SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfOffset(const QString &)));
	connect(m_pScrollBar_NirfOffset, SIGNAL(valueChanged(int)), this, SLOT(adjustNirfOffset(int)));
#endif
}

void QResultTab::createEnFaceMapTab()
{    
    // En face map tab widgets
	m_pGroupBox_EnFace = new QGroupBox;
	m_pGroupBox_EnFace->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QGridLayout *pGridLayout_EnFace = new QGridLayout;
    pGridLayout_EnFace->setSpacing(0);

    uint8_t color[256 * 4];
    for (int i = 0; i < 256 * 4; i++)
        color[i] = 255 - i / 4;

    // Create widgets for OCT projection map
    m_pImageView_OctProjection = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), m_pConfig->nAlines, 1, m_pConfig->octDbGamma);
#ifdef OCT_NIRF
	m_pImageView_OctProjection->setMinimumHeight(150);
#endif
	m_pImageView_OctProjection->setHLineChangeCallback([&](int frame) { m_pSlider_SelectFrame->setValue(frame); });
	m_visOctProjection = np::Uint8Array2(m_pConfig->nAlines4, 1);

    m_pLineEdit_OctDbMax = new QLineEdit(this);
    m_pLineEdit_OctDbMax->setText(QString::number(m_pConfig->octDbRange.max));
	m_pLineEdit_OctDbMax->setAlignment(Qt::AlignCenter);
    m_pLineEdit_OctDbMax->setFixedWidth(30);
	m_pLineEdit_OctDbMax->setDisabled(true);
    m_pLineEdit_OctDbMin = new QLineEdit(this);
    m_pLineEdit_OctDbMin->setText(QString::number(m_pConfig->octDbRange.min));
	m_pLineEdit_OctDbMin->setAlignment(Qt::AlignCenter);
    m_pLineEdit_OctDbMin->setFixedWidth(30);
	m_pLineEdit_OctDbMin->setDisabled(true);

	m_pLabel_OctDbGamma = new QLabel(this);
	m_pLabel_OctDbGamma->setText(QChar(0xb3, 0x03));
	m_pLabel_OctDbGamma->setDisabled(true);
	m_pLineEdit_OctDbGamma = new QLineEdit(this);
	m_pLineEdit_OctDbGamma->setText(QString::number(m_pConfig->octDbGamma, 'f', 2));
	m_pLineEdit_OctDbGamma->setAlignment(Qt::AlignCenter);
	m_pLineEdit_OctDbGamma->setFixedWidth(30);
	m_pLineEdit_OctDbGamma->setDisabled(true);

    m_pImageView_ColorbarOctProjection = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), 4, 256, m_pConfig->octDbGamma);
	m_pImageView_ColorbarOctProjection->getRender()->setFixedWidth(15);
    m_pImageView_ColorbarOctProjection->drawImage(color);
    m_pImageView_ColorbarOctProjection->setFixedWidth(30);

    m_pLabel_OctProjection = new QLabel(this);
    m_pLabel_OctProjection->setText("OCT Maximum Projection Map");
	m_pLabel_OctProjection->setDisabled(true);

#ifdef OCT_FLIM
    // Create widgets for FLIM intensity map
    m_pImageView_IntensityMap = new QImageView(ColorTable::colortable(INTENSITY_COLORTABLE), m_pConfig->n4Alines, 1);
	m_pImageView_IntensityMap->setHLineChangeCallback([&](int frame) { m_pSlider_SelectFrame->setValue(frame); });
	m_pImageView_IntensityMap->getRender()->m_colorLine = 0x00ff00;

    m_pLineEdit_IntensityMax = new QLineEdit(this);
    m_pLineEdit_IntensityMax->setText(QString::number(m_pConfig->flimIntensityRange.max, 'f', 1));
	m_pLineEdit_IntensityMax->setAlignment(Qt::AlignCenter);
    m_pLineEdit_IntensityMax->setFixedWidth(30);
	m_pLineEdit_IntensityMax->setDisabled(true);
    m_pLineEdit_IntensityMin = new QLineEdit(this);
    m_pLineEdit_IntensityMin->setText(QString::number(m_pConfig->flimIntensityRange.min, 'f', 1));
	m_pLineEdit_IntensityMin->setAlignment(Qt::AlignCenter);
    m_pLineEdit_IntensityMin->setFixedWidth(30);
	m_pLineEdit_IntensityMin->setDisabled(true);

    m_pImageView_ColorbarIntensityMap = new QImageView(ColorTable::colortable(INTENSITY_COLORTABLE), 4, 256);
	m_pImageView_ColorbarIntensityMap->getRender()->setFixedWidth(15);
    m_pImageView_ColorbarIntensityMap->drawImage(color);
    m_pImageView_ColorbarIntensityMap->setFixedWidth(30);

    m_pLabel_IntensityMap = new QLabel(this);
    m_pLabel_IntensityMap->setText("FLIM Intensity Map");

    // Create widgets for FLIM lifetime map
	ColorTable temp_ctable;
    m_pImageView_LifetimeMap = new QImageView(ColorTable::colortable(m_pConfig->flimLifetimeColorTable), m_pConfig->n4Alines, 1, 1.0f, true);
	m_pImageView_LifetimeMap->setHLineChangeCallback([&](int frame) { m_pSlider_SelectFrame->setValue(frame); });
	m_pImageView_LifetimeMap->getRender()->m_colorLine = 0xffffff;

    m_pLineEdit_LifetimeMax = new QLineEdit(this);
    m_pLineEdit_LifetimeMax->setText(QString::number(m_pConfig->flimLifetimeRange.max, 'f', 1));
	m_pLineEdit_LifetimeMax->setAlignment(Qt::AlignCenter);
    m_pLineEdit_LifetimeMax->setFixedWidth(30);
	m_pLineEdit_LifetimeMax->setDisabled(true);
    m_pLineEdit_LifetimeMin = new QLineEdit(this);
    m_pLineEdit_LifetimeMin->setText(QString::number(m_pConfig->flimLifetimeRange.min, 'f', 1));
	m_pLineEdit_LifetimeMin->setAlignment(Qt::AlignCenter);
    m_pLineEdit_LifetimeMin->setFixedWidth(30);
	m_pLineEdit_LifetimeMin->setDisabled(true);

    m_pImageView_ColorbarLifetimeMap = new QImageView(ColorTable::colortable(m_pConfig->flimLifetimeColorTable), 4, 256);
	m_pImageView_ColorbarLifetimeMap->getRender()->setFixedWidth(15);
    m_pImageView_ColorbarLifetimeMap->drawImage(color);
    m_pImageView_ColorbarLifetimeMap->setFixedWidth(30);

    m_pLabel_LifetimeMap = new QLabel(this);
    m_pLabel_LifetimeMap->setText("FLIM Lifetime Map");
#endif

#ifdef OCT_NIRF
	// Create widgets for NIRF map
#ifndef TWO_CHANNEL_NIRF
	m_pImageView_NirfMap = new QImageView(ColorTable::colortable(NIRF_COLORTABLE1), m_pConfig->nAlines4, 1);
	m_pImageView_NirfMap->setMinimumHeight(150);
    m_pImageView_NirfMap->setHLineChangeCallback([&](int frame) { m_pSlider_SelectFrame->setValue(frame); });
    m_pImageView_NirfMap->setDoubleClickedMouseCallback([&]() { createNirfEmissionProfileDlg(); });
	m_pImageView_NirfMap->getRender()->m_colorLine = 0x00ff00;
#else
	m_pImageView_NirfMap1 = new QImageView(ColorTable::colortable(NIRF_COLORTABLE1), m_pConfig->nAlines4, 1);
	m_pImageView_NirfMap1->setMinimumHeight(150);
	m_pImageView_NirfMap1->setHLineChangeCallback([&](int frame) { m_pSlider_SelectFrame->setValue(frame); });
	m_pImageView_NirfMap1->setDoubleClickedMouseCallback([&]() { createNirfEmissionProfileDlg(0); });
	m_pImageView_NirfMap1->getRender()->m_colorLine = 0x00ff00;

	m_pImageView_NirfMap2 = new QImageView(ColorTable::colortable(NIRF_COLORTABLE2), m_pConfig->nAlines4, 1);
	m_pImageView_NirfMap2->setMinimumHeight(150);
	m_pImageView_NirfMap2->setHLineChangeCallback([&](int frame) { m_pSlider_SelectFrame->setValue(frame); });
	m_pImageView_NirfMap2->setDoubleClickedMouseCallback([&]() { createNirfEmissionProfileDlg(1); });
	m_pImageView_NirfMap2->getRender()->m_colorLine = 0x00ff00;
#endif

#ifndef TWO_CHANNEL_NIRF
	m_pLineEdit_NirfMax = new QLineEdit(this);
	m_pLineEdit_NirfMax->setText(QString::number(m_pConfig->nirfRange.max, 'f', 2));
	m_pLineEdit_NirfMax->setAlignment(Qt::AlignCenter);
    m_pLineEdit_NirfMax->setFixedWidth(30);
	m_pLineEdit_NirfMax->setDisabled(true);
	m_pLineEdit_NirfMin = new QLineEdit(this);
	m_pLineEdit_NirfMin->setText(QString::number(m_pConfig->nirfRange.min, 'f', 2));
	m_pLineEdit_NirfMin->setAlignment(Qt::AlignCenter);
	m_pLineEdit_NirfMin->setFixedWidth(30);
	m_pLineEdit_NirfMin->setDisabled(true);

	m_pImageView_ColorbarNirfMap = new QImageView(ColorTable::colortable(NIRF_COLORTABLE1), 4, 256);
	m_pImageView_ColorbarNirfMap->getRender()->setFixedWidth(15);
	m_pImageView_ColorbarNirfMap->drawImage(color);
	m_pImageView_ColorbarNirfMap->setFixedWidth(30);
#else
	for (int i = 0; i < 2; i++)
	{
		m_pLineEdit_NirfMax[i] = new QLineEdit(this);
		m_pLineEdit_NirfMax[i]->setFixedWidth(30);
		m_pLineEdit_NirfMax[i]->setText(QString::number(m_pConfig->nirfRange[i].max, 'f', 1));
		m_pLineEdit_NirfMax[i]->setAlignment(Qt::AlignCenter);
		m_pLineEdit_NirfMax[i]->setDisabled(true);
		m_pLineEdit_NirfMin[i] = new QLineEdit(this);
		m_pLineEdit_NirfMin[i]->setFixedWidth(30);
		m_pLineEdit_NirfMin[i]->setText(QString::number(m_pConfig->nirfRange[i].min, 'f', 1));
		m_pLineEdit_NirfMin[i]->setAlignment(Qt::AlignCenter);
		m_pLineEdit_NirfMin[i]->setDisabled(true);
	}
	
	m_pImageView_ColorbarNirfMap[0] = new QImageView(ColorTable::colortable(NIRF_COLORTABLE1), 4, 256);
	m_pImageView_ColorbarNirfMap[0]->getRender()->setFixedWidth(15);
	m_pImageView_ColorbarNirfMap[0]->drawImage(color);
	m_pImageView_ColorbarNirfMap[0]->setFixedWidth(30);

	m_pImageView_ColorbarNirfMap[1] = new QImageView(ColorTable::colortable(NIRF_COLORTABLE2), 4, 256);
	m_pImageView_ColorbarNirfMap[1]->getRender()->setFixedWidth(15);
	m_pImageView_ColorbarNirfMap[1]->drawImage(color);
	m_pImageView_ColorbarNirfMap[1]->setFixedWidth(30);
#endif
	
#ifndef TWO_CHANNEL_NIRF
	m_pLabel_NirfMap = new QLabel(this);
	m_pLabel_NirfMap->setText("NIRF Map");
	m_pLabel_NirfMap->setDisabled(true);
#else
	m_pLabel_NirfMap1 = new QLabel(this);
	m_pLabel_NirfMap1->setText("NIRF Map Ch1");
	m_pLabel_NirfMap1->setDisabled(true);

	m_pLabel_NirfMap2 = new QLabel(this);
	m_pLabel_NirfMap2->setText("NIRF Map Ch2");
	m_pLabel_NirfMap2->setDisabled(true);
#endif
#endif

    // Set layout
    pGridLayout_EnFace->addWidget(m_pLabel_OctProjection, 0, 0, 1, 3);
    pGridLayout_EnFace->addWidget(m_pImageView_OctProjection, 1, 0);
    pGridLayout_EnFace->addWidget(m_pImageView_ColorbarOctProjection, 1, 1);
    QVBoxLayout *pVBoxLayout_Colorbar1 = new QVBoxLayout;
	pVBoxLayout_Colorbar1->setSpacing(0);
    pVBoxLayout_Colorbar1->addWidget(m_pLineEdit_OctDbMax);
    pVBoxLayout_Colorbar1->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));

	QVBoxLayout *pVBoxLayout_Gamma = new QVBoxLayout;
	pVBoxLayout_Gamma->setSpacing(0);
	pVBoxLayout_Gamma->addWidget(m_pLabel_OctDbGamma);
	pVBoxLayout_Gamma->addWidget(m_pLineEdit_OctDbGamma);

	pVBoxLayout_Colorbar1->addItem(pVBoxLayout_Gamma);
	pVBoxLayout_Colorbar1->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));
    pVBoxLayout_Colorbar1->addWidget(m_pLineEdit_OctDbMin);
    pGridLayout_EnFace->addItem(pVBoxLayout_Colorbar1, 1, 2);

#ifdef OCT_FLIM
    pGridLayout_EnFace->addWidget(m_pLabel_IntensityMap, 2, 0, 1, 3);
    pGridLayout_EnFace->addWidget(m_pImageView_IntensityMap, 3, 0);
    pGridLayout_EnFace->addWidget(m_pImageView_ColorbarIntensityMap, 3, 1);
    QVBoxLayout *pVBoxLayout_Colorbar2 = new QVBoxLayout;
	pVBoxLayout_Colorbar2->setSpacing(0);
    pVBoxLayout_Colorbar2->addWidget(m_pLineEdit_IntensityMax);
    pVBoxLayout_Colorbar2->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));
    pVBoxLayout_Colorbar2->addWidget(m_pLineEdit_IntensityMin);
    pGridLayout_EnFace->addItem(pVBoxLayout_Colorbar2, 3, 2);

    pGridLayout_EnFace->addWidget(m_pLabel_LifetimeMap, 4, 0, 1, 3);
    pGridLayout_EnFace->addWidget(m_pImageView_LifetimeMap, 5, 0);
    pGridLayout_EnFace->addWidget(m_pImageView_ColorbarLifetimeMap, 5, 1);
    QVBoxLayout *pVBoxLayout_Colorbar3 = new QVBoxLayout;
	pVBoxLayout_Colorbar3->setSpacing(0);
    pVBoxLayout_Colorbar3->addWidget(m_pLineEdit_LifetimeMax);
    pVBoxLayout_Colorbar3->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));
    pVBoxLayout_Colorbar3->addWidget(m_pLineEdit_LifetimeMin);
    pGridLayout_EnFace->addItem(pVBoxLayout_Colorbar3, 5, 2);
#endif

#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	pGridLayout_EnFace->addWidget(m_pLabel_NirfMap, 2, 0, 1, 3);
	pGridLayout_EnFace->addWidget(m_pImageView_NirfMap, 3, 0);
#else
	pGridLayout_EnFace->addWidget(m_pLabel_NirfMap1, 2, 0, 1, 3);
	pGridLayout_EnFace->addWidget(m_pImageView_NirfMap1, 3, 0);
	
	pGridLayout_EnFace->addWidget(m_pLabel_NirfMap2, 4, 0, 1, 3);
	pGridLayout_EnFace->addWidget(m_pImageView_NirfMap2, 5, 0);
#endif
#ifndef TWO_CHANNEL_NIRF
	pGridLayout_EnFace->addWidget(m_pImageView_ColorbarNirfMap, 3, 1);
	QVBoxLayout *pVBoxLayout_Colorbar2 = new QVBoxLayout;
	pVBoxLayout_Colorbar2->setSpacing(0);
	pVBoxLayout_Colorbar2->addWidget(m_pLineEdit_NirfMax);
	pVBoxLayout_Colorbar2->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));
	pVBoxLayout_Colorbar2->addWidget(m_pLineEdit_NirfMin);
	pGridLayout_EnFace->addItem(pVBoxLayout_Colorbar2, 3, 2);
#else
	pGridLayout_EnFace->addWidget(m_pImageView_ColorbarNirfMap[0], 3, 1);
	QVBoxLayout *pVBoxLayout_Colorbar2 = new QVBoxLayout;
	pVBoxLayout_Colorbar2->setSpacing(0);
	pVBoxLayout_Colorbar2->addWidget(m_pLineEdit_NirfMax[0]);
	pVBoxLayout_Colorbar2->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));
	pVBoxLayout_Colorbar2->addWidget(m_pLineEdit_NirfMin[0]);
	pGridLayout_EnFace->addItem(pVBoxLayout_Colorbar2, 3, 2);

	pGridLayout_EnFace->addWidget(m_pImageView_ColorbarNirfMap[1], 5, 1);
	QVBoxLayout *pVBoxLayout_Colorbar3 = new QVBoxLayout;
	pVBoxLayout_Colorbar3->setSpacing(0);
	pVBoxLayout_Colorbar3->addWidget(m_pLineEdit_NirfMax[1]);
	pVBoxLayout_Colorbar3->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));
	pVBoxLayout_Colorbar3->addWidget(m_pLineEdit_NirfMin[1]);
	pGridLayout_EnFace->addItem(pVBoxLayout_Colorbar3, 5, 2);
#endif
#endif
	
    m_pGroupBox_EnFace->setLayout(pGridLayout_EnFace);
	m_pGroupBox_EnFace->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");

	// Connect signal and slot
	connect(this, SIGNAL(paintOctProjection(uint8_t*)), m_pImageView_OctProjection, SLOT(drawImage(uint8_t*)));
#ifdef OCT_FLIM
	connect(this, SIGNAL(paintIntensityMap(uint8_t*)), m_pImageView_IntensityMap, SLOT(drawImage(uint8_t*)));
	connect(this, SIGNAL(paintLifetimeMap(uint8_t*)), m_pImageView_LifetimeMap, SLOT(drawImage(uint8_t*)));
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	connect(this, SIGNAL(paintNirfMap(uint8_t*)), m_pImageView_NirfMap, SLOT(drawImage(uint8_t*)));
#else
	connect(this, SIGNAL(paintNirfMap1(uint8_t*)), m_pImageView_NirfMap1, SLOT(drawImage(uint8_t*)));
	connect(this, SIGNAL(paintNirfMap2(uint8_t*)), m_pImageView_NirfMap2, SLOT(drawImage(uint8_t*)));
#endif
#endif

	connect(m_pLineEdit_OctDbMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustOctContrast()));
	connect(m_pLineEdit_OctDbMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustOctContrast()));
	connect(m_pLineEdit_OctDbGamma, SIGNAL(textEdited(const QString &)), this, SLOT(adjustOctContrast()));
#ifdef OCT_FLIM
	connect(m_pLineEdit_IntensityMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustFlimContrast()));
	connect(m_pLineEdit_IntensityMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustFlimContrast()));
	connect(m_pLineEdit_LifetimeMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustFlimContrast()));
	connect(m_pLineEdit_LifetimeMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustFlimContrast()));
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	connect(m_pLineEdit_NirfMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast()));
	connect(m_pLineEdit_NirfMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast()));
#else
	connect(m_pLineEdit_NirfMax[0], SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast1()));
	connect(m_pLineEdit_NirfMin[0], SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast1()));
	connect(m_pLineEdit_NirfMax[1], SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast2()));
	connect(m_pLineEdit_NirfMin[1], SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast2()));
#endif
#endif
}


void QResultTab::changeDataSelection(int id)
{
	switch (id)
	{
	case IN_BUFFER_DATA:
		m_pCheckBox_UserDefinedAlines->setDisabled(true);
		if (m_pCheckBox_UserDefinedAlines->isChecked())
			m_pCheckBox_UserDefinedAlines->setChecked(false);
		break;
	case EXTERNAL_DATA:
		m_pCheckBox_UserDefinedAlines->setEnabled(true);
		break;
	}	
}

void QResultTab::createSaveResultDlg()
{
	if (m_pSaveResultDlg == nullptr)
	{
		m_pSaveResultDlg = new SaveResultDlg(this);
		connect(m_pSaveResultDlg, SIGNAL(finished(int)), this, SLOT(deleteSaveResultDlg()));
		m_pSaveResultDlg->show();

		m_pSaveResultDlg->m_defaultTransformation = Qt::SmoothTransformation;
	}
	m_pSaveResultDlg->raise();
	m_pSaveResultDlg->activateWindow();
}

void QResultTab::deleteSaveResultDlg()
{
	m_pSaveResultDlg->deleteLater();
	m_pSaveResultDlg = nullptr;
}

void QResultTab::createLongitudinalViewDlg()
{
	if (m_pLongitudinalViewDlg == nullptr)
	{
		m_pLongitudinalViewDlg = new LongitudinalViewDlg(this);
		connect(m_pLongitudinalViewDlg, SIGNAL(finished(int)), this, SLOT(deleteLongitudinalViewDlg()));
		m_pLongitudinalViewDlg->show();

		m_pImageView_RectImage->setVLineChangeCallback([&](int aline) { if (aline > m_pConfigTemp->nAlines / 2) aline -= m_pConfigTemp->nAlines / 2; m_pLongitudinalViewDlg->setCurrentAline(aline); });
		m_pImageView_RectImage->setVerticalLine(1, 0);
		m_pImageView_RectImage->getRender()->m_bDiametric = true;
		m_pImageView_RectImage->getRender()->update();

		m_pImageView_CircImage->setRLineChangeCallback([&](int aline) { if (aline > m_pConfigTemp->nAlines / 2) aline -= m_pConfigTemp->nAlines / 2; m_pLongitudinalViewDlg->setCurrentAline(aline); });
		m_pImageView_CircImage->setVerticalLine(1, 0); 
		m_pImageView_CircImage->getRender()->m_bRadial = true;
		m_pImageView_CircImage->getRender()->m_bDiametric = true;
		m_pImageView_CircImage->getRender()->m_rMax = m_pConfigTemp->nAlines;
		m_pImageView_CircImage->getRender()->update();
	}
	m_pLongitudinalViewDlg->raise();
	m_pLongitudinalViewDlg->activateWindow();

	m_pLongitudinalViewDlg->drawLongitudinalImage(0);

	m_pImageView_RectImage->setDoubleClickedMouseCallback([&]() {});
	m_pImageView_CircImage->setDoubleClickedMouseCallback([&]() {});

	m_pToggleButton_MeasureDistance->setChecked(false);
	m_pToggleButton_MeasureDistance->setDisabled(true);
}

void QResultTab::deleteLongitudinalViewDlg()
{
	m_pImageView_RectImage->setVerticalLine(0);
	m_pImageView_RectImage->getRender()->update();
	m_pImageView_RectImage->getRender()->m_bDiametric = false;

	m_pImageView_CircImage->setVerticalLine(0);
	m_pImageView_CircImage->getRender()->update();
	m_pImageView_CircImage->getRender()->m_bRadial = false;
	m_pImageView_CircImage->getRender()->m_bDiametric = false;

	m_pLongitudinalViewDlg->deleteLater();
	m_pLongitudinalViewDlg = nullptr;

#ifdef OCT_FLIM
	m_pImageView_RectImage->setDoubleClickedMouseCallback([&]() { createPulseReviewDlg(); });
	m_pImageView_CircImage->setDoubleClickedMouseCallback([&]() { createPulseReviewDlg(); });
#endif

	m_pToggleButton_MeasureDistance->setEnabled(true);
}

void QResultTab::createSpectroOCTDlg()
{
	if (m_pSpectroOCTDlg == nullptr)
	{
		m_pSpectroOCTDlg = new SpectroOCTDlg(this);
		connect(m_pSpectroOCTDlg, SIGNAL(finished(int)), this, SLOT(deleteSpectroOCTDlg()));
		m_pSpectroOCTDlg->show();

		m_pImageView_RectImage->setVLineChangeCallback([&](int aline) { m_pSpectroOCTDlg->setCurrentAline(aline); });
		m_pImageView_RectImage->setVerticalLine(1, 0);
		m_pImageView_RectImage->getRender()->update();

		m_pImageView_CircImage->setRLineChangeCallback([&](int aline) { m_pSpectroOCTDlg->setCurrentAline(aline); });
		m_pImageView_CircImage->setVerticalLine(1, 0);
		m_pImageView_CircImage->getRender()->m_bRadial = true;
		m_pImageView_CircImage->getRender()->m_rMax = m_pConfigTemp->nAlines;
		m_pImageView_CircImage->getRender()->update();
	}
	m_pSpectroOCTDlg->raise();
	m_pSpectroOCTDlg->activateWindow();

	m_pToggleButton_MeasureDistance->setChecked(false);
	m_pToggleButton_MeasureDistance->setDisabled(true);
}

void QResultTab::deleteSpectroOCTDlg()
{
	m_pImageView_RectImage->setVerticalLine(0);
	m_pImageView_RectImage->getRender()->update();

	m_pImageView_CircImage->setVerticalLine(0);
	m_pImageView_CircImage->getRender()->update();
	m_pImageView_CircImage->getRender()->m_bRadial = false;

	m_pSpectroOCTDlg->deleteLater();
	m_pSpectroOCTDlg = nullptr;

	m_pToggleButton_MeasureDistance->setEnabled(true);
}

#ifdef OCT_FLIM
void QResultTab::createPulseReviewDlg()
{
	if (m_pPulseReviewDlg == nullptr)
	{
		m_pPulseReviewDlg = new PulseReviewDlg(this);
		connect(m_pPulseReviewDlg, SIGNAL(finished(int)), this, SLOT(deletePulseReviewDlg()));
		m_pPulseReviewDlg->show();

		m_pImageView_RectImage->setVLineChangeCallback([&](int aline) { m_pPulseReviewDlg->setCurrentAline(aline / 4); });
		m_pImageView_RectImage->setVerticalLine(1, 0);
		m_pImageView_RectImage->getRender()->update();

		m_pImageView_CircImage->setRLineChangeCallback([&](int aline) { m_pPulseReviewDlg->setCurrentAline(aline / 4); });
		m_pImageView_CircImage->setVerticalLine(1, 0);
		m_pImageView_CircImage->getRender()->m_bRadial = true;
		m_pImageView_CircImage->getRender()->m_rMax = m_pFLIMpost->_resize.ny * 4;
		m_pImageView_CircImage->getRender()->update();
	}
	m_pPulseReviewDlg->raise();
	m_pPulseReviewDlg->activateWindow();

	m_pPushButton_LongitudinalView->setDisabled(true);
	m_pToggleButton_MeasureDistance->setChecked(false);
	m_pToggleButton_MeasureDistance->setDisabled(true);
}

void QResultTab::deletePulseReviewDlg()
{
	m_pImageView_RectImage->setVerticalLine(0);
	m_pImageView_RectImage->getRender()->update();

	m_pImageView_CircImage->setVerticalLine(0);
	m_pImageView_CircImage->getRender()->update();
	m_pImageView_CircImage->getRender()->m_bRadial = false;

	m_pPulseReviewDlg->deleteLater();
	m_pPulseReviewDlg = nullptr;

	m_pPushButton_LongitudinalView->setEnabled(true);
	m_pToggleButton_MeasureDistance->setEnabled(true);
}
#endif

#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
void QResultTab::createNirfEmissionProfileDlg()
#else
void QResultTab::createNirfEmissionProfileDlg(int ch)
#endif
{
    if (m_pNirfEmissionProfileDlg == nullptr)
    {
        m_pNirfEmissionProfileDlg = new NirfEmissionProfileDlg(false, this);
        connect(m_pNirfEmissionProfileDlg, SIGNAL(finished(int)), this, SLOT(deleteNirfEmissionProfileDlg()));
        m_pNirfEmissionProfileDlg->show();
    }
#ifdef TWO_CHANNEL_NIRF
	if (ch == 0)
		m_pNirfEmissionProfileDlg->getScope()->resetAxis({ 0, (double)m_nirfMap1.size(0) }, { min(m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[1].min), max(m_pConfig->nirfRange[0].max, m_pConfig->nirfRange[1].max) });
	else
		m_pNirfEmissionProfileDlg->getScope()->resetAxis({ 0, (double)m_nirfMap2.size(0) }, { min(m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[1].min), max(m_pConfig->nirfRange[0].max, m_pConfig->nirfRange[1].max) });
#endif
    m_pNirfEmissionProfileDlg->raise();
    m_pNirfEmissionProfileDlg->activateWindow();
	
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
}

void QResultTab::deleteNirfEmissionProfileDlg()
{
    m_pNirfEmissionProfileDlg->deleteLater();
    m_pNirfEmissionProfileDlg = nullptr;

	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());	
}

void QResultTab::createNirfDistCompDlg()
{
	if (m_pNirfDistCompDlg == nullptr)
	{
		m_pNirfDistCompDlg = new NirfDistCompDlg(this);
		connect(m_pNirfDistCompDlg, SIGNAL(finished(int)), this, SLOT(deleteNirfDistCompDlg()));
		m_pNirfDistCompDlg->show();        

        visualizeEnFaceMap(true);
        visualizeImage(m_pSlider_SelectFrame->value());
	}
	m_pNirfDistCompDlg->raise();
	m_pNirfDistCompDlg->activateWindow();
}

void QResultTab::deleteNirfDistCompDlg()
{
	m_pNirfDistCompDlg->deleteLater();
	m_pNirfDistCompDlg = nullptr;

    m_pImageView_RectImage->setContour(0, nullptr);
    m_pImageView_RectImage->setHorizontalLine(0);
	m_pImageView_RectImage->setHorizontalLineColor(0);

	m_pImageView_CircImage->setContour(0, nullptr);

    visualizeEnFaceMap(true);
    visualizeImage(m_pSlider_SelectFrame->value());
}

#ifdef TWO_CHANNEL_NIRF
void QResultTab::createNirfCrossTalkCompDlg()
{
	if (m_pNirfCrossTalkCompDlg == nullptr)
	{
		m_pNirfCrossTalkCompDlg = new NirfCrossTalkCompDlg(this);
		connect(m_pNirfCrossTalkCompDlg, SIGNAL(finished(int)), this, SLOT(deleteNirfCrossTalkCompDlg()));
		m_pNirfCrossTalkCompDlg->show();

		visualizeEnFaceMap(true);
		visualizeImage(m_pSlider_SelectFrame->value());
	}
	m_pNirfCrossTalkCompDlg->raise();
	m_pNirfCrossTalkCompDlg->activateWindow();
}

void QResultTab::deleteNirfCrossTalkCompDlg()
{
	m_pNirfCrossTalkCompDlg->deleteLater();
	m_pNirfCrossTalkCompDlg = nullptr;

	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
}
#endif
#endif

void QResultTab::enableUserDefinedAlines(bool checked)
{ 
	m_pLineEdit_UserDefinedAlines->setEnabled(checked); 
}

void QResultTab::enableDiscomValue(bool checked)
{
	m_pLineEdit_DiscomValue->setEnabled(checked);
	if (checked)
		if (!m_pConfigTemp)
			m_pLineEdit_DiscomValue->setText(QString::number(m_pConfig->octDiscomVal));
		else
			m_pLineEdit_DiscomValue->setText(QString::number(m_pConfigTemp->octDiscomVal));
	else
		if (m_pConfigTemp)
			m_pLineEdit_DiscomValue->setText(QString::number(m_pConfigTemp->octDiscomVal));
}

void QResultTab::visualizeImage(int frame)
{
	if (m_vectorOctImage.size() != 0)
	{
		IppiSize roi_oct = { m_pImgObjRectImage->getHeight(), m_pImgObjRectImage->getWidth() };

		// OCT Visualization
		np::Uint8Array2 scale_temp(roi_oct.width, roi_oct.height);
		ippiScale_32f8u_C1R(m_vectorOctImage.at(frame), roi_oct.width * sizeof(float),
			scale_temp.raw_ptr(), roi_oct.width * sizeof(uint8_t), roi_oct, m_pConfig->octDbRange.min, m_pConfig->octDbRange.max);
//#if defined(OCT_VERTICAL_MIRRORING)
//		ippiMirror_8u_C1IR(scale_temp.raw_ptr(), sizeof(uint8_t) * roi_oct.width, roi_oct, ippAxsVertical);
//#endif
		ippiTranspose_8u_C1R(scale_temp.raw_ptr(), roi_oct.width * sizeof(uint8_t), m_pImgObjRectImage->arr.raw_ptr(), roi_oct.height * sizeof(uint8_t), roi_oct);
#ifdef GALVANO_MIRROR
		if (m_pConfig->galvoHorizontalShift)
		{
            int roi_oct_height_non4 = m_pImageView_RectImage->getRender()->m_pImage->width();
			for (int i = 0; i < roi_oct.width; i++)
			{
                uint8_t* pImg = m_pImgObjRectImage->arr.raw_ptr() + i * roi_oct.height;
                std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_oct_height_non4);
			}
		}
#endif
		(*m_pMedfiltRect)(m_pImgObjRectImage->arr.raw_ptr());
		
#ifdef OCT_FLIM		
		// FLIM Visualization		
		uint8_t* rectIntensity = &m_pImgObjIntensityMap->arr(0, m_pImgObjIntensityMap->arr.size(1) - 1 - frame);
		uint8_t* rectLifetime = &m_pImgObjLifetimeMap->arr(0, m_pImgObjLifetimeMap->arr.size(1) - 1 - frame);
		
		for (int i = 0; i < m_pConfig->ringThickness; i++)
		{
			memcpy(&m_pImgObjIntensity->arr(0, i), rectIntensity, sizeof(uint8_t) * m_pImgObjIntensityMap->arr.size(0));
			memcpy(&m_pImgObjLifetime->arr(0, i), rectLifetime, sizeof(uint8_t) * m_pImgObjLifetimeMap->arr.size(0));
		}
		emit makeRgb(m_pImgObjRectImage, m_pImgObjCircImage, m_pImgObjIntensity, m_pImgObjLifetime);
		
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
		// NIRF Visualization
#ifndef TWO_CHANNEL_NIRF
		uint8_t* rectNirf = &m_pImgObjNirfMap->arr(0, m_pImgObjNirfMap->arr.size(1) - 1 - frame);

		for (int i = 0; i < m_pConfig->ringThickness; i++)
			memcpy(&m_pImgObjNirf->arr(0, i), rectNirf, sizeof(uint8_t) * m_pImgObjNirfMap->arr.size(0));
#else
		uint8_t* rectNirf1 = &m_pImgObjNirfMap1->arr(0, m_pImgObjNirfMap1->arr.size(1) - 1 - frame);
		uint8_t* rectNirf2 = &m_pImgObjNirfMap2->arr(0, m_pImgObjNirfMap2->arr.size(1) - 1 - frame);

		for (int i = 0; i < m_pConfig->ringThickness; i++)
			memcpy(&m_pImgObjNirf1->arr(0, i), rectNirf1, sizeof(uint8_t) * m_pImgObjNirfMap1->arr.size(0));
		for (int i = 0; i < m_pConfig->ringThickness; i++)
			memcpy(&m_pImgObjNirf2->arr(0, i), rectNirf2, sizeof(uint8_t) * m_pImgObjNirfMap2->arr.size(0));
		
#ifdef CH_DIVIDING_LINE
		np::Uint8Array boundary(m_pImgObjNirfMap1->arr.size(0));
		ippsSet_8u(255, boundary.raw_ptr(), m_pImgObjNirfMap1->arr.size(0));

		memcpy(&m_pImgObjNirf1->arr(0, 0), boundary.raw_ptr(), sizeof(uint8_t) * m_pImgObjNirfMap1->arr.size(0));
		memcpy(&m_pImgObjNirf1->arr(0, m_pConfig->ringThickness - 1), boundary.raw_ptr(), sizeof(uint8_t) * m_pImgObjNirfMap1->arr.size(0));
		memcpy(&m_pImgObjNirf2->arr(0, m_pConfig->ringThickness - 1), boundary.raw_ptr(), sizeof(uint8_t) * m_pImgObjNirfMap2->arr.size(0));
#endif
#endif

        if (m_pNirfEmissionProfileDlg)
        {
#ifndef TWO_CHANNEL_NIRF
            np::FloatArray nirf_emssion(m_nirfMap.size(0));
            memcpy(nirf_emssion.raw_ptr(), &m_nirfMap0(0, frame), sizeof(float) * nirf_emssion.length());

#ifdef GALVANO_MIRROR
            if (m_pConfig->galvoHorizontalShift)
                std::rotate(nirf_emssion.raw_ptr(), nirf_emssion.raw_ptr() + m_pConfig->galvoHorizontalShift, nirf_emssion.raw_ptr() + nirf_emssion.length());
#endif
            m_pNirfEmissionProfileDlg->drawData(nirf_emssion.raw_ptr());
#else
			np::FloatArray nirf_emssion1(m_nirfMap1.size(0));
			memcpy(nirf_emssion1.raw_ptr(), &m_nirfMap1_0(0, frame), sizeof(float) * nirf_emssion1.length());
			np::FloatArray nirf_emssion2(m_nirfMap2.size(0));
			memcpy(nirf_emssion2.raw_ptr(), &m_nirfMap2_0(0, frame), sizeof(float) * nirf_emssion2.length());

#ifdef GALVANO_MIRROR
			if (m_pConfig->galvoHorizontalShift)
			{
				std::rotate(nirf_emssion1.raw_ptr(), nirf_emssion1.raw_ptr() + m_pConfig->galvoHorizontalShift, nirf_emssion1.raw_ptr() + nirf_emssion1.length());
				std::rotate(nirf_emssion2.raw_ptr(), nirf_emssion2.raw_ptr() + m_pConfig->galvoHorizontalShift, nirf_emssion2.raw_ptr() + nirf_emssion2.length());
			}
#endif
			m_pNirfEmissionProfileDlg->drawData(nirf_emssion1.raw_ptr(), nirf_emssion2.raw_ptr());
#endif
        }
		
        if (m_pNirfDistCompDlg)
        {
			if (m_pNirfDistCompDlg->isShowLumContour())
			{
#ifndef TWO_CHANNEL_NIRF
				np::Uint16Array contour(m_nirfMap.size(0));
#else
				np::Uint16Array contour(m_nirfMap1.size(0));
#endif
				if (m_pNirfDistCompDlg->distMap.length() > 0)
				{
					memcpy(contour, &m_pNirfDistCompDlg->distMap(0, frame), sizeof(uint16_t) * contour.length());
#ifdef GALVANO_MIRROR
					if (m_pConfig->galvoHorizontalShift)
						std::rotate(contour.raw_ptr(), contour.raw_ptr() + m_pConfig->galvoHorizontalShift, contour.raw_ptr() + contour.length());
#endif
					ippsAddC_16u_ISfs((Ipp16u)m_pConfig->nirfLumContourOffset, contour.raw_ptr(), contour.length(), 0);	
					if (!m_pCheckBox_CircularizeImage->isChecked())
					{
						if (!m_pNirfEmissionProfileDlg)
							m_pImageView_RectImage->setContour(contour.length(), contour.raw_ptr());
						else
							m_pImageView_RectImage->setContour(contour.length(), contour.raw_ptr(), m_pNirfEmissionProfileDlg->getScope()->getRender()->m_pSelectedRegion);
						m_pImageView_RectImage->setHorizontalLine(2, m_pConfig->nirfOuterSheathPos, m_pConfig->nirfOuterSheathPos);
						m_pImageView_RectImage->setHorizontalLineColor(2, 0xff0000, 0xff0000);
					}
					else
					{
						int center = m_pConfig->circCenter; // (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? m_pConfig->circCenter :
							//(m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3) + m_polishedSurface(frame) - m_pConfig->ballRadius;
						if (!m_pNirfEmissionProfileDlg)
							m_pImageView_CircImage->setContour(contour.length(), contour.raw_ptr());
						else
							m_pImageView_CircImage->setContour(contour.length(), contour.raw_ptr(), m_pNirfEmissionProfileDlg->getScope()->getRender()->m_pSelectedRegion);
						m_pImageView_CircImage->getRender()->m_contour_offset = -center;
					}
				}
			}
			else
			{
				if (!m_pCheckBox_CircularizeImage->isChecked())
				{
					m_pImageView_RectImage->setContour(0, nullptr);
					m_pImageView_RectImage->setHorizontalLine(0);
					m_pImageView_RectImage->setHorizontalLineColor(0);
				}
				else
					m_pImageView_CircImage->setContour(0, nullptr);
			}
        }

#ifndef TWO_CHANNEL_NIRF
		emit makeRgb(m_pImgObjRectImage, m_pImgObjCircImage, m_pImgObjNirf);
#else
		emit makeRgb(m_pImgObjRectImage, m_pImgObjCircImage, m_pImgObjNirf1, m_pImgObjNirf2);
#endif
#else
		//int center = (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? m_pConfig->circCenter :
		//	(m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3) + m_polishedSurface(m_pSlider_SelectFrame->value()) - m_pConfig->ballRadius;
		int center = m_pConfig->circCenter;			
		if (m_pCheckBox_ShowGuideLine->isChecked())
		{
			//int polished_surface = (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? center - m_pConfig->circCenter : center + m_pConfig->ballRadius;
			int polished_surface = center - m_pConfig->circCenter;

			if (!m_pCheckBox_CircularizeImage->isChecked())
			{
				m_pImageView_RectImage->setHorizontalLine(6, m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3, center, center + m_pConfig->sheathRadius,
					polished_surface, center + m_pConfigTemp->circRadius, m_pConfigTemp->n2ScansFFT / 2 + m_pConfigTemp->nScans / 3);
				m_pImageView_RectImage->setHorizontalLineColor(6, 0xff0000, 0x00ff00, 0xffff00, 0xffff00, 0x00ff00, 0xff0000);
			}
			else
				//m_pImageView_CircImage->setCircle(2, m_pConfig->sheathRadius, (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? 1 : m_pConfig->ballRadius);
				m_pImageView_CircImage->setCircle(2, m_pConfig->sheathRadius, 1);
		}

		if (!m_pCheckBox_CircularizeImage->isChecked())
		{
			if (m_pImageView_RectImage->isEnabled()) emit paintRectImage(m_pImgObjRectImage->qindeximg.bits());
		}
		else
		{
#ifndef CUDA_ENABLED
			(*m_pCirc)(m_pImgObjRectImage->arr, m_pImgObjCircImage->arr.raw_ptr(), "vertical", center);
#else
			(*m_pCirc)(m_pImgObjRectImage->arr.raw_ptr(), m_pImgObjCircImage->arr.raw_ptr(), center);
#endif
			if (m_pImageView_CircImage->isEnabled()) emit paintCircImage(m_pImgObjCircImage->qindeximg.bits());
		}
#endif
#endif
		if (m_pToggleButton_AutoContour->isChecked())
		{
			if (m_contourMap(0, frame) == 0)
				m_pToggleButton_AutoContour->setChecked(false);
			else
			{
				if (m_pCheckBox_CircularizeImage->isChecked())
					m_pImageView_CircImage->setContour(m_pConfig->nAlines, &m_contourMap(0, getCurrentFrame()));
				else
				{
					np::Uint16Array contour_16u1(m_pConfig->nAlines);
					ippsAddC_16u_Sfs(&m_contourMap(0, getCurrentFrame()), m_pConfig->circCenter, contour_16u1, m_pConfig->nAlines, 0);
					m_pImageView_RectImage->setContour(m_pConfig->nAlines, contour_16u1);
				}
			}
		}

		m_pImageView_OctProjection->setHorizontalLine(1, m_visOctProjection.size(1) - frame);
		m_pImageView_OctProjection->setHorizontalLineColor(1, 0xff0000);
		m_pImageView_OctProjection->getRender()->update();
#ifdef OCT_FLIM
		m_pImageView_IntensityMap->setHorizontalLine(1, m_pImgObjIntensityMap->arr.size(1) - frame);
		m_pImageView_IntensityMap->setHorizontalLineColor(1, 0x00ff00);
		m_pImageView_IntensityMap->getRender()->update();
		m_pImageView_LifetimeMap->setHorizontalLine(1, m_pImgObjLifetimeMap->arr.size(1) - frame);
		m_pImageView_LifetimeMap->setHorizontalLineColor(1, 0xffffff);
		m_pImageView_LifetimeMap->getRender()->update();
#endif	
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
		m_pImageView_NirfMap->setHorizontalLine(1, m_pImgObjNirfMap->arr.size(1) - frame);
		m_pImageView_NirfMap->setHorizontalLineColor(1, 0x00ff00);
		m_pImageView_NirfMap->getRender()->update();
#else
		m_pImageView_NirfMap1->setHorizontalLine(1, m_pImgObjNirfMap1->arr.size(1) - frame);
		m_pImageView_NirfMap1->setHorizontalLineColor(1, 0x00ff00);
		m_pImageView_NirfMap1->getRender()->update();
		m_pImageView_NirfMap2->setHorizontalLine(1, m_pImgObjNirfMap2->arr.size(1) - frame);
		m_pImageView_NirfMap2->setHorizontalLineColor(1, 0x00ff00);
		m_pImageView_NirfMap2->getRender()->update();
#endif
#endif
		if (m_pLongitudinalViewDlg)
		{
			m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
			m_pLongitudinalViewDlg->getImageView()->setVerticalLine(1, frame);
			m_pLongitudinalViewDlg->getImageView()->getRender()->update();
		}

#ifdef OCT_NIRF
		if (m_pNirfDistCompDlg)	m_pNirfDistCompDlg->updateCorrelation(m_pSlider_SelectFrame->value());		
#endif

#ifdef OCT_FLIM
		if (m_pPulseReviewDlg)
			m_pPulseReviewDlg->drawPulse(m_pPulseReviewDlg->getCurrentAline());
#endif

		QString str; str.sprintf("Current Frame : %3d / %3d", frame + 1, (int)m_vectorOctImage.size());
		m_pLabel_SelectFrame->setText(str);

		if (m_pSpectroOCTDlg)
		{
			if (m_pSpectroOCTDlg->isAutoExtraction())
				m_pSpectroOCTDlg->spectrumExtract();
			else
				m_pSpectroOCTDlg->drawSpectroImage(m_pSpectroOCTDlg->getCurrentAline());
		}
	}
}

#ifdef OCT_FLIM
void QResultTab::constructRgbImage(ImageObject *pRectObj, ImageObject *pCircObj, ImageObject *pIntObj, ImageObject *pLftObj)
{	
	int center = (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? m_pConfig->circCenter :
		(m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3) + m_polishedSurface(m_pSlider_SelectFrame->value()) - m_pConfig->ballRadius;
	if (m_pCheckBox_ShowGuideLine->isChecked())
	{
		int polished_surface = (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? center - m_pConfig->circCenter : center + m_pConfig->ballRadius;

		m_pImageView_RectImage->setHorizontalLine(7, m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3, center, center + m_pConfig->sheathRadius,
			polished_surface, center + m_pConfigTemp->circRadius, center + m_pConfigTemp->circRadius - m_pConfig->ringThickness, m_pConfigTemp->n2ScansFFT / 2 + m_pConfigTemp->nScans / 3);
		m_pImageView_RectImage->setHorizontalLineColor(7, 0xff0000, 0x00ff00, 0xffff00, 0xff00ff, 0x00ff00, 0x00ff00, 0xff0000);
		m_pImageView_CircImage->setCircle(2, m_pConfig->sheathRadius, (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? 1 : m_pConfig->ballRadius);
	}
	
	// Convert RGB
	pRectObj->convertRgb();
	pIntObj->convertScaledRgb();
	pLftObj->convertScaledRgb();	

	// HSV Enhancing  or  Classification
	ImageObject hsvObj(pLftObj->getWidth(), 1, pLftObj->getColorTable());
	if (m_pCheckBox_HsvEnhancedMap->isChecked() || m_pCheckBox_Classification->isChecked())
	{
		memcpy(hsvObj.qrgbimg.bits(),
			m_pImgObjHsvEnhancedMap->qrgbimg.bits() + (m_pImgObjHsvEnhancedMap->qrgbimg.height() - m_pSlider_SelectFrame->value() - 1) * 3 * m_pImgObjHsvEnhancedMap->qrgbimg.width(),
			3 * m_pImgObjHsvEnhancedMap->qrgbimg.width());
		hsvObj.scaledRgb4();
	}

	// Rect View
	if (!m_pCheckBox_CircularizeImage->isChecked())
	{
		if (!m_pCheckBox_HsvEnhancedMap->isChecked() && !m_pCheckBox_Classification->isChecked() && !m_pCheckBox_IntensityRatio->isChecked())
		{
			// Paste FLIM color ring to RGB rect image
			for (int i = 0; i < m_pConfig->ringThickness; i++)
			{
				memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (pRectObj->arr.size(1) - 2 * m_pConfig->ringThickness + i), pIntObj->qrgbimg.bits(), 4 * pIntObj->qrgbimg.width());
				memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (pRectObj->arr.size(1) - 1 * m_pConfig->ringThickness + i), pLftObj->qrgbimg.bits(), 4 * pLftObj->qrgbimg.width());
			}
		}
		else		
			// Paste FLIM color ring to RGB rect image
			if (m_pCheckBox_IntensityRatio->isChecked())
				for (int i = 0; i < m_pConfig->ringThickness; i++)
					memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (pRectObj->arr.size(1) - i - 1), pIntObj->qrgbimg.bits(), 4 * pIntObj->qrgbimg.width());
			else
				for (int i = 0; i < m_pConfig->ringThickness; i++)
					memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (pRectObj->arr.size(1) - i - 1), hsvObj.qrgbimg.bits(), hsvObj.qrgbimg.byteCount());

		// Draw image
        if (m_pImageView_RectImage->isEnabled()) emit paintRectImage(pRectObj->qrgbimg.bits());
	}
	// Circ View
	else
	{
		if (!m_pCheckBox_HsvEnhancedMap->isChecked() && !m_pCheckBox_Classification->isChecked() && !m_pCheckBox_IntensityRatio->isChecked())
		{
			// Paste FLIM color ring to RGB rect image
			for (int i = 0; i < m_pConfig->ringThickness; i++)
			{
				memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (center + m_pConfigTemp->circRadius - 2 * m_pConfig->ringThickness + i), pIntObj->qrgbimg.bits(), 4 * pIntObj->qrgbimg.width());
				memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (center + m_pConfigTemp->circRadius - 1 * m_pConfig->ringThickness + i), pLftObj->qrgbimg.bits(), 4 * pLftObj->qrgbimg.width());
			}
			//memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (center + m_pConfigTemp->circRadius - 2 * m_pConfig->ringThickness), pIntObj->qrgbimg.bits(), 4 * pIntObj->qrgbimg.width());
			//memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (center + m_pConfigTemp->circRadius - 1 * m_pConfig->ringThickness), pLftObj->qrgbimg.bits(), 4 * pLftObj->qrgbimg.width());
		}
		else
			// Paste FLIM color ring to RGB rect image
			if (m_pCheckBox_IntensityRatio->isChecked())
				for (int i = 0; i < m_pConfig->ringThickness; i++)
					memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (center + m_pConfigTemp->circRadius - i), pIntObj->qrgbimg.bits(), 4 * pIntObj->qrgbimg.width());
			else
				for (int i = 0; i < m_pConfig->ringThickness; i++)
					memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (center + m_pConfigTemp->circRadius - i), hsvObj.qrgbimg.bits(), hsvObj.qrgbimg.byteCount());
		
				
		np::Uint8Array2 rect_temp(pRectObj->qrgbimg.bits(), 3 * pRectObj->arr.size(0), pRectObj->arr.size(1));
#ifndef CUDA_ENABLED
		(*m_pCirc)(rect_temp, pCircObj->qrgbimg.bits(), "vertical", "rgb", center);
#else
		(*m_pCirc)(rect_temp.raw_ptr(), pCircObj->qrgbimg.bits(), "rgb", center);
#endif
		
		// Draw image        
        if (m_pImageView_CircImage->isEnabled()) emit paintCircImage(pCircObj->qrgbimg.bits());
	}
}
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
void QResultTab::constructRgbImage(ImageObject *pRectObj, ImageObject *pCircObj, ImageObject *pNirfObj)
#else
void QResultTab::constructRgbImage(ImageObject *pRectObj, ImageObject *pCircObj, ImageObject *pNirfObj1, ImageObject *pNirfObj2)
#endif
{	
	int center = m_pConfig->circCenter; // (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? m_pConfig->circCenter :
		//(m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3) + m_polishedSurface(m_pSlider_SelectFrame->value()) - m_pConfig->ballRadius;
	if (m_pCheckBox_ShowGuideLine->isChecked())
	{
		int polished_surface = center - m_pConfig->circCenter; //  (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? center - m_pConfig->circCenter : center + m_pConfig->ballRadius;

		m_pImageView_RectImage->setHorizontalLine(7, m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3, center, center + m_pConfig->sheathRadius,
			polished_surface, center + m_pConfigTemp->circRadius, center + m_pConfigTemp->circRadius - m_pConfig->ringThickness, m_pConfigTemp->n2ScansFFT / 2 + m_pConfigTemp->nScans / 3);
		m_pImageView_RectImage->setHorizontalLineColor(7, 0xff0000, 0x00ff00, 0xffff00, 0xff00ff, 0x00ff00, 0x00ff00, 0xff0000);
		m_pImageView_CircImage->setCircle(2, m_pConfig->sheathRadius, 1); // (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? 1 : m_pConfig->ballRadius);
	}

	// Convert RGB
	pRectObj->convertRgb();
#ifndef TWO_CHANNEL_NIRF
	pNirfObj->convertNonScaledRgb();
#else
	pNirfObj1->convertNonScaledRgb();
	pNirfObj2->convertNonScaledRgb();
#endif

	// Rect View
	if (!m_pCheckBox_CircularizeImage->isChecked())
	{
		// Paste NIRF color ring to RGB rect image
#ifndef TWO_CHANNEL_NIRF
        memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (pRectObj->arr.size(1) - 1 * m_pConfig->ringThickness), pNirfObj->qrgbimg.bits(), pNirfObj->qrgbimg.byteCount());
#else
		memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (pRectObj->arr.size(1) - 2 * m_pConfig->ringThickness), pNirfObj1->qrgbimg.bits(), pNirfObj1->qrgbimg.byteCount());
		memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (pRectObj->arr.size(1) - 1 * m_pConfig->ringThickness), pNirfObj2->qrgbimg.bits(), pNirfObj2->qrgbimg.byteCount());
#endif
	
		// Draw image
        if (m_pImageView_RectImage->isEnabled()) emit paintRectImage(pRectObj->qrgbimg.bits());
	}
	// Circ View
	else
	{		
		// Paste NIRF color ring to RGB rect image
#ifndef TWO_CHANNEL_NIRF
		memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (center + m_pConfigTemp->circRadius - 1 * m_pConfig->ringThickness), pNirfObj->qrgbimg.bits(), pNirfObj->qrgbimg.byteCount());
#else
		memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (center + m_pConfigTemp->circRadius - 2 * m_pConfig->ringThickness), pNirfObj1->qrgbimg.bits(), pNirfObj1->qrgbimg.byteCount());
		memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (center + m_pConfigTemp->circRadius - 1 * m_pConfig->ringThickness), pNirfObj2->qrgbimg.bits(), pNirfObj2->qrgbimg.byteCount());
#endif

		// Circularize
		np::Uint8Array2 rect_temp(pRectObj->qrgbimg.bits(), 3 * pRectObj->arr.size(0), pRectObj->arr.size(1));
#ifndef CUDA_ENABLED
		(*m_pCirc)(rect_temp, pCircObj->qrgbimg.bits(), "vertical", "rgb", center);
#else
		(*m_pCirc)(rect_temp.raw_ptr(), pCircObj->qrgbimg.bits(), "rgb", center);
#endif

		// Draw image        
		if (m_pImageView_CircImage->isEnabled()) emit paintCircImage(pCircObj->qrgbimg.bits());
	}
}
#endif	

void QResultTab::visualizeEnFaceMap(bool scaling)
{
	if (m_octProjection.size(0) != 0)
	{
		if (scaling)
		{
			IppiSize roi_proj = { m_octProjection.size(0), m_octProjection.size(1) };

			// Scaling OCT projection
			ippiScale_32f8u_C1R(m_octProjection, sizeof(float) * roi_proj.width, m_visOctProjection, sizeof(uint8_t) * roi_proj.width,
				roi_proj, m_pConfig->octDbRange.min, m_pConfig->octDbRange.max);
			ippiMirror_8u_C1IR(m_visOctProjection, sizeof(uint8_t) * roi_proj.width, roi_proj, ippAxsHorizontal);
#ifdef GALVANO_MIRROR
			if (m_pConfig->galvoHorizontalShift)
			{
				int roi_proj_width_non4 = m_pImageView_RectImage->getRender()->m_pImage->width();
				for (int i = 0; i < roi_proj.height; i++)
				{
					uint8_t* pImg = m_visOctProjection.raw_ptr() + i * roi_proj.width;
					std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_proj_width_non4);
				}
			}
#endif

#ifdef OCT_NIRF
			// Adjusting NIRF offset	
#ifndef TWO_CHANNEL_NIRF
			if (m_nirfSignal.length() != 0)
			{
				int sig_len = m_nirfSignal.length();
				int map_len = m_nirfMap.length();
				int diff;

				memset(m_nirfMap.raw_ptr(), 0, sizeof(float) * m_nirfMap.length());

				if (m_nirfOffset > 0)
				{
					if (sig_len > map_len)
					{
						diff = sig_len - map_len;
						memcpy(&m_nirfMap(m_nirfOffset, 0), m_nirfSignal + diff, sizeof(float) * (map_len - m_nirfOffset));
					}
					else if (sig_len == map_len)
					{
						memcpy(&m_nirfMap(m_nirfOffset, 0), m_nirfSignal, sizeof(float) * (map_len - m_nirfOffset));
					}
					else
					{
						diff = map_len - sig_len;
						memcpy(&m_nirfMap(diff + m_nirfOffset, 0), m_nirfSignal, sizeof(float) * (sig_len - m_nirfOffset));
					}
				}
				else
				{
					if (sig_len > map_len)
					{
						diff = sig_len - map_len;
						memcpy(&m_nirfMap(0, 0), m_nirfSignal + diff - m_nirfOffset, sizeof(float) * (map_len + m_nirfOffset));
					}
					else if (sig_len == map_len)
					{
						memcpy(&m_nirfMap(0, 0), m_nirfSignal - m_nirfOffset, sizeof(float) * (map_len + m_nirfOffset));
					}
					else
					{
						diff = map_len - sig_len;
						memcpy(&m_nirfMap(diff, 0), m_nirfSignal - m_nirfOffset, sizeof(float) * (sig_len + m_nirfOffset));
					}
				}
			}

			m_nirfMap0 = np::FloatArray2(roi_proj.width, roi_proj.height);
			memset(m_nirfMap0.raw_ptr(), 0, sizeof(float) * m_nirfMap0.length());
			ippiCopy_32f_C1R(m_nirfMap.raw_ptr(), sizeof(float) * m_nirfMap.size(0), m_nirfMap0.raw_ptr(), sizeof(float) * m_nirfMap0.size(0), roi_proj);
#else
			if (m_nirfSignal1.length() != 0)
			{
				int sig_len = m_nirfSignal1.length();
				int map_len = m_nirfMap1.length();
				int diff;

				memset(m_nirfMap1.raw_ptr(), 0, sizeof(float) * m_nirfMap1.length());

				if (m_nirfOffset > 0)
				{
					if (sig_len > map_len)
					{
						diff = sig_len - map_len;
						memcpy(&m_nirfMap1(m_nirfOffset, 0), m_nirfSignal1 + diff, sizeof(float) * (map_len - m_nirfOffset));
					}
					else if (sig_len == map_len)
					{
						memcpy(&m_nirfMap1(m_nirfOffset, 0), m_nirfSignal1, sizeof(float) * (map_len - m_nirfOffset));
					}
					else
					{
						diff = map_len - sig_len;
						memcpy(&m_nirfMap1(diff + m_nirfOffset, 0), m_nirfSignal1, sizeof(float) * (sig_len - m_nirfOffset));
					}
				}
				else
				{
					if (sig_len > map_len)
					{
						diff = sig_len - map_len;
						memcpy(&m_nirfMap1(0, 0), m_nirfSignal1 + diff - m_nirfOffset, sizeof(float) * (map_len + m_nirfOffset));
					}
					else if (sig_len == map_len)
					{
						memcpy(&m_nirfMap1(0, 0), m_nirfSignal1 - m_nirfOffset, sizeof(float) * (map_len + m_nirfOffset));
					}
					else
					{
						diff = map_len - sig_len;
						memcpy(&m_nirfMap1(diff, 0), m_nirfSignal1 - m_nirfOffset, sizeof(float) * (sig_len + m_nirfOffset));
					}
				}
			}

			m_nirfMap1_0 = np::FloatArray2(roi_proj.width, roi_proj.height);
			memset(m_nirfMap1_0.raw_ptr(), 0, sizeof(float) * m_nirfMap1_0.length());
			ippiCopy_32f_C1R(m_nirfMap1.raw_ptr(), sizeof(float) * m_nirfMap1.size(0), m_nirfMap1_0.raw_ptr(), sizeof(float) * m_nirfMap1_0.size(0), roi_proj);

			if (m_nirfSignal2.length() != 0)
			{
				int sig_len = m_nirfSignal2.length();
				int map_len = m_nirfMap2.length();
				int diff;

				memset(m_nirfMap2.raw_ptr(), 0, sizeof(float) * m_nirfMap2.length());

				if (m_nirfOffset > 0)
				{
					if (sig_len > map_len)
					{
						diff = sig_len - map_len;
						memcpy(&m_nirfMap2(m_nirfOffset, 0), m_nirfSignal2 + diff, sizeof(float) * (map_len - m_nirfOffset));
					}
					else if (sig_len == map_len)
					{
						memcpy(&m_nirfMap2(m_nirfOffset, 0), m_nirfSignal2, sizeof(float) * (map_len - m_nirfOffset));
					}
					else
					{
						diff = map_len - sig_len;
						memcpy(&m_nirfMap2(diff + m_nirfOffset, 0), m_nirfSignal2, sizeof(float) * (sig_len - m_nirfOffset));
					}
				}
				else
				{
					if (sig_len > map_len)
					{
						diff = sig_len - map_len;
						memcpy(&m_nirfMap2(0, 0), m_nirfSignal2 + diff - m_nirfOffset, sizeof(float) * (map_len + m_nirfOffset));
					}
					else if (sig_len == map_len)
					{
						memcpy(&m_nirfMap2(0, 0), m_nirfSignal2 - m_nirfOffset, sizeof(float) * (map_len + m_nirfOffset));
					}
					else
					{
						diff = map_len - sig_len;
						memcpy(&m_nirfMap2(diff, 0), m_nirfSignal2 - m_nirfOffset, sizeof(float) * (sig_len + m_nirfOffset));
					}
				}
			}

			m_nirfMap2_0 = np::FloatArray2(roi_proj.width, roi_proj.height);
			memset(m_nirfMap2_0.raw_ptr(), 0, sizeof(float) * m_nirfMap2_0.length());
			ippiCopy_32f_C1R(m_nirfMap2.raw_ptr(), sizeof(float) * m_nirfMap2.size(0), m_nirfMap2_0.raw_ptr(), sizeof(float) * m_nirfMap2_0.size(0), roi_proj);
#endif
			// Scaling NIRF map
            if (m_pNirfDistCompDlg)
            {
#ifndef TWO_CHANNEL_NIRF
                ippsSubC_32f_I(m_pNirfDistCompDlg->nirfBg, m_nirfMap0.raw_ptr(), m_nirfMap0.length()); // BG subtraction
                if (m_pNirfDistCompDlg->isCompensating())
                    ippsMul_32f_I(m_pNirfDistCompDlg->compMap.raw_ptr(), m_nirfMap0.raw_ptr(), m_nirfMap0.length()); // Distance compensation
                if (m_pNirfDistCompDlg->isTBRMode())
                {
                    if (m_pNirfDistCompDlg->tbrBg > 0)
                    {
						if (m_pNirfDistCompDlg->isZeroTbr())
	                        ippsSubC_32f_I(m_pNirfDistCompDlg->tbrBg, m_nirfMap0.raw_ptr(), m_nirfMap0.length());
                        ippsDivC_32f_I(m_pNirfDistCompDlg->tbrBg, m_nirfMap0.raw_ptr(), m_nirfMap0.length());

						if (m_pNirfDistCompDlg->compConst != 1.0f)
						{
							ippsSubC_32f_I(1.0f, m_nirfMap0.raw_ptr(), m_nirfMap0.length());
							ippsMulC_32f_I(m_pNirfDistCompDlg->compConst, m_nirfMap0.raw_ptr(), m_nirfMap0.length());
							ippsAddC_32f_I(1.0f, m_nirfMap0.raw_ptr(), m_nirfMap0.length());
						}
                    }
                }
				if (m_pNirfDistCompDlg->isGwMasked())
				{
					if (m_pNirfDistCompDlg->gwMap.raw_ptr())
					{
						ippsMul_32f_I(m_pNirfDistCompDlg->gwMap, m_nirfMap0.raw_ptr(), m_nirfMap0.length());
					}
				}
#else
				// Ch 1
				ippsSubC_32f_I(m_pNirfDistCompDlg->nirfBg[0], m_nirfMap1_0.raw_ptr(), m_nirfMap1_0.length()); // BG subtraction
				if (m_pNirfDistCompDlg->isCompensating())
					ippsMul_32f_I(m_pNirfDistCompDlg->compMap[0].raw_ptr(), m_nirfMap1_0.raw_ptr(), m_nirfMap1_0.length()); // Distance compensation
				if (m_pNirfDistCompDlg->isTBRMode())
				{
					if (m_pNirfDistCompDlg->tbrBg[0] > 0)
					{
						if (m_pNirfDistCompDlg->isZeroTbr())
							ippsSubC_32f_I(m_pNirfDistCompDlg->tbrBg[0], m_nirfMap1_0.raw_ptr(), m_nirfMap1_0.length());
						ippsDivC_32f_I(m_pNirfDistCompDlg->tbrBg[0], m_nirfMap1_0.raw_ptr(), m_nirfMap1_0.length());

						if (m_pNirfDistCompDlg->compConst != 1.0f)
						{
							ippsSubC_32f_I(1.0f, m_nirfMap1_0.raw_ptr(), m_nirfMap1_0.length());
							ippsMulC_32f_I(m_pNirfDistCompDlg->compConst, m_nirfMap1_0.raw_ptr(), m_nirfMap1_0.length());
							ippsAddC_32f_I(1.0f, m_nirfMap1_0.raw_ptr(), m_nirfMap1_0.length());
						}
					}
				}
				if (m_pNirfDistCompDlg->isGwMasked())
				{
					if (m_pNirfDistCompDlg->gwMap.raw_ptr())
					{
						ippsMul_32f_I(m_pNirfDistCompDlg->gwMap, m_nirfMap1_0.raw_ptr(), m_nirfMap1_0.length());
					}
				}

				// Ch 2
				ippsSubC_32f_I(m_pNirfDistCompDlg->nirfBg[1], m_nirfMap2_0.raw_ptr(), m_nirfMap2_0.length()); // BG subtraction
				if (m_pNirfDistCompDlg->isCompensating())
					ippsMul_32f_I(m_pNirfDistCompDlg->compMap[1].raw_ptr(), m_nirfMap2_0.raw_ptr(), m_nirfMap2_0.length()); // Distance compensation
				if (m_pNirfDistCompDlg->isTBRMode())
				{
					if (m_pNirfDistCompDlg->tbrBg[1] > 0)
					{
						if (m_pNirfDistCompDlg->isZeroTbr())
							ippsSubC_32f_I(m_pNirfDistCompDlg->tbrBg[1], m_nirfMap2_0.raw_ptr(), m_nirfMap2_0.length());
						ippsDivC_32f_I(m_pNirfDistCompDlg->tbrBg[1], m_nirfMap2_0.raw_ptr(), m_nirfMap2_0.length());

						if (m_pNirfDistCompDlg->compConst != 1.0f)
						{
							ippsSubC_32f_I(1.0f, m_nirfMap2_0.raw_ptr(), m_nirfMap2_0.length());
							ippsMulC_32f_I(m_pNirfDistCompDlg->compConst, m_nirfMap2_0.raw_ptr(), m_nirfMap2_0.length());
							ippsAddC_32f_I(1.0f, m_nirfMap2_0.raw_ptr(), m_nirfMap2_0.length());
						}
					}
				}

				// 2 Ch NIRF Cross-Talk Compensation
				if (m_pNirfDistCompDlg->crossTalkRatio > 0)
				{
					np::FloatArray2 corr_array(m_nirfMap1_0.size(0), m_nirfMap1_0.size(1));
					memcpy(corr_array.raw_ptr(), m_nirfMap1_0.raw_ptr(), sizeof(float) * m_nirfMap1_0.length());
					ippsMulC_32f_I(m_pNirfDistCompDlg->crossTalkRatio, corr_array.raw_ptr(), corr_array.length());

					ippsSub_32f_I(corr_array.raw_ptr(), m_nirfMap2_0.raw_ptr(), m_nirfMap2_0.length());
				}

				// GW map
				if (m_pNirfDistCompDlg->isGwMasked())
				{
					if (m_pNirfDistCompDlg->gwMap.raw_ptr())
					{
						ippsMul_32f_I(m_pNirfDistCompDlg->gwMap, m_nirfMap2_0.raw_ptr(), m_nirfMap2_0.length());
					}
				}
#endif
				QString infoPath = m_path + QString("/dist_comp_info.log");
				m_pNirfDistCompDlg->setCompInfo(infoPath.toUtf8().constData());
            }

#ifndef TWO_CHANNEL_NIRF
            ippiScale_32f8u_C1R(m_nirfMap0.raw_ptr(), sizeof(float) * roi_proj.width, m_pImgObjNirfMap->arr.raw_ptr(), sizeof(uint8_t) * roi_proj.width,
                roi_proj, m_pConfig->nirfRange.min, m_pConfig->nirfRange.max);
			ippiMirror_8u_C1IR(m_pImgObjNirfMap->arr.raw_ptr(), sizeof(uint8_t) * roi_proj.width, roi_proj, ippAxsHorizontal);
#else
			ippiScale_32f8u_C1R(m_nirfMap1_0.raw_ptr(), sizeof(float) * roi_proj.width, m_pImgObjNirfMap1->arr.raw_ptr(), sizeof(uint8_t) * roi_proj.width,
				roi_proj, m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[0].max);
			ippiMirror_8u_C1IR(m_pImgObjNirfMap1->arr.raw_ptr(), sizeof(uint8_t) * roi_proj.width, roi_proj, ippAxsHorizontal);

			ippiScale_32f8u_C1R(m_nirfMap2_0.raw_ptr(), sizeof(float) * roi_proj.width, m_pImgObjNirfMap2->arr.raw_ptr(), sizeof(uint8_t) * roi_proj.width,
				roi_proj, m_pConfig->nirfRange[1].min, m_pConfig->nirfRange[1].max);
			ippiMirror_8u_C1IR(m_pImgObjNirfMap2->arr.raw_ptr(), sizeof(uint8_t) * roi_proj.width, roi_proj, ippAxsHorizontal);
#endif
#ifdef GALVANO_MIRROR
			if (m_pConfig->galvoHorizontalShift)
			{
                int roi_proj_width_non4 = m_pImageView_RectImage->getRender()->m_pImage->width();
				for (int i = 0; i < roi_proj.height; i++)
				{
#ifndef TWO_CHANNEL_NIRF
                    uint8_t* pImg = m_pImgObjNirfMap->arr.raw_ptr() + i * roi_proj.width;
                    std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_proj_width_non4);
#else
					uint8_t* pImg1 = m_pImgObjNirfMap1->arr.raw_ptr() + i * roi_proj.width;
					std::rotate(pImg1, pImg1 + m_pConfig->galvoHorizontalShift, pImg1 + roi_proj_width_non4);
					uint8_t* pImg2 = m_pImgObjNirfMap2->arr.raw_ptr() + i * roi_proj.width;
					std::rotate(pImg2, pImg2 + m_pConfig->galvoHorizontalShift, pImg2 + roi_proj_width_non4);
#endif
				}
			}
#endif
			if (m_pNirfDistCompDlg)
			{
				if (m_pNirfDistCompDlg->isFiltered())
				{
#ifndef TWO_CHANNEL_NIRF
					(*m_pMedfiltNirf)(m_pImgObjNirfMap->arr.raw_ptr());
#else
					(*m_pMedfiltNirf)(m_pImgObjNirfMap1->arr.raw_ptr());
					(*m_pMedfiltNirf)(m_pImgObjNirfMap2->arr.raw_ptr());
#endif
				}
			}
#endif

#ifdef OCT_FLIM
			// Scaling FLIM projection
			IppiSize roi_flimproj = { m_intensityMap.at(0).size(0), m_intensityMap.at(0).size(1) };

			ippiScale_32f8u_C1R(m_intensityMap.at(m_pComboBox_EmissionChannel->currentIndex()), sizeof(float) * roi_flimproj.width,
				m_pImgObjIntensityMap->arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width,
				roi_flimproj, m_pConfig->flimIntensityRange.min, m_pConfig->flimIntensityRange.max);
			ippiMirror_8u_C1IR(m_pImgObjIntensityMap->arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width, roi_flimproj, ippAxsHorizontal);
#ifdef GALVANO_MIRROR
			if (m_pConfig->galvoHorizontalShift)
			{
				for (int i = 0; i < roi_proj.height; i++)
				{
					uint8_t* pImg = m_pImgObjIntensityMap->arr.raw_ptr() + i * roi_proj.width / 4;
					std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift / 4, pImg + roi_proj.width / 4);
				}
			}
#endif
			IppiSize roi_flimfilt = { m_pMedfiltIntensityMap->getWidth(), roi_flimproj.height };
			Uint8Array2 temp_intensity(roi_flimfilt.width, roi_flimfilt.height);
			ippiCopy_8u_C1R(m_pImgObjIntensityMap->arr.raw_ptr(), roi_flimproj.width, temp_intensity.raw_ptr(), roi_flimfilt.width, roi_flimfilt);
			(*m_pMedfiltIntensityMap)(temp_intensity);
			ippiCopy_8u_C1R(temp_intensity.raw_ptr(), roi_flimfilt.width, m_pImgObjIntensityMap->arr.raw_ptr(), roi_flimproj.width, roi_flimproj);

			ippiScale_32f8u_C1R(m_lifetimeMap.at(m_pComboBox_EmissionChannel->currentIndex()), sizeof(float) * roi_flimproj.width,
				m_pImgObjLifetimeMap->arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width,
				roi_flimproj, m_pConfig->flimLifetimeRange.min, m_pConfig->flimLifetimeRange.max);
			ippiMirror_8u_C1IR(m_pImgObjLifetimeMap->arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width, roi_flimproj, ippAxsHorizontal);
#ifdef GALVANO_MIRROR
			if (m_pConfig->galvoHorizontalShift)
			{
				for (int i = 0; i < roi_proj.height; i++)
				{
					uint8_t* pImg = m_pImgObjLifetimeMap->arr.raw_ptr() + i * roi_proj.width / 4;
					std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift / 4, pImg + roi_proj.width / 4);
				}
			}
#endif
			Uint8Array2 temp_lifetime(roi_flimfilt.width, roi_flimfilt.height);
			ippiCopy_8u_C1R(m_pImgObjLifetimeMap->arr.raw_ptr(), roi_flimproj.width, temp_lifetime.raw_ptr(), roi_flimfilt.width, roi_flimfilt);
			(*m_pMedfiltLifetimeMap)(temp_lifetime);
			ippiCopy_8u_C1R(temp_lifetime.raw_ptr(), roi_flimfilt.width, m_pImgObjLifetimeMap->arr.raw_ptr(), roi_flimproj.width, roi_flimproj);

			m_pImgObjLifetimeMap->convertRgb();

			// Make intensity-weighted lifetime map
			if (m_pCheckBox_HsvEnhancedMap->isChecked())
			{
#if (LIFETIME_COLORTABLE == 7)
				// HSV channel setting
				ImageObject tempImgObj(m_pImgObjHsvEnhancedMap->getWidth(), m_pImgObjHsvEnhancedMap->getHeight(), m_pImgObjHsvEnhancedMap->getColorTable());

				memset(tempImgObj.qrgbimg.bits(), 255, tempImgObj.qrgbimg.byteCount()); // Saturation is set to be 255.
				tempImgObj.setRgbChannelData(m_pImgObjLifetimeMap->qindeximg.bits(), 0); // Hue	
				uint8_t *pIntensity = new uint8_t[m_pImgObjIntensityMap->qindeximg.byteCount()];
				memcpy(pIntensity, m_pImgObjIntensityMap->qindeximg.bits(), m_pImgObjIntensityMap->qindeximg.byteCount());
				ippsMulC_8u_ISfs(1.0, pIntensity, m_pImgObjIntensityMap->qindeximg.byteCount(), 0);
				tempImgObj.setRgbChannelData(pIntensity, 2); // Value
				delete[] pIntensity;

				ippiHSVToRGB_8u_C3R(tempImgObj.qrgbimg.bits(), 3 * roi_flimproj.width, m_pImgObjHsvEnhancedMap->qrgbimg.bits(), 3 * roi_flimproj.width, roi_flimproj);
#else
				// Non HSV intensity-weight map
				ColorTable temp_ctable;
				ImageObject tempImgObj(m_pImgObjHsvEnhancedMap->getWidth(), m_pImgObjHsvEnhancedMap->getHeight(), temp_ctable.m_colorTableVector.at(ColorTable::gray));

				m_pImgObjLifetimeMap->convertRgb();
				memcpy(tempImgObj.qindeximg.bits(), m_pImgObjIntensityMap->arr.raw_ptr(), tempImgObj.qindeximg.byteCount());
				ippsMulC_8u_ISfs(5, tempImgObj.qindeximg.bits(), tempImgObj.qindeximg.byteCount(), 0);
				tempImgObj.convertRgb();

				ippsMul_8u_Sfs(m_pImgObjLifetimeMap->qrgbimg.bits(), tempImgObj.qrgbimg.bits(), m_pImgObjHsvEnhancedMap->qrgbimg.bits(), tempImgObj.qrgbimg.byteCount(), 8);
#endif
			}

			// Make intensity ratio map
			if (m_pCheckBox_IntensityRatio->isChecked())
			{
				np::FloatArray2 ch1_intensity(m_intensityMap.at(0).size(0), m_intensityMap.at(0).size(1));
				np::FloatArray2 ch2_intensity(m_intensityMap.at(1).size(0), m_intensityMap.at(1).size(1));
				np::FloatArray2 intensity_ratio(m_intensityMap.at(1).size(0), m_intensityMap.at(1).size(1));
				
				memcpy(ch1_intensity.raw_ptr(), m_intensityMap.at(0).raw_ptr(), sizeof(float) * m_intensityMap.at(0).length());
				memcpy(ch2_intensity.raw_ptr(), m_intensityMap.at(1).raw_ptr(), sizeof(float) * m_intensityMap.at(1).length());

				medfilt *pMedfiltIntensityMap = m_pMedfiltIntensityMap;
				
				(*pMedfiltIntensityMap)(ch1_intensity.raw_ptr());
				(*pMedfiltIntensityMap)(ch2_intensity.raw_ptr());

				ippsDiv_32f(ch1_intensity.raw_ptr(), ch2_intensity.raw_ptr(), intensity_ratio.raw_ptr(), intensity_ratio.length());
				
				ippiScale_32f8u_C1R(intensity_ratio, sizeof(float) * roi_flimproj.width,
					m_pImgObjIntensityMap->arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width, roi_flimproj, 0.0, 2.0);
				ippiMirror_8u_C1IR(m_pImgObjIntensityMap->arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width, roi_flimproj, ippAxsHorizontal);
#ifdef GALVANO_MIRROR
				if (m_pConfig->galvoHorizontalShift)
				{
					for (int i = 0; i < roi_proj.height; i++)
					{
						uint8_t* pImg = m_pImgObjIntensityMap->arr.raw_ptr() + i * roi_proj.width / 4;
						std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift / 4, pImg + roi_proj.width / 4);
					}
				}
#endif
				Uint8Array2 temp_intensity(roi_flimfilt.width, roi_flimfilt.height);
				ippiCopy_8u_C1R(m_pImgObjIntensityMap->arr.raw_ptr(), roi_flimproj.width, temp_intensity.raw_ptr(), roi_flimfilt.width, roi_flimfilt);
				//(*m_pMedfiltIntensityMap)(temp_intensity);
				ippiCopy_8u_C1R(temp_intensity.raw_ptr(), roi_flimfilt.width, m_pImgObjIntensityMap->arr.raw_ptr(), roi_flimproj.width, roi_flimproj);
			}

			// Make classification map
			if (m_pCheckBox_Classification->isChecked())
			{				
				Uint8Array2 clsf_map(3 * m_lifetimeMap.at(0).size(0), m_lifetimeMap.at(0).size(1));
				QFile file("clsf_map.bin");
				file.open(QFile::ReadOnly);
				file.read(reinterpret_cast<char*>(clsf_map.raw_ptr()), sizeof(uint8_t) * clsf_map.length());
				file.close();
					// Read Ini Fil


//				// Ch1, ch2 lifetime and intensity ratio
//				np::FloatArray2 ch1_lifetime(m_lifetimeMap.at(0).size(0), m_lifetimeMap.at(0).size(1));
//				np::FloatArray2 ch2_lifetime(m_lifetimeMap.at(1).size(0), m_lifetimeMap.at(1).size(1));
//				np::FloatArray2 ch1_intensity(m_intensityMap.at(0).size(0), m_intensityMap.at(0).size(1));
//				np::FloatArray2 ch2_intensity(m_intensityMap.at(1).size(0), m_intensityMap.at(1).size(1));
//				np::FloatArray2 intensity_ratio(m_intensityMap.at(1).size(0), m_intensityMap.at(1).size(1));
//
//				memcpy(ch1_lifetime.raw_ptr(), m_lifetimeMap.at(0).raw_ptr(), sizeof(float) * m_lifetimeMap.at(0).length());
//				memcpy(ch2_lifetime.raw_ptr(), m_lifetimeMap.at(1).raw_ptr(), sizeof(float) * m_lifetimeMap.at(1).length());
//				memcpy(ch1_intensity.raw_ptr(), m_intensityMap.at(0).raw_ptr(), sizeof(float) * m_intensityMap.at(0).length());
//				memcpy(ch2_intensity.raw_ptr(), m_intensityMap.at(1).raw_ptr(), sizeof(float) * m_intensityMap.at(1).length());
//							
//				medfilt *pMedfiltIntensityMap = new medfilt(ch1_lifetime.size(0), ch1_lifetime.size(1), 5, 3);
//				medfilt *pMedfiltLifetimeMap = new medfilt(ch1_lifetime.size(0), ch1_lifetime.size(1), 11, 7);
//				
//				(*pMedfiltLifetimeMap)(ch1_lifetime.raw_ptr());
//				(*pMedfiltLifetimeMap)(ch2_lifetime.raw_ptr());
//				(*pMedfiltIntensityMap)(ch1_intensity.raw_ptr());
//				(*pMedfiltIntensityMap)(ch2_intensity.raw_ptr());
//
//				ippsDiv_32f(ch1_intensity.raw_ptr(), ch2_intensity.raw_ptr(), intensity_ratio.raw_ptr(), intensity_ratio.length());
//
//				// Make ann object
//				int w = ch1_lifetime.size(0), h = ch1_lifetime.size(1);
//
//				if (m_pAnn) delete m_pAnn;
//				m_pAnn = new ann(w, h, m_pConfig->clfAnnXNode, m_pConfig->clfAnnHNode, m_pConfig->clfAnnYNode);
//
//				(*m_pAnn)(ch1_lifetime, ch2_lifetime, intensity_ratio);
//
//#ifdef GALVANO_MIRROR
//				if (m_pConfig->galvoHorizontalShift)
//				{
//					for (int i = 0; i < roi_proj.height; i++)
//					{
//						uint8_t* pImg = m_pAnn->GetClfMapPtr() + i * 3 * roi_proj.width / 4;
//						std::rotate(pImg, pImg + 3 * int(m_pConfig->galvoHorizontalShift / 4), pImg + 3 * roi_proj.width / 4);
//					}
//				}
//#endif										
//				// Classification map
//				memcpy(m_pImgObjHsvEnhancedMap->qrgbimg.bits(), m_pAnn->GetClfMapPtr(), m_pImgObjHsvEnhancedMap->qrgbimg.byteCount());
				memcpy(m_pImgObjHsvEnhancedMap->qrgbimg.bits(), clsf_map.raw_ptr(), m_pImgObjHsvEnhancedMap->qrgbimg.byteCount());
				printf("%d\n", m_pImgObjHsvEnhancedMap->qrgbimg.byteCount());

				// Intensity weighting
				if (m_pCheckBox_HsvEnhancedMap->isChecked())
				{
					ColorTable temp_ctable;
					ImageObject tempImgObj(m_pImgObjHsvEnhancedMap->getWidth(), m_pImgObjHsvEnhancedMap->getHeight(), temp_ctable.m_colorTableVector.at(ColorTable::gray));

					memcpy(tempImgObj.qindeximg.bits(), m_pImgObjIntensityMap->arr.raw_ptr(), tempImgObj.qindeximg.byteCount());
					ippsMulC_8u_ISfs(5, tempImgObj.qindeximg.bits(), tempImgObj.qindeximg.byteCount(), 0);
					tempImgObj.convertRgb();

					ippsMul_8u_ISfs(tempImgObj.qrgbimg.bits(), m_pImgObjHsvEnhancedMap->qrgbimg.bits(), tempImgObj.qrgbimg.byteCount(), 8);
				}
				///ColorTable temp_ctable;
				///ImageObject tempImgObj(m_pImgObjHsvEnhancedMap->getWidth(), m_pImgObjHsvEnhancedMap->getHeight(), temp_ctable.m_colorTableVector.at(ColorTable::clf));

				///memcpy(tempImgObj.qindeximg.bits(), m_pAnn->GetClfMapPtr(), tempImgObj.qindeximg.byteCount());
				///tempImgObj.convertRgb();

				///memcpy(m_pImgObjHsvEnhancedMap->qrgbimg.bits(), tempImgObj.qrgbimg.bits(), tempImgObj.qrgbimg.byteCount());
			}
#endif
		}
		emit paintOctProjection(m_visOctProjection);
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
		emit paintNirfMap(m_pImgObjNirfMap->arr.raw_ptr());
#else
		emit paintNirfMap1(m_pImgObjNirfMap1->arr.raw_ptr());
		emit paintNirfMap2(m_pImgObjNirfMap2->arr.raw_ptr());
#endif
#endif
#ifdef OCT_FLIM
		emit paintIntensityMap(m_pImgObjIntensityMap->arr.raw_ptr());
		emit paintLifetimeMap((!m_pCheckBox_HsvEnhancedMap->isChecked() && !m_pCheckBox_Classification->isChecked()) ? 
								m_pImgObjLifetimeMap->qrgbimg.bits() : m_pImgObjHsvEnhancedMap->qrgbimg.bits());
#endif
}
}

void QResultTab::measureDistance(bool toggled)
{
	m_pImageView_CircImage->getRender()->m_bMeasureDistance = toggled;
	if (!toggled)
	{
		m_pImageView_CircImage->getRender()->m_nClicked = 0;
		m_pImageView_CircImage->getRender()->update();
	}
}

void QResultTab::showGuideLine(bool toggled)
{
	if (toggled)
	{
		if (m_pCheckBox_ShowGuideLine->isChecked())
		{
			//int center = (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? m_pConfig->circCenter :
			//	(m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3) + m_polishedSurface(m_pSlider_SelectFrame->value()) - m_pConfig->ballRadius;
			int center = m_pConfig->circCenter;				
			//int polished_surface = (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? center - m_pConfig->circCenter : center + m_pConfig->ballRadius;
			int polished_surface = center - m_pConfig->circCenter;

#if (defined(OCT_FLIM) || defined(OCT_NIRF))
			m_pImageView_RectImage->setHorizontalLine(7, m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3, center, center + m_pConfig->sheathRadius,
				polished_surface, center + m_pConfigTemp->circRadius, center + m_pConfigTemp->circRadius - m_pConfig->ringThickness, m_pConfigTemp->n2ScansFFT / 2 + m_pConfigTemp->nScans / 3);
			m_pImageView_RectImage->setHorizontalLineColor(7, 0xff0000, 0x00ff00, 0xffff00, 0xff00ff, 0x00ff00, 0x00ff00, 0xff0000);
#else
			m_pImageView_RectImage->setHorizontalLine(6, m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3, center, center + m_pConfig->sheathRadius,
				polished_surface, center + m_pConfigTemp->circRadius, m_pConfigTemp->n2ScansFFT / 2 + m_pConfigTemp->nScans / 3);
			m_pImageView_RectImage->setHorizontalLineColor(6, 0xff0000, 0x00ff00, 0xffff00, 0xffff00, 0x00ff00, 0xff0000);
#endif
			//m_pImageView_CircImage->setCircle(2, m_pConfig->sheathRadius, (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? 1 : m_pConfig->ballRadius);
			m_pImageView_CircImage->setCircle(2, m_pConfig->sheathRadius, 1);
		}
	}
	else
	{
		m_pImageView_RectImage->setHorizontalLine(0);
		m_pImageView_RectImage->setHorizontalLineColor(0);
		m_pImageView_CircImage->setCircle(0);
	}

	m_pImageView_RectImage->getRender()->update();
	m_pImageView_CircImage->getRender()->update();
}

void QResultTab::changeVisImage(bool toggled)
{	
	if (toggled) // circ view
	{
		m_pImageView_CircImage->setVisible(toggled);
		m_pImageView_RectImage->setVisible(!toggled);

		if (m_pLongitudinalViewDlg)
			m_pImageView_CircImage->setVerticalLine(1, m_pLongitudinalViewDlg->getCurrentAline());
		if (m_pSpectroOCTDlg)
			m_pImageView_CircImage->setVerticalLine(1, m_pSpectroOCTDlg->getCurrentAline());
	}
	else // rect view
	{
		m_pToggleButton_MeasureDistance->setChecked(false);
		m_pImageView_CircImage->setVisible(toggled);
		m_pImageView_RectImage->setVisible(!toggled);

		if (m_pLongitudinalViewDlg)
			m_pImageView_RectImage->setVerticalLine(1, m_pLongitudinalViewDlg->getCurrentAline());
		if (m_pSpectroOCTDlg)
			m_pImageView_RectImage->setVerticalLine(1, m_pSpectroOCTDlg->getCurrentAline());
	}
	visualizeImage(m_pSlider_SelectFrame->value());

	m_pToggleButton_MeasureDistance->setEnabled(toggled);		
}

void QResultTab::findPolishedSurface(bool toggled)
{
	//if (toggled)
	//{		
	//	m_pProgressBar_PostProcessing->setFormat("Finding polished surfaces... %p%");
	//	m_pProgressBar_PostProcessing->setRange(0, m_pConfigTemp->nFrames - 1);
	//	m_pProgressBar_PostProcessing->setValue(0);

	//	m_polishedSurface = np::Array<int>(m_pConfigTemp->nFrames);
	//	memset(m_polishedSurface, 0, sizeof(int) * m_polishedSurface.length());
	//	
	//	// Ball lens polished surface detection
	//	IppiSize roi_oct = { m_pImgObjRectImage->getHeight(), m_pImgObjRectImage->getWidth() };
	//	for (int i = 0; i < m_pConfigTemp->nFrames; i++)
	//	{
	//		// OCT Visualization
	//		np::FloatArray2 tranp_temp(roi_oct.height, roi_oct.width);
	//		ippiTranspose_32f_C1R(m_vectorOctImage.at(i), roi_oct.width * sizeof(float), tranp_temp.raw_ptr(), roi_oct.height * sizeof(float), roi_oct);
	//		
	//		np::FloatArray mean_profile(tranp_temp.size(1) / 2 - (m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3));
	//		np::FloatArray mean_profile_filt(mean_profile.length());
	//		memset(mean_profile_filt, 0, sizeof(float) * mean_profile_filt.length());
	//		tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)mean_profile.length()),
	//			[&](const tbb::blocked_range<size_t>& r) {
	//			for (size_t j = r.begin(); j != r.end(); ++j)
	//			{
	//				ippsMean_32f((const Ipp32f*)&tranp_temp(0, m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3 + (int)j), 
	//																			tranp_temp.size(0), &mean_profile((int)j), ippAlgHintFast);
	//			}
	//		});
	//		tbb::parallel_for(tbb::blocked_range<size_t>(2, (size_t)mean_profile.length() - 2),
	//			[&](const tbb::blocked_range<size_t>& r) {
	//			for (size_t j = r.begin(); j != r.end(); ++j)
	//			{
	//				for (int k = 0; k < 5; k++)
	//					mean_profile_filt((int)j) += mean_profile((int)j - 2 + k);
	//				mean_profile_filt((int)j) /= 5;
	//			}
	//		});
	//					
	//		np::FloatArray drv_profile(mean_profile.length() - 1);
	//		memset(drv_profile, 0, sizeof(float) * drv_profile.length());
	//		tbb::parallel_for(tbb::blocked_range<size_t>(2, (size_t)drv_profile.length() - 2),
	//			[&](const tbb::blocked_range<size_t>& r) {
	//			for (size_t j = r.begin(); j != r.end(); ++j)
	//			{
	//				drv_profile((int)j) = mean_profile_filt((int)j + 1) - mean_profile_filt((int)j);
	//			}
	//		});
	//		
	//		QFile file1("profile.dat");
	//		if (true == file1.open(QFile::WriteOnly))
	//			file1.write(reinterpret_cast<char*>(mean_profile_filt.raw_ptr()), sizeof(float) * mean_profile_filt.length());
	//		file1.close();
	//		QFile file("d_profile.dat");
	//		if (true == file.open(QFile::WriteOnly))			
	//			file.write(reinterpret_cast<char*>(drv_profile.raw_ptr()), sizeof(float) * drv_profile.length());
	//		file.close();

	//		float max;
	//		ippsMax_32f(mean_profile_filt.raw_ptr(), mean_profile_filt.length(), &max);
	//		//printf("%f\n", max);
	//		for (int j = 0; j < drv_profile.length() - 1; j++)
	//		{
	//			bool det = (drv_profile(j + 1) * drv_profile(j) < 0) ? true : false;
	//			if (det && (mean_profile_filt(j + 1) > max * 0.9))
	//			{
	//				//printf("[%d %f]\n", j + 1, mean_profile_filt(j + 1 ));
	//				m_polishedSurface(i) = j + 1;
	//				break;
	//			}
	//		}

	//		m_pProgressBar_PostProcessing->setValue(i);

	//		//break;
	//	}
	//	QFile file("polished.dat");
	//	if (true == file.open(QFile::WriteOnly))
	//		file.write(reinterpret_cast<char*>(m_polishedSurface.raw_ptr()), sizeof(int) * m_polishedSurface.length());
	//	file.close();

	//	m_pProgressBar_PostProcessing->setFormat("");
	//	m_pProgressBar_PostProcessing->setValue(0);
	//	m_pToggleButton_FindPolishedSurfaces->setText("Fixed Center Mode");
	//	m_pLabel_CircCenter->setText("Ball Radius");
	//	m_pLineEdit_CircCenter->setText(QString::number(m_pConfig->ballRadius));
	//}
	//else
	//{		
	//	m_pToggleButton_FindPolishedSurfaces->setText("Find Polished Surfaces");
	//	m_pLabel_CircCenter->setText("Circ Center");
	//	m_pLineEdit_CircCenter->setText(QString::number(m_pConfig->circCenter));
	//}

	//getOctProjection(m_vectorOctImage, m_octProjection);
	//visualizeEnFaceMap(true);
	//visualizeImage(m_pSlider_SelectFrame->value());
	//if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
}

void QResultTab::autoContouring(bool toggled)
{
	QString contour_map_path = m_path + "/contour_map.bin";

	QFileInfo check_file(contour_map_path);

	if (toggled)
	{
		// Load first
		if (check_file.exists())
		{
			QFile file(contour_map_path);
			file.open(QIODevice::ReadOnly);
			file.read(reinterpret_cast<char*>(m_contourMap.raw_ptr()), sizeof(uint16_t) * m_contourMap.length());
			file.close();
		}
		else
		{
			// Threading
			QMessageBox msg_box(QMessageBox::NoIcon, "Lumen Contour Detection...", "", QMessageBox::NoButton, this);
			msg_box.setStandardButtons(0);
			msg_box.setWindowModality(Qt::WindowModal);
			msg_box.setWindowFlags(Qt::Window | Qt::WindowTitleHint | Qt::CustomizeWindowHint);
			msg_box.move(QApplication::desktop()->screen()->rect().center() - msg_box.rect().center());
			msg_box.setFixedSize(msg_box.width(), msg_box.height());
			msg_box.show();
						
			int n_pieces = 12;
			np::Array<int> pieces(n_pieces + 1);
			pieces[0] = 0;
			for (int p = 0; p < n_pieces; p++)
				pieces[p + 1] = (int)((double)m_vectorOctImage.size() / (double)n_pieces * (double)(p + 1));

			std::thread *plumdet = new std::thread[n_pieces];
			for (int p = 0; p < n_pieces; p++)
			{
				plumdet[p] = std::thread([&, p, pieces]() {

					LumenDetection *pLumenDetection = new LumenDetection(m_pConfig->sheathRadius, 50, false);

					for (int i = pieces[p]; i < pieces[p + 1]; i++)
					{
						np::FloatArray contour(m_contourMap.size(0));

						np::Uint8Array2 scale_temp(m_vectorOctImage.at(i).size(0), m_vectorOctImage.at(i).size(1));
						ippiScale_32f8u_C1R(m_vectorOctImage.at(i), m_vectorOctImage.at(i).size(0) * sizeof(float),
							scale_temp.raw_ptr(), scale_temp.size(0) * sizeof(uint8_t), { scale_temp.size(0), scale_temp.size(1) },
							m_pConfig->octDbRange.min, m_pConfig->octDbRange.max);

						np::Uint8Array2 crop_img(m_pConfig->circRadius, scale_temp.size(1));
						ippiCopy_8u_C1R(&scale_temp(m_pConfig->circCenter, 0), scale_temp.size(0), crop_img, crop_img.size(0), { crop_img.size(0), crop_img.size(1) });

						(*pLumenDetection)(crop_img, contour);

						ippsConvert_32f16u_Sfs(contour, &m_contourMap(0, i), contour.length(), ippRndNear, 0);

						printf("%d\n", i);
					}

					delete pLumenDetection;
					pLumenDetection = nullptr;
				});
			}
			for (int p = 0; p < n_pieces; p++)
				plumdet[p].join();
			
			QFile file(m_path + "/contour_map.bin");
			file.open(QIODevice::WriteOnly);
			file.write(reinterpret_cast<const char*>(m_contourMap.raw_ptr()), sizeof(uint16_t) * m_contourMap.length());
			file.close();

			//// Set lumen contour in circ image
			//np::Uint16Array contour_16u(&m_contourMap(0, getCurrentFrame()), m_pConfig->nAlines);
			//if (contour_16u[0] == 0)
			//{
			//	if (!m_pLumenDetection)
			//	{
			//		np::FloatArray contour;
			//		m_pLumenDetection = new LumenDetection(m_pConfig->sheathRadius, 50, true);

			//		np::Uint8Array2 scale_temp(m_vectorOctImage.at(getCurrentFrame()).size(0), m_vectorOctImage.at(getCurrentFrame()).size(1));
			//		ippiScale_32f8u_C1R(m_vectorOctImage.at(getCurrentFrame()), m_vectorOctImage.at(getCurrentFrame()).size(0) * sizeof(float),
			//			scale_temp.raw_ptr(), scale_temp.size(0) * sizeof(uint8_t), { scale_temp.size(0), scale_temp.size(1) },
			//			m_pConfig->octDbRange.min, m_pConfig->octDbRange.max);
			//		
			//		np::Uint8Array2 crop_img(m_pConfig->circRadius, scale_temp.size(1));
			//		ippiCopy_8u_C1R(&scale_temp(m_pConfig->circCenter, 0), scale_temp.size(0), crop_img, crop_img.size(0), { crop_img.size(0), crop_img.size(1) });

			//		(*m_pLumenDetection)(crop_img, contour);

			//		ippsConvert_32f16u_Sfs(contour, contour_16u, contour.length(), ippRndNear, 0);

			//		QFile file(m_path + "/contour_map.bin");
			//		file.open(QIODevice::WriteOnly);
			//		file.write(reinterpret_cast<const char*>(m_contourMap.raw_ptr()), sizeof(uint16_t) * m_contourMap.length());
			//		file.close();
			//	}
			//}		
		}

		if (m_pCheckBox_CircularizeImage->isChecked())
			m_pImageView_CircImage->setContour(m_pConfig->nAlines, &m_contourMap(0, getCurrentFrame()));
		else
		{
			np::Uint16Array contour_16u1(m_pConfig->nAlines);
			ippsAddC_16u_Sfs(&m_contourMap(0, getCurrentFrame()), m_pConfig->circCenter, contour_16u1, m_pConfig->nAlines, 0);
			m_pImageView_RectImage->setContour(m_pConfig->nAlines, contour_16u1);
		}
	}
	else
	{
		if (m_pLumenDetection)
		{
			delete m_pLumenDetection;
			m_pLumenDetection = nullptr;
		}
		
		if (m_pCheckBox_CircularizeImage->isChecked())
			m_pImageView_CircImage->setContour(0, nullptr);
		else
			m_pImageView_RectImage->setContour(0, nullptr);
	}

	if (m_pCheckBox_CircularizeImage->isChecked())
		m_pImageView_CircImage->getRender()->update();
	else
		m_pImageView_RectImage->getRender()->update();
}

void QResultTab::checkCircCenter(const QString &str)
{
	//if (!m_pToggleButton_FindPolishedSurfaces->isChecked())
	//{
		int circCenter = str.toInt();
		if (circCenter + m_pConfigTemp->circRadius + 1 > m_pImageView_RectImage->getRender()->m_pImage->height())
		{
			circCenter = m_pImageView_RectImage->getRender()->m_pImage->height() - m_pConfigTemp->circRadius - 1;
			m_pLineEdit_CircCenter->setText(QString::number(circCenter));
		}
		if (circCenter < 0)
		{
			circCenter = 0;
			m_pLineEdit_CircCenter->setText(QString::number(circCenter));
		}
		m_pConfig->circCenter = circCenter;
	//}
	//else
	//{
	//	int ballRadius = str.toInt();
	//	if (ballRadius > m_polishedSurface(m_pSlider_SelectFrame->value()))
	//	{
	//		ballRadius = m_polishedSurface(m_pSlider_SelectFrame->value());
	//		m_pLineEdit_CircCenter->setText(QString::number(ballRadius));
	//	}
	//	if (ballRadius < 0)
	//	{
	//		ballRadius = 0;
	//		m_pLineEdit_CircCenter->setText(QString::number(ballRadius));
	//	}
	//	m_pConfig->ballRadius = ballRadius;
	//}
	
#ifdef OCT_NIRF
		if (m_pCheckBox_CircularizeImage->isChecked())
			if (m_pNirfDistCompDlg)
				if (m_pImageView_CircImage->getRender()->m_contour.length() > 0)
					m_pImageView_CircImage->getRender()->m_contour_offset = m_pConfig->circCenter;// (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? m_pConfig->circCenter :
				//(m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3) + m_polishedSurface(m_pSlider_SelectFrame->value()) - m_pConfig->ballRadius;
#endif

	getOctProjection(m_vectorOctImage, m_octProjection);
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());

	if (m_pCheckBox_ShowGuideLine->isChecked())
	{
		//int center = (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? m_pConfig->circCenter :
		//	(m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3) + m_polishedSurface(m_pSlider_SelectFrame->value()) - m_pConfig->ballRadius;
		//int polished_surface = (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? center - m_pConfig->circCenter : center + m_pConfig->ballRadius;
		int center = m_pConfig->circCenter;			
		int polished_surface = center - m_pConfig->circCenter;

#if (defined(OCT_FLIM) || defined(OCT_NIRF))
		m_pImageView_RectImage->setHorizontalLine(7, m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3, center, center + m_pConfig->sheathRadius,
			polished_surface, center + m_pConfigTemp->circRadius, center + m_pConfigTemp->circRadius - m_pConfig->ringThickness, m_pConfigTemp->n2ScansFFT / 2 + m_pConfigTemp->nScans / 3);
		m_pImageView_RectImage->setHorizontalLineColor(7, 0xff0000, 0x00ff00, 0xffff00, 0xff00ff, 0x00ff00, 0x00ff00, 0xff0000);
#else
		m_pImageView_RectImage->setHorizontalLine(6, m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3, center, center + m_pConfig->sheathRadius,
			polished_surface, center + m_pConfigTemp->circRadius, m_pConfigTemp->n2ScansFFT / 2 + m_pConfigTemp->nScans / 3);
		m_pImageView_RectImage->setHorizontalLineColor(6, 0xff0000, 0x00ff00, 0xffff00, 0xffff00, 0x00ff00, 0xff0000);
#endif
		//m_pImageView_CircImage->setCircle(2, m_pConfig->sheathRadius, (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? 1 : m_pConfig->ballRadius);		
		m_pImageView_CircImage->setCircle(2, m_pConfig->sheathRadius, 1);
	}	
}

void QResultTab::checkCircRadius(const QString &str)
{
	// Range and value test
	int circRadius = str.toInt();
	//if (!m_pToggleButton_FindPolishedSurfaces->isChecked())
	//{
		if (circRadius + m_pConfig->circCenter + 1 > m_pConfigTemp->n2ScansFFT)
		{
			circRadius = m_pConfigTemp->n2ScansFFT - m_pConfig->circCenter - 1;
			circRadius = (circRadius % 2) ? circRadius - 1 : circRadius;
			m_pLineEdit_CircRadius->setText(QString::number(circRadius));
		}
	//}
	//else
	//{
	//	int pol_surface_max;
	//	ippsMax_32s(m_polishedSurface.raw_ptr(), m_polishedSurface.length(), &pol_surface_max);
	//	pol_surface_max = (m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3) + pol_surface_max;

	//	if (circRadius + pol_surface_max + 1 > m_pConfigTemp->n2ScansFFT)
	//	{
	//		circRadius = m_pConfigTemp->n2ScansFFT - pol_surface_max - 1;
	//		circRadius = (circRadius % 2) ? circRadius - 1 : circRadius;
	//		m_pLineEdit_CircRadius->setText(QString::number(circRadius));
	//	}
	//}

#if (defined(OCT_NIRF) || defined(OCT_FLIM))
#ifndef TWO_CHANNEL_NIRF
	if (circRadius < m_pConfig->ringThickness)
	{
		circRadius = m_pConfig->ringThickness + 2;
		circRadius = (circRadius % 2) ? circRadius - 1 : circRadius;
		m_pLineEdit_CircRadius->setText(QString::number(circRadius));
	}
#else
	if (circRadius < 2 * m_pConfig->ringThickness)
	{
		circRadius = 2 * m_pConfig->ringThickness + 2;
		m_pLineEdit_CircRadius->setText(QString::number(circRadius));
}
#endif
#else
	if (circRadius < 2)
	{
		circRadius = 2;
		m_pLineEdit_CircRadius->setText(QString::number(circRadius));
	}
#endif
	if (circRadius % 2)
	{
		circRadius--;
		m_pLineEdit_CircRadius->setText(QString::number(circRadius));
	}
	if (m_pConfig->n2ScansFFT == m_pConfigTemp->n2ScansFFT) m_pConfig->circRadius = circRadius;
	m_pConfigTemp->circRadius = circRadius;

	// Reset rect image size
	m_pImageView_CircImage->resetSize(2 * circRadius, 2 * circRadius);

	// Create image visualization buffers
	ColorTable temp_ctable;

	if (m_pImgObjCircImage) delete m_pImgObjCircImage;
	m_pImgObjCircImage = new ImageObject(2 * circRadius, 2 * circRadius, temp_ctable.m_colorTableVector.at(m_pComboBox_OctColorTable->currentIndex()), m_pConfig->octDbGamma);

	// Create circ object
	if (m_pCirc)
	{
		delete m_pCirc;
#ifndef CUDA_ENABLED
		m_pCirc = new circularize(circRadius, m_pConfigTemp->nAlines, false);
#else
		m_pCirc = new CudaCircularize(circRadius, m_pConfigTemp->nAlines, m_pConfigTemp->n2ScansFFT);
#endif
	}

	// Set Dialog
	if (m_pSaveResultDlg)
		m_pSaveResultDlg->setCircRadius(circRadius);

	if (m_pLongitudinalViewDlg)
		m_pLongitudinalViewDlg->setLongiRadius(circRadius);
	
	// Renewal
	getOctProjection(m_vectorOctImage, m_octProjection);
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());

	if (m_pCheckBox_ShowGuideLine->isChecked())
	{
		//int center = (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? m_pConfig->circCenter :
		//	(m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3) + m_polishedSurface(m_pSlider_SelectFrame->value()) - m_pConfig->ballRadius;
		//int polished_surface = (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? center - m_pConfig->circCenter : center + m_pConfig->ballRadius;
		int center = m_pConfig->circCenter;			
		int polished_surface = center - m_pConfig->circCenter;

#if (defined(OCT_FLIM) || defined(OCT_NIRF))
		m_pImageView_RectImage->setHorizontalLine(7, m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3, center, center + m_pConfig->sheathRadius,
			polished_surface, center + m_pConfigTemp->circRadius, center + m_pConfigTemp->circRadius - m_pConfig->ringThickness, m_pConfigTemp->n2ScansFFT / 2 + m_pConfigTemp->nScans / 3);
		m_pImageView_RectImage->setHorizontalLineColor(7, 0xff0000, 0x00ff00, 0xffff00, 0xff00ff, 0x00ff00, 0x00ff00, 0xff0000);
#else
		m_pImageView_RectImage->setHorizontalLine(6, m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3, center, center + m_pConfig->sheathRadius,
			polished_surface, center + m_pConfigTemp->circRadius, m_pConfigTemp->n2ScansFFT / 2 + m_pConfigTemp->nScans / 3);
		m_pImageView_RectImage->setHorizontalLineColor(6, 0xff0000, 0x00ff00, 0xffff00, 0xffff00, 0x00ff00, 0xff0000);
#endif
		//m_pImageView_CircImage->setCircle(2, m_pConfig->sheathRadius, (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? 1 : m_pConfig->ballRadius);
		m_pImageView_CircImage->setCircle(2, m_pConfig->sheathRadius, 1);
	}
}

void QResultTab::checkSheathRadius(const QString &str)
{
	// Range and value test
	int sheathRadius = str.toInt(); 
	if (sheathRadius > m_pConfigTemp->circRadius - 2)
	{
		sheathRadius = m_pConfigTemp->circRadius - 2;
		m_pLineEdit_SheathRadius->setText(QString::number(sheathRadius));
	}
	if (sheathRadius < 1)
	{
		sheathRadius = 1;
		m_pLineEdit_SheathRadius->setText(QString::number(sheathRadius));
	}
	m_pConfig->sheathRadius = sheathRadius;
	
	// Renewal
	getOctProjection(m_vectorOctImage, m_octProjection);
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
}

#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
void QResultTab::checkRingThickness(const QString &str)
{
	// Range and value test
	int ringThickness = str.toInt();
	if (ringThickness > m_pConfigTemp->circRadius - 2)
	{
		ringThickness = m_pConfigTemp->circRadius - 2;
		m_pLineEdit_RingThickness->setText(QString::number(ringThickness));
	}
	if (ringThickness < 1)
	{
		ringThickness = 1;
		m_pLineEdit_RingThickness->setText(QString::number(ringThickness));
	}
	m_pConfig->ringThickness = ringThickness;
	
	// Create image visualization buffers
	ColorTable temp_ctable;

#ifdef OCT_FLIM
	if (m_pImgObjIntensity) delete m_pImgObjIntensity;
	m_pImgObjIntensity = new ImageObject(m_pConfigTemp->n4Alines4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
	if (m_pImgObjLifetime) delete m_pImgObjLifetime;
	m_pImgObjLifetime = new ImageObject(m_pConfigTemp->n4Alines4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(m_pComboBox_LifetimeColorTable->currentIndex()));
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	if (m_pImgObjNirf) delete m_pImgObjNirf;
	m_pImgObjNirf = new ImageObject(m_pConfigTemp->nAlines4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
#else
	if (m_pImgObjNirf1) delete m_pImgObjNirf1;
	m_pImgObjNirf1 = new ImageObject(m_pConfigTemp->nAlines4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
	if (m_pImgObjNirf2) delete m_pImgObjNirf2;
	m_pImgObjNirf2 = new ImageObject(m_pConfigTemp->nAlines4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE2));
#endif
#endif

	// Set Dialog
	if (m_pLongitudinalViewDlg)
		m_pLongitudinalViewDlg->setLongiRingThickness(ringThickness);

	// Renewal
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());

	if (m_pCheckBox_ShowGuideLine->isChecked())
	{
		int center = m_pConfig->circCenter;// (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? m_pConfig->circCenter :
			//(m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3) + m_polishedSurface(m_pSlider_SelectFrame->value()) - m_pConfig->ballRadius;
		int polished_surface = center - m_pConfig->circCenter; /// (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? center - m_pConfig->circCenter : center + m_pConfig->ballRadius;

#if (defined(OCT_FLIM) || defined(OCT_NIRF))
		m_pImageView_RectImage->setHorizontalLine(7, m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3, center, center + m_pConfig->sheathRadius,
			polished_surface, center + m_pConfigTemp->circRadius, center + m_pConfigTemp->circRadius - m_pConfig->ringThickness, m_pConfigTemp->n2ScansFFT / 2 + m_pConfigTemp->nScans / 3);
		m_pImageView_RectImage->setHorizontalLineColor(7, 0xff0000, 0x00ff00, 0xffff00, 0xff00ff, 0x00ff00, 0x00ff00, 0xff0000);
#else
		m_pImageView_RectImage->setHorizontalLine(6, m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3, center, center + m_pConfig->sheathRadius,
			polished_surface, center + m_pConfigTemp->circRadius, m_pConfigTemp->n2ScansFFT / 2 + m_pConfigTemp->nScans / 3);
		m_pImageView_RectImage->setHorizontalLineColor(6, 0xff0000, 0x00ff00, 0xffff00, 0xffff00, 0x00ff00, 0xff0000);
#endif
		m_pImageView_CircImage->setCircle(2, m_pConfig->sheathRadius, 1); // (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? 1 : m_pConfig->ballRadius);
	}
}
#endif

void QResultTab::adjustOctContrast()
{
	int min_dB = m_pLineEdit_OctDbMin->text().toInt();
	int max_dB = m_pLineEdit_OctDbMax->text().toInt();
	float gamma = m_pLineEdit_OctDbGamma->text().toFloat();
	int ctable_ind = m_pComboBox_OctColorTable->currentIndex();

	m_pConfig->octDbRange.min = min_dB;
	m_pConfig->octDbRange.max = max_dB;
	m_pConfig->octDbGamma = gamma;
	m_pConfig->octColorTable = ctable_ind;

	m_pImageView_RectImage->resetColormap(ColorTable::colortable(ctable_ind), gamma);
	m_pImageView_CircImage->resetColormap(ColorTable::colortable(ctable_ind), gamma);
	m_pImageView_OctProjection->resetColormap(ColorTable::colortable(ctable_ind), gamma);
	m_pImageView_ColorbarOctProjection->resetColormap(ColorTable::colortable(ctable_ind), gamma);

	if (m_pLongitudinalViewDlg)
		m_pLongitudinalViewDlg->getImageView()->resetColormap(ColorTable::colortable(ctable_ind), gamma);

	ColorTable temp_ctable;

    int rect_width4 = 0;
    if (m_pImgObjRectImage)
    {
        rect_width4 = m_pImgObjRectImage->getWidth();
        delete m_pImgObjRectImage;
    }
    m_pImgObjRectImage = new ImageObject(rect_width4, m_pImageView_RectImage->getRender()->m_pImage->height(), temp_ctable.m_colorTableVector.at(ctable_ind), gamma);

    int circ_width4 = 0;
    if (m_pImgObjCircImage)
    {
        circ_width4 = m_pImgObjCircImage->getWidth();
        delete m_pImgObjCircImage;
    }
    m_pImgObjCircImage = new ImageObject(circ_width4, m_pImageView_CircImage->getRender()->m_pImage->height(), temp_ctable.m_colorTableVector.at(ctable_ind), gamma);
	
	if (m_pLongitudinalViewDlg)
	{
		if (m_pLongitudinalViewDlg->m_pImgObjOctLongiImage)
			delete m_pLongitudinalViewDlg->m_pImgObjOctLongiImage;
		m_pLongitudinalViewDlg->m_pImgObjOctLongiImage = new ImageObject(((m_pConfigTemp->nFrames + 3) >> 2) << 2, 2 * m_pConfigTemp->circRadius, 
			temp_ctable.m_colorTableVector.at(ctable_ind), gamma);
	}

	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
}

#ifdef OCT_FLIM
void QResultTab::changeFlimCh(int)
{
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
}

void QResultTab::enableHsvEnhancingMode(bool)
{
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
}

void QResultTab::adjustFlimContrast()
{
	float min_intensity = m_pLineEdit_IntensityMin->text().toFloat();
	float max_intensity = m_pLineEdit_IntensityMax->text().toFloat();
	float min_lifetime = m_pLineEdit_LifetimeMin->text().toFloat();
	float max_lifetime = m_pLineEdit_LifetimeMax->text().toFloat();
	
	m_pConfig->flimIntensityRange.min = min_intensity;
	m_pConfig->flimIntensityRange.max = max_intensity;
	m_pConfig->flimLifetimeRange.min = min_lifetime;
	m_pConfig->flimLifetimeRange.max = max_lifetime;

	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
}

void QResultTab::changeLifetimeColorTable(int ctable_ind)
{
	m_pConfig->flimLifetimeColorTable = ctable_ind;

	m_pImageView_LifetimeMap->resetColormap(ColorTable::colortable(ctable_ind));
	m_pImageView_ColorbarLifetimeMap->resetColormap(ColorTable::colortable(ctable_ind));
	
	ColorTable temp_ctable;

	if (m_pImgObjLifetime) delete m_pImgObjLifetime;
	m_pImgObjLifetime = new ImageObject(m_pImageView_LifetimeMap->getRender()->m_pImage->width(), m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ctable_ind));
	if (m_pImgObjLifetimeMap) delete m_pImgObjLifetimeMap;
	m_pImgObjLifetimeMap = new ImageObject(m_pImageView_LifetimeMap->getRender()->m_pImage->width(), m_pImageView_LifetimeMap->getRender()->m_pImage->height(), temp_ctable.m_colorTableVector.at(ctable_ind));
	if (m_pImgObjHsvEnhancedMap) delete m_pImgObjHsvEnhancedMap;
	m_pImgObjHsvEnhancedMap = new ImageObject(m_pImageView_LifetimeMap->getRender()->m_pImage->width(), m_pImageView_LifetimeMap->getRender()->m_pImage->height(), temp_ctable.m_colorTableVector.at(ctable_ind));
	
	if (m_pLongitudinalViewDlg)
	{
		if (m_pLongitudinalViewDlg->m_pImgObjLifetime)
			delete m_pLongitudinalViewDlg->m_pImgObjLifetime;
		m_pLongitudinalViewDlg->m_pImgObjLifetime = new ImageObject(((m_pConfigTemp->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ctable_ind));

		if (m_pLongitudinalViewDlg->m_pImgObjHsvEnhanced)
			delete m_pLongitudinalViewDlg->m_pImgObjHsvEnhanced;
		m_pLongitudinalViewDlg->m_pImgObjHsvEnhanced = new ImageObject(((m_pConfigTemp->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ctable_ind));
	}

	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
}

void QResultTab::enableIntensityRatio(bool toggled)
{
	// Set enable state of associated widgets
	m_pLabel_EmissionChannel->setEnabled(!toggled);
	m_pComboBox_EmissionChannel->setEnabled(!toggled);

	ColorTable temp_ctable;

	if (toggled)
	{
		m_pImageView_IntensityMap->resetColormap(ColorTable::colortable::bwr);
		m_pImageView_ColorbarIntensityMap->resetColormap(ColorTable::colortable::bwr);

		if (m_pImgObjIntensity) delete m_pImgObjIntensity;
		m_pImgObjIntensity = new ImageObject(m_pImageView_IntensityMap->getRender()->m_pImage->width(), m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ColorTable::bwr));

		m_pLineEdit_IntensityMax->setDisabled(true); 
		m_pLineEdit_IntensityMin->setDisabled(true);
		m_pLineEdit_IntensityMax->setText(QString::number(2, 'f', 1));
		m_pLineEdit_IntensityMin->setText(QString::number(0, 'f', 1));
	}
	else
	{
		m_pImageView_IntensityMap->resetColormap(ColorTable::colortable::fire);
		m_pImageView_ColorbarIntensityMap->resetColormap(ColorTable::colortable::fire);

		if (m_pImgObjLifetime) delete m_pImgObjLifetime;
		m_pImgObjLifetime = new ImageObject(m_pImageView_LifetimeMap->getRender()->m_pImage->width(), m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ColorTable::fire));

		m_pLineEdit_IntensityMax->setEnabled(true);
		m_pLineEdit_IntensityMin->setEnabled(true);
		m_pLineEdit_IntensityMax->setText(QString::number(m_pConfig->flimIntensityRange.max, 'f', 1));
		m_pLineEdit_IntensityMin->setText(QString::number(m_pConfig->flimIntensityRange.min, 'f', 1));
	}
	
	// Visualization
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
}

void QResultTab::enableClassification(bool toggled)
{
	// Set enable state of associated widgets 	
	//if (toggled) m_pCheckBox_HsvEnhancedMap->setChecked(false);
	//m_pCheckBox_HsvEnhancedMap->setEnabled(!toggled);
	m_pLabel_EmissionChannel->setEnabled(!toggled);
	m_pComboBox_EmissionChannel->setEnabled(!toggled);
	m_pLabel_LifetimeColorTable->setEnabled(!toggled);
	m_pComboBox_LifetimeColorTable->setEnabled(!toggled);
	//m_pLineEdit_IntensityMax->setEnabled(!toggled);
	//m_pLineEdit_IntensityMin->setEnabled(!toggled);
	m_pLineEdit_LifetimeMax->setEnabled(!toggled);
	m_pLineEdit_LifetimeMin->setEnabled(!toggled);
	if (toggled)
		m_pImageView_ColorbarLifetimeMap->resetColormap(ColorTable::colortable::clf);
	else
		m_pImageView_ColorbarLifetimeMap->resetColormap(ColorTable::colortable(m_pConfig->flimLifetimeColorTable));
		
	// Visualization
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
}
#endif

#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
void QResultTab::adjustNirfContrast()
{
	float min_intensity = m_pLineEdit_NirfMin->text().toFloat();
	float max_intensity = m_pLineEdit_NirfMax->text().toFloat();

	m_pConfig->nirfRange.min = min_intensity;
    m_pConfig->nirfRange.max = max_intensity;

    if (m_pNirfEmissionProfileDlg)
        m_pNirfEmissionProfileDlg->getScope()->resetAxis({ 0, (double)m_nirfMap.size(0) }, { m_pConfig->nirfRange.min, m_pConfig->nirfRange.max });

	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
	if (m_pNirfDistCompDlg) m_pNirfDistCompDlg->updateCorrelation(m_pSlider_SelectFrame->value());
}
#else
void QResultTab::adjustNirfContrast1()
{
	float min_intensity = m_pLineEdit_NirfMin[0]->text().toFloat();
	float max_intensity = m_pLineEdit_NirfMax[0]->text().toFloat();

	m_pConfig->nirfRange[0].min = min_intensity;
	m_pConfig->nirfRange[0].max = max_intensity;

	if (m_pNirfEmissionProfileDlg)
		m_pNirfEmissionProfileDlg->getScope()->resetAxis({ 0, (double)m_nirfMap1.size(0) }, { min(m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[1].min), max(m_pConfig->nirfRange[0].max, m_pConfig->nirfRange[1].max) });

	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
}

void QResultTab::adjustNirfContrast2()
{
	float min_intensity = m_pLineEdit_NirfMin[1]->text().toFloat();
	float max_intensity = m_pLineEdit_NirfMax[1]->text().toFloat();

	m_pConfig->nirfRange[1].min = min_intensity;
	m_pConfig->nirfRange[1].max = max_intensity;

	if (m_pNirfEmissionProfileDlg)
		m_pNirfEmissionProfileDlg->getScope()->resetAxis({ 0, (double)m_nirfMap1.size(0) }, { min(m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[1].min), max(m_pConfig->nirfRange[0].max, m_pConfig->nirfRange[1].max) });

	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
}
#endif

void QResultTab::adjustNirfOffset(const QString &str)
{
	m_nirfOffset = str.toInt();
	if (m_nirfOffset > m_pConfigTemp->nAlines - 1)
	{
		m_nirfOffset = m_pConfigTemp->nAlines - 1;
		m_pLineEdit_NirfOffset->setText(QString::number(m_nirfOffset));
	}
	if (m_nirfOffset < -m_pConfigTemp->nAlines + 1)
	{
		m_nirfOffset = -m_pConfigTemp->nAlines + 1;
		m_pLineEdit_NirfOffset->setText(QString::number(m_nirfOffset));
	}
	m_pScrollBar_NirfOffset->setValue(m_nirfOffset);
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
	if (m_pNirfDistCompDlg) m_pNirfDistCompDlg->updateCorrelation(m_pSlider_SelectFrame->value());
}

void QResultTab::adjustNirfOffset(int offset)
{
	m_nirfOffset = offset;
	m_pLineEdit_NirfOffset->setText(QString::number(m_nirfOffset));
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
	if (m_pLongitudinalViewDlg)	m_pLongitudinalViewDlg->drawLongitudinalImage(m_pLongitudinalViewDlg->getCurrentAline());
	if (m_pNirfDistCompDlg) m_pNirfDistCompDlg->updateCorrelation(m_pSlider_SelectFrame->value());
}
#endif


void QResultTab::startProcessing()
{	
	m_pSlider_SelectFrame->setValue(0);

    if (m_pSaveResultDlg)
        m_pSaveResultDlg->close();
	if (m_pLongitudinalViewDlg)
		m_pLongitudinalViewDlg->close();
	if (m_pSpectroOCTDlg)
		m_pSpectroOCTDlg->close();
#ifdef OCT_FLIM
    if (m_pPulseReviewDlg)
        m_pPulseReviewDlg->close();
#endif
#ifdef OCT_NIRF
    if (m_pNirfEmissionProfileDlg)
        m_pNirfEmissionProfileDlg->close();
    if (m_pNirfDistCompDlg)
        m_pNirfDistCompDlg->close();
#ifdef TWO_CHANNEL_NIRF
	if (m_pNirfCrossTalkCompDlg)
		m_pNirfCrossTalkCompDlg->close();
#endif
#endif

	int id = m_pButtonGroup_DataSelection->checkedId();
	switch (id)
	{
	case IN_BUFFER_DATA:
		inBufferDataProcessing();
		break;
	case EXTERNAL_DATA:
		externalDataProcessing();
		break;
	}
}




void QResultTab::inBufferDataProcessing()
{
	std::thread t1([&]() {

		std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

		// Load configuration data ///////////////////////////////////////////////////////////////////
		Configuration *pConfig = m_pMainWnd->m_pConfiguration;
        if (m_pCheckBox_SingleFrame->isChecked()) pConfig->nFrames = 1;
		printf("Start in-buffer image processing... (Total nFrame: %d)\n", pConfig->nFrames);

		// Set Widgets //////////////////////////////////////////////////////////////////////////////
		emit setWidgets(false, pConfig);
        m_pImageView_RectImage->setUpdatesEnabled(false);
        m_pImageView_CircImage->setUpdatesEnabled(false);

        // Set Buffers //////////////////////////////////////////////////////////////////////////////
		setObjects(pConfig);

        int bufferSize = (false == m_pCheckBox_SingleFrame->isChecked()) ? PROCESSING_BUFFER_SIZE : 1;

        m_syncCh1Processing.allocate_queue_buffer(pConfig->nScans, pConfig->nAlines, bufferSize);
        m_syncCh2Processing.allocate_queue_buffer(pConfig->nScans, pConfig->nAlines, bufferSize);
#ifdef STANDALONE_OCT
#ifdef DUAL_CHANNEL
        m_syncCh1Visualization.allocate_queue_buffer(pConfig->n2ScansFFT, pConfig->nAlines, bufferSize);
        m_syncCh2Visualization.allocate_queue_buffer(pConfig->n2ScansFFT, pConfig->nAlines, bufferSize);
#endif
#endif
		// Set OCT FLIM Object //////////////////////////////////////////////////////////////////////
#ifdef OCT_FLIM
#ifndef CUDA_ENABLED
		OCTProcess* pOCT = m_pMainWnd->m_pStreamTab->m_pOCT;
#else
		CudaOCTProcess* pOCT = m_pMainWnd->m_pStreamTab->m_pOCT;
#endif
		m_pFLIMpost = m_pMainWnd->m_pStreamTab->m_pFLIM;
#elif defined (STANDALONE_OCT)
#ifndef CUDA_ENABLED
		OCTProcess* pOCT1 = m_pMainWnd->m_pStreamTab->m_pOCT1;
#else
		CudaOCTProcess* pOCT1 = m_pMainWnd->m_pStreamTab->m_pOCT1;
#endif
#ifdef DUAL_CHANNEL
#ifndef CUDA_ENABLED
		OCTProcess* pOCT2 = m_pMainWnd->m_pStreamTab->m_pOCT2;
#else
		CudaOCTProcess* pOCT2 = m_pMainWnd->m_pStreamTab->m_pOCT2;
#endif
#endif
#endif				
		// Data DeInterleaving & FLIM Process ///////////////////////////////////////////////////////
		std::thread deinterleave([&]() { deinterleavingInBuffer(pConfig); });

#ifdef OCT_FLIM
		// Ch1 Process //////////////////////////////////////////////////////////////////////////////
		std::thread ch1_proc([&]() { octProcessing(pOCT, pConfig); });

		// Ch2 Process //////////////////////////////////////////////////////////////////////////////		
		std::thread ch2_proc([&]() { flimProcessing(m_pFLIMpost, pConfig); });
#elif defined (STANDALONE_OCT)
		// Ch1 Process //////////////////////////////////////////////////////////////////////////////
		std::thread ch1_proc([&]() { octProcessing1(pOCT1, pConfig); });

		// Ch2 Process //////////////////////////////////////////////////////////////////////////////	
#ifdef DUAL_CHANNEL
		std::thread ch2_proc([&]() { octProcessing2(pOCT2, pConfig); });
#else
		std::thread ch2_proc([&]() { octProcessing2(pOCT1, pConfig); });
#endif
#ifdef DUAL_CHANNEL
		// Image Merge //////////////////////////////////////////////////////////////////////////////		
		std::thread img_merge([&]() { imageMerge(pConfig); });
#endif
#endif
		// Wait for threads end /////////////////////////////////////////////////////////////////////
		deinterleave.join();
		ch1_proc.join();
		ch2_proc.join();
#ifdef STANDALONE_OCT
#ifdef DUAL_CHANNEL
		img_merge.join();
#endif
#endif
		// Generate en face maps ////////////////////////////////////////////////////////////////////
		getOctProjection(m_vectorOctImage, m_octProjection);

#ifdef OCT_NIRF
		// Load NIRF data ///////////////////////////////////////////////////////////////////////////
		if (!m_pCheckBox_SingleFrame->isChecked())
		{
			m_pMemBuff->circulation_nirf(WRITING_BUFFER_SIZE - m_pMemBuff->m_nRecordedFrames);
#ifndef TWO_CHANNEL_NIRF
			m_nirfSignal = np::FloatArray(pConfig->nAlines * pConfig->nFrames);
#else
			m_nirfSignal1 = np::FloatArray(pConfig->nAlines * pConfig->nFrames);
			m_nirfSignal2 = np::FloatArray(pConfig->nAlines * pConfig->nFrames);
#endif

			double* buffer_nirf = nullptr;
			for (int i = 0; i < m_pMemBuff->m_nRecordedFrames; i++)
			{
				buffer_nirf = m_pMemBuff->pop_front_nirf();
#ifndef TWO_CHANNEL_NIRF
				ippsConvert_64f32f(buffer_nirf, &m_nirfSignal(pConfig->nAlines * i), pConfig->nAlines);
#else
				ippsConvert_64f32f(buffer_nirf, &m_nirfSignal1(pConfig->nAlines * i), pConfig->nAlines);
				ippsConvert_64f32f(buffer_nirf + pConfig->nAlines, &m_nirfSignal2(pConfig->nAlines * i), pConfig->nAlines);				
#endif
				m_pMemBuff->push_back_nirf(buffer_nirf);
			}
		}
#endif

		// Delete threading sync buffers ////////////////////////////////////////////////////////////
		m_syncCh1Processing.deallocate_queue_buffer();
		m_syncCh2Processing.deallocate_queue_buffer();
#ifdef STANDALONE_OCT
#ifdef DUAL_CHANNEL
		m_syncCh1Visualization.deallocate_queue_buffer();
		m_syncCh2Visualization.deallocate_queue_buffer();
#endif
#endif
		// Reset Widgets /////////////////////////////////////////////////////////////////////////////
		emit setWidgets(true, pConfig);

		// Visualization /////////////////////////////////////////////////////////////////////////////
		visualizeEnFaceMap(true);	
		visualizeImage(0);

		std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
		printf("elapsed time : %.2f sec\n\n", elapsed.count());
	});

	t1.detach();
}

void QResultTab::externalDataProcessing()
{	
	// Get path to read
	QString fileName = QFileDialog::getOpenFileName(nullptr, "Load external data", "", "Raw data (*.data)");
	if (fileName != "")
	{
		std::thread t1([&, fileName]() {

			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
			
			QFile file(fileName);
			if (false == file.open(QFile::ReadOnly))
				printf("[ERROR] Invalid external data!\n");
			else
			{
				// Read Ini File & Initialization ///////////////////////////////////////////////////////////
				m_fileName = fileName;

				QString fileTitle;
				for (int i = 0; i < fileName.length(); i++)
					if (fileName.at(i) == QChar('.')) fileTitle = fileName.left(i);
				for (int i = 0; i < fileName.length(); i++)
					if (fileName.at(i) == QChar('/')) m_path = fileName.left(i);

				QString iniName = fileTitle + ".ini";
				QString bgName = fileTitle + ".background";
				QString calibName = fileTitle + ".calibration";
#ifdef OCT_FLIM
				QString maskName = fileTitle + ".flim_mask";
#endif							
#ifdef OCT_NIRF
				QString nirfName = m_path + "/nirf.bin";
#endif
				qDebug() << iniName;

				if (m_pConfigTemp) delete m_pConfigTemp;
				m_pConfigTemp = new Configuration;

				m_pConfigTemp->getConfigFile(iniName);
				if (m_pCheckBox_UserDefinedAlines->isChecked())
				{
					m_pConfigTemp->nAlines = m_pLineEdit_UserDefinedAlines->text().toInt();
					m_pConfigTemp->n4Alines = m_pConfigTemp->nAlines / 4;
					m_pConfigTemp->nAlines4 = ((m_pConfigTemp->nAlines + 3) >> 2) << 2;
					m_pConfigTemp->n4Alines4 = ((m_pConfigTemp->n4Alines + 3) >> 2) << 2;
#ifdef OCT_FLIM
                    if (m_pConfigTemp->nAlines != m_pConfigTemp->nAlines4)
                    {
                        printf("[ERROR] nAlines should be a multiple of 4.\n");
                        file.close();
                        return;
                    }
#endif
					m_pConfigTemp->nFrameSize = m_pConfigTemp->nChannels * m_pConfigTemp->nScans * m_pConfigTemp->nAlines;
				}

				if (!m_pCheckBox_DiscomValue->isChecked())
				{
					m_pLineEdit_DiscomValue->setText(QString::number(m_pConfigTemp->octDiscomVal));
				}
				else
				{
					m_pConfigTemp->octDiscomVal = m_pLineEdit_DiscomValue->text().toInt();
				}

				m_pConfigTemp->circRadius = m_pConfig->circRadius;
				if (m_pConfigTemp->circRadius + m_pConfig->circCenter + 1 > m_pConfigTemp->n2ScansFFT)
				{
					m_pConfigTemp->circRadius = m_pConfigTemp->n2ScansFFT - m_pConfig->circCenter - 1;
					m_pConfigTemp->circRadius = (m_pConfigTemp->circRadius % 2) ? m_pConfigTemp->circRadius - 1 : m_pConfigTemp->circRadius;
				}
				
				m_pConfigTemp->nFrames = (int)(file.size() / (qint64)m_pConfigTemp->nChannels / (qint64)m_pConfigTemp->nScans / (qint64)m_pConfigTemp->nAlines / sizeof(uint16_t));
				if (m_pCheckBox_SingleFrame->isChecked()) m_pConfigTemp->nFrames = 1;
#ifdef OCT_NIRF
				if (m_pConfigTemp->erasmus)
					nirfName = m_path + "/NIRF.txt";
#endif
				printf("Start external image processing... (Total nFrame: %d)\n", m_pConfigTemp->nFrames);

				// Set Widgets //////////////////////////////////////////////////////////////////////////////
				emit setWidgets(false, m_pConfigTemp);
				m_pImageView_RectImage->setUpdatesEnabled(false);
				m_pImageView_CircImage->setUpdatesEnabled(false);

				// Set Buffers //////////////////////////////////////////////////////////////////////////////
				setObjects(m_pConfigTemp);

				int bufferSize = (false == m_pCheckBox_SingleFrame->isChecked()) ? PROCESSING_BUFFER_SIZE : 1;

				m_syncDeinterleaving.allocate_queue_buffer(m_pConfigTemp->nChannels * m_pConfigTemp->nScans, m_pConfigTemp->nAlines, bufferSize);
				m_syncCh1Processing.allocate_queue_buffer(m_pConfigTemp->nScans, m_pConfigTemp->nAlines, bufferSize);
#ifdef OCT_FLIM
				m_syncCh2Processing.allocate_queue_buffer(m_pConfigTemp->nScans * 4, m_pConfigTemp->n4Alines, bufferSize);
#elif defined (STANDALONE_OCT)
				m_syncCh2Processing.allocate_queue_buffer(m_pConfigTemp->nScans, m_pConfigTemp->nAlines, bufferSize);
#endif

#ifdef STANDALONE_OCT
#ifdef DUAL_CHANNEL
				m_syncCh1Visualization.allocate_queue_buffer(m_pConfigTemp->n2ScansFFT, m_pConfigTemp->nAlines, bufferSize);
				m_syncCh2Visualization.allocate_queue_buffer(m_pConfigTemp->n2ScansFFT, m_pConfigTemp->nAlines, bufferSize);
#endif
#endif
				// Set OCT FLIM Object //////////////////////////////////////////////////////////////////////
#ifdef OCT_FLIM
#ifndef CUDA_ENABLED
				OCTProcess* pOCT = new OCTProcess(m_pConfigTemp->nScans, m_pConfigTemp->nAlines);
				pOCT->loadCalibration(CH_1, calibName.toUtf8().constData(), bgName.toUtf8().constData(), m_pConfigTemp->erasmus);
				pOCT->changeDiscomValue(m_pConfigTemp->octDiscomVal);
#else
				CudaOCTProcess* pOCT = new CudaOCTProcess(m_pConfigTemp->nScans, m_pConfigTemp->nAlines);
				pOCT->loadCalibration(CH_1, calibName.toUtf8().constData(), bgName.toUtf8().constData(), m_pConfigTemp->erasmus);
				pOCT->changeDiscomValue(m_pConfigTemp->octDiscomVal);
				pOCT->initialize();
#endif
				if (m_pFLIMpost) if (m_pFLIMpost != m_pMainWnd->m_pStreamTab->m_pFLIM) delete m_pFLIMpost;
				m_pFLIMpost = new FLIMProcess;
				m_pFLIMpost->setParameters(m_pConfigTemp);
				m_pFLIMpost->_resize(np::Uint16Array2(m_pConfigTemp->nScans * 4, m_pConfigTemp->n4Alines), m_pFLIMpost->_params);
				m_pFLIMpost->loadMaskData(maskName);

#elif defined (STANDALONE_OCT)
#ifndef CUDA_ENABLED
				OCTProcess* pOCT1 = new OCTProcess(m_pConfigTemp->nScans, m_pConfigTemp->nAlines);
#ifndef K_CLOCKING
				pOCT1->loadCalibration(CH_1, calibName.toUtf8().constData(), bgName.toUtf8().constData(), m_pConfigTemp->erasmus);
#endif
				pOCT1->changeDiscomValue(m_pConfigTemp->octDiscomVal);
#else
				CudaOCTProcess* pOCT1 = new CudaOCTProcess(m_pConfigTemp->nScans, m_pConfigTemp->nAlines);
#ifndef K_CLOCKING
				pOCT1->loadCalibration(CH_1, calibName.toUtf8().constData(), bgName.toUtf8().constData(), m_pConfigTemp->erasmus);
#endif
				pOCT1->changeDiscomValue(m_pConfigTemp->octDiscomVal);
				pOCT1->initialize();
#endif
#ifdef DUAL_CHANNEL
#ifndef CUDA_ENABLED
				OCTProcess* pOCT2 = new OCTProcess(m_pConfigTemp->nScans, m_pConfigTemp->nAlines);
				pOCT1->loadCalibration(CH_2, calibName.toUtf8().constData(), bgName.toUtf8().constData(), m_pConfigTemp->erasmus);
				pOCT1->changeDiscomValue(m_pConfigTemp->octDiscomVal);
#else
				CudaOCTProcess* pOCT2 = new CudaOCTProcess(m_pConfigTemp->nScans, m_pConfigTemp->nAlines);
				pOCT1->loadCalibration(CH_2, calibName.toUtf8().constData(), bgName.toUtf8().constData(), m_pConfigTemp->erasmus);
				pOCT1->changeDiscomValue(m_pConfigTemp->octDiscomVal);
				pOCT2->initialize();
#endif
#endif
#endif				
				// Get external data ////////////////////////////////////////////////////////////////////////
				std::thread load_data([&]() { loadingRawData(&file, m_pConfigTemp); });

				// Data DeInterleaving & FLIM Process ///////////////////////////////////////////////////////
				std::thread deinterleave([&]() { deinterleaving(m_pConfigTemp); });

#ifdef OCT_FLIM
				// Ch1 Process //////////////////////////////////////////////////////////////////////////////
				std::thread ch1_proc([&]() { octProcessing(pOCT, m_pConfigTemp); });

				// Ch2 Process //////////////////////////////////////////////////////////////////////////////		
				std::thread ch2_proc([&]() { flimProcessing(m_pFLIMpost, m_pConfigTemp); });
#elif defined (STANDALONE_OCT)
				// Ch1 Process //////////////////////////////////////////////////////////////////////////////
				std::thread ch1_proc([&]() { octProcessing1(pOCT1, m_pConfigTemp); });

				// Ch2 Process //////////////////////////////////////////////////////////////////////////////	
#ifdef DUAL_CHANNEL
				std::thread ch2_proc([&]() { octProcessing2(pOCT2, m_pConfigTemp); });
#else
				std::thread ch2_proc([&]() { octProcessing2(pOCT1, m_pConfigTemp); });
#endif
#ifdef DUAL_CHANNEL
				// Image Merge //////////////////////////////////////////////////////////////////////////////		
				std::thread img_merge([&]() { imageMerge(&config); });
#endif
#endif
				// Wait for threads end /////////////////////////////////////////////////////////////////////
				load_data.join();
				deinterleave.join();
				ch1_proc.join();
				ch2_proc.join();
#ifdef STANDALONE_OCT
#ifdef DUAL_CHANNEL
				img_merge.join();
#endif
#endif
				// Generate en face maps ////////////////////////////////////////////////////////////////////
				getOctProjection(m_vectorOctImage, m_octProjection);

#ifdef OCT_NIRF
				// Load NIRF data ///////////////////////////////////////////////////////////////////////////
				if (!m_pCheckBox_SingleFrame->isChecked())
				{
					QFile nirfFile(nirfName);
					if (m_pConfigTemp->erasmus)
					{
						if (false == nirfFile.open(QFile::ReadOnly | QFile::Text))
							printf("[ERROR] There is no nirf data or you selected an invalid nirf data!\n");
						else
						{
#ifndef TWO_CHANNEL_NIRF
							QTextStream in(&nirfFile);
							int nLine = 0;
							while (!in.atEnd())
							{
								QString tempLine = in.readLine();
								nLine++;
							}

							int interlace_aline = (int)(round((double)nLine / (double)m_octProjection.length()));

							if (interlace_aline == 2) nLine /= 2;

							in.seek(0);
							m_nirfSignal = np::FloatArray(nLine);
							if (interlace_aline == 2)
							{
								for (int i = 0; i < nLine; i++)
								{
									QString line1 = in.readLine();
									QString line2 = in.readLine();
									m_nirfSignal.at(i) = (line1.toFloat() + line2.toFloat()) / 2.0f;
								}
							}
							else if (interlace_aline == 1)
							{
								for (int i = 0; i < nLine; i++)
								{
									QString line = in.readLine();
									m_nirfSignal.at(i) = line.toFloat();
								}
							}
							nirfFile.close();

							printf("NIRF data was successfully loaded...\n");
							m_pConfigTemp->nirf = true;
#endif
						}
					}
					else
					{
						if (false == nirfFile.open(QFile::ReadOnly))
							printf("[ERROR] Invalid NIRF data!\n");
						else
						{
							int ch = nirfFile.size() / sizeof(double) / (m_pConfigTemp->nAlines * m_pConfigTemp->nFrames);
#ifndef TWO_CHANNEL_NIRF
							m_nirfSignal = np::FloatArray(m_pConfigTemp->nAlines * m_pConfigTemp->nFrames);

							if (ch == 1)
							{
								np::DoubleArray nirf_data(m_pConfigTemp->nAlines * m_pConfigTemp->nFrames);
								nirfFile.read(reinterpret_cast<char *>(nirf_data.raw_ptr()), sizeof(double) * nirf_data.length());
								nirfFile.close();

								ippsConvert_64f32f(nirf_data.raw_ptr(), m_nirfSignal.raw_ptr(), nirf_data.length());
							}
							else if (ch == 2)
							{
								np::DoubleArray nirf_data(2 * m_pConfigTemp->nAlines);
								np::DoubleArray nirf_data1(m_pConfigTemp->nAlines);
								np::DoubleArray nirf_data2(m_pConfigTemp->nAlines);
								for (int i = 0; i < m_pConfigTemp->nFrames; i++)
								{
									nirfFile.read(reinterpret_cast<char *>(nirf_data.raw_ptr()), sizeof(double) * 2 * m_pConfigTemp->nAlines);
									ippsCplxToReal_64fc((const Ipp64fc*)nirf_data.raw_ptr(), nirf_data1, nirf_data2, m_pConfigTemp->nAlines);

									ippsConvert_64f32f(nirf_data1.raw_ptr(), m_nirfSignal.raw_ptr() + i * m_pConfigTemp->nAlines, m_pConfigTemp->nAlines);									
								}
								nirfFile.close();
							}							
#else
							//m_nirfSignal1 = np::FloatArray(m_pConfigTemp->nAlines * m_pConfigTemp->nFrames);
							//m_nirfSignal2 = np::FloatArray(m_pConfigTemp->nAlines * m_pConfigTemp->nFrames);
							//memset(m_nirfSignal2.raw_ptr(), 0, sizeof(float) * m_nirfSignal2.length());
							//
							//np::DoubleArray nirf_data(2 * m_pConfigTemp->nAlines);
							//np::DoubleArray nirf_data1(m_pConfigTemp->nAlines);
							//np::DoubleArray nirf_data2(m_pConfigTemp->nAlines);
							//for (int i = 0; i < m_pConfigTemp->nFrames; i++)
							//{
							//	if (ch == 2)
							//	{
							//		nirfFile.read(reinterpret_cast<char *>(nirf_data.raw_ptr()), sizeof(double) * 2 * m_pConfigTemp->nAlines);
							//		ippsCplxToReal_64fc((const Ipp64fc*)nirf_data.raw_ptr(), nirf_data1, nirf_data2, m_pConfigTemp->nAlines);

							//		ippsConvert_64f32f(nirf_data1.raw_ptr(), m_nirfSignal1.raw_ptr() + i * m_pConfigTemp->nAlines, m_pConfigTemp->nAlines);
							//		ippsConvert_64f32f(nirf_data2.raw_ptr(), m_nirfSignal2.raw_ptr() + i * m_pConfigTemp->nAlines, m_pConfigTemp->nAlines);
							//	}
							//	else if (ch == 1)
							//	{
							//		nirfFile.read(reinterpret_cast<char *>(nirf_data1.raw_ptr()), sizeof(double) * m_pConfigTemp->nAlines);
							//		ippsConvert_64f32f(nirf_data1.raw_ptr(), m_nirfSignal1.raw_ptr() + i * m_pConfigTemp->nAlines, m_pConfigTemp->nAlines);
							//	}
							//}
							//nirfFile.close();
							//m_pConfigTemp->_2ch_nirf = true;

							m_nirfSignal1 = np::FloatArray(m_pConfigTemp->nAlines * m_pConfigTemp->nFrames);
							m_nirfSignal2 = np::FloatArray(m_pConfigTemp->nAlines * m_pConfigTemp->nFrames);
							memset(m_nirfSignal2.raw_ptr(), 0, sizeof(float) * m_nirfSignal2.length());

							np::DoubleArray2 nirf_data(2 * MODULATION_FREQ * NIRF_SCANS, m_pConfigTemp->nAlines / MODULATION_FREQ);
							np::DoubleArray2 nirf_data1(MODULATION_FREQ * NIRF_SCANS, m_pConfigTemp->nAlines / MODULATION_FREQ);
							np::DoubleArray2 nirf_data2(MODULATION_FREQ * NIRF_SCANS, m_pConfigTemp->nAlines / MODULATION_FREQ);
							for (int i = 0; i < m_pConfigTemp->nFrames; i++)
							{
								nirfFile.read(reinterpret_cast<char *>(nirf_data.raw_ptr()), sizeof(double) * 2 * NIRF_SCANS * m_pConfigTemp->nAlines);
								ippsCplxToReal_64fc((const Ipp64fc*)nirf_data.raw_ptr(), nirf_data1, nirf_data2, NIRF_SCANS * m_pConfigTemp->nAlines);
									
								// Averaging				
								tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)(m_pConfig->nAlines / MODULATION_FREQ)),
									[&](const tbb::blocked_range<size_t>& r) {
									for (size_t j = r.begin(); j != r.end(); ++j)
									{
										Ipp64f m1, m2;
										ippsMean_64f(&nirf_data1(m_pConfigTemp->nirfIntegWindow[0].min, j), m_pConfigTemp->nirfIntegWindow[0].max - m_pConfigTemp->nirfIntegWindow[0].min, &m1);
										ippsMean_64f(&nirf_data2(m_pConfigTemp->nirfIntegWindow[1].min, j), m_pConfigTemp->nirfIntegWindow[1].max - m_pConfigTemp->nirfIntegWindow[1].min, &m2);

										for (int k = 0; k < MODULATION_FREQ; k++)
										{
											m_nirfSignal1(m_pConfig->nAlines * i + MODULATION_FREQ * j + k) = (float)m1;
											m_nirfSignal2(m_pConfig->nAlines * i + MODULATION_FREQ * j + k) = (float)m2;
										}											
									}
								});
							}
							nirfFile.close();
							m_pConfigTemp->_2ch_nirf = true;
#endif
							printf("NIRF data was successfully loaded...\n");
							m_pConfigTemp->nirf = true;
						}
					}
				}
#endif

				// Delete OCT FLIM Object & threading sync buffers //////////////////////////////////////////
#ifdef OCT_FLIM
				delete pOCT; 
#elif defined (STANDALONE_OCT)
				delete pOCT1; 
#ifdef DUAL_CHANNEL
				delete pOCT2;
#endif
#endif
				m_syncDeinterleaving.deallocate_queue_buffer();
				m_syncCh1Processing.deallocate_queue_buffer();
				m_syncCh2Processing.deallocate_queue_buffer();
#ifdef STANDALONE_OCT
#ifdef DUAL_CHANNEL
				m_syncCh1Visualization.deallocate_queue_buffer();
				m_syncCh2Visualization.deallocate_queue_buffer();
#endif
#endif
				// Reset Widgets /////////////////////////////////////////////////////////////////////////////
				emit setWidgets(true, m_pConfigTemp);
		
				// Visualization /////////////////////////////////////////////////////////////////////////////
                visualizeEnFaceMap(true);
				visualizeImage(0);
			}

			std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
			printf("elapsed time : %.2f sec\n\n", elapsed.count());
		});

		t1.detach();
	}	
}

void QResultTab::setWidgetsEnabled(bool enabled, Configuration* pConfig)
{
	if (enabled)
	{
		// Reset visualization widgets
		m_pImageView_RectImage->setEnabled(true);
        m_pImageView_RectImage->setUpdatesEnabled(true);
		m_pImageView_CircImage->setEnabled(true);
        m_pImageView_CircImage->setUpdatesEnabled(true);
		if (enabled)
		{
			m_pImageView_RectImage->setMagnDefault();
			m_pImageView_CircImage->setMagnDefault();
		}

        m_pImageView_RectImage->resetSize(pConfig->nAlines, pConfig->n2ScansFFT);
		m_pImageView_CircImage->resetSize(2 * m_pConfigTemp->circRadius, 2 * m_pConfigTemp->circRadius);

        m_pImageView_OctProjection->resetSize(pConfig->nAlines, pConfig->nFrames);
#ifdef OCT_FLIM
		m_pImageView_IntensityMap->resetSize(pConfig->n4Alines, pConfig->nFrames);
		m_pImageView_LifetimeMap->resetSize(pConfig->n4Alines, pConfig->nFrames);
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
        m_pImageView_NirfMap->resetSize(pConfig->nAlines, pConfig->nFrames);
#else
		m_pImageView_NirfMap1->resetSize(pConfig->nAlines, pConfig->nFrames);
		m_pImageView_NirfMap2->resetSize(pConfig->nAlines, pConfig->nFrames);
#endif
#endif
		// Reset widgets
		m_pLabel_OctProjection->setEnabled(true);
		m_pImageView_OctProjection->setEnabled(true);        
        m_pImageView_OctProjection->setUpdatesEnabled(true);
#ifdef OCT_FLIM
		m_pLabel_IntensityMap->setEnabled(true);
		m_pImageView_IntensityMap->setEnabled(true);        
        m_pImageView_IntensityMap->setUpdatesEnabled(true);
		m_pLabel_LifetimeMap->setEnabled(true);
		m_pImageView_LifetimeMap->setEnabled(true);        
        m_pImageView_LifetimeMap->setUpdatesEnabled(true);
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
		m_pLabel_NirfMap->setEnabled(true);
		m_pImageView_NirfMap->setEnabled(true);
		m_pImageView_NirfMap->setUpdatesEnabled(true);
#else
		m_pLabel_NirfMap1->setEnabled(true);
		m_pImageView_NirfMap1->setEnabled(true);
		m_pImageView_NirfMap1->setUpdatesEnabled(true);

		m_pLabel_NirfMap2->setEnabled(true);
		m_pImageView_NirfMap2->setEnabled(true);
		m_pImageView_NirfMap2->setUpdatesEnabled(true);
#endif
#endif
		m_pPushButton_StartProcessing->setEnabled(true);
        if (m_pRadioButton_External->isChecked() && (false == m_pCheckBox_SingleFrame->isChecked()))
            m_pPushButton_SaveResults->setEnabled(true);

		if ((m_pMemBuff->m_nRecordedFrames != 0) && (!m_pMemBuff->m_bIsSaved))
            m_pRadioButton_InBuffer->setEnabled(true);
		m_pRadioButton_External->setEnabled(true);

		m_pCheckBox_SingleFrame->setEnabled(true);
		m_pCheckBox_UserDefinedAlines->setEnabled(true);
		if (m_pCheckBox_UserDefinedAlines->isChecked())
			m_pLineEdit_UserDefinedAlines->setEnabled(true);
		m_pCheckBox_DiscomValue->setEnabled(true);
		if (m_pCheckBox_DiscomValue->isChecked())
			m_pLineEdit_DiscomValue->setEnabled(true);

		m_pProgressBar_PostProcessing->setFormat("");
        m_pProgressBar_PostProcessing->setValue(0);

		if (m_pCheckBox_CircularizeImage->isChecked())
			m_pToggleButton_MeasureDistance->setEnabled(true);
		m_pLabel_SelectFrame->setEnabled(true);
		m_pSlider_SelectFrame->setEnabled(true);
		m_pSlider_SelectFrame->setRange(0, pConfig->nFrames - 1);
		m_pSlider_SelectFrame->setValue(0);
		
#ifdef OCT_NIRF
		m_pPushButton_NirfDistanceCompensation->setEnabled(true);
#ifdef TWO_CHANNEL_NIRF
		m_pPushButton_NirfCrossTalkCompensation->setEnabled(true);
#endif
#endif
		m_pPushButton_LongitudinalView->setEnabled(true);
		m_pPushButton_SpectroscopicView->setEnabled(true);
		//m_pToggleButton_FindPolishedSurfaces->setChecked(false);
		//m_pToggleButton_FindPolishedSurfaces->setEnabled(true);
		m_pToggleButton_AutoContour->setEnabled(true);
		m_pCheckBox_CircularizeImage->setEnabled(true);
		m_pCheckBox_ShowGuideLine->setEnabled(true);
		m_pLabel_CircCenter->setEnabled(true);
		m_pLineEdit_CircCenter->setEnabled(true);
		m_pLabel_CircRadius->setEnabled(true);
		m_pLineEdit_CircRadius->setEnabled(true);		
		m_pLineEdit_CircRadius->setText(QString::number(m_pConfigTemp->circRadius));
		m_pLabel_SheathRadius->setEnabled(true);
		m_pLineEdit_SheathRadius->setEnabled(true);
#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
		m_pLabel_RingThickness->setEnabled(true);
		m_pLineEdit_RingThickness->setEnabled(true);
#endif		
		m_pLabel_OctColorTable->setEnabled(true);
		m_pComboBox_OctColorTable->setEnabled(true);
#ifdef OCT_FLIM
		m_pCheckBox_HsvEnhancedMap->setEnabled(true);
		m_pLabel_EmissionChannel->setEnabled(true);
		m_pComboBox_EmissionChannel->setEnabled(true);
		m_pLabel_LifetimeColorTable->setEnabled(true);
		m_pComboBox_LifetimeColorTable->setEnabled(true);
		m_pCheckBox_IntensityRatio->setEnabled(true);
		m_pCheckBox_Classification->setEnabled(true);
#endif
#ifdef OCT_NIRF
		if (pConfig->nirf)
		{
			m_pLabel_NirfOffset->setEnabled(true);
			m_pLineEdit_NirfOffset->setText(QString::number(0));
			m_pLineEdit_NirfOffset->setEnabled(true);
			m_pScrollBar_NirfOffset->setRange(-pConfig->nAlines + 1, pConfig->nAlines - 1);
			m_pScrollBar_NirfOffset->setValue(0);
			m_pScrollBar_NirfOffset->setEnabled(true);
		}
#endif
		m_pLabel_OctProjection->setEnabled(true);
		m_pLineEdit_OctDbMax->setEnabled(true);
		m_pLineEdit_OctDbMin->setEnabled(true);
		m_pLabel_OctDbGamma->setEnabled(true);
		m_pLineEdit_OctDbGamma->setEnabled(true);
#ifdef OCT_FLIM
		m_pLineEdit_IntensityMax->setEnabled(true);
		m_pLineEdit_IntensityMin->setEnabled(true);
		m_pLineEdit_LifetimeMax->setEnabled(true);
		m_pLineEdit_LifetimeMin->setEnabled(true);
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
		m_pLabel_NirfMap->setEnabled(true);
		m_pLineEdit_NirfMax->setEnabled(true);
		m_pLineEdit_NirfMin->setEnabled(true);
#else
		m_pLabel_NirfMap1->setEnabled(true);
		m_pLineEdit_NirfMax[0]->setEnabled(true);
		m_pLineEdit_NirfMin[0]->setEnabled(true);
		m_pLabel_NirfMap2->setEnabled(true);
		m_pLineEdit_NirfMax[1]->setEnabled(true);
		m_pLineEdit_NirfMin[1]->setEnabled(true);
#endif
#endif

#ifdef GALVANO_MIRROR
		m_pDeviceControlTab->setScrollBarRange(pConfig->nAlines);
		m_pDeviceControlTab->setScrollBarEnabled(true);
		m_pDeviceControlTab->setScrollBarValue(pConfig->galvoHorizontalShift);
#endif

		//m_pToggleButton_FindPolishedSurfaces->setChecked(true);
	}
	else
	{
		// Set widgets
        m_pImageView_RectImage->setDisabled(true);
        m_pImageView_RectImage->setUpdatesEnabled(false);
        m_pImageView_CircImage->setDisabled(true);
        m_pImageView_CircImage->setUpdatesEnabled(false);

		m_pLabel_OctProjection->setDisabled(true);
		m_pImageView_OctProjection->setDisabled(true);
        m_pImageView_OctProjection->setUpdatesEnabled(false);
#ifdef OCT_FLIM
		m_pLabel_IntensityMap->setDisabled(true);
		m_pImageView_IntensityMap->setDisabled(true);
        m_pImageView_IntensityMap->setUpdatesEnabled(false);
		m_pLabel_LifetimeMap->setDisabled(true);
		m_pImageView_LifetimeMap->setDisabled(true);
        m_pImageView_LifetimeMap->setUpdatesEnabled(false);
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
		m_pLabel_NirfMap->setDisabled(true);
		m_pImageView_NirfMap->setDisabled(true);
		m_pImageView_NirfMap->setUpdatesEnabled(false);
#else
		m_pLabel_NirfMap1->setDisabled(true);
		m_pImageView_NirfMap1->setDisabled(true);
		m_pImageView_NirfMap1->setUpdatesEnabled(false);

		m_pLabel_NirfMap2->setDisabled(true);
		m_pImageView_NirfMap2->setDisabled(true);
		m_pImageView_NirfMap2->setUpdatesEnabled(false);
#endif
#endif
		m_pPushButton_StartProcessing->setDisabled(true);
		m_pPushButton_SaveResults->setDisabled(true);
		m_pRadioButton_InBuffer->setDisabled(true);
		m_pRadioButton_External->setDisabled(true);

		m_pCheckBox_SingleFrame->setDisabled(true);
		m_pCheckBox_UserDefinedAlines->setDisabled(true);
		m_pLineEdit_UserDefinedAlines->setDisabled(true);
		m_pCheckBox_DiscomValue->setDisabled(true);
		m_pLineEdit_DiscomValue->setDisabled(true);

		int id = m_pButtonGroup_DataSelection->checkedId();
		switch (id)
		{
		case IN_BUFFER_DATA:
			m_pProgressBar_PostProcessing->setFormat("In-buffer data processing... %p%");
			break;
		case EXTERNAL_DATA:
			m_pProgressBar_PostProcessing->setFormat("External data processing... %p%");
			break;
		}		
        m_pProgressBar_PostProcessing->setRange(0, (pConfig->nFrames != 1) ? pConfig->nFrames - 1 : 1);
		m_pProgressBar_PostProcessing->setValue(0);

		m_pToggleButton_MeasureDistance->setDisabled(true);
		m_pLabel_SelectFrame->setDisabled(true);
		QString str; str.sprintf("Current Frame : %3d / %3d", 1, pConfig->nFrames);
		m_pLabel_SelectFrame->setText(str);

		m_pSlider_SelectFrame->setDisabled(true);

#ifdef OCT_NIRF
		m_pPushButton_NirfDistanceCompensation->setDisabled(true);
#ifdef TWO_CHANNEL_NIRF
		m_pPushButton_NirfCrossTalkCompensation->setDisabled(true);
#endif
#endif
		m_pPushButton_LongitudinalView->setDisabled(true);
		m_pPushButton_SpectroscopicView->setDisabled(true);
		//m_pToggleButton_FindPolishedSurfaces->setChecked(false);
		//m_pToggleButton_FindPolishedSurfaces->setDisabled(true);
		m_pToggleButton_AutoContour->setDisabled(true);
		m_pCheckBox_CircularizeImage->setDisabled(true);
		m_pCheckBox_ShowGuideLine->setDisabled(true);
		m_pLabel_CircCenter->setDisabled(true);
		m_pLineEdit_CircCenter->setDisabled(true);
		m_pLabel_CircRadius->setDisabled(true);
		m_pLineEdit_CircRadius->setDisabled(true);
		m_pLabel_SheathRadius->setDisabled(true);
		m_pLineEdit_SheathRadius->setDisabled(true);
#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
		m_pLabel_RingThickness->setDisabled(true);
		m_pLineEdit_RingThickness->setDisabled(true);
#endif
		m_pLabel_OctColorTable->setDisabled(true);
		m_pComboBox_OctColorTable->setDisabled(true);
#ifdef OCT_FLIM
		m_pCheckBox_HsvEnhancedMap->setDisabled(true);
		m_pLabel_EmissionChannel->setDisabled(true);
		m_pComboBox_EmissionChannel->setDisabled(true);
		m_pLabel_LifetimeColorTable->setDisabled(true);
		m_pComboBox_LifetimeColorTable->setDisabled(true);
		m_pCheckBox_IntensityRatio->setDisabled(true);
		m_pCheckBox_Classification->setDisabled(true);
#endif
#ifdef OCT_NIRF
		m_pLabel_NirfOffset->setDisabled(true);
		//m_pLineEdit_NirfOffset->setText(QString::number(0));
		m_pLineEdit_NirfOffset->setDisabled(true);
		//m_pScrollBar_NirfOffset->setValue(0);
		m_pScrollBar_NirfOffset->setDisabled(true);
#endif
		m_pLabel_OctProjection->setDisabled(true);
		m_pLineEdit_OctDbMax->setDisabled(true);
		m_pLineEdit_OctDbMin->setDisabled(true);
		m_pLabel_OctDbGamma->setDisabled(true);
		m_pLineEdit_OctDbGamma->setDisabled(true);
#ifdef OCT_FLIM
		m_pLineEdit_IntensityMax->setDisabled(true);
		m_pLineEdit_IntensityMin->setDisabled(true);
		m_pLineEdit_LifetimeMax->setDisabled(true);
		m_pLineEdit_LifetimeMin->setDisabled(true);
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
		m_pLabel_NirfMap->setDisabled(true);
		m_pLineEdit_NirfMax->setDisabled(true);
		m_pLineEdit_NirfMin->setDisabled(true);
#else
		m_pLabel_NirfMap1->setDisabled(true);
		m_pLineEdit_NirfMax[0]->setDisabled(true);
		m_pLineEdit_NirfMin[0]->setDisabled(true);
		m_pLabel_NirfMap2->setDisabled(true);
		m_pLineEdit_NirfMax[1]->setDisabled(true);
		m_pLineEdit_NirfMin[1]->setDisabled(true);
#endif
#endif

#ifdef GALVANO_MIRROR
		m_pDeviceControlTab->setScrollBarEnabled(false);
		m_pDeviceControlTab->setScrollBarValue(0);
#endif
	}
}

void QResultTab::setWidgetsEnabled(bool enabled)
{
	if (enabled)
	{
		// Reset visualization widgets
		m_pImageView_RectImage->setEnabled(true);
		m_pImageView_CircImage->setEnabled(true);
		if (enabled)
		{
			m_pImageView_RectImage->setMagnDefault();
			m_pImageView_CircImage->setMagnDefault();	
		}

		// Reset widgets
		m_pImageView_OctProjection->setEnabled(true);
#ifdef OCT_FLIM
		m_pImageView_IntensityMap->setEnabled(true);
		m_pImageView_LifetimeMap->setEnabled(true);
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
		m_pImageView_NirfMap->setEnabled(true);
#else
		m_pImageView_NirfMap1->setEnabled(true);
		m_pImageView_NirfMap2->setEnabled(true);
#endif
#endif

		m_pPushButton_StartProcessing->setEnabled(true);
		m_pPushButton_SaveResults->setEnabled(true);
		if ((m_pMemBuff->m_nRecordedFrames != 0) && (!m_pMemBuff->m_bIsSaved))
			m_pRadioButton_InBuffer->setEnabled(true);
		m_pRadioButton_External->setEnabled(true);

		m_pCheckBox_SingleFrame->setEnabled(true);
		m_pCheckBox_UserDefinedAlines->setEnabled(true);
		if (m_pCheckBox_UserDefinedAlines->isChecked())
			m_pLineEdit_UserDefinedAlines->setEnabled(true);
		m_pCheckBox_DiscomValue->setEnabled(true);
		if (m_pCheckBox_DiscomValue->isChecked())
			m_pLineEdit_DiscomValue->setEnabled(true);

		m_pProgressBar_PostProcessing->setFormat("");

		if (m_pCheckBox_CircularizeImage->isChecked())
			m_pToggleButton_MeasureDistance->setEnabled(true);
		m_pLabel_SelectFrame->setEnabled(true);
		m_pSlider_SelectFrame->setEnabled(true);
		m_pSlider_SelectFrame->setValue(0);

#ifdef OCT_NIRF
		m_pPushButton_NirfDistanceCompensation->setEnabled(true);
#ifdef TWO_CHANNEL_NIRF
		m_pPushButton_NirfCrossTalkCompensation->setEnabled(true);
#endif
#endif
		m_pPushButton_LongitudinalView->setEnabled(true);
		m_pPushButton_SpectroscopicView->setEnabled(true);
		//m_pToggleButton_FindPolishedSurfaces->setEnabled(true);
		m_pToggleButton_AutoContour->setEnabled(true);
		m_pCheckBox_CircularizeImage->setEnabled(true);
		m_pCheckBox_ShowGuideLine->setEnabled(true);
		m_pLabel_CircCenter->setEnabled(true);
		m_pLineEdit_CircCenter->setEnabled(true);
		m_pLabel_CircRadius->setEnabled(true);
		m_pLineEdit_CircRadius->setEnabled(true);
		m_pLabel_SheathRadius->setEnabled(true);
		m_pLineEdit_SheathRadius->setEnabled(true);
#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
		m_pLabel_RingThickness->setEnabled(true);
		m_pLineEdit_RingThickness->setEnabled(true);
#endif
		m_pLabel_OctColorTable->setEnabled(true);
		m_pComboBox_OctColorTable->setEnabled(true);
#ifdef OCT_FLIM
		m_pCheckBox_HsvEnhancedMap->setEnabled(true);
		m_pLabel_EmissionChannel->setEnabled(true);
		m_pComboBox_EmissionChannel->setEnabled(true);
		m_pLabel_LifetimeColorTable->setEnabled(true);
		m_pComboBox_LifetimeColorTable->setEnabled(true);
		m_pCheckBox_IntensityRatio->setEnabled(true);
		m_pCheckBox_Classification->setEnabled(true);
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
		if (m_nirfSignal.length() != 0)
		{
			m_pLabel_NirfOffset->setEnabled(true);
			m_pLineEdit_NirfOffset->setEnabled(true);
			m_pScrollBar_NirfOffset->setEnabled(true);
		}
#else
		if ((m_nirfSignal1.length() != 0) && (m_nirfSignal2.length() != 0))
		{
			m_pLabel_NirfOffset->setEnabled(true);
			m_pLineEdit_NirfOffset->setEnabled(true);
			m_pScrollBar_NirfOffset->setEnabled(true);
		}
#endif
#endif
		m_pLabel_OctProjection->setEnabled(true);
		m_pLineEdit_OctDbMax->setEnabled(true);
		m_pLineEdit_OctDbMin->setEnabled(true);
		m_pLabel_OctDbGamma->setEnabled(true);
		m_pLineEdit_OctDbGamma->setEnabled(true);
#ifdef OCT_FLIM
		m_pLineEdit_IntensityMax->setEnabled(true);
		m_pLineEdit_IntensityMin->setEnabled(true);
		m_pLineEdit_LifetimeMax->setEnabled(true);
		m_pLineEdit_LifetimeMin->setEnabled(true);
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
		m_pLabel_NirfMap->setEnabled(true);
		m_pLineEdit_NirfMax->setEnabled(true);
		m_pLineEdit_NirfMin->setEnabled(true);
#else
		m_pLabel_NirfMap1->setEnabled(true);
		m_pLineEdit_NirfMax[0]->setEnabled(true);
		m_pLineEdit_NirfMin[0]->setEnabled(true);
		m_pLabel_NirfMap2->setEnabled(true);
		m_pLineEdit_NirfMax[1]->setEnabled(true);
		m_pLineEdit_NirfMin[1]->setEnabled(true);
#endif
#endif
#ifdef GALVANO_MIRROR
		m_pDeviceControlTab->setScrollBarEnabled(true);
#endif
	}
	else
	{
		// Set widgets
        m_pImageView_RectImage->setDisabled(true);
        m_pImageView_CircImage->setDisabled(true);

		m_pImageView_OctProjection->setDisabled(true);
#ifdef OCT_FLIM
		m_pImageView_IntensityMap->setDisabled(true);
		m_pImageView_LifetimeMap->setDisabled(true);
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
		m_pImageView_NirfMap->setDisabled(true);
#else
		m_pImageView_NirfMap1->setDisabled(true);
		m_pImageView_NirfMap2->setDisabled(true);
#endif
#endif

		m_pPushButton_StartProcessing->setDisabled(true);
		m_pPushButton_SaveResults->setDisabled(true);
		m_pRadioButton_InBuffer->setDisabled(true);
		m_pRadioButton_External->setDisabled(true);

		m_pCheckBox_SingleFrame->setDisabled(true);
		m_pCheckBox_UserDefinedAlines->setDisabled(true);
		m_pLineEdit_UserDefinedAlines->setDisabled(true);
		m_pCheckBox_DiscomValue->setDisabled(true);
		m_pLineEdit_DiscomValue->setDisabled(true);

		m_pProgressBar_PostProcessing->setFormat("Saving results... %p%");
		m_pProgressBar_PostProcessing->setRange(0, (int)m_vectorOctImage.size() * 2 - 1);
		m_pProgressBar_PostProcessing->setValue(0);

		m_pToggleButton_MeasureDistance->setDisabled(true);
		m_pLabel_SelectFrame->setDisabled(true);
		m_pSlider_SelectFrame->setDisabled(true);

#ifdef OCT_NIRF
		m_pPushButton_NirfDistanceCompensation->setDisabled(true);
#ifdef TWO_CHANNEL_NIRF
		m_pPushButton_NirfCrossTalkCompensation->setDisabled(true);
#endif
#endif
		m_pPushButton_LongitudinalView->setDisabled(true);
		m_pPushButton_SpectroscopicView->setDisabled(true);
		//m_pToggleButton_FindPolishedSurfaces->setDisabled(true);
		m_pToggleButton_AutoContour->setDisabled(true);
		m_pCheckBox_CircularizeImage->setDisabled(true);
		m_pCheckBox_ShowGuideLine->setDisabled(true);
		m_pLabel_CircCenter->setDisabled(true);
		m_pLineEdit_CircCenter->setDisabled(true);
		m_pLabel_CircRadius->setDisabled(true);
		m_pLineEdit_CircRadius->setDisabled(true);
		m_pLabel_SheathRadius->setDisabled(true);
		m_pLineEdit_SheathRadius->setDisabled(true);
#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
		m_pLabel_RingThickness->setDisabled(true);
		m_pLineEdit_RingThickness->setDisabled(true);
#endif
		m_pLabel_OctColorTable->setDisabled(true);
		m_pComboBox_OctColorTable->setDisabled(true);
#ifdef OCT_FLIM
		m_pCheckBox_HsvEnhancedMap->setDisabled(true);
		m_pLabel_EmissionChannel->setDisabled(true);
		m_pComboBox_EmissionChannel->setDisabled(true);
		m_pLabel_LifetimeColorTable->setDisabled(true);
		m_pComboBox_LifetimeColorTable->setDisabled(true);
		m_pCheckBox_IntensityRatio->setDisabled(true);
		m_pCheckBox_Classification->setDisabled(true);
#endif
#ifdef OCT_NIRF
		m_pLabel_NirfOffset->setDisabled(true);
		m_pLineEdit_NirfOffset->setDisabled(true);
		m_pScrollBar_NirfOffset->setDisabled(true);
#endif
		m_pLabel_OctProjection->setDisabled(true);
		m_pLineEdit_OctDbMax->setDisabled(true);
		m_pLineEdit_OctDbMin->setDisabled(true);
		m_pLabel_OctDbGamma->setDisabled(true);
		m_pLineEdit_OctDbGamma->setDisabled(true);
#ifdef OCT_FLIM
		m_pLineEdit_IntensityMax->setDisabled(true);
		m_pLineEdit_IntensityMin->setDisabled(true);
		m_pLineEdit_LifetimeMax->setDisabled(true);
		m_pLineEdit_LifetimeMin->setDisabled(true);
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
		m_pLabel_NirfMap->setDisabled(true);
		m_pLineEdit_NirfMax->setDisabled(true);
		m_pLineEdit_NirfMin->setDisabled(true);
#else
		m_pLabel_NirfMap1->setDisabled(true);
		m_pLineEdit_NirfMax[0]->setDisabled(true);
		m_pLineEdit_NirfMin[0]->setDisabled(true);
		m_pLabel_NirfMap2->setDisabled(true);
		m_pLineEdit_NirfMax[1]->setDisabled(true);
		m_pLineEdit_NirfMin[1]->setDisabled(true);
#endif
#endif
#ifdef GALVANO_MIRROR
		m_pDeviceControlTab->setScrollBarEnabled(false);
#endif
	}
}

void QResultTab::setObjects(Configuration* pConfig)
{
	// Clear existed buffers
	std::vector<np::FloatArray2> clear_vector;
	clear_vector.swap(m_vectorOctImage);
#ifdef OCT_FLIM
	std::vector<np::FloatArray2> clear_vector1;
	clear_vector1.swap(m_intensityMap);
	std::vector<np::FloatArray2> clear_vector2;
	clear_vector2.swap(m_lifetimeMap);	
	std::vector<np::FloatArray2> clear_vector3;
	clear_vector3.swap(m_vectorPulseCrop);
	std::vector<np::FloatArray2> clear_vector4;
	clear_vector4.swap(m_vectorPulseMask);
#endif

	// Data buffers
	for (int i = 0; i < pConfig->nFrames; i++)
	{
		np::FloatArray2 buffer = np::FloatArray2(pConfig->n2ScansFFT, pConfig->nAlines4);
		m_vectorOctImage.push_back(buffer);
	}
	m_octProjection = np::FloatArray2(pConfig->nAlines4, pConfig->nFrames);
#ifdef OCT_FLIM
	for (int i = 0; i < 3; i++)
	{
		np::FloatArray2 intensity = np::FloatArray2(pConfig->n4Alines4, pConfig->nFrames);
		np::FloatArray2 lifetime = np::FloatArray2(pConfig->n4Alines4, pConfig->nFrames);
		m_intensityMap.push_back(intensity);
		m_lifetimeMap.push_back(lifetime);
	}
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	m_nirfSignal = np::FloatArray();
    m_nirfMap = np::FloatArray2(pConfig->nAlines, pConfig->nFrames);
#else
	m_nirfSignal1 = np::FloatArray();
	m_nirfMap1 = np::FloatArray2(pConfig->nAlines, pConfig->nFrames);
	m_nirfSignal2 = np::FloatArray();
	m_nirfMap2 = np::FloatArray2(pConfig->nAlines, pConfig->nFrames);
#endif
	m_nirfOffset = 0;
#endif
	m_contourMap = np::Uint16Array2(pConfig->nAlines, pConfig->nFrames);
	memset(m_contourMap, 0, sizeof(uint16_t) * m_contourMap.length());

	// Visualization buffers
	ColorTable temp_ctable;

	if (m_pImgObjRectImage) delete m_pImgObjRectImage;
    m_pImgObjRectImage = new ImageObject(pConfig->nAlines4, pConfig->n2ScansFFT, temp_ctable.m_colorTableVector.at(m_pComboBox_OctColorTable->currentIndex()), m_pConfig->octDbGamma);
	if (m_pImgObjCircImage) delete m_pImgObjCircImage;
    m_pImgObjCircImage = new ImageObject(2 * m_pConfigTemp->circRadius, 2 * m_pConfigTemp->circRadius, temp_ctable.m_colorTableVector.at(m_pComboBox_OctColorTable->currentIndex()), m_pConfig->octDbGamma);
#ifdef OCT_FLIM
	if (m_pImgObjIntensity) delete m_pImgObjIntensity;
	m_pImgObjIntensity = new ImageObject(pConfig->n4Alines4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
	if (m_pImgObjLifetime) delete m_pImgObjLifetime;
	m_pImgObjLifetime = new ImageObject(pConfig->n4Alines4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(m_pComboBox_LifetimeColorTable->currentIndex()));
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	if (m_pImgObjNirf) delete m_pImgObjNirf;
	m_pImgObjNirf = new ImageObject(pConfig->nAlines4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
#else
	if (m_pImgObjNirf1) delete m_pImgObjNirf1;
	m_pImgObjNirf1 = new ImageObject(pConfig->nAlines4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
	if (m_pImgObjNirf2) delete m_pImgObjNirf2;
	m_pImgObjNirf2 = new ImageObject(pConfig->nAlines4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE2));
#endif
#endif

	// En face map visualization buffers
	m_visOctProjection = np::Uint8Array2(pConfig->nAlines4, pConfig->nFrames);
#ifdef OCT_FLIM
	if (m_pImgObjIntensityMap) delete m_pImgObjIntensityMap;
	m_pImgObjIntensityMap = new ImageObject(pConfig->n4Alines4, pConfig->nFrames, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
	if (m_pImgObjLifetimeMap) delete m_pImgObjLifetimeMap;
	m_pImgObjLifetimeMap = new ImageObject(pConfig->n4Alines4, pConfig->nFrames, temp_ctable.m_colorTableVector.at(m_pComboBox_LifetimeColorTable->currentIndex()));
	if (m_pImgObjHsvEnhancedMap) delete m_pImgObjHsvEnhancedMap;
	m_pImgObjHsvEnhancedMap = new ImageObject(pConfig->n4Alines4, pConfig->nFrames, temp_ctable.m_colorTableVector.at(m_pComboBox_LifetimeColorTable->currentIndex()));
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	if (m_pImgObjNirfMap) delete m_pImgObjNirfMap;
	m_pImgObjNirfMap = new ImageObject(pConfig->nAlines4, pConfig->nFrames, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
#else
	if (m_pImgObjNirfMap1) delete m_pImgObjNirfMap1;
	m_pImgObjNirfMap1 = new ImageObject(pConfig->nAlines4, pConfig->nFrames, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
	if (m_pImgObjNirfMap2) delete m_pImgObjNirfMap2;
	m_pImgObjNirfMap2 = new ImageObject(pConfig->nAlines4, pConfig->nFrames, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
#endif
#endif

	// Circ & Medfilt objects
    if (m_pCirc) delete m_pCirc;
#ifndef CUDA_ENABLED
    m_pCirc = new circularize(m_pConfigTemp->circRadius, pConfig->nAlines, false);
#else
	m_pCirc = new CudaCircularize(m_pConfigTemp->circRadius, m_pConfigTemp->nAlines, m_pConfigTemp->n2ScansFFT);
#endif

	if (m_pMedfiltRect) delete m_pMedfiltRect;
	m_pMedfiltRect = new medfilt(pConfig->nAlines, pConfig->n2ScansFFT, 3, 3); // median filtering kernel
#ifdef OCT_FLIM
	if (m_pMedfiltIntensityMap) delete m_pMedfiltIntensityMap;
    if (m_pMedfiltLifetimeMap) delete m_pMedfiltLifetimeMap;
    if (m_pCheckBox_SingleFrame->isChecked() == false)
    {
        m_pMedfiltIntensityMap = new medfilt(pConfig->n4Alines, pConfig->nFrames, 5, 3);
        m_pMedfiltLifetimeMap = new medfilt(pConfig->n4Alines, pConfig->nFrames, 11, 7);
    }
    else
    {
        m_pMedfiltIntensityMap = new medfilt(pConfig->n4Alines, pConfig->nFrames, 5, 1);
        m_pMedfiltLifetimeMap = new medfilt(pConfig->n4Alines, pConfig->nFrames, 11, 1);
    }
#endif
#ifdef OCT_NIRF
    if (m_pMedfiltNirf) delete m_pMedfiltNirf;
    if (m_pCheckBox_SingleFrame->isChecked() == false)
        m_pMedfiltNirf = new medfilt(pConfig->nAlines4, pConfig->nFrames, 7, 3); // 7 3
    else
        m_pMedfiltNirf = new medfilt(pConfig->nAlines4, pConfig->nFrames, 7, 1);
#endif
}

void QResultTab::loadingRawData(QFile* pFile, Configuration* pConfig)
{
	int frameCount = 0;

	while (frameCount < pConfig->nFrames)
	{
		// Get buffers from threading queues
		uint16_t* frame_data = nullptr;
		do
		{
			{
				std::unique_lock<std::mutex> lock(m_syncDeinterleaving.mtx);
				if (!m_syncDeinterleaving.queue_buffer.empty())
				{
					frame_data = m_syncDeinterleaving.queue_buffer.front();
					m_syncDeinterleaving.queue_buffer.pop();
				}
			}

			if (frame_data)
			{
				// Read data from the external data 
				pFile->read(reinterpret_cast<char *>(frame_data), sizeof(uint16_t) * pConfig->nFrameSize);
				frameCount++;

				// Push the buffers to sync Queues``
				m_syncDeinterleaving.Queue_sync.push(frame_data);
			}
		} while (frame_data == nullptr);
	}
}

void QResultTab::deinterleaving(Configuration* pConfig)
{
	int frameCount = 0;
	while (frameCount < pConfig->nFrames)
	{
		// Get the buffer from the previous sync Queue
		uint16_t* frame_ptr = m_syncDeinterleaving.Queue_sync.pop();
		if (frame_ptr != nullptr)
		{
			// Get buffers from threading queues
			uint16_t* ch1_ptr = nullptr;
			uint16_t* ch2_ptr = nullptr;
			do
			{
				if (ch1_ptr == nullptr)
				{
					std::unique_lock<std::mutex> lock(m_syncCh1Processing.mtx);

					if (!m_syncCh1Processing.queue_buffer.empty())
					{
						ch1_ptr = m_syncCh1Processing.queue_buffer.front();
						m_syncCh1Processing.queue_buffer.pop();
					}
				}
				if (ch2_ptr == nullptr)
				{
					std::unique_lock<std::mutex> lock(m_syncCh2Processing.mtx);

					if (!m_syncCh2Processing.queue_buffer.empty())
					{
						ch2_ptr = m_syncCh2Processing.queue_buffer.front();
						m_syncCh2Processing.queue_buffer.pop();
					}
				}

				if ((ch1_ptr != nullptr) && (ch2_ptr != nullptr))
				{
					// Data deinterleaving
					int frame_length = pConfig->nFrameSize / pConfig->nChannels;
					if (pConfig->nChannels == 2)
						ippsCplxToReal_16sc((Ipp16sc *)frame_ptr, (Ipp16s *)ch1_ptr, (Ipp16s *)ch2_ptr, frame_length);
					else
						memcpy(ch1_ptr, frame_ptr, sizeof(uint16_t) * frame_length);
					//printf("ch1,2proc: %d %d [%d]\n", m_syncCh1Processing.queue_buffer.size(), m_syncCh2Processing.queue_buffer.size(), frameCount);
					frameCount++;

					// Push the buffers to sync Queues
					m_syncCh1Processing.Queue_sync.push(ch1_ptr);
					m_syncCh2Processing.Queue_sync.push(ch2_ptr);

					// Return (push) the buffer to the previous threading queue
					{
						std::unique_lock<std::mutex> lock(m_syncDeinterleaving.mtx);
						m_syncDeinterleaving.queue_buffer.push(frame_ptr);
					}
				}
			} while ((ch1_ptr == nullptr) || (ch2_ptr == nullptr));
		}
		else
		{
			printf("deinterleaving is halted.\n");
			break;
		}
	}
}

void QResultTab::deinterleavingInBuffer(Configuration* pConfig)
{
	MemoryBuffer* pMemBuff = m_pMainWnd->m_pOperationTab->m_pMemoryBuffer;
	pMemBuff->circulation(WRITING_BUFFER_SIZE - pConfig->nFrames);

	int frameCount = 0;
	while (frameCount < pConfig->nFrames)
	{
		// Pop front the buffer from the writing buffer queue
		uint16_t* frame_ptr = pMemBuff->pop_front();
	
		// Get buffers from threading queues
		uint16_t* ch1_ptr = nullptr;
		uint16_t* ch2_ptr = nullptr;
		do
		{
			if (ch1_ptr == nullptr)
			{
				std::unique_lock<std::mutex> lock(m_syncCh1Processing.mtx);

				if (!m_syncCh1Processing.queue_buffer.empty())
				{
					ch1_ptr = m_syncCh1Processing.queue_buffer.front();
					m_syncCh1Processing.queue_buffer.pop();
				}
			}
			if (ch2_ptr == nullptr)
			{
				std::unique_lock<std::mutex> lock(m_syncCh2Processing.mtx);

				if (!m_syncCh2Processing.queue_buffer.empty())
				{
					ch2_ptr = m_syncCh2Processing.queue_buffer.front();
					m_syncCh2Processing.queue_buffer.pop();
				}
			}

			if ((ch1_ptr != nullptr) && (ch2_ptr != nullptr))
			{
				// Data deinterleaving
				int frame_length = pConfig->nFrameSize / pConfig->nChannels;
				if (pConfig->nChannels == 2)
					ippsCplxToReal_16sc((Ipp16sc *)frame_ptr, (Ipp16s *)ch1_ptr, (Ipp16s *)ch2_ptr, frame_length);
				else
					memcpy(ch1_ptr, frame_ptr, sizeof(uint16_t) * frame_length);
				//printf("ch1,2proc: %d %d [%d]\n", m_syncCh1Processing.queue_buffer.size(), m_syncCh2Processing.queue_buffer.size(), frameCount);
				frameCount++;

				// Push the buffers to sync Queues
				m_syncCh1Processing.Queue_sync.push(ch1_ptr);
				m_syncCh2Processing.Queue_sync.push(ch2_ptr);
			}
		} while ((ch1_ptr == nullptr) || (ch2_ptr == nullptr));		

		// Push back the buffer to the writing buffer queue
		pMemBuff->push_back(frame_ptr);
	}
}

#ifdef OCT_FLIM
#ifndef CUDA_ENABLED
void QResultTab::octProcessing(OCTProcess* pOCT, Configuration* pConfig)
#else
void QResultTab::octProcessing(CudaOCTProcess* pOCT, Configuration* pConfig)
#endif
{	
	int frameCount = 0;
	while (frameCount < pConfig->nFrames)
	{
		// Get the buffer from the previous sync Queue
		uint16_t* fringe_data = m_syncCh1Processing.Queue_sync.pop();
		if (fringe_data != nullptr)
		{			
			// OCT processing
			(*pOCT)(m_vectorOctImage.at(frameCount).raw_ptr(), fringe_data);
			if (pConfig->erasmus)
			{
				IppiSize roi = { pConfig->n2ScansFFT, pConfig->nAlines };
				ippiMirror_32f_C1IR(m_vectorOctImage.at(frameCount), sizeof(float) * roi.width, roi, ippAxsVertical);
			}
			emit processedSingleFrame(frameCount);
			frameCount++;
						
			// Return (push) the buffer to the previous threading queue
			{
				std::unique_lock<std::mutex> lock(m_syncCh1Processing.mtx);
				m_syncCh1Processing.queue_buffer.push(fringe_data);
			}
		}		
		else
		{
			printf("octProcessing is halted.\n");
			break;
		}
	}
}

void QResultTab::flimProcessing(FLIMProcess* pFLIM, Configuration* pConfig)
{	
	np::Array<float, 2> itn(pConfig->n4Alines, 4); // temp intensity
	np::Array<float, 2> md(pConfig->n4Alines, 4); // temp mean delay
	np::Array<float, 2> ltm(pConfig->n4Alines, 3); // temp lifetime

	int frameCount = 0;
	while (frameCount < pConfig->nFrames)
	{
		// Get the buffer from the previous sync Queue
		uint16_t* pulse_data = m_syncCh2Processing.Queue_sync.pop();
		if (pulse_data != nullptr)
		{
			// FLIM Process
			np::Uint16Array2 pulse(pulse_data, pConfig->nScans * 4, pConfig->n4Alines);
			(*pFLIM)(itn, md, ltm, pulse);

			// Copy for Pulse Review
			np::Array<float, 2> crop(pFLIM->_resize.nx, pFLIM->_resize.ny);
			np::Array<float, 2> mask(pFLIM->_resize.nx, pFLIM->_resize.ny);
			memcpy(crop, pFLIM->_resize.crop_src, crop.length() * sizeof(float));
			memcpy(mask, pFLIM->_resize.mask_src, mask.length() * sizeof(float));
			m_vectorPulseCrop.push_back(crop);
			m_vectorPulseMask.push_back(mask);

			// Intensity compensation
			for (int i = 0; i < 3; i++)
				ippsDivC_32f_I(pConfig->flimIntensityComp[i], &itn(0, i + 1), pConfig->n4Alines);

			// Copy for Intensity & Lifetime
			memcpy(&m_intensityMap.at(0)(0, frameCount), &itn(0, 1), sizeof(float) * pConfig->n4Alines);
			memcpy(&m_intensityMap.at(1)(0, frameCount), &itn(0, 2), sizeof(float) * pConfig->n4Alines);
			memcpy(&m_intensityMap.at(2)(0, frameCount), &itn(0, 3), sizeof(float) * pConfig->n4Alines);
			memcpy(&m_lifetimeMap.at(0)(0, frameCount), &ltm(0, 0), sizeof(float) * pConfig->n4Alines);
			memcpy(&m_lifetimeMap.at(1)(0, frameCount), &ltm(0, 1), sizeof(float) * pConfig->n4Alines);
			memcpy(&m_lifetimeMap.at(2)(0, frameCount), &ltm(0, 2), sizeof(float) * pConfig->n4Alines);			
			frameCount++;

			// Return (push) the buffer to the previous threading queue
			{
				std::unique_lock<std::mutex> lock(m_syncCh2Processing.mtx);
				m_syncCh2Processing.queue_buffer.push(pulse_data);
			}
		}
		else
		{
			printf("flimProcessing is halted.\n");
			break;
		}
	}
}
#elif defined (STANDALONE_OCT)
#ifndef CUDA_ENABLED
void QResultTab::octProcessing1(OCTProcess* pOCT, Configuration* pConfig)
#else
void QResultTab::octProcessing1(CudaOCTProcess* pOCT, Configuration* pConfig)
#endif
{
	int frameCount = 0;
	while (frameCount < pConfig->nFrames)
	{
		// Get the buffer from the previous sync Queue
		uint16_t* fringe_data = m_syncCh1Processing.Queue_sync.pop();		
		if (fringe_data != nullptr)
		{
#ifdef DUAL_CHANNEL
			// Get buffers from threading queues
			float* res_data = nullptr;
			do
			{
				{
					std::unique_lock<std::mutex> lock(m_syncCh1Visualization.mtx);
					if (!m_syncCh1Visualization.queue_buffer.empty())
					{
						res_data = m_syncCh1Visualization.queue_buffer.front();
						m_syncCh1Visualization.queue_buffer.pop();
					}
				}

				if (res_data != nullptr)
				{
					// OCT processing
					(*pOCT)(res_data, fringe_data, "linear");
#else
					(*pOCT)(m_vectorOctImage.at(frameCount), fringe_data);
#if defined(OCT_VERTICAL_MIRRORING)
					IppiSize roi = { pConfig->n2ScansFFT, pConfig->nAlines };
					if (!pConfig->oldUhs)
						ippiMirror_32f_C1IR(m_vectorOctImage.at(frameCount), sizeof(float) * roi.width, roi, ippAxsVertical);
#endif
					if (pConfig->erasmus)
					{
						IppiSize roi = { pConfig->n2ScansFFT, pConfig->nAlines };
						if (!pConfig->oldUhs)
							ippiMirror_32f_C1IR(m_vectorOctImage.at(frameCount), sizeof(float) * roi.width, roi, ippAxsVertical);
					}
					emit processedSingleFrame(frameCount);
#endif
					//printf("ch1vis: %d [%d]\n", m_syncCh2Visualization.queue_buffer.size(), frameCount);
					frameCount++;

#ifdef DUAL_CHANNEL
					// Push the buffers to sync Queues
					m_syncCh1Visualization.Queue_sync.push(res_data);
#endif
					// Return (push) the buffer to the previous threading queue
					{
						std::unique_lock<std::mutex> lock(m_syncCh1Processing.mtx);
						m_syncCh1Processing.queue_buffer.push(fringe_data);
					}
#ifdef DUAL_CHANNEL
				}
			} while (res_data == nullptr);
#endif
		}
		else
		{
			printf("octProcessing1 is halted.\n");
			break;
		}
	}
}

#ifndef CUDA_ENABLED
void QResultTab::octProcessing2(OCTProcess* pOCT, Configuration* pConfig)
#else
void QResultTab::octProcessing2(CudaOCTProcess* pOCT, Configuration* pConfig)
#endif
{
	int frameCount = 0;
	while (frameCount < pConfig->nFrames)
	{
		// Get the buffer from the previous sync Queue
		uint16_t* fringe_data = m_syncCh2Processing.Queue_sync.pop();
		if (fringe_data != nullptr)
		{
#ifdef DUAL_CHANNEL
			// Get buffers from threading queues
			float* res_data = nullptr;
			do
			{
				{
					std::unique_lock<std::mutex> lock(m_syncCh2Visualization.mtx);
					if (!m_syncCh2Visualization.queue_buffer.empty())
					{
						res_data = m_syncCh2Visualization.queue_buffer.front();
						m_syncCh2Visualization.queue_buffer.pop();
					}
				}

				if (res_data != nullptr)
				{
					// OCT processing
					(*pOCT)(res_data, fringe_data, "linear");
#endif
					//printf("ch2vis: %d [%d]\n", m_syncCh2Visualization.queue_buffer.size(), frameCount);
					frameCount++;

#ifdef DUAL_CHANNEL
					// Push the buffers to sync Queues
					m_syncCh2Visualization.Queue_sync.push(res_data);
#endif
					// Return (push) the buffer to the previous threading queue
					{
						std::unique_lock<std::mutex> lock(m_syncCh2Processing.mtx);
						m_syncCh2Processing.queue_buffer.push(fringe_data);
					}
#ifdef DUAL_CHANNEL
				}
			} while (res_data == nullptr);
#endif
		}
		else
		{
			printf("octProcessing2 is halted.\n");
			break;
		}
	}
#ifndef DUAL_CHANNEL
    (void)pOCT;
#endif
}

#ifdef DUAL_CHANNEL
void QResultTab::imageMerge(Configuration* pConfig)
{
	int frameCount = 0;
	while (frameCount < pConfig->nFrames)
	{
		// Get the buffer from the previous sync Queue
		float* res1_data = m_syncCh1Visualization.Queue_sync.pop();
		float* res2_data = m_syncCh2Visualization.Queue_sync.pop();
		if ((res1_data != nullptr) && (res2_data != nullptr))
		{
			// Body
			np::FloatArray2 temp(pConfig->n2ScansFFT, pConfig->nAlines);
			ippsAdd_32f(res1_data, res2_data, temp.raw_ptr(), temp.length());
			ippsLog10_32f_A11(temp.raw_ptr(), m_vectorOctImage.at(frameCount), temp.length());
			ippsMulC_32f_I(10.0f, m_vectorOctImage.at(frameCount), temp.length());
			emit processedSingleFrame(frameCount++);

			// Return (push) the buffer to the previous threading queue
			{
				std::unique_lock<std::mutex> lock(m_syncCh1Visualization.mtx);
				m_syncCh1Visualization.queue_buffer.push(res1_data);
			}
			{
				std::unique_lock<std::mutex> lock(m_syncCh2Visualization.mtx);
				m_syncCh2Visualization.queue_buffer.push(res2_data);
			}
		}
		else
		{
			printf("imageMerge is halted.\n");
			break;
		}
	}
}
#endif
#endif

void QResultTab::getOctProjection(std::vector<np::FloatArray2>& vecImg, np::FloatArray2& octProj)
{
	tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)vecImg.size()),
		[&](const tbb::blocked_range<size_t>& r) {
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			//int center = (!m_pToggleButton_FindPolishedSurfaces->isChecked()) ? m_pConfig->circCenter :
			//	(m_pConfigTemp->n2ScansFFT / 2 - m_pConfigTemp->nScans / 3) + m_polishedSurface((int)i) - m_pConfig->ballRadius;
			int center = m_pConfig->circCenter;
			int len = m_pConfigTemp->circRadius - m_pConfig->sheathRadius;

			float maxVal;
			for (int j = 0; j < octProj.size(0); j++)
			{
				ippsMax_32f(&vecImg.at((int)i)(center + m_pConfig->sheathRadius, j), len, &maxVal);
				octProj(j, (int)i) = maxVal;
			}
		}
	});
}