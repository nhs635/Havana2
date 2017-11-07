
#include "QResultTab.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QOperationTab.h>
#include <Havana2/QStreamTab.h>
#include <Havana2/Viewer/QImageView.h>
#include <Havana2/Dialog/SaveResultDlg.h>
#ifdef OCT_FLIM
#include <Havana2/Dialog/PulseReviewDlg.h>
#endif

#include <MemoryBuffer/MemoryBuffer.h>

#include <DataProcess/OCTProcess/OCTProcess.h>
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
	m_pImgObjRectImage(nullptr), m_pImgObjCircImage(nullptr), m_pCirc(nullptr),
	m_pMedfiltRect(nullptr), 
	m_pSaveResultDlg(nullptr)
#ifdef OCT_FLIM
	, m_pFLIMpost(nullptr), m_pPulseReviewDlg(nullptr)
	, m_pImgObjIntensity(nullptr), m_pImgObjLifetime(nullptr)
	, m_pImgObjIntensityMap(nullptr), m_pImgObjLifetimeMap(nullptr), m_pImgObjHsvEnhancedMap(nullptr)
	, m_pMedfiltIntensityMap(nullptr), m_pMedfiltLifetimeMap(nullptr)
#endif
#ifdef OCT_NIRF
	, m_pImgObjNirf(nullptr), m_pImgObjNirfMap(nullptr), m_nirfOffset(0)
#endif
{
	// Set main window objects
	m_pMainWnd = (MainWindow*)parent;
	m_pConfig = m_pMainWnd->m_pConfiguration;
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
    bool rgb_used = false;
#ifdef OCT_NIRF
	rgb_used = true;
#endif
#endif
    m_pImageView_RectImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), m_pConfig->nAlines, m_pConfig->n2ScansFFT, rgb_used);
    m_pImageView_RectImage->setMinimumWidth(600);
	m_pImageView_RectImage->setDisabled(true);
	m_pImageView_RectImage->setMovedMouseCallback([&](QPoint& p) { m_pMainWnd->m_pStatusLabel_ImagePos->setText(QString("(%1, %2)").arg(p.x(), 4).arg(p.y(), 4)); });
#ifdef OCT_FLIM
	m_pImageView_RectImage->setDoubleClickedMouseCallback([&]() { createPulseReviewDlg(); });
#endif
	m_pImageView_RectImage->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

    m_pImageView_CircImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), 2 * CIRC_RADIUS, 2 * CIRC_RADIUS, rgb_used);
    m_pImageView_CircImage->setMinimumWidth(600);
	m_pImageView_CircImage->setDisabled(true);
	m_pImageView_CircImage->setMovedMouseCallback([&](QPoint& p) { m_pMainWnd->m_pStatusLabel_ImagePos->setText(QString("(%1, %2)").arg(p.x(), 4).arg(p.y(), 4)); });
#ifdef OCT_FLIM
	m_pImageView_CircImage->setDoubleClickedMouseCallback([&]() { createPulseReviewDlg(); });
#endif
	m_pImageView_CircImage->setSquare(true);
	m_pImageView_CircImage->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
	m_pImageView_CircImage->setVisible(false);

	QLabel *pNullLabel = new QLabel("", this);
	pNullLabel->setMinimumWidth(600);
	pNullLabel->setFixedHeight(0);
	pNullLabel->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

	// Create image view buffers
	ColorTable temp_ctable;
	m_pImgObjRectImage = new ImageObject(m_pConfig->nAlines4, m_pConfig->n2ScansFFT, temp_ctable.m_colorTableVector.at(m_pConfig->octColorTable));
	m_pImgObjCircImage = new ImageObject(2 * CIRC_RADIUS, 2 * CIRC_RADIUS, temp_ctable.m_colorTableVector.at(m_pConfig->octColorTable));

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
	connect(this, SIGNAL(makeRgb(ImageObject*, ImageObject*, ImageObject*)),
		this, SLOT(constructRgbImage(ImageObject*, ImageObject*, ImageObject*)));
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
	if (m_pImgObjNirf) delete m_pImgObjNirf;
	if (m_pImgObjNirfMap) delete m_pImgObjNirfMap;
#endif
	if (m_pMedfiltRect) delete m_pMedfiltRect;
#ifdef OCT_FLIM
	if (m_pMedfiltIntensityMap) delete m_pMedfiltIntensityMap;
	if (m_pMedfiltLifetimeMap) delete m_pMedfiltLifetimeMap;
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
	m_pLineEdit_CircCenter->setText(QString::number(m_pConfig->circCenter));

	m_pComboBox_OctColorTable->setCurrentIndex(m_pConfig->octColorTable);
	m_pLineEdit_OctDbMax->setText(QString::number(m_pConfig->octDbRange.max));
	m_pLineEdit_OctDbMin->setText(QString::number(m_pConfig->octDbRange.min));

#ifdef OCT_FLIM
	m_pComboBox_LifetimeColorTable->setCurrentIndex(m_pConfig->flimLifetimeColorTable);
	m_pLineEdit_IntensityMax->setText(QString::number(m_pConfig->flimIntensityRange.max, 'f', 1));
	m_pLineEdit_IntensityMin->setText(QString::number(m_pConfig->flimIntensityRange.min, 'f', 1));
	m_pLineEdit_LifetimeMax->setText(QString::number(m_pConfig->flimLifetimeRange.max, 'f', 1));
	m_pLineEdit_LifetimeMin->setText(QString::number(m_pConfig->flimLifetimeRange.min, 'f', 1));
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

	m_pLabel_DiscomValue = new QLabel("    Discom Value", this);

	m_pLineEdit_DiscomValue = new QLineEdit(this);
	m_pLineEdit_DiscomValue->setText(QString::number(m_pConfig->octDiscomVal));
	m_pLineEdit_DiscomValue->setFixedWidth(30);
	m_pLineEdit_DiscomValue->setAlignment(Qt::AlignCenter);

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
	pHBoxLayout_SingleFrameDiscomValue->addWidget(m_pLabel_DiscomValue);
	pHBoxLayout_SingleFrameDiscomValue->addWidget(m_pLineEdit_DiscomValue);

	pGridLayout_DataLoadingWriting->addItem(pHBoxLayout_UserDefined, 2, 1, 1, 2);
	pGridLayout_DataLoadingWriting->addItem(pHBoxLayout_SingleFrameDiscomValue, 3, 1, 1, 2);

	pGridLayout_DataLoadingWriting->addWidget(m_pProgressBar_PostProcessing, 4, 1, 1, 2);

	pGridLayout_DataLoadingWriting->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 3, 3, 1);

    m_pGroupBox_DataLoadingWriting->setLayout(pGridLayout_DataLoadingWriting);

	// Connect signal and slot
	connect(m_pButtonGroup_DataSelection, SIGNAL(buttonClicked(int)), this, SLOT(changeDataSelection(int)));
	connect(m_pPushButton_StartProcessing, SIGNAL(clicked(bool)), this, SLOT(startProcessing(void)));
	connect(m_pPushButton_SaveResults, SIGNAL(clicked(bool)), this, SLOT(createSaveResultDlg()));
	connect(m_pCheckBox_UserDefinedAlines, SIGNAL(toggled(bool)), this, SLOT(enableUserDefinedAlines(bool)));

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
	
    // Create checkboxs for OCT operation
    m_pCheckBox_ShowGuideLine = new QCheckBox(this);
    m_pCheckBox_ShowGuideLine->setText("Show Guide Line");
	m_pCheckBox_ShowGuideLine->setDisabled(true);
    m_pCheckBox_CircularizeImage = new QCheckBox(this);
    m_pCheckBox_CircularizeImage->setText("Circularize Image");
	m_pCheckBox_CircularizeImage->setDisabled(true);

    // Create widgets for OCT circularizing
    m_pLineEdit_CircCenter = new QLineEdit(this);
    m_pLineEdit_CircCenter->setFixedWidth(30);
    m_pLineEdit_CircCenter->setText(QString::number(m_pConfig->circCenter));
	m_pLineEdit_CircCenter->setAlignment(Qt::AlignCenter);
	m_pLineEdit_CircCenter->setDisabled(true);
    m_pLabel_CircCenter = new QLabel("Circ Center", this);
    m_pLabel_CircCenter->setBuddy(m_pLabel_CircCenter);
	m_pLabel_CircCenter->setDisabled(true);

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
	m_pComboBox_EmissionChannel->setDisabled(true);
    m_pLabel_EmissionChannel = new QLabel("Em Channel ", this);
    m_pLabel_EmissionChannel->setBuddy(m_pComboBox_EmissionChannel);
	m_pLabel_EmissionChannel->setDisabled(true);

    m_pCheckBox_HsvEnhancedMap = new QCheckBox(this);
    m_pCheckBox_HsvEnhancedMap->setText("HSV Enhanced Map");
	m_pCheckBox_HsvEnhancedMap->setDisabled(true);

	ColorTable temp_ctable;
	m_pComboBox_LifetimeColorTable = new QComboBox(this);
	for (int i = 0; i < temp_ctable.m_cNameVector.size(); i++)
		m_pComboBox_LifetimeColorTable->addItem(temp_ctable.m_cNameVector.at(i));
	m_pComboBox_LifetimeColorTable->setCurrentIndex(m_pConfig->flimLifetimeColorTable);
	m_pComboBox_LifetimeColorTable->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	m_pComboBox_LifetimeColorTable->setDisabled(true);
	m_pLabel_LifetimeColorTable = new QLabel("Lifetime Colortable ", this);
	m_pLabel_LifetimeColorTable->setBuddy(m_pComboBox_LifetimeColorTable);
	m_pLabel_LifetimeColorTable->setDisabled(true);
#endif
	
#ifdef OCT_NIRF
	// Create widgets for NIRF offset control
	m_pLineEdit_NirfOffset = new QLineEdit(this);
	m_pLineEdit_NirfOffset->setFixedWidth(35);
	m_pLineEdit_NirfOffset->setReadOnly(true);
	m_pLineEdit_NirfOffset->setText(QString::number(0));
	m_pLineEdit_NirfOffset->setAlignment(Qt::AlignCenter);
	m_pLineEdit_NirfOffset->setDisabled(true);
	
	m_pLabel_NirfOffset = new QLabel("NIRF Offset ", this);
	m_pLabel_NirfOffset->setBuddy(m_pLineEdit_NirfOffset);
	m_pLabel_NirfOffset->setDisabled(true);

	m_pScrollBar_NirfOffset = new QScrollBar(this);
	m_pScrollBar_NirfOffset->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	m_pScrollBar_NirfOffset->setFocusPolicy(Qt::StrongFocus);
	m_pScrollBar_NirfOffset->setOrientation(Qt::Horizontal);
	m_pScrollBar_NirfOffset->setRange(0, m_pConfig->nAlines - 1);
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

    pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 2, 0);
    pGridLayout_Visualization->addWidget(m_pCheckBox_CircularizeImage, 2, 1);
    pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 2, 2);
    pGridLayout_Visualization->addWidget(m_pLabel_CircCenter, 2, 3);
    pGridLayout_Visualization->addWidget(m_pLineEdit_CircCenter, 2, 4);
    pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 2, 5);

    pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 3, 0);
    pGridLayout_Visualization->addWidget(m_pCheckBox_ShowGuideLine, 3, 1);
    pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 3, 2);
    pGridLayout_Visualization->addWidget(m_pLabel_OctColorTable, 3, 3);
    pGridLayout_Visualization->addWidget(m_pComboBox_OctColorTable, 3, 4);
    pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 3, 5);

#ifdef OCT_FLIM
    pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 4, 0);
    pGridLayout_Visualization->addWidget(m_pCheckBox_HsvEnhancedMap, 4, 1);
    pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 4, 2);
    pGridLayout_Visualization->addWidget(m_pLabel_EmissionChannel, 4, 3);
    pGridLayout_Visualization->addWidget(m_pComboBox_EmissionChannel, 4, 4);
    pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 4, 5);

	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 5, 0, 1, 3);
	pGridLayout_Visualization->addWidget(m_pLabel_LifetimeColorTable, 5, 3);
	pGridLayout_Visualization->addWidget(m_pComboBox_LifetimeColorTable, 5, 4);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 5, 5);
#endif

#ifdef OCT_NIRF
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 4, 0);
	QHBoxLayout *pHBoxLayout_Nirf = new QHBoxLayout;
	pHBoxLayout_Nirf->setSpacing(3);
	pHBoxLayout_Nirf->addWidget(m_pLabel_NirfOffset);
	pHBoxLayout_Nirf->addWidget(m_pLineEdit_NirfOffset);
	pHBoxLayout_Nirf->addWidget(new QLabel("  ", this));
	pHBoxLayout_Nirf->addWidget(m_pScrollBar_NirfOffset);
	pGridLayout_Visualization->addItem(pHBoxLayout_Nirf, 4, 1, 1, 4);
	pGridLayout_Visualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 4, 5);
#endif

    m_pGroupBox_Visualization->setLayout(pGridLayout_Visualization);

	// Connect signal and slot
	connect(m_pSlider_SelectFrame, SIGNAL(valueChanged(int)), this, SLOT(visualizeImage(int)));
	connect(m_pToggleButton_MeasureDistance, SIGNAL(toggled(bool)), this, SLOT(measureDistance(bool)));
	connect(m_pCheckBox_ShowGuideLine, SIGNAL(toggled(bool)), this, SLOT(showGuideLine(bool)));
	connect(m_pCheckBox_CircularizeImage, SIGNAL(toggled(bool)), this, SLOT(changeVisImage(bool)));
	connect(m_pLineEdit_CircCenter, SIGNAL(textEdited(const QString &)), this, SLOT(checkCircCenter(const QString &)));
	connect(m_pComboBox_OctColorTable, SIGNAL(currentIndexChanged(int)), this, SLOT(adjustOctContrast()));
#ifdef OCT_FLIM
	connect(m_pCheckBox_HsvEnhancedMap, SIGNAL(toggled(bool)), this, SLOT(enableHsvEnhancingMode(bool)));
	connect(m_pComboBox_EmissionChannel, SIGNAL(currentIndexChanged(int)), this, SLOT(changeFlimCh(int)));	
	connect(m_pComboBox_LifetimeColorTable, SIGNAL(currentIndexChanged(int)), this, SLOT(changeLifetimeColorTable(int)));
#endif
#ifdef OCT_NIRF
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
    m_pImageView_OctProjection = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), m_pConfig->nAlines, 1);
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

    m_pImageView_ColorbarOctProjection = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), 4, 256);
	m_pImageView_ColorbarOctProjection->getRender()->setFixedWidth(15);
    m_pImageView_ColorbarOctProjection->drawImage(color);
    m_pImageView_ColorbarOctProjection->setFixedWidth(30);

    m_pLabel_OctProjection = new QLabel(this);
    m_pLabel_OctProjection->setText("OCT Maximum Projection Map");

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

    m_pImageView_ColorbarIntensityMap = new QImageView(ColorTable::colortable(INTENSITY_COLORTABLE), 4, 256, false);
	m_pImageView_ColorbarIntensityMap->getRender()->setFixedWidth(15);
    m_pImageView_ColorbarIntensityMap->drawImage(color);
    m_pImageView_ColorbarIntensityMap->setFixedWidth(30);

    m_pLabel_IntensityMap = new QLabel(this);
    m_pLabel_IntensityMap->setText("FLIM Intensity Map");

    // Create widgets for FLIM lifetime map
	ColorTable temp_ctable;
    m_pImageView_LifetimeMap = new QImageView(ColorTable::colortable(m_pConfig->flimLifetimeColorTable), m_pConfig->n4Alines, 1, true);
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

    m_pImageView_ColorbarLifetimeMap = new QImageView(ColorTable::colortable(m_pConfig->flimLifetimeColorTable), 4, 256, false);
	m_pImageView_ColorbarLifetimeMap->getRender()->setFixedWidth(15);
    m_pImageView_ColorbarLifetimeMap->drawImage(color);
    m_pImageView_ColorbarLifetimeMap->setFixedWidth(30);

    m_pLabel_LifetimeMap = new QLabel(this);
    m_pLabel_LifetimeMap->setText("FLIM Lifetime Map");
#endif

#ifdef OCT_NIRF
	// Create widgets for NIRF map
	m_pImageView_NirfMap = new QImageView(ColorTable::colortable(ColorTable::hot), m_pConfig->nAlines, 1);
	m_pImageView_NirfMap->setMinimumHeight(150);
	m_pImageView_NirfMap->setHLineChangeCallback([&](int frame) { m_pSlider_SelectFrame->setValue(frame); });
	m_pImageView_NirfMap->getRender()->m_colorLine = 0x00ff00;

	m_pLineEdit_NirfMax = new QLineEdit(this);
	m_pLineEdit_NirfMax->setText(QString::number(m_pConfig->nirfRange.max, 'f', 1));
	m_pLineEdit_NirfMax->setAlignment(Qt::AlignCenter);
	m_pLineEdit_NirfMax->setFixedWidth(30);
	m_pLineEdit_NirfMax->setDisabled(true);
	m_pLineEdit_NirfMin = new QLineEdit(this);
	m_pLineEdit_NirfMin->setText(QString::number(m_pConfig->nirfRange.min, 'f', 1));
	m_pLineEdit_NirfMin->setAlignment(Qt::AlignCenter);
	m_pLineEdit_NirfMin->setFixedWidth(30);
	m_pLineEdit_NirfMin->setDisabled(true);

	m_pImageView_ColorbarNirfMap = new QImageView(ColorTable::colortable(ColorTable::hot), 4, 256, false);
	m_pImageView_ColorbarNirfMap->getRender()->setFixedWidth(15);
	m_pImageView_ColorbarNirfMap->drawImage(color);
	m_pImageView_ColorbarNirfMap->setFixedWidth(30);

	m_pLabel_NirfMap = new QLabel(this);
	m_pLabel_NirfMap->setText("NIRF Map");
#endif

    // Set layout
    pGridLayout_EnFace->addWidget(m_pLabel_OctProjection, 0, 0, 1, 3);
    pGridLayout_EnFace->addWidget(m_pImageView_OctProjection, 1, 0);
    pGridLayout_EnFace->addWidget(m_pImageView_ColorbarOctProjection, 1, 1);
    QVBoxLayout *pVBoxLayout_Colorbar1 = new QVBoxLayout;
	pVBoxLayout_Colorbar1->setSpacing(0);
    pVBoxLayout_Colorbar1->addWidget(m_pLineEdit_OctDbMax);
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
	pGridLayout_EnFace->addWidget(m_pLabel_NirfMap, 2, 0, 1, 3);
	pGridLayout_EnFace->addWidget(m_pImageView_NirfMap, 3, 0);
	pGridLayout_EnFace->addWidget(m_pImageView_ColorbarNirfMap, 3, 1);
	QVBoxLayout *pVBoxLayout_Colorbar2 = new QVBoxLayout;
	pVBoxLayout_Colorbar2->setSpacing(0);
	pVBoxLayout_Colorbar2->addWidget(m_pLineEdit_NirfMax);
	pVBoxLayout_Colorbar2->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));
	pVBoxLayout_Colorbar2->addWidget(m_pLineEdit_NirfMin);
	pGridLayout_EnFace->addItem(pVBoxLayout_Colorbar2, 3, 2);
#endif
	
    m_pGroupBox_EnFace->setLayout(pGridLayout_EnFace);

	// Connect signal and slot
	connect(this, SIGNAL(paintOctProjection(uint8_t*)), m_pImageView_OctProjection, SLOT(drawImage(uint8_t*)));
#ifdef OCT_FLIM
	connect(this, SIGNAL(paintIntensityMap(uint8_t*)), m_pImageView_IntensityMap, SLOT(drawImage(uint8_t*)));
	connect(this, SIGNAL(paintLifetimeMap(uint8_t*)), m_pImageView_LifetimeMap, SLOT(drawImage(uint8_t*)));
#endif
#ifdef OCT_NIRF
	connect(this, SIGNAL(paintNirfMap(uint8_t*)), m_pImageView_NirfMap, SLOT(drawImage(uint8_t*)));
#endif

	connect(m_pLineEdit_OctDbMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustOctContrast()));
	connect(m_pLineEdit_OctDbMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustOctContrast()));
#ifdef OCT_FLIM
	connect(m_pLineEdit_IntensityMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustFlimContrast()));
	connect(m_pLineEdit_IntensityMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustFlimContrast()));
	connect(m_pLineEdit_LifetimeMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustFlimContrast()));
	connect(m_pLineEdit_LifetimeMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustFlimContrast()));
#endif
#ifdef OCT_NIRF
	connect(m_pLineEdit_NirfMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast()));
	connect(m_pLineEdit_NirfMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast()));
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
	}
	m_pSaveResultDlg->raise();
	m_pSaveResultDlg->activateWindow();
}

void QResultTab::deleteSaveResultDlg()
{
	m_pSaveResultDlg->deleteLater();
	m_pSaveResultDlg = nullptr;
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
}

void QResultTab::deletePulseReviewDlg()
{
	m_pImageView_RectImage->setVerticalLine(0);
	m_pImageView_RectImage->getRender()->update();

	m_pImageView_CircImage->setVerticalLine(0);
	m_pImageView_CircImage->getRender()->update();

	m_pPulseReviewDlg->deleteLater();
	m_pPulseReviewDlg = nullptr;
}
#endif


void QResultTab::enableUserDefinedAlines(bool checked)
{ 
	m_pLineEdit_UserDefinedAlines->setEnabled(checked); 
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
		ippiTranspose_8u_C1R(scale_temp.raw_ptr(), roi_oct.width * sizeof(uint8_t), m_pImgObjRectImage->arr.raw_ptr(), roi_oct.height * sizeof(uint8_t), roi_oct);
		(*m_pMedfiltRect)(m_pImgObjRectImage->arr.raw_ptr());

#ifdef OCT_FLIM		
		// FLIM Visualization		
		uint8_t* rectIntensity = &m_pImgObjIntensityMap->arr(0, m_pImgObjIntensityMap->arr.size(1) - 1 - frame);
		uint8_t* rectLifetime = &m_pImgObjLifetimeMap->arr(0, m_pImgObjLifetimeMap->arr.size(1) - 1 - frame);
		for (int i = 0; i < RING_THICKNESS; i++)
		{
			memcpy(&m_pImgObjIntensity->arr(0, i), rectIntensity, sizeof(uint8_t) * m_pImgObjIntensityMap->arr.size(0));
			memcpy(&m_pImgObjLifetime->arr(0, i), rectLifetime, sizeof(uint8_t) * m_pImgObjLifetimeMap->arr.size(0));
		}

		emit makeRgb(m_pImgObjRectImage, m_pImgObjCircImage, m_pImgObjIntensity, m_pImgObjLifetime);
		
#elif defined (STANDALONE_OCT)

#ifdef OCT_NIRF
		// NIRF Visualization
		uint8_t* rectNirf = &m_pImgObjNirfMap->arr(0, m_pImgObjNirfMap->arr.size(1) - 1 - frame);
		for (int i = 0; i < RING_THICKNESS; i++)
			memcpy(&m_pImgObjNirf->arr(0, i), rectNirf, sizeof(uint8_t) * m_pImgObjNirfMap->arr.size(0));

		emit makeRgb(m_pImgObjRectImage, m_pImgObjCircImage, m_pImgObjNirf);
#else
		if (!m_pCheckBox_CircularizeImage->isChecked())
			if (m_pImageView_RectImage->isEnabled()) emit paintRectImage(m_pImgObjRectImage->qindeximg.bits());
		else
		{
			(*m_pCirc)(m_pImgObjRectImage->arr, m_pImgObjCircImage->arr.raw_ptr(), "vertical", m_pConfig->circCenter);
			if (m_pImageView_CircImage->isEnabled()) emit paintCircImage(m_pImgObjCircImage->qindeximg.bits());
		}
#endif
#endif
		m_pImageView_OctProjection->setHorizontalLine(1, m_visOctProjection.size(1) - frame);
		m_pImageView_OctProjection->getRender()->update();
#ifdef OCT_FLIM
		m_pImageView_IntensityMap->setHorizontalLine(1, m_pImgObjIntensityMap->arr.size(1) - frame);
		m_pImageView_IntensityMap->getRender()->update();
		m_pImageView_LifetimeMap->setHorizontalLine(1, m_pImgObjLifetimeMap->arr.size(1) - frame);
		m_pImageView_LifetimeMap->getRender()->update();
#endif	
#ifdef OCT_NIRF
		m_pImageView_NirfMap->setHorizontalLine(1, m_pImgObjNirfMap->arr.size(1) - frame);
		m_pImageView_NirfMap->getRender()->update();
#endif

#ifdef OCT_FLIM
		if (m_pPulseReviewDlg)
			m_pPulseReviewDlg->drawPulse(m_pPulseReviewDlg->getCurrentAline());
#endif

		QString str; str.sprintf("Current Frame : %3d / %3d", frame + 1, (int)m_vectorOctImage.size());
		m_pLabel_SelectFrame->setText(str);
	}
}

#ifdef OCT_FLIM
void QResultTab::constructRgbImage(ImageObject *pRectObj, ImageObject *pCircObj, ImageObject *pIntObj, ImageObject *pLftObj)
{	
	// Convert RGB
	pRectObj->convertRgb();
	pIntObj->convertScaledRgb();
	pLftObj->convertScaledRgb();	

	// HSV Enhancing
	ImageObject hsvObj(pLftObj->getWidth(), 1, pLftObj->getColorTable());
	if (m_pCheckBox_HsvEnhancedMap->isChecked())
	{
		memcpy(hsvObj.qrgbimg.bits(),
			m_pImgObjHsvEnhancedMap->qrgbimg.bits() + (m_pImgObjHsvEnhancedMap->qrgbimg.height() - m_pSlider_SelectFrame->value() - 1) * 3 * m_pImgObjHsvEnhancedMap->qrgbimg.width(),
			3 * m_pImgObjHsvEnhancedMap->qrgbimg.width());
		hsvObj.scaledRgb4();
	}

	// Rect View
	if (!m_pCheckBox_CircularizeImage->isChecked())
	{
		if (!m_pCheckBox_HsvEnhancedMap->isChecked())
		{
			// Paste FLIM color ring to RGB rect image
			memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (pRectObj->arr.size(1) - 2 * RING_THICKNESS - 1), pIntObj->qrgbimg.bits(), pIntObj->qrgbimg.byteCount());
			memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (pRectObj->arr.size(1) - 1 * RING_THICKNESS - 1), pLftObj->qrgbimg.bits(), pLftObj->qrgbimg.byteCount());
		}
		else		
			// Paste FLIM color ring to RGB rect image
			for (int i = 0; i < RING_THICKNESS; i++)
				memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (pRectObj->arr.size(1) - i - 1), hsvObj.qrgbimg.bits(), hsvObj.qrgbimg.byteCount());

		// Draw image
        if (m_pImageView_RectImage->isEnabled()) emit paintRectImage(pRectObj->qrgbimg.bits());
	}
	// Circ View
	else
	{
		if (!m_pCheckBox_HsvEnhancedMap->isChecked())
		{
			// Paste FLIM color ring to RGB rect image
			memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (m_pConfig->circCenter + CIRC_RADIUS - 2 * RING_THICKNESS), pIntObj->qrgbimg.bits(), pIntObj->qrgbimg.byteCount());
			memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (m_pConfig->circCenter + CIRC_RADIUS - 1 * RING_THICKNESS), pLftObj->qrgbimg.bits(), pLftObj->qrgbimg.byteCount());
		}
		else
			// Paste FLIM color ring to RGB rect image
			for (int i = 0; i < RING_THICKNESS; i++)
				memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (m_pConfig->circCenter + CIRC_RADIUS - i), hsvObj.qrgbimg.bits(), hsvObj.qrgbimg.byteCount());
		
				
		np::Uint8Array2 rect_temp(pRectObj->qrgbimg.bits(), 3 * pRectObj->arr.size(0), pRectObj->arr.size(1));
		(*m_pCirc)(rect_temp, pCircObj->qrgbimg.bits(), "vertical", "rgb", m_pConfig->circCenter);
		
		// Draw image        
        if (m_pImageView_CircImage->isEnabled()) emit paintCircImage(pCircObj->qrgbimg.bits());
	}
}
#endif
#ifdef OCT_NIRF
void QResultTab::constructRgbImage(ImageObject *pRectObj, ImageObject *pCircObj, ImageObject *pNirfObj)
{
	// Convert RGB
	pRectObj->convertRgb();
	pNirfObj->convertNonScaledRgb();

	// Rect View
	if (!m_pCheckBox_CircularizeImage->isChecked())
	{
		// Paste NIRF color ring to RGB rect image
		memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (pRectObj->arr.size(1) - 1 * RING_THICKNESS - 1), pNirfObj->qrgbimg.bits(), pNirfObj->qrgbimg.byteCount());
	
		// Draw image
		if (m_pImageView_RectImage->isEnabled()) emit paintRectImage(pRectObj->qrgbimg.bits());
	}
	// Circ View
	else
	{		
		// Paste NIRF color ring to RGB rect image
		memcpy(pRectObj->qrgbimg.bits() + 3 * pRectObj->arr.size(0) * (m_pConfig->circCenter + CIRC_RADIUS - 1 * RING_THICKNESS), pNirfObj->qrgbimg.bits(), pNirfObj->qrgbimg.byteCount());

		// Circularize
		np::Uint8Array2 rect_temp(pRectObj->qrgbimg.bits(), 3 * pRectObj->arr.size(0), pRectObj->arr.size(1));
		(*m_pCirc)(rect_temp, pCircObj->qrgbimg.bits(), "vertical", "rgb", m_pConfig->circCenter);

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

#ifdef OCT_NIRF
			// Adjusting NIRF offset
			float* pNirfMap = m_nirfMap.raw_ptr();		
			if (m_nirfSignal.length() != 0)
			{
				float* pNirfSignal = m_nirfSignal.raw_ptr();

				int sig_len = m_nirfSignal.length();
				int map_len = m_nirfMap.length();
				int diff;

				memset(m_nirfMap.raw_ptr(), 0, sizeof(float) * m_nirfMap.length());

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

			// Scaling NIRF map
			ippiScale_32f8u_C1R(m_nirfMap, sizeof(float) * roi_proj.width, m_pImgObjNirfMap->arr.raw_ptr(), sizeof(uint8_t) * roi_proj.width,
				roi_proj, m_pConfig->nirfRange.min, m_pConfig->nirfRange.max);
			ippiMirror_8u_C1IR(m_pImgObjNirfMap->arr.raw_ptr(), sizeof(uint8_t) * roi_proj.width, roi_proj, ippAxsHorizontal);
#endif

#ifdef OCT_FLIM
			IppiSize roi_flimproj = { m_intensityMap.at(0).size(0), m_intensityMap.at(0).size(1) };

			ippiScale_32f8u_C1R(m_intensityMap.at(m_pComboBox_EmissionChannel->currentIndex()), sizeof(float) * roi_flimproj.width,
				m_pImgObjIntensityMap->arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width,
				roi_flimproj, m_pConfig->flimIntensityRange.min, m_pConfig->flimIntensityRange.max);
			ippiMirror_8u_C1IR(m_pImgObjIntensityMap->arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width, roi_flimproj, ippAxsHorizontal);
			(*m_pMedfiltIntensityMap)(m_pImgObjIntensityMap->arr.raw_ptr());

			ippiScale_32f8u_C1R(m_lifetimeMap.at(m_pComboBox_EmissionChannel->currentIndex()), sizeof(float) * roi_flimproj.width,
				m_pImgObjLifetimeMap->arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width,
				roi_flimproj, m_pConfig->flimLifetimeRange.min, m_pConfig->flimLifetimeRange.max);
			ippiMirror_8u_C1IR(m_pImgObjLifetimeMap->arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width, roi_flimproj, ippAxsHorizontal);
			(*m_pMedfiltLifetimeMap)(m_pImgObjLifetimeMap->arr.raw_ptr());
			m_pImgObjLifetimeMap->convertRgb();

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
				tempImgObj.convertRgb();

				ippsMul_8u_Sfs(m_pImgObjLifetimeMap->qrgbimg.bits(), tempImgObj.qrgbimg.bits(), m_pImgObjHsvEnhancedMap->qrgbimg.bits(), tempImgObj.qrgbimg.byteCount(), 8);
#endif
		}
#endif
	}
		emit paintOctProjection(m_visOctProjection);
#ifdef OCT_NIRF
		emit paintNirfMap(m_pImgObjNirfMap->arr.raw_ptr());
#endif
#ifdef OCT_FLIM
		emit paintIntensityMap(m_pImgObjIntensityMap->arr.raw_ptr());
		emit paintLifetimeMap((!m_pCheckBox_HsvEnhancedMap->isChecked()) ? m_pImgObjLifetimeMap->qrgbimg.bits() : m_pImgObjHsvEnhancedMap->qrgbimg.bits());
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
		m_pImageView_RectImage->setHorizontalLine(3, m_pConfig->circCenter, m_pConfig->circCenter + PROJECTION_OFFSET, m_pConfig->circCenter + CIRC_RADIUS);
		m_pImageView_CircImage->setCircle(1, PROJECTION_OFFSET);
	}
	else
	{
		m_pImageView_RectImage->setHorizontalLine(0);
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
	}
	else // rect view
	{
		m_pToggleButton_MeasureDistance->setChecked(false);
		m_pImageView_CircImage->setVisible(toggled);
		m_pImageView_RectImage->setVisible(!toggled);
	}
	visualizeImage(m_pSlider_SelectFrame->value());

	m_pToggleButton_MeasureDistance->setEnabled(toggled);
}

void QResultTab::checkCircCenter(const QString &str)
{
	int circCenter = str.toInt();
	if (circCenter + CIRC_RADIUS + 1 > m_pImageView_RectImage->getRender()->m_pImage->height())
	{
		circCenter = m_pImageView_RectImage->getRender()->m_pImage->height() - CIRC_RADIUS - 1;
		m_pLineEdit_CircCenter->setText(QString("%1").arg(circCenter));
	}
	m_pConfig->circCenter = circCenter;

	if (m_pCheckBox_ShowGuideLine->isChecked())
		m_pImageView_RectImage->setHorizontalLine(3, m_pConfig->circCenter, m_pConfig->circCenter + PROJECTION_OFFSET, m_pConfig->circCenter + CIRC_RADIUS);

	getOctProjection(m_vectorOctImage, m_octProjection, circCenter);
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
}

void QResultTab::adjustOctContrast()
{
	int min_dB = m_pLineEdit_OctDbMin->text().toInt();
	int max_dB = m_pLineEdit_OctDbMax->text().toInt();
	int ctable_ind = m_pComboBox_OctColorTable->currentIndex();

	m_pConfig->octDbRange.min = min_dB;
	m_pConfig->octDbRange.max = max_dB;
	
	m_pConfig->octColorTable = ctable_ind;

	m_pImageView_RectImage->resetColormap(ColorTable::colortable(ctable_ind));
	m_pImageView_CircImage->resetColormap(ColorTable::colortable(ctable_ind));
	m_pImageView_OctProjection->resetColormap(ColorTable::colortable(ctable_ind));	
	m_pImageView_ColorbarOctProjection->resetColormap(ColorTable::colortable(ctable_ind));

	ColorTable temp_ctable;

	if (m_pImgObjRectImage) delete m_pImgObjRectImage;
	m_pImgObjRectImage = new ImageObject(m_pImageView_RectImage->getRender()->m_pImage->width(), m_pImageView_RectImage->getRender()->m_pImage->height(), temp_ctable.m_colorTableVector.at(ctable_ind));
	if (m_pImgObjCircImage) delete m_pImgObjCircImage;
	m_pImgObjCircImage = new ImageObject(m_pImageView_CircImage->getRender()->m_pImage->width(), m_pImageView_CircImage->getRender()->m_pImage->height(), temp_ctable.m_colorTableVector.at(ctable_ind));

	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
}

#ifdef OCT_FLIM
void QResultTab::changeFlimCh(int)
{
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());	
}

void QResultTab::enableHsvEnhancingMode(bool)
{
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
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
}

void QResultTab::changeLifetimeColorTable(int ctable_ind)
{
	m_pConfig->flimLifetimeColorTable = ctable_ind;

	m_pImageView_LifetimeMap->resetColormap(ColorTable::colortable(ctable_ind));
	m_pImageView_ColorbarLifetimeMap->resetColormap(ColorTable::colortable(ctable_ind));
	
	ColorTable temp_ctable;

	if (m_pImgObjLifetime) delete m_pImgObjLifetime;
	m_pImgObjLifetime = new ImageObject(m_pImageView_LifetimeMap->getRender()->m_pImage->width(), RING_THICKNESS, temp_ctable.m_colorTableVector.at(ctable_ind));
	if (m_pImgObjLifetimeMap) delete m_pImgObjLifetimeMap;
	m_pImgObjLifetimeMap = new ImageObject(m_pImageView_LifetimeMap->getRender()->m_pImage->width(), m_pImageView_LifetimeMap->getRender()->m_pImage->height(), temp_ctable.m_colorTableVector.at(ctable_ind));
	if (m_pImgObjHsvEnhancedMap) delete m_pImgObjHsvEnhancedMap;
	m_pImgObjHsvEnhancedMap = new ImageObject(m_pImageView_LifetimeMap->getRender()->m_pImage->width(), m_pImageView_LifetimeMap->getRender()->m_pImage->height(), temp_ctable.m_colorTableVector.at(ctable_ind));

	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
}
#endif

#ifdef OCT_NIRF
void QResultTab::adjustNirfContrast()
{
	float min_intensity = m_pLineEdit_NirfMin->text().toFloat();
	float max_intensity = m_pLineEdit_NirfMax->text().toFloat();

	m_pConfig->nirfRange.min = min_intensity;
	m_pConfig->nirfRange.max = max_intensity;

	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
}

void QResultTab::adjustNirfOffset(int offset)
{
	m_nirfOffset = offset;
	m_pLineEdit_NirfOffset->setText(QString::number(m_nirfOffset));
	visualizeEnFaceMap(true);
	visualizeImage(m_pSlider_SelectFrame->value());
}
#endif


void QResultTab::startProcessing()
{	
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
		OCTProcess* pOCT = m_pMainWnd->m_pStreamTab->m_pOCT;
		m_pFLIMpost = m_pMainWnd->m_pStreamTab->m_pFLIM;
#elif defined (STANDALONE_OCT)
		OCTProcess* pOCT1 = m_pMainWnd->m_pStreamTab->m_pOCT1;
#ifdef DUAL_CHANNEL
		OCTProcess* pOCT2 = m_pMainWnd->m_pStreamTab->m_pOCT2;
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
		getOctProjection(m_vectorOctImage, m_octProjection, m_pConfig->circCenter);

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
	QString fileName = QFileDialog::getOpenFileName(nullptr, "Load external OCT FLIM data", "", "OCT FLIM raw data (*.data)");
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
				QString nirfName = m_path + "/NIRF.txt";
#endif
				qDebug() << iniName;

				static Configuration config;
				config.getConfigFile(iniName);
				if (m_pCheckBox_UserDefinedAlines->isChecked())
				{
					config.nAlines = m_pLineEdit_UserDefinedAlines->text().toInt();
					config.nAlines4 = ((config.nAlines + 3) >> 2) << 2;
					config.nFrameSize = config.nChannels * config.nScans * config.nAlines;
				}
				config.octDiscomVal = m_pLineEdit_DiscomValue->text().toInt();
				config.nFrames = (int)(file.size() / (qint64)config.nChannels / (qint64)config.nScans / (qint64)config.nAlines / sizeof(uint16_t));
				if (m_pCheckBox_SingleFrame->isChecked()) config.nFrames = 1;

				printf("Start external image processing... (Total nFrame: %d)\n", config.nFrames);

				// Set Widgets //////////////////////////////////////////////////////////////////////////////
				emit setWidgets(false, &config);
				m_pImageView_RectImage->setUpdatesEnabled(false);
				m_pImageView_CircImage->setUpdatesEnabled(false);

				// Set Buffers //////////////////////////////////////////////////////////////////////////////
				setObjects(&config);

				int bufferSize = (false == m_pCheckBox_SingleFrame->isChecked()) ? PROCESSING_BUFFER_SIZE : 1;

				m_syncDeinterleaving.allocate_queue_buffer(config.nChannels * config.nScans, config.nAlines, bufferSize);
				m_syncCh1Processing.allocate_queue_buffer(config.nScans, config.nAlines, bufferSize);
#ifdef OCT_FLIM
				m_syncCh2Processing.allocate_queue_buffer(config.nScans * 4, config.n4Alines, bufferSize);
#elif defined (STANDALONE_OCT)
				m_syncCh2Processing.allocate_queue_buffer(config.nScans, config.nAlines, bufferSize);
#endif

#ifdef STANDALONE_OCT
#ifdef DUAL_CHANNEL
				m_syncCh1Visualization.allocate_queue_buffer(config.n2ScansFFT, config.nAlines, bufferSize);
				m_syncCh2Visualization.allocate_queue_buffer(config.n2ScansFFT, config.nAlines, bufferSize);
#endif
#endif
				// Set OCT FLIM Object //////////////////////////////////////////////////////////////////////
#ifdef OCT_FLIM
				OCTProcess* pOCT = new OCTProcess(config.nScans, config.nAlines);
				pOCT->loadCalibration(CH_1, calibName, bgName, config.erasmus);
				pOCT->changeDiscomValue(config.octDiscomVal);

				if (m_pFLIMpost) if (m_pFLIMpost != m_pMainWnd->m_pStreamTab->m_pFLIM) delete m_pFLIMpost;
				m_pFLIMpost = new FLIMProcess;
				m_pFLIMpost->setParameters(&config);
				m_pFLIMpost->_resize(np::Uint16Array2(config.nScans * 4, config.n4Alines), m_pFLIMpost->_params);
				m_pFLIMpost->loadMaskData(maskName);

#elif defined (STANDALONE_OCT)
				OCTProcess* pOCT1 = new OCTProcess(config.nScans, config.nAlines);
				pOCT1->loadCalibration(CH_1, calibName.toUtf8().constData(), bgName.toUtf8().constData(), config.erasmus);
				pOCT1->changeDiscomValue(config.octDiscomVal);

#ifdef DUAL_CHANNEL
				OCTProcess* pOCT2 = new OCTProcess(config.nScans, config.nAlines);
				pOCT2->loadCalibration(CH_2, calibName.toUtf8().constData(), bgName.toUtf8().constData(), config.erasmus);
				pOCT2->changeDiscomValue(config.octDiscomVal);
#endif
#endif				
				// Get external data ////////////////////////////////////////////////////////////////////////
				std::thread load_data([&]() { loadingRawData(&file, &config); });

				// Data DeInterleaving & FLIM Process ///////////////////////////////////////////////////////
				std::thread deinterleave([&]() { deinterleaving(&config); });

#ifdef OCT_FLIM
				// Ch1 Process //////////////////////////////////////////////////////////////////////////////
				std::thread ch1_proc([&]() { octProcessing(pOCT, &config); });

				// Ch2 Process //////////////////////////////////////////////////////////////////////////////		
				std::thread ch2_proc([&]() { flimProcessing(m_pFLIMpost, &config); });
#elif defined (STANDALONE_OCT)
				// Ch1 Process //////////////////////////////////////////////////////////////////////////////
				std::thread ch1_proc([&]() { octProcessing1(pOCT1, &config); });

				// Ch2 Process //////////////////////////////////////////////////////////////////////////////	
#ifdef DUAL_CHANNEL
				std::thread ch2_proc([&]() { octProcessing2(pOCT2, &config); });
#else
				std::thread ch2_proc([&]() { octProcessing2(pOCT1, &config); });
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
				getOctProjection(m_vectorOctImage, m_octProjection, m_pConfig->circCenter);

#ifdef OCT_NIRF
				// Load NIRF data ///////////////////////////////////////////////////////////////////////////
				if (!m_pCheckBox_SingleFrame->isChecked())
				{
					QFile nirfFile(nirfName);
					if (false == nirfFile.open(QFile::ReadOnly | QFile::Text))
						printf("[ERROR] There is no nirf data or you selected an invalid nirf data!\n");
					else
					{
						QTextStream in(&nirfFile);
						int nLine = 0;
						while (!in.atEnd())
						{
							QString tempLine = in.readLine();
							nLine++;
						}
						nLine /= 2;

						in.seek(0);
						m_nirfSignal = np::FloatArray(nLine);
						for (int i = 0; i < nLine; i++)
						{
							QString line1 = in.readLine();
							QString line2 = in.readLine();
							m_nirfSignal.at(i) = (line1.toFloat() + line2.toFloat()) / 2.0f;
						}
						nirfFile.close();

						printf("NIRF data was successfully loaded...\n");
						config.nirf = true;
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
				emit setWidgets(true, &config);
		
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

		m_pImageView_RectImage->resetSize(pConfig->nAlines4, pConfig->n2ScansFFT);
		m_pImageView_CircImage->resetSize(2 * CIRC_RADIUS, 2 * CIRC_RADIUS);

		m_pImageView_OctProjection->resetSize(pConfig->nAlines4, pConfig->nFrames);
#ifdef OCT_FLIM
		m_pImageView_IntensityMap->resetSize(pConfig->n4Alines, pConfig->nFrames);
		m_pImageView_LifetimeMap->resetSize(pConfig->n4Alines, pConfig->nFrames);
#endif
#ifdef OCT_NIRF
		m_pImageView_NirfMap->resetSize(pConfig->nAlines4, pConfig->nFrames);
#endif
		// Reset widgets
		m_pImageView_OctProjection->setEnabled(true);        
        m_pImageView_OctProjection->setUpdatesEnabled(true);
#ifdef OCT_FLIM
		m_pImageView_IntensityMap->setEnabled(true);        
        m_pImageView_IntensityMap->setUpdatesEnabled(true);
		m_pImageView_LifetimeMap->setEnabled(true);        
        m_pImageView_LifetimeMap->setUpdatesEnabled(true);
#endif
#ifdef OCT_NIRF
		m_pImageView_NirfMap->setEnabled(true);
		m_pImageView_NirfMap->setUpdatesEnabled(true);
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
		m_pLabel_DiscomValue->setEnabled(true);
		m_pLineEdit_DiscomValue->setEnabled(true);

		m_pProgressBar_PostProcessing->setFormat("");
        m_pProgressBar_PostProcessing->setValue(0);

		if (m_pCheckBox_CircularizeImage->isChecked())
			m_pToggleButton_MeasureDistance->setEnabled(true);
		m_pLabel_SelectFrame->setEnabled(true);
		m_pSlider_SelectFrame->setEnabled(true);
		m_pSlider_SelectFrame->setRange(0, pConfig->nFrames - 1);
		m_pSlider_SelectFrame->setValue(0);
		
		m_pCheckBox_CircularizeImage->setEnabled(true);
		m_pCheckBox_ShowGuideLine->setEnabled(true);
		m_pLabel_CircCenter->setEnabled(true);
		m_pLineEdit_CircCenter->setEnabled(true);
		m_pLabel_OctColorTable->setEnabled(true);
		m_pComboBox_OctColorTable->setEnabled(true);
#ifdef OCT_FLIM
		m_pCheckBox_HsvEnhancedMap->setEnabled(true);
		m_pLabel_EmissionChannel->setEnabled(true);
		m_pComboBox_EmissionChannel->setEnabled(true);
		m_pLabel_LifetimeColorTable->setEnabled(true);
		m_pComboBox_LifetimeColorTable->setEnabled(true);
#endif
#ifdef OCT_NIRF
		if (pConfig->nirf)
		{
			m_pLabel_NirfOffset->setEnabled(true);
			m_pLineEdit_NirfOffset->setText(QString::number(0));
			m_pLineEdit_NirfOffset->setEnabled(true);
			m_pScrollBar_NirfOffset->setRange(0, pConfig->nAlines - 1);
			m_pScrollBar_NirfOffset->setValue(0);
			m_pScrollBar_NirfOffset->setEnabled(true);
		}
#endif

		m_pLineEdit_OctDbMax->setEnabled(true);
		m_pLineEdit_OctDbMin->setEnabled(true);
#ifdef OCT_FLIM
		m_pLineEdit_IntensityMax->setEnabled(true);
		m_pLineEdit_IntensityMin->setEnabled(true);
		m_pLineEdit_LifetimeMax->setEnabled(true);
		m_pLineEdit_LifetimeMin->setEnabled(true);
#endif
#ifdef OCT_NIRF
		m_pLineEdit_NirfMax->setEnabled(true);
		m_pLineEdit_NirfMin->setEnabled(true);
#endif
	}
	else
	{
		// Set widgets
        m_pImageView_RectImage->setDisabled(true);
        m_pImageView_RectImage->setUpdatesEnabled(false);
        m_pImageView_CircImage->setDisabled(true);
        m_pImageView_CircImage->setUpdatesEnabled(false);

		m_pImageView_OctProjection->setDisabled(true);
        m_pImageView_OctProjection->setUpdatesEnabled(false);
#ifdef OCT_FLIM
		m_pImageView_IntensityMap->setDisabled(true);
        m_pImageView_IntensityMap->setUpdatesEnabled(false);
		m_pImageView_LifetimeMap->setDisabled(true);
        m_pImageView_LifetimeMap->setUpdatesEnabled(false);
#endif
#ifdef OCT_NIRF
		m_pImageView_NirfMap->setDisabled(true);
		m_pImageView_NirfMap->setUpdatesEnabled(false);
#endif
		m_pPushButton_StartProcessing->setDisabled(true);
		m_pPushButton_SaveResults->setDisabled(true);
		m_pRadioButton_InBuffer->setDisabled(true);
		m_pRadioButton_External->setDisabled(true);

		m_pCheckBox_SingleFrame->setDisabled(true);
		m_pCheckBox_UserDefinedAlines->setDisabled(true);
		m_pLineEdit_UserDefinedAlines->setDisabled(true);
		m_pLabel_DiscomValue->setDisabled(true);
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

		m_pCheckBox_CircularizeImage->setDisabled(true);
		m_pCheckBox_ShowGuideLine->setDisabled(true);
		m_pLabel_CircCenter->setDisabled(true);
		m_pLineEdit_CircCenter->setDisabled(true);
		m_pLabel_OctColorTable->setDisabled(true);
		m_pComboBox_OctColorTable->setDisabled(true);
#ifdef OCT_FLIM
		m_pCheckBox_HsvEnhancedMap->setDisabled(true);
		m_pLabel_EmissionChannel->setDisabled(true);
		m_pComboBox_EmissionChannel->setDisabled(true);
		m_pLabel_LifetimeColorTable->setDisabled(true);
		m_pComboBox_LifetimeColorTable->setDisabled(true);
#endif
#ifdef OCT_NIRF
		m_pLabel_NirfMap->setDisabled(true);
		m_pLineEdit_NirfOffset->setText(QString::number(0));
		m_pLineEdit_NirfOffset->setDisabled(true);
		m_pScrollBar_NirfOffset->setValue(0);
		m_pScrollBar_NirfOffset->setDisabled(true);
#endif

		m_pLineEdit_OctDbMax->setDisabled(true);
		m_pLineEdit_OctDbMin->setDisabled(true);
#ifdef OCT_FLIM
		m_pLineEdit_IntensityMax->setDisabled(true);
		m_pLineEdit_IntensityMin->setDisabled(true);
		m_pLineEdit_LifetimeMax->setDisabled(true);
		m_pLineEdit_LifetimeMin->setDisabled(true);
#endif
#ifdef OCT_NIRF
		m_pLineEdit_NirfMax->setDisabled(true);
		m_pLineEdit_NirfMin->setDisabled(true);
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

		// Reset widgets
		m_pImageView_OctProjection->setEnabled(true);
#ifdef OCT_FLIM
		m_pImageView_IntensityMap->setEnabled(true);
		m_pImageView_LifetimeMap->setEnabled(true);
#endif
#ifdef OCT_NIRF
		m_pImageView_NirfMap->setEnabled(true);
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
		m_pLabel_DiscomValue->setEnabled(true);
		m_pLineEdit_DiscomValue->setEnabled(true);

		m_pProgressBar_PostProcessing->setFormat("");

		if (m_pCheckBox_CircularizeImage->isChecked())
			m_pToggleButton_MeasureDistance->setEnabled(true);
		m_pLabel_SelectFrame->setEnabled(true);
		m_pSlider_SelectFrame->setEnabled(true);

		m_pCheckBox_CircularizeImage->setEnabled(true);
		m_pCheckBox_ShowGuideLine->setEnabled(true);
		m_pLabel_CircCenter->setEnabled(true);
		m_pLineEdit_CircCenter->setEnabled(true);
		m_pLabel_OctColorTable->setEnabled(true);
		m_pComboBox_OctColorTable->setEnabled(true);
#ifdef OCT_FLIM
		m_pCheckBox_HsvEnhancedMap->setEnabled(true);
		m_pLabel_EmissionChannel->setEnabled(true);
		m_pComboBox_EmissionChannel->setEnabled(true);
		m_pLabel_LifetimeColorTable->setEnabled(true);
		m_pComboBox_LifetimeColorTable->setEnabled(true);
#endif
#ifdef OCT_NIRF
		if (m_nirfSignal.length() != 0)
		{
			m_pLabel_NirfOffset->setEnabled(true);
			m_pLineEdit_NirfOffset->setEnabled(true);
			m_pScrollBar_NirfOffset->setEnabled(true);
		}
#endif

		m_pLineEdit_OctDbMax->setEnabled(true);
		m_pLineEdit_OctDbMin->setEnabled(true);
#ifdef OCT_FLIM
		m_pLineEdit_IntensityMax->setEnabled(true);
		m_pLineEdit_IntensityMin->setEnabled(true);
		m_pLineEdit_LifetimeMax->setEnabled(true);
		m_pLineEdit_LifetimeMin->setEnabled(true);
#endif
#ifdef OCT_NIRF
		m_pLineEdit_NirfMax->setEnabled(true);
		m_pLineEdit_NirfMin->setEnabled(true);
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
		m_pImageView_NirfMap->setDisabled(true);
#endif

		m_pPushButton_StartProcessing->setDisabled(true);
		m_pPushButton_SaveResults->setDisabled(true);
		m_pRadioButton_InBuffer->setDisabled(true);
		m_pRadioButton_External->setDisabled(true);

		m_pCheckBox_SingleFrame->setDisabled(true);
		m_pCheckBox_UserDefinedAlines->setDisabled(true);
		m_pLineEdit_UserDefinedAlines->setDisabled(true);
		m_pLabel_DiscomValue->setDisabled(true);
		m_pLineEdit_DiscomValue->setDisabled(true);

		m_pProgressBar_PostProcessing->setFormat("Saving results... %p%");
		m_pProgressBar_PostProcessing->setRange(0, (int)m_vectorOctImage.size() * 2 - 1);
		m_pProgressBar_PostProcessing->setValue(0);

		m_pToggleButton_MeasureDistance->setDisabled(true);
		m_pLabel_SelectFrame->setDisabled(true);
		m_pSlider_SelectFrame->setDisabled(true);

		m_pCheckBox_CircularizeImage->setDisabled(true);
		m_pCheckBox_ShowGuideLine->setDisabled(true);
		m_pLabel_CircCenter->setDisabled(true);
		m_pLineEdit_CircCenter->setDisabled(true);
		m_pLabel_OctColorTable->setDisabled(true);
		m_pComboBox_OctColorTable->setDisabled(true);
#ifdef OCT_FLIM
		m_pCheckBox_HsvEnhancedMap->setDisabled(true);
		m_pLabel_EmissionChannel->setDisabled(true);
		m_pComboBox_EmissionChannel->setDisabled(true);
		m_pLabel_LifetimeColorTable->setDisabled(true);
		m_pComboBox_LifetimeColorTable->setDisabled(true);
#endif
#ifdef OCT_NIRF
		m_pLabel_NirfOffset->setDisabled(true);
		m_pLineEdit_NirfOffset->setDisabled(true);
		m_pScrollBar_NirfOffset->setDisabled(true);
#endif

		m_pLineEdit_OctDbMax->setDisabled(true);
		m_pLineEdit_OctDbMin->setDisabled(true);
#ifdef OCT_FLIM
		m_pLineEdit_IntensityMax->setDisabled(true);
		m_pLineEdit_IntensityMin->setDisabled(true);
		m_pLineEdit_LifetimeMax->setDisabled(true);
		m_pLineEdit_LifetimeMin->setDisabled(true);
#endif
#ifdef OCT_NIRF
		m_pLineEdit_NirfMax->setDisabled(true);
		m_pLineEdit_NirfMin->setDisabled(true);
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
		np::FloatArray2 intensity = np::FloatArray2(pConfig->n4Alines, pConfig->nFrames);
		np::FloatArray2 lifetime = np::FloatArray2(pConfig->n4Alines, pConfig->nFrames);
		m_intensityMap.push_back(intensity);
		m_lifetimeMap.push_back(lifetime);
	}
#endif
#ifdef OCT_NIRF
	m_nirfSignal = np::FloatArray();
	m_nirfMap = np::FloatArray2(pConfig->nAlines4, pConfig->nFrames);
	m_nirfOffset = 0;
#endif

	// Visualization buffers
	ColorTable temp_ctable;

	if (m_pImgObjRectImage) delete m_pImgObjRectImage;
	m_pImgObjRectImage = new ImageObject(pConfig->nAlines4, pConfig->n2ScansFFT, temp_ctable.m_colorTableVector.at(m_pComboBox_OctColorTable->currentIndex()));
	if (m_pImgObjCircImage) delete m_pImgObjCircImage;
	m_pImgObjCircImage = new ImageObject(2 * CIRC_RADIUS, 2 * CIRC_RADIUS, temp_ctable.m_colorTableVector.at(m_pComboBox_OctColorTable->currentIndex()));
#ifdef OCT_FLIM
	if (m_pImgObjIntensity) delete m_pImgObjIntensity;
	m_pImgObjIntensity = new ImageObject(pConfig->n4Alines, RING_THICKNESS, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
	if (m_pImgObjLifetime) delete m_pImgObjLifetime;
	m_pImgObjLifetime = new ImageObject(pConfig->n4Alines, RING_THICKNESS, temp_ctable.m_colorTableVector.at(m_pComboBox_LifetimeColorTable->currentIndex()));
#endif
#ifdef OCT_NIRF
	if (m_pImgObjNirf) delete m_pImgObjNirf;
	m_pImgObjNirf = new ImageObject(pConfig->nAlines, RING_THICKNESS, temp_ctable.m_colorTableVector.at(ColorTable::hot));
#endif

	// En face map visualization buffers
	m_visOctProjection = np::Uint8Array2(pConfig->nAlines4, pConfig->nFrames);
#ifdef OCT_FLIM
	if (m_pImgObjIntensityMap) delete m_pImgObjIntensityMap;
	m_pImgObjIntensityMap = new ImageObject(pConfig->n4Alines, pConfig->nFrames, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
	if (m_pImgObjLifetimeMap) delete m_pImgObjLifetimeMap;
	m_pImgObjLifetimeMap = new ImageObject(pConfig->n4Alines, pConfig->nFrames, temp_ctable.m_colorTableVector.at(m_pComboBox_LifetimeColorTable->currentIndex()));
	if (m_pImgObjHsvEnhancedMap) delete m_pImgObjHsvEnhancedMap;
	m_pImgObjHsvEnhancedMap = new ImageObject(pConfig->n4Alines, pConfig->nFrames, temp_ctable.m_colorTableVector.at(m_pComboBox_LifetimeColorTable->currentIndex()));
#endif
#ifdef OCT_NIRF
	if (m_pImgObjNirfMap) delete m_pImgObjNirfMap;
	m_pImgObjNirfMap = new ImageObject(pConfig->nAlines, pConfig->nFrames, temp_ctable.m_colorTableVector.at(ColorTable::hot));
#endif

	// Circ & Medfilt objects
    if (m_pCirc) delete m_pCirc;
    m_pCirc = new circularize(CIRC_RADIUS, pConfig->nAlines, false);

	if (m_pMedfiltRect) delete m_pMedfiltRect;
    m_pMedfiltRect = new medfilt(pConfig->nAlines4, pConfig->n2ScansFFT, 3, 3);
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
					ippsCplxToReal_16sc((Ipp16sc *)frame_ptr, (Ipp16s *)ch1_ptr, (Ipp16s *)ch2_ptr, frame_length);
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
				ippsCplxToReal_16sc((Ipp16sc *)frame_ptr, (Ipp16s *)ch1_ptr, (Ipp16s *)ch2_ptr, frame_length);
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
void QResultTab::octProcessing(OCTProcess* pOCT, Configuration* pConfig)
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

void QResultTab::octProcessing1(OCTProcess* pOCT, Configuration* pConfig)
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
					if (pConfig->erasmus)
					{
						IppiSize roi = { pConfig->n2ScansFFT, pConfig->nAlines };
						if (~pConfig->oldUhs)
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

void QResultTab::octProcessing2(OCTProcess* pOCT, Configuration* pConfig)
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

void QResultTab::getOctProjection(std::vector<np::FloatArray2>& vecImg, np::FloatArray2& octProj, int offset)
{
	int len = CIRC_RADIUS;
	tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)vecImg.size()),
		[&](const tbb::blocked_range<size_t>& r) {
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			float maxVal;
			for (int j = 0; j < octProj.size(0); j++)
			{
				ippsMax_32f(&vecImg.at((int)i)(offset + PROJECTION_OFFSET, j), len, &maxVal);
				octProj(j, (int)i) = maxVal;
			}
		}
	});
}
