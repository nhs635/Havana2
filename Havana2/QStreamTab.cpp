
#include "QStreamTab.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QOperationTab.h>
#include <Havana2/QDeviceControlTab.h>

#include <DataAcquisition/DataAcquisition.h>
#include <MemoryBuffer/MemoryBuffer.h>

#ifdef OCT_FLIM
#include <Havana2/Viewer/QScope.h>
#elif defined (STANDALONE_OCT)
#include <Havana2/Viewer/QScope2.h>
#endif
#include <Havana2/Viewer/QImageView.h>

#include <DataProcess/OCTProcess/OCTProcess.h>
#ifdef OCT_FLIM
#include <DataProcess/FLIMProcess/FLIMProcess.h>
#endif
#include <DataProcess/ThreadManager.h>

#include <Havana2/Dialog/OctCalibDlg.h>
#ifdef OCT_FLIM
#include <Havana2/Dialog/FlimCalibDlg.h>
#endif


QStreamTab::QStreamTab(QWidget *parent) :
    QDialog(parent), m_pOctCalibDlg(nullptr), m_pImgObjRectImage(nullptr), m_pImgObjCircImage(nullptr),
	m_pCirc(nullptr), m_pMedfilt(nullptr)
#ifdef OCT_FLIM
	, m_pFlimCalibDlg(nullptr), m_pImgObjIntensity(nullptr), m_pImgObjLifetime(nullptr)
#endif
{
	// Set main window objects
	m_pMainWnd = (MainWindow*)parent;
	m_pConfig = m_pMainWnd->m_pConfiguration;
	m_pOperationTab = m_pMainWnd->m_pOperationTab;
	m_pDataAcq = m_pOperationTab->m_pDataAcquisition;
	m_pMemBuff = m_pOperationTab->m_pMemoryBuffer;

	// Create data process object
#ifdef OCT_FLIM
	m_pOCT = new OCTProcess(m_pConfig->nScans, m_pConfig->nAlines);
	m_pOCT->loadCalibration();

	m_pFLIM = new FLIMProcess;
	m_pFLIM->setParameters(m_pConfig);
	m_pFLIM->_resize(np::Uint16Array2(m_pConfig->fnScans, m_pConfig->n4Alines), m_pFLIM->_params);
	m_pFLIM->loadMaskData();
#elif defined (STANDALONE_OCT)
	m_pOCT1 = new OCTProcess(m_pConfig->nScans, m_pConfig->nAlines);
	m_pOCT1->loadCalibration(CH_1);

	m_pOCT2 = new OCTProcess(m_pConfig->nScans, m_pConfig->nAlines);
	m_pOCT2->loadCalibration(CH_2);
#endif	

	// Create thread managers for data processing
	m_pThreadDeinterleave = new ThreadManager("Deinterleave process");
#ifdef OCT_FLIM
	m_pThreadCh1Process = new ThreadManager("OCT image process");
	m_pThreadCh2Process = new ThreadManager("FLIM image process");
#elif defined (STANDALONE_OCT)
	m_pThreadCh1Process = new ThreadManager("Ch1 OCT image process");
	m_pThreadCh2Process = new ThreadManager("Ch2 OCT image process");
#endif
	m_pThreadVisualization = new ThreadManager("Visualization process");

	// Create buffers for threading operation
	m_syncDeinterleaving.allocate_queue_buffer(m_pConfig->nChannels * m_pConfig->nScans, m_pConfig->nAlines, PROCECSSING_BUFFER_SIZE);
	m_syncCh1Processing.allocate_queue_buffer(m_pConfig->nScans, m_pConfig->nAlines, PROCECSSING_BUFFER_SIZE); // Ch1 Processing
	m_syncCh1Visualization.allocate_queue_buffer(m_pConfig->n2ScansFFT, m_pConfig->nAlines, PROCECSSING_BUFFER_SIZE); // Ch1 Visualization
#ifdef OCT_FLIM
	m_syncCh2Processing.allocate_queue_buffer(m_pConfig->fnScans, m_pConfig->n4Alines, PROCECSSING_BUFFER_SIZE); // Ch2 Processing
	m_syncCh2Visualization.allocate_queue_buffer(11, m_pConfig->n4Alines, PROCECSSING_BUFFER_SIZE); // FLIM Visualization
#elif defined (STANDALONE_OCT)
	m_syncCh2Processing.allocate_queue_buffer(m_pConfig->nScans, m_pConfig->nAlines, PROCECSSING_BUFFER_SIZE); // Ch2 Processing
	m_syncCh2Visualization.allocate_queue_buffer(m_pConfig->n2ScansFFT, m_pConfig->nAlines, PROCECSSING_BUFFER_SIZE); // Ch2 OCT Visualization
#endif
	
	// Set signal object
	setDataAcquisitionCallback();
	setDeinterleavingCallback();
	setCh1ProcessingCallback();
	setCh2ProcessingCallback();
	setVisualizationCallback();

	// Create visualization buffers
#ifdef OCT_FLIM
	m_visFringe = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visFringeBg = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visFringeRm = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visImage = np::FloatArray2(m_pConfig->n2ScansFFT, m_pConfig->nAlines);

	m_visPulse = np::FloatArray2(m_pConfig->fnScans, m_pConfig->n4Alines);
	m_visIntensity = np::FloatArray2(m_pConfig->n4Alines, 4);
	m_visMeanDelay = np::FloatArray2(m_pConfig->n4Alines, 4);
	m_visLifetime = np::FloatArray2(m_pConfig->n4Alines, 3);
	
#elif defined (STANDALONE_OCT)
	m_visFringe1 = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visFringeBg1 = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visFringe1Rm = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visImage1 = np::FloatArray2(m_pConfig->n2ScansFFT, m_pConfig->nAlines);

	m_visFringe2 = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visFringeBg2 = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visFringe2Rm = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visImage2 = np::FloatArray2(m_pConfig->n2ScansFFT, m_pConfig->nAlines);
#endif 

	// Create image visualization buffers
	ColorTable temp_ctable;

	m_pImgObjRectImage = new ImageObject(m_pConfig->nAlines, m_pConfig->n2ScansFFT, temp_ctable.m_colorTableVector.at(temp_ctable.gray));
	m_pImgObjCircImage = new ImageObject(2 * CIRC_RADIUS, 2 * CIRC_RADIUS, temp_ctable.m_colorTableVector.at(temp_ctable.gray));
#ifdef OCT_FLIM
	m_pImgObjIntensity = new ImageObject(m_pConfig->n4Alines, RING_THICKNESS, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
	m_pImgObjLifetime = new ImageObject(m_pConfig->n4Alines, RING_THICKNESS, temp_ctable.m_colorTableVector.at(m_pConfig->flimLifetimeColorTable));
#endif
	
	m_pCirc = new circularize(CIRC_RADIUS, m_pConfig->nAlines, false);
	m_pMedfilt = new medfilt(m_pConfig->nAlines, m_pConfig->n2ScansFFT, 3, 3);


    // Create layout
    QHBoxLayout* pHBoxLayout = new QHBoxLayout;
	pHBoxLayout->setSpacing(0);

    // Create graph view
	double voltageCh1 = DIGITIZER_VOLTAGE;
	double voltageCh2 = DIGITIZER_VOLTAGE;
	for (int i = 0; i < m_pConfig->ch1VoltageRange; i++)
		voltageCh1 *= DIGITIZER_VOLTAGE_RATIO;
	for (int i = 0; i < m_pConfig->ch2VoltageRange; i++)
		voltageCh2 *= DIGITIZER_VOLTAGE_RATIO;

#ifdef OCT_FLIM
    m_pScope_FlimPulse = new QScope({ 0, N_VIS_SAMPS_FLIM }, { 0, POWER_2(16) }, 2, 3, 1, voltageCh2 / (double)POWER_2(16), 0, -voltageCh2 / 2, "", "V");
    m_pScope_FlimPulse->setMinimumSize(600, 250);
    m_pScope_OctFringe= new QScope({ 0, (double)m_pConfig->nScans }, { -POWER_2(15), POWER_2(15) }, 2, 3, 1, voltageCh1 / (double)POWER_2(16), 0, 0, "", "V");
    m_pScope_OctFringe->setMinimumSize(600, 250);
    m_pScope_OctDepthProfile = new QScope({ 0, (double)m_pConfig->n2ScansFFT }, {(double)m_pConfig->octDbRange.min, (double)m_pConfig->octDbRange.max}, 2, 2, 1, 1, 0, 0, "", "dB");
    m_pScope_OctDepthProfile->setMinimumSize(600, 250);
#elif defined (STANDALONE_OCT)
	m_pScope_OctFringe = new QScope2({ 0, (double)m_pConfig->nScans }, { -POWER_2(15), POWER_2(15) }, 2, 3, 1, voltageCh1 / (double)POWER_2(16), 0, 0, "", "V");
	m_pScope_OctFringe->setMinimumSize(600, 250);
	m_pScope_OctDepthProfile = new QScope2({ 0, (double)m_pConfig->n2ScansFFT }, { (double)m_pConfig->octDbRange.min, (double)m_pConfig->octDbRange.max }, 2, 2, 1, 1, 0, 0, "", "dB");
	m_pScope_OctDepthProfile->setMinimumSize(600, 250);
#endif
	
    // Create slider for exploring a-lines
    m_pSlider_SelectAline = new QSlider(this);
    m_pSlider_SelectAline->setOrientation(Qt::Horizontal);
    m_pSlider_SelectAline->setRange(0, m_pConfig->nAlines - 1);

    m_pLabel_SelectAline = new QLabel(this);
    QString str; str.sprintf("Current A-line : %4d / %4d   ", 1, m_pConfig->nAlines);
    m_pLabel_SelectAline->setText(str);
    m_pLabel_SelectAline->setBuddy(m_pSlider_SelectAline);

    // Set layout for left panel
    QVBoxLayout *pVBoxLayout_GraphView = new QVBoxLayout;
    pVBoxLayout_GraphView->setSpacing(0);

	QHBoxLayout *pHBoxLayout_SelectAline = new QHBoxLayout;
	pHBoxLayout_SelectAline->addWidget(m_pLabel_SelectAline);
	pHBoxLayout_SelectAline->addWidget(m_pSlider_SelectAline);

#ifdef OCT_FLIM
    pVBoxLayout_GraphView->addWidget(m_pScope_FlimPulse);
    pVBoxLayout_GraphView->addItem(pHBoxLayout_SelectAline);
    pVBoxLayout_GraphView->addWidget(m_pScope_OctFringe);
    pVBoxLayout_GraphView->addWidget(m_pScope_OctDepthProfile);
#elif defined (STANDALONE_OCT)	
	pVBoxLayout_GraphView->addWidget(m_pScope_OctFringe);
	pVBoxLayout_GraphView->addItem(pHBoxLayout_SelectAline);
	pVBoxLayout_GraphView->addWidget(m_pScope_OctDepthProfile);
#endif

    pHBoxLayout->addItem(pVBoxLayout_GraphView);

#ifdef OCT_FLIM
    // Create FLIM visualization option tab
    createFlimVisualizationOptionTab();
#endif
    // Create OCT visualization option tab
    createOctVisualizationOptionTab();
	
    // Create image view
#ifdef OCT_FLIM
	m_pImageView_RectImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), m_pConfig->nAlines, m_pConfig->n2ScansFFT, true);
	m_pImageView_CircImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), 2 * CIRC_RADIUS, 2 * CIRC_RADIUS, true);
#elif defined (STANDALONE_OCT)
	m_pImageView_RectImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), m_pConfig->nAlines, m_pConfig->n2ScansFFT);
	m_pImageView_CircImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), 2 * CIRC_RADIUS, 2 * CIRC_RADIUS);
#endif

    m_pImageView_RectImage->setMinimumSize(350, 350);
    m_pImageView_RectImage->setMovedMouseCallback([&] (QPoint& p) { m_pMainWnd->m_pStatusLabel_ImagePos->setText(QString("(%1, %2)").arg(p.x(), 4).arg(p.y(), 4)); });

    m_pImageView_CircImage->setMinimumSize(350, 350);
    m_pImageView_CircImage->setSquare(true);
	m_pImageView_CircImage->hide();
    //m_pImageView_CircImage->setVisible(false);
    m_pImageView_CircImage->setMovedMouseCallback([&] (QPoint& p) { m_pMainWnd->m_pStatusLabel_ImagePos->setText(QString("(%1, %2)").arg(p.x(), 4).arg(p.y(), 4)); });

    // Set layout for right panel
	QVBoxLayout *pVBoxLayout_RightPanel = new QVBoxLayout;
	pVBoxLayout_RightPanel->setSpacing(0);

#ifdef OCT_FLIM
	pVBoxLayout_RightPanel->addWidget(m_pGroupBox_FlimVisualization);
#endif
	pVBoxLayout_RightPanel->addWidget(m_pGroupBox_OctVisualization);
	pVBoxLayout_RightPanel->addWidget(m_pImageView_RectImage);
	pVBoxLayout_RightPanel->addWidget(m_pImageView_CircImage);

    pHBoxLayout->addItem(pVBoxLayout_RightPanel);

    this->setLayout(pHBoxLayout);


	// Connect signal and slot
#ifdef OCT_FLIM
	connect(this, SIGNAL(plotPulse(float*)), m_pScope_FlimPulse, SLOT(drawData(float*)));
	connect(this, SIGNAL(plotFringe(float*)), m_pScope_OctFringe, SLOT(drawData(float*)));
	connect(this, SIGNAL(plotAline(float*)), m_pScope_OctDepthProfile, SLOT(drawData(float*)));

	connect(this, SIGNAL(makeRgb(ImageObject*, ImageObject*, ImageObject*, ImageObject*)),
		this, SLOT(constructRgbImage(ImageObject*, ImageObject*, ImageObject*, ImageObject*)));
#elif defined (STANDALONE_OCT)	
	connect(this, SIGNAL(plotFringe(float*, float*)), m_pScope_OctFringe, SLOT(drawData(float*, float*)));
	connect(this, SIGNAL(plotAline(float*, float*)), m_pScope_OctDepthProfile, SLOT(drawData(float*, float*)));

	connect(this, SIGNAL(paintRectImage(uint8_t*)), m_pImageView_RectImage, SLOT(drawImage(uint8_t*)));
	connect(this, SIGNAL(paintCircImage(uint8_t*)), m_pImageView_CircImage, SLOT(drawImage(uint8_t*)));
#endif	

	connect(m_pSlider_SelectAline, SIGNAL(valueChanged(int)), this, SLOT(updateAlinePos(int)));
}

QStreamTab::~QStreamTab()
{
	if (m_pImgObjRectImage) delete m_pImgObjRectImage;
	if (m_pImgObjCircImage) delete m_pImgObjCircImage;
#ifdef OCT_FLIM
	if (m_pImgObjIntensity) delete m_pImgObjIntensity;
	if (m_pImgObjLifetime) delete m_pImgObjLifetime;
#endif
	if (m_pMedfilt) delete m_pMedfilt;
	if (m_pCirc) delete m_pCirc;

	if (m_pThreadVisualization) delete m_pThreadVisualization;
	if (m_pThreadCh1Process) delete m_pThreadCh1Process;
	if (m_pThreadCh2Process) delete m_pThreadCh2Process;
	if (m_pThreadDeinterleave) delete m_pThreadDeinterleave;

#ifdef OCT_FLIM
	if (m_pFLIM) delete m_pFLIM;
	if (m_pOCT) delete m_pOCT;
#elif defined (STANDALONE_OCT)
	if (m_pOCT1) delete m_pOCT1;
	if (m_pOCT2) delete m_pOCT2;
#endif
}

void QStreamTab::keyPressEvent(QKeyEvent *e)
{
	if (e->key() != Qt::Key_Escape)
		QDialog::keyPressEvent(e);
}

void QStreamTab::setWidgetsText()
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


#ifdef OCT_FLIM
void QStreamTab::createFlimVisualizationOptionTab()
{
    // Create widgets for FLIM visualization option tab
    m_pGroupBox_FlimVisualization = new QGroupBox;
	m_pGroupBox_FlimVisualization->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    QGridLayout *pGridLayout_FlimVisualization = new QGridLayout;
	pGridLayout_FlimVisualization->setSpacing(3);

    // Create widgets for FLIM calibration
    m_pPushButton_FlimCalibration = new QPushButton(this);
	m_pPushButton_FlimCalibration->setText("FLIM Calibration...");
	m_pPushButton_FlimCalibration->setFixedWidth(120);

	// Create widgets for FLIM visualization
    m_pSpinBox_DataChannel = new QSpinBox(this);
    m_pSpinBox_DataChannel->setRange(0, 3);
    m_pSpinBox_DataChannel->setSingleStep(1);
    m_pSpinBox_DataChannel->setValue(0);
	m_pSpinBox_DataChannel->setAlignment(Qt::AlignCenter);
	m_pSpinBox_DataChannel->setFixedWidth(35);
    m_pLabel_DataChannel = new QLabel("Data Channel ", this);
    m_pLabel_DataChannel->setBuddy(m_pSpinBox_DataChannel);

    // Create widgets for FLIM emission control
    m_pComboBox_EmissionChannel = new QComboBox(this);
    m_pComboBox_EmissionChannel->addItem("Ch 1");
    m_pComboBox_EmissionChannel->addItem("Ch 2");
    m_pComboBox_EmissionChannel->addItem("Ch 3");
    m_pLabel_EmissionChannel = new QLabel("Em Channel   ", this);
    m_pLabel_EmissionChannel->setBuddy(m_pComboBox_EmissionChannel);

	// Create widgets for FLIM lifetime colortable selection
	ColorTable temp_ctable;
	m_pComboBox_LifetimeColorTable = new QComboBox(this);
	for (int i = 0; i < temp_ctable.m_cNameVector.size(); i++)
		m_pComboBox_LifetimeColorTable->addItem(temp_ctable.m_cNameVector.at(i));
	m_pComboBox_LifetimeColorTable->setCurrentIndex(m_pConfig->flimLifetimeColorTable);
	m_pLabel_LifetimeColorTable = new QLabel("Lifetime Colortable ", this);
	m_pLabel_LifetimeColorTable->setBuddy(m_pComboBox_LifetimeColorTable);

    // Create line edit widgets for FLIM contrast adjustment
    m_pLineEdit_IntensityMax = new QLineEdit(this);
    m_pLineEdit_IntensityMax->setFixedWidth(30);
    m_pLineEdit_IntensityMax->setText(QString::number(m_pConfig->flimIntensityRange.max, 'f', 1));
	m_pLineEdit_IntensityMax->setAlignment(Qt::AlignCenter);
    m_pLineEdit_IntensityMin = new QLineEdit(this);
    m_pLineEdit_IntensityMin->setFixedWidth(30);
    m_pLineEdit_IntensityMin->setText(QString::number(m_pConfig->flimIntensityRange.min, 'f', 1));
	m_pLineEdit_IntensityMin->setAlignment(Qt::AlignCenter);
    m_pLineEdit_LifetimeMax = new QLineEdit(this);
    m_pLineEdit_LifetimeMax->setFixedWidth(30);
    m_pLineEdit_LifetimeMax->setText(QString::number(m_pConfig->flimLifetimeRange.max, 'f', 1));
	m_pLineEdit_LifetimeMax->setAlignment(Qt::AlignCenter);
    m_pLineEdit_LifetimeMin = new QLineEdit(this);
    m_pLineEdit_LifetimeMin->setFixedWidth(30);
    m_pLineEdit_LifetimeMin->setText(QString::number(m_pConfig->flimLifetimeRange.min, 'f', 1));
	m_pLineEdit_LifetimeMin->setAlignment(Qt::AlignCenter);

    // Create color bar for FLIM visualization
    uint8_t color[256];
    for (int i = 0; i < 256; i++)
        color[i] = i;
	
    m_pImageView_IntensityColorbar = new QImageView(ColorTable::colortable(INTENSITY_COLORTABLE), 256, 1);
	m_pImageView_IntensityColorbar->setFixedHeight(15);
    m_pImageView_IntensityColorbar->drawImage(color);
    m_pImageView_LifetimeColorbar = new QImageView(ColorTable::colortable(m_pConfig->flimLifetimeColorTable), 256, 1);
	m_pImageView_LifetimeColorbar->setFixedHeight(15);
    m_pImageView_LifetimeColorbar->drawImage(color);
    m_pLabel_NormIntensity = new QLabel("N Intensity", this);
    m_pLabel_NormIntensity->setFixedWidth(60);
    m_pLabel_Lifetime = new QLabel("Lifetime", this);
    m_pLabel_Lifetime->setFixedWidth(60);

    // Set layout
	QHBoxLayout *pHBoxLayout_DataChannel = new QHBoxLayout;
	pHBoxLayout_DataChannel->setSpacing(3);
	pHBoxLayout_DataChannel->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));
	pHBoxLayout_DataChannel->addWidget(m_pLabel_DataChannel);
	pHBoxLayout_DataChannel->addWidget(m_pSpinBox_DataChannel);

	pGridLayout_FlimVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 0);
	pGridLayout_FlimVisualization->addItem(pHBoxLayout_DataChannel, 0, 1, 1, 2);
	pGridLayout_FlimVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Fixed), 0, 3);
	pGridLayout_FlimVisualization->addWidget(m_pPushButton_FlimCalibration, 0, 4, 1, 2);	
	
	pGridLayout_FlimVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
	pGridLayout_FlimVisualization->addWidget(m_pLabel_LifetimeColorTable, 1, 1);
	pGridLayout_FlimVisualization->addWidget(m_pComboBox_LifetimeColorTable, 1, 2);
	pGridLayout_FlimVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Fixed), 1, 3);
	pGridLayout_FlimVisualization->addWidget(m_pLabel_EmissionChannel, 1, 4);
	pGridLayout_FlimVisualization->addWidget(m_pComboBox_EmissionChannel, 1, 5);

    QHBoxLayout *pHBoxLayout_IntensityColorbar = new QHBoxLayout;
    pHBoxLayout_IntensityColorbar->setSpacing(3);
    pHBoxLayout_IntensityColorbar->addWidget(m_pLabel_NormIntensity);
    pHBoxLayout_IntensityColorbar->addWidget(m_pLineEdit_IntensityMin);
    pHBoxLayout_IntensityColorbar->addWidget(m_pImageView_IntensityColorbar);
    pHBoxLayout_IntensityColorbar->addWidget(m_pLineEdit_IntensityMax);
    QHBoxLayout *pHBoxLayout_LifetimeColorbar = new QHBoxLayout;
    pHBoxLayout_LifetimeColorbar->setSpacing(3);
    pHBoxLayout_LifetimeColorbar->addWidget(m_pLabel_Lifetime);
    pHBoxLayout_LifetimeColorbar->addWidget(m_pLineEdit_LifetimeMin);
    pHBoxLayout_LifetimeColorbar->addWidget(m_pImageView_LifetimeColorbar);
    pHBoxLayout_LifetimeColorbar->addWidget(m_pLineEdit_LifetimeMax);

	pGridLayout_FlimVisualization->addItem(pHBoxLayout_IntensityColorbar, 3, 0, 1, 6);
	pGridLayout_FlimVisualization->addItem(pHBoxLayout_LifetimeColorbar, 4, 0, 1, 6);

	m_pGroupBox_FlimVisualization->setLayout(pGridLayout_FlimVisualization);

	// Connect signal and slot
	connect(m_pSpinBox_DataChannel, SIGNAL(valueChanged(int)), this, SLOT(changeFlimCh(int)));
	connect(m_pComboBox_LifetimeColorTable, SIGNAL(currentIndexChanged(int)), this, SLOT(changeLifetimeColorTable(int)));
	connect(m_pLineEdit_IntensityMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustFlimContrast()));
	connect(m_pLineEdit_IntensityMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustFlimContrast()));
	connect(m_pLineEdit_LifetimeMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustFlimContrast()));
	connect(m_pLineEdit_LifetimeMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustFlimContrast()));

	connect(m_pPushButton_FlimCalibration, SIGNAL(clicked(bool)), this, SLOT(createFlimCalibDlg()));
}
#endif

void QStreamTab::createOctVisualizationOptionTab()
{
    // Create widgets for OCT visualization option tab
    m_pGroupBox_OctVisualization = new QGroupBox;
	m_pGroupBox_OctVisualization->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    QGridLayout *pGridLayout_OctVisualization = new QGridLayout;
	pGridLayout_OctVisualization->setSpacing(3);

	// Create widgets for OCT calibration
	m_pPushButton_OctCalibration = new QPushButton(this);
	m_pPushButton_OctCalibration->setText("OCT Calibration...");

    // Create widgets for OCT visualization
    m_pCheckBox_CircularizeImage = new QCheckBox(this);
    m_pCheckBox_CircularizeImage->setText("Circularize Image");
    m_pCheckBox_ShowBgRemovedSignal = new QCheckBox(this);
    m_pCheckBox_ShowBgRemovedSignal->setText("BG Removed Fringe");
    m_pCheckBox_ShowBgRemovedSignal->setChecked(true);
	changeFringeBg(true);

    m_pLineEdit_CircCenter = new QLineEdit(this);
    m_pLineEdit_CircCenter->setFixedWidth(35);
    m_pLineEdit_CircCenter->setText(QString("%1").arg(m_pConfig->circCenter));
	m_pLineEdit_CircCenter->setAlignment(Qt::AlignCenter);
    m_pLabel_CircCenter = new QLabel("Circ Center", this);
    m_pLabel_CircCenter->setBuddy(m_pLabel_CircCenter);

    // Create widgets for OCT color table
    m_pComboBox_OctColorTable = new QComboBox(this);
    m_pComboBox_OctColorTable->addItem("gray");
    m_pComboBox_OctColorTable->addItem("invgray");
    m_pComboBox_OctColorTable->addItem("sepia");
	m_pComboBox_OctColorTable->setCurrentIndex(m_pConfig->octColorTable);
    m_pLabel_OctColorTable = new QLabel("OCT Colortable", this);
    m_pLabel_OctColorTable->setBuddy(m_pComboBox_OctColorTable);

    // Create line edit widgets for OCT contrast adjustment
    m_pLineEdit_OctDbMax = new QLineEdit(this);
    m_pLineEdit_OctDbMax->setFixedWidth(30);
	m_pLineEdit_OctDbMax->setText(QString::number(m_pConfig->octDbRange.max));
	m_pLineEdit_OctDbMax->setAlignment(Qt::AlignCenter);
    m_pLineEdit_OctDbMin = new QLineEdit(this);
    m_pLineEdit_OctDbMin->setFixedWidth(30);
    m_pLineEdit_OctDbMin->setText(QString::number(m_pConfig->octDbRange.min));
	m_pLineEdit_OctDbMin->setAlignment(Qt::AlignCenter);

    // Create color bar for OCT visualization
    uint8_t color[256];
    for (int i = 0; i < 256; i++)
        color[i] = i;

    m_pImageView_OctDbColorbar = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), 256, 1);
	m_pImageView_OctDbColorbar->setFixedHeight(15);
    m_pImageView_OctDbColorbar->drawImage(color);
    m_pLabel_OctDb = new QLabel("OCT dB", this);
    m_pLabel_OctDb->setFixedWidth(60);

    // Set layout
	pGridLayout_OctVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 0);
	pGridLayout_OctVisualization->addWidget(m_pPushButton_OctCalibration, 0, 3, 1, 2);

	pGridLayout_OctVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
	pGridLayout_OctVisualization->addWidget(m_pCheckBox_CircularizeImage, 1, 1);
	pGridLayout_OctVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Fixed), 1, 2);
	pGridLayout_OctVisualization->addWidget(m_pLabel_CircCenter, 1, 3);
	pGridLayout_OctVisualization->addWidget(m_pLineEdit_CircCenter, 1, 4);

	pGridLayout_OctVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 2, 0);
	pGridLayout_OctVisualization->addWidget(m_pCheckBox_ShowBgRemovedSignal, 2, 1);
	pGridLayout_OctVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Fixed), 2, 2);
	pGridLayout_OctVisualization->addWidget(m_pLabel_OctColorTable, 2, 3);
	pGridLayout_OctVisualization->addWidget(m_pComboBox_OctColorTable, 2, 4);

    QHBoxLayout *pHBoxLayout_OctDbColorbar = new QHBoxLayout;
    pHBoxLayout_OctDbColorbar->setSpacing(3);
    pHBoxLayout_OctDbColorbar->addWidget(m_pLabel_OctDb);
    pHBoxLayout_OctDbColorbar->addWidget(m_pLineEdit_OctDbMin);
    pHBoxLayout_OctDbColorbar->addWidget(m_pImageView_OctDbColorbar);
    pHBoxLayout_OctDbColorbar->addWidget(m_pLineEdit_OctDbMax);

	pGridLayout_OctVisualization->addItem(pHBoxLayout_OctDbColorbar, 3, 0, 1, 5);

    m_pGroupBox_OctVisualization->setLayout(pGridLayout_OctVisualization);
    
	// Connect signal and slot
	connect(m_pCheckBox_CircularizeImage, SIGNAL(toggled(bool)), this, SLOT(changeVisImage(bool)));
	connect(m_pCheckBox_ShowBgRemovedSignal, SIGNAL(toggled(bool)), this, SLOT(changeFringeBg(bool)));
	connect(m_pLineEdit_CircCenter, SIGNAL(textEdited(const QString &)), this, SLOT(checkCircCenter(const QString &)));
	connect(m_pComboBox_OctColorTable, SIGNAL(currentIndexChanged(int)), this, SLOT(changeOctColorTable(int)));
	connect(m_pLineEdit_OctDbMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustOctContrast()));
	connect(m_pLineEdit_OctDbMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustOctContrast()));

	connect(m_pPushButton_OctCalibration, SIGNAL(clicked(bool)), this, SLOT(createOctCalibDlg()));
}


void QStreamTab::setDataAcquisitionCallback()
{
#if PX14_ENABLE
	m_pDataAcq->ConnectDaqAcquiredData([&](int frame_count, const np::Array<uint16_t, 2>& frame) {

		// Data halving
		/* To be updated*/

		// Data transfer		
		if (!(frame_count % RENEWAL_COUNT))
		{
			// Data transfer for data deinterleaving
			const uint16_t* frame_ptr = frame.raw_ptr();

			// Get buffer from threading queue
			uint16_t* fulldata_ptr = nullptr;
			{
				std::unique_lock<std::mutex> lock(m_syncDeinterleaving.mtx);

				if (!m_syncDeinterleaving.queue_buffer.empty())
				{
					fulldata_ptr = m_syncDeinterleaving.queue_buffer.front();
					m_syncDeinterleaving.queue_buffer.pop();
				}
			}

			if (fulldata_ptr != nullptr)
			{
				// Body
				int frame_length = m_pConfig->nFrameSize;
				memcpy(fulldata_ptr, frame_ptr, sizeof(uint16_t) * frame_length);

				// Push the buffer to sync Queue
				m_syncDeinterleaving.Queue_sync.push(fulldata_ptr);
			}
		}

		// Buffering (When recording)
		if (m_pMemBuff->m_bIsRecording)
		{
			if (m_pMemBuff->m_nRecordedFrames < WRITING_BUFFER_SIZE)
			{
				// Push to the copy queue for copying transfered data in copy thread
				m_pMemBuff->m_syncBuffering.Queue_sync.push(frame.raw_ptr());
				m_pMemBuff->m_nRecordedFrames++;
			}
			else
			{
				// Finish recording when the buffer is full
				m_pMemBuff->m_bIsRecording = false;
				m_pOperationTab->setRecordingButton(false);
			}
		}
	});

	m_pDataAcq->ConnectDaqStopData([&]() {
		m_syncDeinterleaving.Queue_sync.push(nullptr);
	});

	m_pDataAcq->ConnectDaqSendStatusMessage([&](const char * msg) {
		QMessageBox MsgBox(QMessageBox::Critical, "Error", msg);
		MsgBox.exec();
	});
#endif
}

void QStreamTab::setDeinterleavingCallback()
{
	// Deinterleave Process Signal Objects ///////////////////////////////////////////////////////////////////////////////////////////
	m_pThreadDeinterleave->DidAcquireData += [&](int frame_count) {

		// Get the buffer from the previous sync Queue
		uint16_t* fulldata_ptr = m_syncDeinterleaving.Queue_sync.pop();
		if (fulldata_ptr != nullptr)
		{
			// Get buffers from threading queues
			uint16_t* ch1_ptr = nullptr;
			uint16_t* ch2_ptr = nullptr;
			{
				std::unique_lock<std::mutex> lock(m_syncCh1Processing.mtx);

				if (!m_syncCh1Processing.queue_buffer.empty())
				{
					ch1_ptr = m_syncCh1Processing.queue_buffer.front();
					m_syncCh1Processing.queue_buffer.pop();
				}
			}
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
				int frame_length = m_pConfig->nFrameSize / m_pConfig->nChannels;
				ippsCplxToReal_16sc((Ipp16sc *)fulldata_ptr, (Ipp16s *)ch1_ptr, (Ipp16s *)ch2_ptr, frame_length);
#ifdef OCT_FLIM
				// Draw fringe & pulse	
				ippsConvert_16u32f(ch1_ptr, m_visFringe.raw_ptr(), m_visFringe.length());
				ippsSub_32f(m_visFringe.raw_ptr(), m_visFringeBg.raw_ptr(), m_visFringeRm.raw_ptr(), m_visFringe.length());
				emit plotFringe(&m_visFringeRm(0, m_pSlider_SelectAline->value()));
				
				ippsConvert_16u32f(ch2_ptr, m_visPulse.raw_ptr(), m_visPulse.length());
				emit plotPulse(&m_visPulse(m_pConfig->preTrigSamps + m_pConfig->nScans * m_pConfig->flimCh, m_pSlider_SelectAline->value() / 4));

				// Transfer to OCT calibration dlg
				if (m_pOctCalibDlg)
					emit m_pOctCalibDlg->catchFringe(ch1_ptr);

#elif defined (STANDALONE_OCT)
				// Draw fringes	
				ippsConvert_16u32f(ch1_ptr, m_visFringe1.raw_ptr(), m_visFringe1.length());
				ippsSub_32f(m_visFringe1.raw_ptr(), m_visFringeBg1.raw_ptr(), m_visFringe1Rm.raw_ptr(), m_visFringe1Rm.length());
				
				ippsConvert_16u32f(ch2_ptr, m_visFringe2.raw_ptr(), m_visFringe2.length());
				ippsSub_32f(m_visFringe2.raw_ptr(), m_visFringeBg2.raw_ptr(), m_visFringe2Rm.raw_ptr(), m_visFringe2Rm.length());

				emit plotFringe(&m_visFringe1Rm(0, m_pSlider_SelectAline->value()), &m_visFringe2Rm(0, m_pSlider_SelectAline->value()));

				// Transfer to OCT calibration dlg
				if (m_pOctCalibDlg)
					emit m_pOctCalibDlg->catchFringe(ch1_ptr, ch2_ptr);
#endif			
				// Push the buffers to sync Queues
				m_syncCh1Processing.Queue_sync.push(ch1_ptr);
				m_syncCh2Processing.Queue_sync.push(ch2_ptr);

				// Return (push) the buffer to the previous threading queue
				{
					std::unique_lock<std::mutex> lock(m_syncDeinterleaving.mtx);
					m_syncDeinterleaving.queue_buffer.push(fulldata_ptr);
				}
			}
		}
		else
			m_pThreadDeinterleave->_running = false;

        (void)frame_count;
	};

	m_pThreadDeinterleave->DidStopData += [&]() {
		m_syncCh1Processing.queue_buffer.push(nullptr);
		m_syncCh2Processing.queue_buffer.push(nullptr);
	};

	m_pThreadDeinterleave->SendStatusMessage += [&](const char* msg) {
		QMessageBox MsgBox(QMessageBox::Critical, "Error", msg);
		MsgBox.exec();
	};
}

void QStreamTab::setCh1ProcessingCallback()
{
	// Ch1 Process Signal Objects ///////////////////////////////////////////////////////////////////////////////////////////
	m_pThreadCh1Process->DidAcquireData += [&](int frame_count) {

		// Get the buffer from the previous sync Queue
		uint16_t* ch1_data = m_syncCh1Processing.Queue_sync.pop();
		if (ch1_data != nullptr)
		{
			// Get buffers from threading queues
			float* res_ptr = nullptr;
			{
				std::unique_lock<std::mutex> lock(m_syncCh1Visualization.mtx);

				if (!m_syncCh1Visualization.queue_buffer.empty())
				{
					res_ptr = m_syncCh1Visualization.queue_buffer.front();
					m_syncCh1Visualization.queue_buffer.pop();
				}
			}

			if (res_ptr != nullptr)
            {
				// Body
#ifdef OCT_FLIM
				(*m_pOCT)(res_ptr, ch1_data);
#elif defined (STANDALONE_OCT)
				(*m_pOCT1)(res_ptr, ch1_data);
#endif
				// Push the buffers to sync Queues
				m_syncCh1Visualization.Queue_sync.push(res_ptr);

				// Return (push) the buffer to the previous threading queue
				{
					std::unique_lock<std::mutex> lock(m_syncCh1Processing.mtx);
					m_syncCh1Processing.queue_buffer.push(ch1_data);
				}
			}
		}
		else
			m_pThreadCh1Process->_running = false;

        (void)frame_count;
	};

	m_pThreadCh1Process->DidStopData += [&]() {
		m_syncCh1Processing.Queue_sync.push(nullptr);
	};

	m_pThreadCh1Process->SendStatusMessage += [&](const char* msg) {
		QMessageBox MsgBox(QMessageBox::Critical, "Error", msg);
		MsgBox.exec();
	};
}

void QStreamTab::setCh2ProcessingCallback()
{
	// Ch2 Process Signal Objects ///////////////////////////////////////////////////////////////////////////////////////////
	m_pThreadCh2Process->DidAcquireData += [&](int frame_count) {

		// Get the buffer from the previous sync Queue
		uint16_t* ch2_data = m_syncCh2Processing.Queue_sync.pop();
		if (ch2_data != nullptr)
		{
			// Get buffers from threading queues
			float* res_ptr = nullptr;
			{
				std::unique_lock<std::mutex> lock(m_syncCh2Visualization.mtx);

				if (!m_syncCh2Visualization.queue_buffer.empty())
				{
					res_ptr = m_syncCh2Visualization.queue_buffer.front();
					m_syncCh2Visualization.queue_buffer.pop();
				}
			}

			if (res_ptr != nullptr)
			{
				// Body
#ifdef OCT_FLIM
				np::FloatArray2 intensity (res_ptr + 0 * m_pConfig->n4Alines, m_pConfig->n4Alines, 4);
				np::FloatArray2 mean_delay(res_ptr + 4 * m_pConfig->n4Alines, m_pConfig->n4Alines, 4);
				np::FloatArray2 lifetime  (res_ptr + 8 * m_pConfig->n4Alines, m_pConfig->n4Alines, 3);
				np::Uint16Array2 pulse(ch2_data, m_pConfig->fnScans, m_pConfig->n4Alines);

				(*m_pFLIM)(intensity, mean_delay, lifetime, pulse);

				// Transfer to FLIM calibration dlg
				if (m_pFlimCalibDlg)
					emit m_pFlimCalibDlg->plotRoiPulse(m_pFLIM, m_pSlider_SelectAline->value() / 4);

#elif defined (STANDALONE_OCT)
				(*m_pOCT2)(res_ptr, ch2_data);
#endif		
				// Push the buffers to sync Queues
				m_syncCh2Visualization.Queue_sync.push(res_ptr);

				// Return (push) the buffer to the previous threading queue
				{
					std::unique_lock<std::mutex> lock(m_syncCh2Processing.mtx);
					m_syncCh2Processing.queue_buffer.push(ch2_data);
				}
			}
		}
		else
			m_pThreadCh2Process->_running = false;

        (void)frame_count;
	};

	m_pThreadCh2Process->DidStopData += [&]() {
		m_syncCh2Processing.Queue_sync.push(nullptr);
	};

	m_pThreadCh2Process->SendStatusMessage += [&](const char* msg) {
		QMessageBox MsgBox(QMessageBox::Critical, "Error", msg);
		MsgBox.exec();
	};
}

void QStreamTab::setVisualizationCallback()
{
	// Visualization Signal Objects ///////////////////////////////////////////////////////////////////////////////////////////
	m_pThreadVisualization->DidAcquireData += [&](int frame_count) {

		// Get the buffers from the previous sync Queues
		float* res1_data = m_syncCh1Visualization.Queue_sync.pop();
		float* res2_data = m_syncCh2Visualization.Queue_sync.pop();
		if ((res1_data != nullptr) && (res2_data != nullptr))
		{
			// Body	
			if (m_pOperationTab->isAcquisitionButtonToggled()) // Only valid if acquisition is running
            {
				// Draw A-lines
#ifdef OCT_FLIM
				m_visImage = np::FloatArray2(res1_data, m_pConfig->n2ScansFFT, m_pConfig->nAlines);
				emit plotAline(&m_visImage(0, m_pSlider_SelectAline->value()));

				m_visIntensity = np::FloatArray2(res2_data + 0 * m_pConfig->n4Alines, m_pConfig->n4Alines, 4);
				m_visMeanDelay = np::FloatArray2(res2_data + 4 * m_pConfig->n4Alines, m_pConfig->n4Alines, 4);
				m_visLifetime = np::FloatArray2(res2_data + 8 * m_pConfig->n4Alines, m_pConfig->n4Alines, 3);

                // Draw Images
                visualizeImage(m_visImage.raw_ptr(), m_visIntensity.raw_ptr(), m_visMeanDelay.raw_ptr(), m_visLifetime.raw_ptr());

#elif defined (STANDALONE_OCT)
				m_visImage1 = np::FloatArray2(res1_data, m_pConfig->n2ScansFFT, m_pConfig->nAlines);
				m_visImage2 = np::FloatArray2(res2_data, m_pConfig->n2ScansFFT, m_pConfig->nAlines);
				emit plotAline(&m_visImage1(0, m_pSlider_SelectAline->value()), &m_visImage2(0, m_pSlider_SelectAline->value()));

                // Draw Images
                visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr());
#endif
			}

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
			if (res1_data != nullptr)
			{
				float* res1_temp = res1_data;
				do
				{
					m_syncCh1Visualization.queue_buffer.push(res1_temp);
					res1_temp = m_syncCh1Visualization.Queue_sync.pop();
				} while (res1_temp != nullptr);
			}
			else if (res2_data != nullptr)
			{
				float* res2_temp = res2_data;
				do
				{
					m_syncCh1Visualization.queue_buffer.push(res2_temp);
					res2_temp = m_syncCh1Visualization.Queue_sync.pop();
				} while (res2_temp != nullptr);
			}

			m_pThreadVisualization->_running = false;

            (void)frame_count;
		}
	};

	m_pThreadVisualization->DidStopData += [&]() {
		m_syncCh1Visualization.Queue_sync.push(nullptr);
		m_syncCh2Visualization.Queue_sync.push(nullptr);
	};

	m_pThreadVisualization->SendStatusMessage += [&](const char* msg) {
		QMessageBox MsgBox(QMessageBox::Critical, "Error", msg);
		MsgBox.exec();
	};
}


void QStreamTab::setCh1ScopeVoltRange(int idx)
{
	double voltage = DIGITIZER_VOLTAGE;
	for (int i = 0; i < idx; i++)
		voltage *= DIGITIZER_VOLTAGE_RATIO;
		
	m_pScope_OctFringe->resetAxis({ 0, (double)m_pConfig->nScans }, { -POWER_2(15), POWER_2(15) }, 1, voltage / (double)POWER_2(16), 0, 0, "", "V");
}

#ifdef OCT_FLIM
void QStreamTab::setCh2ScopeVoltRange(int idx)
{
	double voltage = DIGITIZER_VOLTAGE;
	for (int i = 0; i < idx; i++)
		voltage *= DIGITIZER_VOLTAGE_RATIO;

	m_pScope_FlimPulse->resetAxis({ 0, N_VIS_SAMPS_FLIM }, { 0, POWER_2(16) }, 1, voltage / (double)POWER_2(16), 0, -voltage / 2, "", "V");
}
#endif

void QStreamTab::resetObjectsForAline(int nAlines) // need modification
{	
	// Create data process object
#ifdef OCT_FLIM
	if (m_pOCT)
	{
		delete m_pOCT;
		m_pOCT = new OCTProcess(m_pConfig->nScans, nAlines);
		m_pOCT->loadCalibration();
	}
	if (m_pFLIM)
	{
		delete m_pFLIM;
		m_pFLIM = new FLIMProcess;
		m_pFLIM->setParameters(m_pConfig);
	}
#elif defined (STANDALONE_OCT)
	if (m_pOCT1)
	{
		delete m_pOCT1;
		m_pOCT1 = new OCTProcess(m_pConfig->nScans, nAlines);
		m_pOCT1->loadCalibration(CH_1);
	}
	if (m_pOCT2)
	{
		delete m_pOCT2;
		m_pOCT2 = new OCTProcess(m_pConfig->nScans, nAlines);
		m_pOCT2->loadCalibration(CH_2);
	}
#endif

	// Create buffers for threading operation
	m_syncDeinterleaving.deallocate_queue_buffer();
	m_syncCh1Processing.deallocate_queue_buffer();
	m_syncCh1Visualization.deallocate_queue_buffer();
	m_syncCh2Processing.deallocate_queue_buffer();
	m_syncCh2Visualization.deallocate_queue_buffer();

	m_syncDeinterleaving.allocate_queue_buffer(m_pConfig->nChannels * m_pConfig->nScans, nAlines, PROCECSSING_BUFFER_SIZE);
	m_syncCh1Processing.allocate_queue_buffer(m_pConfig->nScans, nAlines, PROCECSSING_BUFFER_SIZE);
	m_syncCh1Visualization.allocate_queue_buffer(m_pConfig->n2ScansFFT, nAlines, PROCECSSING_BUFFER_SIZE);	
#ifdef OCT_FLIM
	m_syncCh2Processing.allocate_queue_buffer(m_pConfig->fnScans, nAlines / 4, PROCECSSING_BUFFER_SIZE);
	m_syncCh2Visualization.allocate_queue_buffer(11, nAlines / 4, PROCECSSING_BUFFER_SIZE);
#elif defined (STANDALONE_OCT)
	m_syncCh2Processing.allocate_queue_buffer(m_pConfig->nScans, nAlines, PROCECSSING_BUFFER_SIZE);
	m_syncCh2Visualization.allocate_queue_buffer(m_pConfig->n2ScansFFT, nAlines, PROCECSSING_BUFFER_SIZE);
#endif

	// Reset rect image size
	m_pImageView_RectImage->resetSize(nAlines, m_pConfig->n2ScansFFT);

	// Reset scan adjust range
#ifdef GALVANO_MIRROR
	m_pMainWnd->m_pDeviceControlTab->setScrollBarRange(nAlines);
#endif
	
	// Create visualization buffers
#ifdef OCT_FLIM
	m_visFringe = np::FloatArray2(m_pConfig->nScans, nAlines);
	m_visFringeBg = np::FloatArray2(m_pConfig->nScans, nAlines);
	m_visFringeRm = np::FloatArray2(m_pConfig->nScans, nAlines);
	m_visImage = np::FloatArray2(m_pConfig->n2ScansFFT, nAlines);

	m_visPulse = np::FloatArray2(m_pConfig->fnScans, nAlines / 4);
	m_visIntensity = np::FloatArray2(nAlines / 4, 4);
	m_visMeanDelay = np::FloatArray2(nAlines / 4, 4);
	m_visLifetime = np::FloatArray2(nAlines / 4, 3);

	// FLIM initialization
	ippiSet_32f_C1R(m_pFLIM->_params.bg, m_visPulse.raw_ptr(),
		sizeof(float) * m_pConfig->fnScans, { m_pConfig->fnScans, nAlines / 4 });

	np::Uint16Array2 temp_pulse(m_pConfig->fnScans, nAlines / 4);
	ippiSet_16u_C1R((uint16_t)m_pFLIM->_params.bg, temp_pulse.raw_ptr(),
		sizeof(uint16_t) * m_pConfig->fnScans, { m_pConfig->fnScans, nAlines / 4 });
	(*m_pFLIM)(m_visIntensity, m_visMeanDelay, m_visLifetime, temp_pulse);
	m_pFLIM->loadMaskData();

#elif defined (STANDALONE_OCT)
	m_visFringe1 = np::FloatArray2(m_pConfig->nScans, nAlines);
	m_visFringeBg1 = np::FloatArray2(m_pConfig->nScans, nAlines);
	m_visFringe1Rm = np::FloatArray2(m_pConfig->nScans, nAlines);
	m_visImage1 = np::FloatArray2(m_pConfig->n2ScansFFT, nAlines);

	m_visFringe2 = np::FloatArray2(m_pConfig->nScans, nAlines);
	m_visFringeBg2 = np::FloatArray2(m_pConfig->nScans, nAlines);
	m_visFringe2Rm = np::FloatArray2(m_pConfig->nScans, nAlines);
	m_visImage2 = np::FloatArray2(m_pConfig->n2ScansFFT, nAlines);
#endif 

	// Create image visualization buffers
	ColorTable temp_ctable;

	if (m_pImgObjRectImage) delete m_pImgObjRectImage;
	m_pImgObjRectImage = new ImageObject(nAlines, m_pConfig->n2ScansFFT, temp_ctable.m_colorTableVector.at(m_pComboBox_OctColorTable->currentIndex()));
#ifdef OCT_FLIM
	if (m_pImgObjIntensity) delete m_pImgObjIntensity;
	m_pImgObjIntensity = new ImageObject(nAlines / 4, RING_THICKNESS, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
	if (m_pImgObjLifetime) delete m_pImgObjLifetime;
	m_pImgObjLifetime = new ImageObject(nAlines / 4, RING_THICKNESS, temp_ctable.m_colorTableVector.at(m_pComboBox_LifetimeColorTable->currentIndex()));
#endif

	// Create circularize object
	if (m_pCirc)
	{
		delete m_pCirc;
		m_pCirc = new circularize(CIRC_RADIUS, nAlines, false);
	}
	if (m_pMedfilt)
	{
		delete m_pMedfilt;
		m_pMedfilt = new medfilt(nAlines, m_pConfig->n2ScansFFT, 3, 3);
	}

	// Reset fringe background
	changeFringeBg(m_pCheckBox_ShowBgRemovedSignal->isChecked());

	// Reset slider range
	m_pSlider_SelectAline->setRange(0, nAlines - 1);

	// Reset slider label
	QString str; str.sprintf("Current A-line : %4d / %4d   ", 1, nAlines);
	m_pLabel_SelectAline->setText(str);
}

#ifdef OCT_FLIM
void QStreamTab::visualizeImage(float* res1, float* res2, float* res3, float* res4)
#elif defined (STANDALONE_OCT)
void QStreamTab::visualizeImage(float* res1, float* res2)
#endif
{
	IppiSize roi_oct = { m_pConfig->n2ScansFFT, m_pConfig->nAlines };
	
	// OCT Visualization
	np::Uint8Array2 scale_temp(roi_oct.width, roi_oct.height);
	ippiScale_32f8u_C1R(res1, roi_oct.width * sizeof(float), scale_temp.raw_ptr(), roi_oct.width * sizeof(uint8_t), roi_oct, m_pConfig->octDbRange.min, m_pConfig->octDbRange.max);
	ippiTranspose_8u_C1R(scale_temp.raw_ptr(), roi_oct.width * sizeof(uint8_t), m_pImgObjRectImage->arr.raw_ptr(), roi_oct.height * sizeof(uint8_t), roi_oct);
	(*m_pMedfilt)(m_pImgObjRectImage->arr.raw_ptr());

#ifdef OCT_FLIM
	// FLIM Visualization
	IppiSize roi_flim = { m_pConfig->n4Alines, 1 };

	float* scanIntensity = res2 + (1 + m_pComboBox_EmissionChannel->currentIndex()) * roi_flim.width;
	uint8_t* rectIntensity = m_pImgObjIntensity->arr.raw_ptr();	
	ippiScale_32f8u_C1R(scanIntensity, roi_flim.width * sizeof(float), rectIntensity, roi_flim.width * sizeof(uint8_t), roi_flim, m_pConfig->flimIntensityRange.min, m_pConfig->flimIntensityRange.max);
	
	float* scanLifetime = res4 + (m_pComboBox_EmissionChannel->currentIndex()) * roi_flim.width;
	uint8_t* rectLifetime = m_pImgObjLifetime->arr.raw_ptr();	
	ippiScale_32f8u_C1R(scanLifetime, roi_flim.width * sizeof(float), rectLifetime, roi_flim.width * sizeof(uint8_t), roi_flim, m_pConfig->flimLifetimeRange.min, m_pConfig->flimLifetimeRange.max);

	for (int i = 1; i < RING_THICKNESS; i++)
	{
		memcpy(&m_pImgObjIntensity->arr(0, i), rectIntensity, sizeof(uint8_t) * roi_flim.width);
		memcpy(&m_pImgObjLifetime->arr(0, i), rectLifetime, sizeof(uint8_t) * roi_flim.width);
	}
	
	emit makeRgb(m_pImgObjRectImage, m_pImgObjCircImage, m_pImgObjIntensity, m_pImgObjLifetime);	

    (void)res3;
#elif defined (STANDALONE_OCT)
	
	if (!m_pCheckBox_CircularizeImage->isChecked()) // rect image 
	{        
		emit paintRectImage(m_pImgObjRectImage->arr.raw_ptr());
	}
	else // circ image
    {
		(*m_pCirc)(m_pImgObjRectImage->arr, m_pImgObjCircImage->arr.raw_ptr(), "vertical", m_pConfig->circCenter);
		emit paintCircImage(m_pImgObjCircImage->qindeximg.bits());
	}

    (void)res2;
#endif
}

#ifdef OCT_FLIM
void QStreamTab::constructRgbImage(ImageObject *rectObj, ImageObject *circObj, ImageObject *intObj, ImageObject *lftObj)
{
	// Convert RGB
	rectObj->convertRgb();
	intObj->convertScaledRgb();
	lftObj->convertScaledRgb();

	// Rect View
	if (!m_pCheckBox_CircularizeImage->isChecked())
	{
		// Paste FLIM color ring to RGB rect image
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (rectObj->arr.size(1) - 2 * RING_THICKNESS - 1), intObj->qrgbimg.bits(), intObj->qrgbimg.byteCount());
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (rectObj->arr.size(1) - 1 * RING_THICKNESS - 1), lftObj->qrgbimg.bits(), lftObj->qrgbimg.byteCount());

		// Scan adjust
		for (int i = 0; i < m_pConfig->n2ScansFFT; i++)
		{
			uint8_t* pImg = rectObj->qrgbimg.bits() + 3 * i * m_pConfig->nAlines;
			std::rotate(pImg, pImg + 3 * m_pMainWnd->m_pDeviceControlTab->getScrollBarValue(), pImg + 3 * m_pConfig->nAlines);
		}

		// Draw image
		m_pImageView_RectImage->drawImage(rectObj->qrgbimg.bits());
	}
	// Circ View
	else
	{
		// Paste FLIM color ring to RGB rect image
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (m_pConfig->circCenter + CIRC_RADIUS - 2 * RING_THICKNESS), intObj->qrgbimg.bits(), intObj->qrgbimg.byteCount());
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (m_pConfig->circCenter + CIRC_RADIUS - 1 * RING_THICKNESS), lftObj->qrgbimg.bits(), lftObj->qrgbimg.byteCount());

		np::Uint8Array2 rect_temp(rectObj->qrgbimg.bits(), 3 * rectObj->arr.size(0), rectObj->arr.size(1));
		(*m_pCirc)(rect_temp, circObj->qrgbimg.bits(), "vertical", "rgb", m_pConfig->circCenter);

		// Draw image  
		m_pImageView_CircImage->drawImage(circObj->qrgbimg.bits());
	}
}
#endif	

void QStreamTab::updateAlinePos(int aline)
{
	// Reset channel data
	if (!m_pOperationTab->isAcquisitionButtonToggled())
	{
#ifdef OCT_FLIM
		emit plotFringe(&m_visFringeRm(0, m_pSlider_SelectAline->value()));
		emit plotPulse(&m_visPulse(m_pConfig->preTrigSamps + m_pConfig->nScans * m_pConfig->flimCh, m_pSlider_SelectAline->value() / 4));
		emit plotAline(&m_visImage(0, m_pSlider_SelectAline->value()));
#elif defined (STANDALONE_OCT)
		emit plotFringe(&m_visFringe1Rm(0, m_pSlider_SelectAline->value()), &m_visFringe2Rm(0, m_pSlider_SelectAline->value()));
		emit plotAline(&m_visImage1(0, m_pSlider_SelectAline->value()), &m_visImage2(0, m_pSlider_SelectAline->value()));
#endif
	}

	// Reset slider label
	QString str; str.sprintf("Current A-line : %4d / %4d   ", aline + 1, m_pConfig->nAlines);
	m_pLabel_SelectAline->setText(str);
}


#ifdef OCT_FLIM
void QStreamTab::changeFlimCh(int ch)
{
	m_pConfig->flimCh = ch;
	m_pFLIM->_params.act_ch = ch;

	if (!m_pOperationTab->isAcquisitionButtonToggled())
		emit plotPulse(&m_visPulse(m_pConfig->preTrigSamps + m_pConfig->nScans * m_pConfig->flimCh, m_pSlider_SelectAline->value() / 4));
}

void QStreamTab::changeLifetimeColorTable(int ctable_ind)
{
	m_pConfig->flimLifetimeColorTable = ctable_ind;

	m_pImageView_LifetimeColorbar->resetColormap(ColorTable::colortable(ctable_ind));

	ColorTable temp_ctable;
	if (m_pImgObjLifetime) delete m_pImgObjLifetime;
	m_pImgObjLifetime = new ImageObject(m_pConfig->n4Alines, RING_THICKNESS, temp_ctable.m_colorTableVector.at(ctable_ind));
}

void QStreamTab::adjustFlimContrast()
{
	m_pConfig->flimIntensityRange.min = m_pLineEdit_IntensityMin->text().toFloat();
	m_pConfig->flimIntensityRange.max = m_pLineEdit_IntensityMax->text().toFloat();
	m_pConfig->flimLifetimeRange.min = m_pLineEdit_LifetimeMin->text().toFloat();
	m_pConfig->flimLifetimeRange.max = m_pLineEdit_LifetimeMax->text().toFloat();

	if (!m_pOperationTab->isAcquisitionButtonToggled())
	{
		visualizeImage(m_visImage.raw_ptr(), m_visIntensity.raw_ptr(), m_visMeanDelay.raw_ptr(), m_visLifetime.raw_ptr());
		if (m_pFlimCalibDlg)
			emit m_pFlimCalibDlg->plotRoiPulse(m_pFLIM, m_pSlider_SelectAline->value() / 4);
	}
}

void QStreamTab::createFlimCalibDlg()
{
	if (m_pFlimCalibDlg == nullptr)
	{
		m_pFlimCalibDlg = new FlimCalibDlg(this);
		connect(m_pFlimCalibDlg, SIGNAL(finished(int)), this, SLOT(deleteFlimCalibDlg()));
		m_pFlimCalibDlg->show();
		emit m_pFlimCalibDlg->plotRoiPulse(m_pFLIM, m_pSlider_SelectAline->value() / 4);
	}
	m_pFlimCalibDlg->raise();
	m_pFlimCalibDlg->activateWindow();
}

void QStreamTab::deleteFlimCalibDlg()
{
	m_pFlimCalibDlg->showWindow(false);
	m_pFlimCalibDlg->showMeanDelay(false);
	m_pFlimCalibDlg->showMask(false);

	m_pFlimCalibDlg->deleteLater();
	m_pFlimCalibDlg = nullptr;
}
#endif


void QStreamTab::changeVisImage(bool toggled)
{	
	if (toggled)
	{
		m_pImageView_CircImage->show();// setVisible(toggled);
		m_pImageView_RectImage->hide();// setVisible(!toggled);
	}
	else
	{
		m_pImageView_CircImage->hide(); // setVisible(toggled);
		m_pImageView_RectImage->show(); // etVisible(!toggled);
	}

	if (!m_pOperationTab->isAcquisitionButtonToggled())
	{
#ifdef OCT_FLIM
		visualizeImage(m_visImage.raw_ptr(), m_visIntensity.raw_ptr(), m_visMeanDelay.raw_ptr(), m_visLifetime.raw_ptr());
#elif defined (STANDALONE_OCT)
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr());
#endif
	}
}

void QStreamTab::changeFringeBg(bool toggled)
{
#ifdef OCT_FLIM
	float* bg = (toggled) ? m_pOCT->getBg() : m_pOCT->getBg0();
	for (int i = 0; i < m_pConfig->nAlines; i++)
		memcpy(&m_visFringeBg(0, i), bg, sizeof(float) * m_pConfig->nScans);

#elif defined (STANDALONE_OCT)
	float* bg1 = (toggled) ? m_pOCT1->getBg() : m_pOCT1->getBg0();
	float* bg2 = (toggled) ? m_pOCT2->getBg() : m_pOCT2->getBg0();	
	for (int i = 0; i < m_pConfig->nAlines; i++)
	{
		memcpy(&m_visFringeBg1(0, i), bg1, sizeof(float) * m_pConfig->nScans);
		memcpy(&m_visFringeBg2(0, i), bg2, sizeof(float) * m_pConfig->nScans);
	}
#endif   

	if (!m_pOperationTab->isAcquisitionButtonToggled())
	{
#ifdef OCT_FLIM
		emit plotFringe(&m_visFringeRm(0, m_pSlider_SelectAline->value()));
#elif defined (STANDALONE_OCT)
		emit plotFringe(&m_visFringe1Rm(0, m_pSlider_SelectAline->value()), &m_visFringe2Rm(0, m_pSlider_SelectAline->value()));
#endif
	}
}

void QStreamTab::checkCircCenter(const QString &str)
{
	int circCenter = str.toInt();
	if (circCenter + CIRC_RADIUS + 1 > m_pConfig->n2ScansFFT)
	{
		circCenter = m_pConfig->n2ScansFFT - CIRC_RADIUS - 1;
		m_pLineEdit_CircCenter->setText(QString::number(circCenter));
	}
	m_pConfig->circCenter = circCenter;

	if (!m_pOperationTab->isAcquisitionButtonToggled())
	{
#ifdef OCT_FLIM
		visualizeImage(m_visImage.raw_ptr(), m_visIntensity.raw_ptr(), m_visMeanDelay.raw_ptr(), m_visLifetime.raw_ptr());
#elif defined (STANDALONE_OCT)
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr());
#endif
	}
}

void QStreamTab::changeOctColorTable(int ctable_ind)
{
	m_pConfig->octColorTable = ctable_ind;

	m_pImageView_RectImage->resetColormap(ColorTable::colortable(ctable_ind));
	m_pImageView_CircImage->resetColormap(ColorTable::colortable(ctable_ind));
	m_pImageView_OctDbColorbar->resetColormap(ColorTable::colortable(ctable_ind));

	ColorTable temp_ctable;
	if (m_pImgObjRectImage) delete m_pImgObjRectImage;
	m_pImgObjRectImage = new ImageObject(m_pImageView_RectImage->getRender()->m_pImage->width(), m_pImageView_RectImage->getRender()->m_pImage->height(), temp_ctable.m_colorTableVector.at(ctable_ind));
	if (m_pImgObjCircImage) delete m_pImgObjCircImage;
	m_pImgObjCircImage = new ImageObject(m_pImageView_CircImage->getRender()->m_pImage->width(), m_pImageView_CircImage->getRender()->m_pImage->height(), temp_ctable.m_colorTableVector.at(ctable_ind));

	if (!m_pOperationTab->isAcquisitionButtonToggled())
	{
#ifdef OCT_FLIM
		visualizeImage(m_visImage.raw_ptr(), m_visIntensity.raw_ptr(), m_visMeanDelay.raw_ptr(), m_visLifetime.raw_ptr());
#elif defined (STANDALONE_OCT)
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr());
#endif
	}
}

void QStreamTab::adjustOctContrast()
{
	int min_dB = m_pLineEdit_OctDbMin->text().toInt();
	int max_dB = m_pLineEdit_OctDbMax->text().toInt();
			
	m_pConfig->octDbRange.min = min_dB;
	m_pConfig->octDbRange.max = max_dB;

	m_pScope_OctDepthProfile->resetAxis({ 0, (double)m_pConfig->n2ScansFFT }, { (double)min_dB, (double)max_dB }, 1, 1, 0, 0, "", "dB");
	if (m_pOctCalibDlg != nullptr)
		m_pOctCalibDlg->m_pScope->resetAxis({ 0, (double)m_pConfig->n2ScansFFT }, { (double)min_dB, (double)max_dB });

	if (!m_pOperationTab->isAcquisitionButtonToggled())
	{
#ifdef OCT_FLIM
		emit plotAline(&m_visImage(0, m_pSlider_SelectAline->value()));
		visualizeImage(m_visImage.raw_ptr(), m_visIntensity.raw_ptr(), m_visMeanDelay.raw_ptr(), m_visLifetime.raw_ptr());
#elif defined (STANDALONE_OCT)
		emit plotAline(&m_visImage1(0, m_pSlider_SelectAline->value()), &m_visImage2(0, m_pSlider_SelectAline->value()));
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr());
#endif
	}
}

void QStreamTab::createOctCalibDlg()
{			
    if (m_pOctCalibDlg == nullptr)
    {
        m_pOctCalibDlg = new OctCalibDlg(this);
        connect(m_pOctCalibDlg, SIGNAL(finished(int)), this, SLOT(deleteOctCalibDlg()));
        m_pOctCalibDlg->show();
    }
    m_pOctCalibDlg->raise();
    m_pOctCalibDlg->activateWindow();
}

void QStreamTab::deleteOctCalibDlg()
{
	m_pOctCalibDlg->deleteLater();
	m_pOctCalibDlg = nullptr;
}
