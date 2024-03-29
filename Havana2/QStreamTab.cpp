
#include "QStreamTab.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QOperationTab.h>
#include <Havana2/QDeviceControlTab.h>

#include <DataAcquisition/DataAcquisition.h>
#include <MemoryBuffer/MemoryBuffer.h>
#ifdef OCT_NIRF
#ifndef ALAZAR_NIRF_ACQUISITION
#include <DeviceControl/NirfEmission/NirfEmission.h>
#endif
#endif

#include <Havana2/Viewer/QScope.h>
#ifdef STANDALONE_OCT
#include <Havana2/Viewer/QScope2.h>
#endif
#include <Havana2/Viewer/QImageView.h>

#ifndef CUDA_ENABLED
#include <DataProcess/OCTProcess/OCTProcess.h>
#else
#include <CUDA/CudaOCTProcess.cuh>
#endif
#ifdef OCT_FLIM
#include <DataProcess/FLIMProcess/FLIMProcess.h>
#endif
#include <DataProcess/ThreadManager.h>

#include <Havana2/Dialog/OctCalibDlg.h>
#ifdef OCT_FLIM
#include <Havana2/Dialog/FlimCalibDlg.h>
#endif
#ifdef OCT_NIRF
#include <Havana2/Dialog/NirfEmissionProfileDlg.h>
#endif

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>


QStreamTab::QStreamTab(QWidget *parent) :
    QDialog(parent), m_pOctCalibDlg(nullptr), 
	m_pImgObjRectImage(nullptr), m_pImgObjCircImage(nullptr),
	m_pCirc(nullptr), m_pMedfilt(nullptr)
#ifdef OCT_FLIM
	, m_pFlimCalibDlg(nullptr), m_pImgObjIntensity(nullptr), m_pImgObjLifetime(nullptr)
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	, m_pNirfEmissionProfileDlg(nullptr)
#endif
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
#ifndef CUDA_ENABLED
	m_pOCT = new OCTProcess(m_pConfig->nScans, m_pConfig->nAlines);
	m_pOCT->loadCalibration();
	m_pOCT->changeDiscomValue(m_pConfig->octDiscomVal);
#else
	m_pOCT = new CudaOCTProcess(m_pConfig->nScans, m_pConfig->nAlines);
	m_pOCT->loadCalibration();
	m_pOCT->changeDiscomValue(m_pConfig->octDiscomVal);
	m_pOCT->initialize();
#endif

	m_pFLIM = new FLIMProcess;
	m_pFLIM->setParameters(m_pConfig);
	m_pFLIM->_resize(np::Uint16Array2(m_pConfig->fnScans, m_pConfig->n4Alines), m_pFLIM->_params);
	m_pFLIM->loadMaskData();
#elif defined (STANDALONE_OCT)
#ifndef CUDA_ENABLED
	m_pOCT1 = new OCTProcess(m_pConfig->nScans, m_pConfig->nAlines);
#ifndef K_CLOCKING
	m_pOCT1->loadCalibration(CH_1);
#endif
	m_pOCT1->changeDiscomValue(m_pConfig->octDiscomVal);
#else
	m_pOCT1 = new CudaOCTProcess(m_pConfig->nScans, m_pConfig->nAlines);
#ifndef K_CLOCKING
	m_pOCT1->loadCalibration(CH_1);
#endif
	m_pOCT1->changeDiscomValue(m_pConfig->octDiscomVal);
	m_pOCT1->initialize();
#endif
#ifdef DUAL_CHANNEL
#ifndef CUDA_ENABLED
	m_pOCT2 = new OCTProcess(m_pConfig->nScans, m_pConfig->nAlines);
	m_pOCT2->loadCalibration(CH_2);
	m_pOCT2->changeDiscomValue(m_pConfig->octDiscomVal);
#else
	m_pOCT2 = new CudaOCTProcess(m_pConfig->nScans, m_pConfig->nAlines);
	m_pOCT2->loadCalibration(CH_2);
	m_pOCT2->changeDiscomValue(m_pConfig->octDiscomVal);
	m_pOCT2->initialize();
#endif
#endif
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
    m_pMemBuff->m_syncBuffering.allocate_queue_buffer(m_pConfig->nChannels * m_pConfig->nScans, m_pConfig->nAlines, PROCESSING_BUFFER_SIZE);
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	m_pMemBuff->m_syncBufferingNirf.allocate_queue_buffer(1, m_pConfig->nAlines, PROCESSING_BUFFER_SIZE);
#else
	m_pMemBuff->m_syncBufferingNirf.allocate_queue_buffer(4 * NIRF_SCANS, m_pConfig->nAlines / 2, PROCESSING_BUFFER_SIZE);
#endif
#endif
    m_syncDeinterleaving.allocate_queue_buffer(m_pConfig->nChannels * m_pConfig->nScans, m_pConfig->nAlines, PROCESSING_BUFFER_SIZE);
    m_syncCh1Processing.allocate_queue_buffer(m_pConfig->nScans, m_pConfig->nAlines, PROCESSING_BUFFER_SIZE); // Ch1 Processing
    m_syncCh1Visualization.allocate_queue_buffer(m_pConfig->n2ScansFFT, m_pConfig->nAlines, PROCESSING_BUFFER_SIZE); // Ch1 Visualization
#ifdef OCT_FLIM
    m_syncCh2Processing.allocate_queue_buffer(m_pConfig->fnScans, m_pConfig->n4Alines, PROCESSING_BUFFER_SIZE); // Ch2 Processing
    m_syncCh2Visualization.allocate_queue_buffer(11, m_pConfig->n4Alines, PROCESSING_BUFFER_SIZE); // FLIM Visualization
#elif defined (STANDALONE_OCT)
    m_syncCh2Processing.allocate_queue_buffer(m_pConfig->nScans, m_pConfig->nAlines, PROCESSING_BUFFER_SIZE); // Ch2 Processing
    m_syncCh2Visualization.allocate_queue_buffer(m_pConfig->n2ScansFFT, m_pConfig->nAlines, PROCESSING_BUFFER_SIZE); // Ch2 OCT Visualization
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	m_syncNirfVisualization.allocate_queue_buffer(1, m_pConfig->nAlines, PROCESSING_BUFFER_SIZE); // NIRF Visualization
#else
	m_syncNirfVisualization.allocate_queue_buffer(4 * NIRF_SCANS, m_pConfig->nAlines / 2, PROCESSING_BUFFER_SIZE); // NIRF Visualization
#endif
#endif
#endif
	
	// Set signal object
	setDataAcquisitionCallback();
#ifdef ALAZAR_NIRF_ACQUISITION
	setNirfAcquisitionCallback();
#endif
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
	m_visIntensity = np::FloatArray2(m_pConfig->n4Alines, 4); memset(m_visIntensity.raw_ptr(), 0, sizeof(float) * m_visIntensity.length());
	m_visMeanDelay = np::FloatArray2(m_pConfig->n4Alines, 4); memset(m_visMeanDelay.raw_ptr(), 0, sizeof(float) * m_visMeanDelay.length());
	m_visLifetime = np::FloatArray2(m_pConfig->n4Alines, 3); memset(m_visLifetime.raw_ptr(), 0, sizeof(float) * m_visLifetime.length());
	
#elif defined (STANDALONE_OCT)
	m_visFringe1 = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visFringeBg1 = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visFringe1Rm = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visImage1 = np::FloatArray2(m_pConfig->n2ScansFFT, m_pConfig->nAlines);

	m_visFringe2 = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visFringeBg2 = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	memset(m_visFringeBg2, 0, sizeof(float) * m_visFringeBg2.length());
	m_visFringe2Rm = np::FloatArray2(m_pConfig->nScans, m_pConfig->nAlines);
	m_visImage2 = np::FloatArray2(m_pConfig->n2ScansFFT, m_pConfig->nAlines);

#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	m_visNirf = np::DoubleArray2(1, m_pConfig->nAlines);
#else
	m_visNirf = np::DoubleArray2(2, m_pConfig->nAlines);
#endif
#endif
#endif 

	// Create image visualization buffers
	ColorTable temp_ctable;

	m_pImgObjRectImage = new ImageObject(m_pConfig->nAlines, m_pConfig->n2ScansFFT, temp_ctable.m_colorTableVector.at(temp_ctable.gray));
	m_pImgObjCircImage = new ImageObject(2 * m_pConfig->circRadius, 2 * m_pConfig->circRadius, temp_ctable.m_colorTableVector.at(temp_ctable.gray));
#ifdef OCT_FLIM
	m_pImgObjIntensity = new ImageObject(m_pConfig->n4Alines, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
	m_pImgObjLifetime = new ImageObject(m_pConfig->n4Alines, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(m_pConfig->flimLifetimeColorTable));
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	m_pImgObjNirf = new ImageObject(m_pConfig->nAlines, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ColorTable::colortable(NIRF_COLORTABLE1)));
#else
	m_pImgObjNirf1 = new ImageObject(m_pConfig->nAlines, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ColorTable::colortable(NIRF_COLORTABLE1)));
	m_pImgObjNirf2 = new ImageObject(m_pConfig->nAlines, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ColorTable::colortable(NIRF_COLORTABLE2)));
#endif
#endif
#endif
	
#ifndef CUDA_ENABLED
	m_pCirc = new circularize(m_pConfig->circRadius, m_pConfig->nAlines, false);
#else
	m_pCirc = new CudaCircularize(m_pConfig->circRadius, m_pConfig->nAlines, m_pConfig->n2ScansFFT);
#endif
	m_pMedfilt = new medfilt(m_pConfig->nAlines, m_pConfig->n2ScansFFT, 3, 3);


    // Create layout
    QHBoxLayout* pHBoxLayout = new QHBoxLayout;
	pHBoxLayout->setSpacing(0);

    // Create graph view
#if PX14_ENABLE
	double voltageCh1 = DIGITIZER_VOLTAGE;
	double voltageCh2 = DIGITIZER_VOLTAGE;
	for (int i = 0; i < m_pConfig->ch1VoltageRange; i++)
		voltageCh1 *= DIGITIZER_VOLTAGE_RATIO;
	for (int i = 0; i < m_pConfig->ch2VoltageRange; i++)
		voltageCh2 *= DIGITIZER_VOLTAGE_RATIO;
#elif ALAZAR_ENABLE
    double voltageCh1 = m_pConfig->voltRange[m_pConfig->ch1VoltageRange + 1];
    //double voltageCh2 = m_pConfig->voltRange[m_pConfig->ch2VoltageRange + 1];
#else
	double voltageCh1 = 0.0;
	double voltageCh2 = 0.0;
	for (int i = 0; i < m_pConfig->ch1VoltageRange; i++)
		voltageCh1 *= 0.0;
	for (int i = 0; i < m_pConfig->ch2VoltageRange; i++)
		voltageCh2 *= 0.0;
#endif

#ifdef OCT_FLIM
    m_pScope_FlimPulse = new QScope({ 0, N_VIS_SAMPS_FLIM }, { 0, POWER_2(16) }, 2, 3, 1, voltageCh2 / (double)POWER_2(16), 0, -voltageCh2 / 2, "", "V");
    m_pScope_FlimPulse->setMinimumSize(600, 250);
    m_pScope_OctFringe= new QScope({ 0, (double)m_pConfig->nScans }, { -POWER_2(15), POWER_2(15) }, 2, 3, 1, voltageCh1 / (double)POWER_2(16), 0, 0, "", "V");
    m_pScope_OctFringe->setMinimumSize(600, 250);
    m_pScope_OctDepthProfile = new QScope({ 0, (double)m_pConfig->n2ScansFFT }, {(double)m_pConfig->octDbRange.min, (double)m_pConfig->octDbRange.max}, 2, 2, 1, 1, 0, 0, "", "dB");
    m_pScope_OctDepthProfile->setMinimumSize(600, 250);
	//m_pScope_OctDepthProfile->setWindowLine(2, m_pConfig->n2ScansFFT / 2 - m_pConfig->nScans / 4, m_pConfig->n2ScansFFT / 2 + m_pConfig->nScans / 4);
	m_pScope_OctDepthProfile->getRender()->update();
#elif defined (STANDALONE_OCT)
	m_pScope_OctFringe = new QScope2({ 0, (double)m_pConfig->nScans }, { -POWER_2(15), POWER_2(15) }, 2, 3, 1, voltageCh1 / (double)POWER_2(16), 0, 0, "", "V");
	m_pScope_OctFringe->setMinimumSize(600, 250);
	m_pScope_OctDepthProfile = new QScope2({ 0, (double)m_pConfig->n2ScansFFT }, { (double)m_pConfig->octDbRange.min, (double)m_pConfig->octDbRange.max }, 2, 2, 1, 1, 0, 0, "", "dB");	
	m_pScope_OctDepthProfile->setMinimumSize(600, 250);
	//m_pScope_OctDepthProfile->setVerticalLine(2, m_pConfig->n2ScansFFT / 2 - m_pConfig->nScans / 4, m_pConfig->n2ScansFFT / 2 + m_pConfig->nScans / 4);
	m_pScope_OctDepthProfile->getRender()->setGrid(8, 64, 8); // default 8 64 4
	m_pScope_OctDepthProfile->getRender()->update();
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
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	// Create NIRF visualization option tab
	createNirfVisualizationOptionTab();
#endif
#endif
    // Create OCT visualization option tab
    createOctVisualizationOptionTab();
	
    // Create image view
#ifdef OCT_FLIM
	m_pImageView_RectImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), m_pConfig->nAlines, m_pConfig->n2ScansFFT, 1.0f, true);
	m_pImageView_CircImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), 2 * m_pConfig->circRadius, 2 * m_pConfig->circRadius, 1.0f, true);
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
	m_pImageView_RectImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), m_pConfig->nAlines, m_pConfig->n2ScansFFT);
	m_pImageView_CircImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), 2 * m_pConfig->circRadius, 2 * m_pConfig->circRadius);
#else
	m_pImageView_RectImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), m_pConfig->nAlines, m_pConfig->n2ScansFFT, 1.0f, true);
	m_pImageView_CircImage = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), 2 * m_pConfig->circRadius, 2 * m_pConfig->circRadius, 1.0f, true);
#endif
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
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	pVBoxLayout_RightPanel->addWidget(m_pGroupBox_NirfVisualization);
#endif
#endif
	pVBoxLayout_RightPanel->addWidget(m_pGroupBox_OctVisualization);
	pVBoxLayout_RightPanel->addWidget(m_pImageView_RectImage);
	pVBoxLayout_RightPanel->addWidget(m_pImageView_CircImage);

    pHBoxLayout->addItem(pVBoxLayout_RightPanel);

    this->setLayout(pHBoxLayout);


	// Connect signal and slot
#ifdef OCT_FLIM
	connect(this, SIGNAL(plotPulse(const float*)), m_pScope_FlimPulse, SLOT(drawData(const float*)));
	connect(this, SIGNAL(plotFringe(const float*)), m_pScope_OctFringe, SLOT(drawData(const float*)));
	connect(this, SIGNAL(plotAline(const float*)), m_pScope_OctDepthProfile, SLOT(drawData(const float*)));

	connect(this, SIGNAL(makeRgb(ImageObject*, ImageObject*, ImageObject*, ImageObject*)),
		this, SLOT(constructRgbImage(ImageObject*, ImageObject*, ImageObject*, ImageObject*)));
#elif defined (STANDALONE_OCT)	
	connect(this, SIGNAL(plotFringe(const float*, const float*)), m_pScope_OctFringe, SLOT(drawData(const float*, const float*)));
	connect(this, SIGNAL(plotAline(const float*, const float*)), m_pScope_OctDepthProfile, SLOT(drawData(const float*, const float*)));

#ifndef OCT_NIRF
	connect(this, SIGNAL(paintRectImage(uint8_t*)), m_pImageView_RectImage, SLOT(drawImage(uint8_t*)));
	connect(this, SIGNAL(paintCircImage(uint8_t*)), m_pImageView_CircImage, SLOT(drawImage(uint8_t*)));
#else
#ifndef TWO_CHANNEL_NIRF
	connect(this, SIGNAL(makeRgb(ImageObject*, ImageObject*, ImageObject*)),
		this, SLOT(constructRgbImage(ImageObject*, ImageObject*, ImageObject*)));
#else
	connect(this, SIGNAL(makeRgb(ImageObject*, ImageObject*, ImageObject*, ImageObject*)),
		this, SLOT(constructRgbImage(ImageObject*, ImageObject*, ImageObject*, ImageObject*)));
#endif
#endif
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
#elif defined (STANDALONE_OCT)	
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	if (m_pImgObjNirf) delete m_pImgObjNirf;
#else
	if (m_pImgObjNirf1) delete m_pImgObjNirf1;
	if (m_pImgObjNirf2) delete m_pImgObjNirf2;
#endif
#endif
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
#ifdef DUAL_CHANNEL
	if (m_pOCT2) delete m_pOCT2;
#endif
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

	if (m_pConfig->circCenter + m_pConfig->circRadius + 1 > m_pConfig->n2ScansFFT)
		m_pConfig->circRadius = m_pConfig->n2ScansFFT - m_pConfig->circCenter - 1;
	m_pLineEdit_CircRadius->setText(QString::number(m_pConfig->circRadius));

#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
	if (m_pConfig->ringThickness > m_pConfig->circRadius - 2)
		m_pConfig->ringThickness = m_pConfig->circRadius - 2;
	m_pLineEdit_RingThickness->setText(QString::number(m_pConfig->ringThickness));
#endif

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
	m_pLineEdit_NirfEmissionMax->setText(QString::number(m_pConfig->nirfRange.max, 'f', 2));
	m_pLineEdit_NirfEmissionMin->setText(QString::number(m_pConfig->nirfRange.min, 'f', 2));
#else
	m_pLineEdit_NirfEmissionMax[0]->setText(QString::number(m_pConfig->nirfRange[0].max, 'f', 2));
	m_pLineEdit_NirfEmissionMin[0]->setText(QString::number(m_pConfig->nirfRange[0].min, 'f', 2));
	m_pLineEdit_NirfEmissionMax[1]->setText(QString::number(m_pConfig->nirfRange[1].max, 'f', 2));
	m_pLineEdit_NirfEmissionMin[1]->setText(QString::number(m_pConfig->nirfRange[1].min, 'f', 2));
#endif
#endif
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
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
void QStreamTab::createNirfVisualizationOptionTab()
{
	// Create widgets for NIRF visualization option tab
	m_pGroupBox_NirfVisualization = new QGroupBox;
	m_pGroupBox_NirfVisualization->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
	QGridLayout *pGridLayout_NirfVisualization = new QGridLayout;
	pGridLayout_NirfVisualization->setSpacing(3);

	// Create widgets for NIRF emission
	m_pPushButton_NirfEmissionProfile = new QPushButton(this);
	m_pPushButton_NirfEmissionProfile->setText("NIRF Emission Profile...");
	m_pPushButton_NirfEmissionProfile->setFixedWidth(150);

	// Create line edit widgets for Nirf contrast adjustment
#ifndef TWO_CHANNEL_NIRF
	m_pLineEdit_NirfEmissionMax = new QLineEdit(this);
	m_pLineEdit_NirfEmissionMax->setFixedWidth(30);
	m_pLineEdit_NirfEmissionMax->setText(QString::number(m_pConfig->nirfRange.max, 'f', 2));
	m_pLineEdit_NirfEmissionMax->setAlignment(Qt::AlignCenter);
	m_pLineEdit_NirfEmissionMin = new QLineEdit(this);
	m_pLineEdit_NirfEmissionMin->setFixedWidth(30);
	m_pLineEdit_NirfEmissionMin->setText(QString::number(m_pConfig->nirfRange.min, 'f', 2));
	m_pLineEdit_NirfEmissionMin->setAlignment(Qt::AlignCenter);
#else
	for (int i = 0; i < 2; i++)
	{
		m_pLineEdit_NirfEmissionMax[i] = new QLineEdit(this);
		m_pLineEdit_NirfEmissionMax[i]->setFixedWidth(30);
		m_pLineEdit_NirfEmissionMax[i]->setText(QString::number(m_pConfig->nirfRange[i].max, 'f', 2));
		m_pLineEdit_NirfEmissionMax[i]->setAlignment(Qt::AlignCenter);
		m_pLineEdit_NirfEmissionMin[i] = new QLineEdit(this);
		m_pLineEdit_NirfEmissionMin[i]->setFixedWidth(30);
		m_pLineEdit_NirfEmissionMin[i]->setText(QString::number(m_pConfig->nirfRange[i].min, 'f', 2));
		m_pLineEdit_NirfEmissionMin[i]->setAlignment(Qt::AlignCenter);
	}
#endif

	// Create color bar for FLIM visualization
	uint8_t color[256];
	for (int i = 0; i < 256; i++)
		color[i] = i;

#ifndef TWO_CHANNEL_NIRF
	m_pImageView_NirfEmissionColorbar = new QImageView(ColorTable::colortable(NIRF_COLORTABLE1), 256, 1);
	m_pImageView_NirfEmissionColorbar->setFixedHeight(15);
	m_pImageView_NirfEmissionColorbar->drawImage(color);
	m_pLabel_NirfEmission = new QLabel("NIRF Em", this);
	m_pLabel_NirfEmission->setFixedWidth(60);
#else
	m_pImageView_NirfEmissionColorbar[0] = new QImageView(ColorTable::colortable(NIRF_COLORTABLE1), 256, 1);
	m_pImageView_NirfEmissionColorbar[0]->setFixedHeight(15);
	m_pImageView_NirfEmissionColorbar[0]->drawImage(color);
	m_pLabel_NirfEmission[0] = new QLabel("NIRF Em 1", this);
	m_pLabel_NirfEmission[0]->setFixedWidth(60);

	m_pImageView_NirfEmissionColorbar[1] = new QImageView(ColorTable::colortable(NIRF_COLORTABLE2), 256, 1);
	m_pImageView_NirfEmissionColorbar[1]->setFixedHeight(15);
	m_pImageView_NirfEmissionColorbar[1]->drawImage(color);
	m_pLabel_NirfEmission[1] = new QLabel("NIRF Em 2", this);
	m_pLabel_NirfEmission[1]->setFixedWidth(60);
#endif

	// Set layout
	pGridLayout_NirfVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 0);
	pGridLayout_NirfVisualization->addWidget(m_pPushButton_NirfEmissionProfile, 0, 1);

#ifndef TWO_CHANNEL_NIRF
	QHBoxLayout *pHBoxLayout_NirfEmissionColorbar = new QHBoxLayout;
	pHBoxLayout_NirfEmissionColorbar->setSpacing(3);
	pHBoxLayout_NirfEmissionColorbar->addWidget(m_pLabel_NirfEmission);
	pHBoxLayout_NirfEmissionColorbar->addWidget(m_pLineEdit_NirfEmissionMin);
	pHBoxLayout_NirfEmissionColorbar->addWidget(m_pImageView_NirfEmissionColorbar);
	pHBoxLayout_NirfEmissionColorbar->addWidget(m_pLineEdit_NirfEmissionMax);

	pGridLayout_NirfVisualization->addItem(pHBoxLayout_NirfEmissionColorbar, 1, 0, 1, 2);
#else
	QHBoxLayout *pHBoxLayout_NirfEmissionColorbar1 = new QHBoxLayout;
	pHBoxLayout_NirfEmissionColorbar1->setSpacing(3);
	pHBoxLayout_NirfEmissionColorbar1->addWidget(m_pLabel_NirfEmission[0]);
	pHBoxLayout_NirfEmissionColorbar1->addWidget(m_pLineEdit_NirfEmissionMin[0]);
	pHBoxLayout_NirfEmissionColorbar1->addWidget(m_pImageView_NirfEmissionColorbar[0]);
	pHBoxLayout_NirfEmissionColorbar1->addWidget(m_pLineEdit_NirfEmissionMax[0]);
	
	QHBoxLayout *pHBoxLayout_NirfEmissionColorbar2 = new QHBoxLayout;
	pHBoxLayout_NirfEmissionColorbar2->setSpacing(3);
	pHBoxLayout_NirfEmissionColorbar2->addWidget(m_pLabel_NirfEmission[1]);
	pHBoxLayout_NirfEmissionColorbar2->addWidget(m_pLineEdit_NirfEmissionMin[1]);
	pHBoxLayout_NirfEmissionColorbar2->addWidget(m_pImageView_NirfEmissionColorbar[1]);
	pHBoxLayout_NirfEmissionColorbar2->addWidget(m_pLineEdit_NirfEmissionMax[1]);

	pGridLayout_NirfVisualization->addItem(pHBoxLayout_NirfEmissionColorbar1, 1, 0, 1, 2);
	pGridLayout_NirfVisualization->addItem(pHBoxLayout_NirfEmissionColorbar2, 2, 0, 1, 2);
#endif

	m_pGroupBox_NirfVisualization->setLayout(pGridLayout_NirfVisualization);
	m_pGroupBox_NirfVisualization->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");

	// Connect signal and slot
#ifndef TWO_CHANNEL_NIRF
	connect(m_pLineEdit_NirfEmissionMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast()));
	connect(m_pLineEdit_NirfEmissionMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast()));
#else
	connect(m_pLineEdit_NirfEmissionMax[0], SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast1()));
	connect(m_pLineEdit_NirfEmissionMin[0], SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast1()));
	connect(m_pLineEdit_NirfEmissionMax[1], SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast2()));
	connect(m_pLineEdit_NirfEmissionMin[1], SIGNAL(textEdited(const QString &)), this, SLOT(adjustNirfContrast2()));
#endif

	connect(m_pPushButton_NirfEmissionProfile, SIGNAL(clicked(bool)), this, SLOT(createNirfEmissionProfileDlg()));
}
#endif
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
    m_pLabel_CircCenter->setBuddy(m_pLineEdit_CircCenter);
	
	if (m_pConfig->circRadius + m_pConfig->circCenter + 1 > m_pConfig->n2ScansFFT)
	{
		m_pConfig->circRadius = m_pConfig->n2ScansFFT - m_pConfig->circCenter - 1;
		m_pConfig->circRadius = (m_pConfig->circRadius % 2) ? m_pConfig->circRadius - 1 : m_pConfig->circRadius;
	}

	m_pLineEdit_CircRadius = new QLineEdit(this);
	m_pLineEdit_CircRadius->setFixedWidth(35);
	m_pLineEdit_CircRadius->setText(QString("%1").arg(m_pConfig->circRadius));
	m_pLineEdit_CircRadius->setAlignment(Qt::AlignCenter);
	m_pLabel_CircRadius = new QLabel("Circ Radius", this);
	m_pLabel_CircRadius->setBuddy(m_pLineEdit_CircRadius);

#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
	m_pLineEdit_RingThickness = new QLineEdit(this);
	m_pLineEdit_RingThickness->setFixedWidth(35);
	m_pLineEdit_RingThickness->setText(QString("%1").arg(m_pConfig->ringThickness));
	m_pLineEdit_RingThickness->setAlignment(Qt::AlignCenter);
	m_pLabel_RingThickness = new QLabel("Ring Thickness    ", this);
	m_pLabel_RingThickness->setBuddy(m_pLineEdit_RingThickness);
#endif

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

    // Create label
	m_pLabel_AvgMaxVal = new QLabel(this);
	m_pLabel_AvgMaxVal->setStyleSheet("font: 15pt; color: yellow; font-weight: bold");
	m_pLabel_AvgMaxVal->setText("0.000 dB");


    // Set layout
	pGridLayout_OctVisualization->addWidget(m_pLabel_AvgMaxVal, 0, 0, 3, 3);
	//pGridLayout_OctVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 0, 1, 3);
	pGridLayout_OctVisualization->addWidget(m_pPushButton_OctCalibration, 0, 3, 1, 2);
	
#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
	pGridLayout_OctVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 2, 0);

	QHBoxLayout *pHBoxLayout_RingThickness = new QHBoxLayout;
	pHBoxLayout_RingThickness->setSpacing(3);
	pHBoxLayout_RingThickness->addWidget(m_pLabel_RingThickness);
	pHBoxLayout_RingThickness->addWidget(m_pLineEdit_RingThickness);
	pHBoxLayout_RingThickness->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));

	pGridLayout_OctVisualization->addItem(pHBoxLayout_RingThickness, 2, 1);
#else
	pGridLayout_OctVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 2, 0, 1, 3);
#endif
	pGridLayout_OctVisualization->addWidget(m_pLabel_CircCenter, 2, 3);
	pGridLayout_OctVisualization->addWidget(m_pLineEdit_CircCenter, 2, 4);

	pGridLayout_OctVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 3, 0);
	pGridLayout_OctVisualization->addWidget(m_pCheckBox_CircularizeImage, 3, 1);
	pGridLayout_OctVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Fixed), 3, 2);
	pGridLayout_OctVisualization->addWidget(m_pLabel_CircRadius, 3, 3);
	pGridLayout_OctVisualization->addWidget(m_pLineEdit_CircRadius, 3, 4);

	pGridLayout_OctVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 4, 0);
	pGridLayout_OctVisualization->addWidget(m_pCheckBox_ShowBgRemovedSignal, 4, 1);
	pGridLayout_OctVisualization->addItem(new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Fixed), 4, 2);
	pGridLayout_OctVisualization->addWidget(m_pLabel_OctColorTable, 4, 3);
	pGridLayout_OctVisualization->addWidget(m_pComboBox_OctColorTable, 4, 4);

    QHBoxLayout *pHBoxLayout_OctDbColorbar = new QHBoxLayout;
    pHBoxLayout_OctDbColorbar->setSpacing(3);
    pHBoxLayout_OctDbColorbar->addWidget(m_pLabel_OctDb);
    pHBoxLayout_OctDbColorbar->addWidget(m_pLineEdit_OctDbMin);
    pHBoxLayout_OctDbColorbar->addWidget(m_pImageView_OctDbColorbar);
    pHBoxLayout_OctDbColorbar->addWidget(m_pLineEdit_OctDbMax);

	pGridLayout_OctVisualization->addItem(pHBoxLayout_OctDbColorbar, 5, 0, 1, 5);

    m_pGroupBox_OctVisualization->setLayout(pGridLayout_OctVisualization);
	m_pGroupBox_OctVisualization->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");
    
	// Connect signal and slot
	connect(m_pCheckBox_CircularizeImage, SIGNAL(toggled(bool)), this, SLOT(changeVisImage(bool)));
	connect(m_pCheckBox_ShowBgRemovedSignal, SIGNAL(toggled(bool)), this, SLOT(changeFringeBg(bool)));
	connect(m_pLineEdit_CircCenter, SIGNAL(textEdited(const QString &)), this, SLOT(checkCircCenter(const QString &)));
	connect(m_pLineEdit_CircRadius, SIGNAL(textEdited(const QString &)), this, SLOT(checkCircRadius(const QString &)));
#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
	connect(m_pLineEdit_RingThickness, SIGNAL(textEdited(const QString &)), this, SLOT(checkRingThickness(const QString &)));
#endif
	connect(m_pComboBox_OctColorTable, SIGNAL(currentIndexChanged(int)), this, SLOT(changeOctColorTable(int)));
	connect(m_pLineEdit_OctDbMax, SIGNAL(textEdited(const QString &)), this, SLOT(adjustOctContrast()));
	connect(m_pLineEdit_OctDbMin, SIGNAL(textEdited(const QString &)), this, SLOT(adjustOctContrast()));

	connect(m_pPushButton_OctCalibration, SIGNAL(clicked(bool)), this, SLOT(createOctCalibDlg()));
}


void QStreamTab::setDataAcquisitionCallback()
{
#if PX14_ENABLE || ALAZAR_ENABLE
	m_pDataAcq->ConnectDaqAcquiredData([&](int frame_count, const np::Array<uint16_t, 2>& frame) {

		// Data halving
#ifdef STANDALONE_OCT
		/* To be updated*/
#endif			
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
			if (!(frame_count % RECORDING_SKIP_FRAMES))
			{
				if (m_pMemBuff->m_nRecordedFrames < WRITING_BUFFER_SIZE)
				{
					// Get buffer from writing queue
					uint16_t* frame_ptr = nullptr;
					{
						std::unique_lock<std::mutex> lock(m_pMemBuff->m_syncBuffering.mtx);

						if (!m_pMemBuff->m_syncBuffering.queue_buffer.empty())
						{
							frame_ptr = m_pMemBuff->m_syncBuffering.queue_buffer.front();
							m_pMemBuff->m_syncBuffering.queue_buffer.pop();
						}
					}

					if (frame_ptr != nullptr)
					{
						// Body (Copying the frame data)
						memcpy(frame_ptr, frame.raw_ptr(), sizeof(uint16_t) * m_pConfig->nFrameSize);
						//frame_ptr[0] = m_pMemBuff->m_nRecordedFrames; // for test

						// Push to the copy queue for copying transfered data in copy thread
						m_pMemBuff->m_syncBuffering.Queue_sync.push(frame_ptr);
					}
				}
				else
				{
					// Finish recording when the buffer is full
					m_pMemBuff->m_bIsRecording = false;
					m_pOperationTab->setRecordingButton(false);
				}
			}
		}
	});

	m_pDataAcq->ConnectDaqStopData([&]() {
		m_syncDeinterleaving.Queue_sync.push(nullptr);
	});

	m_pDataAcq->ConnectDaqSendStatusMessage([&](const char * msg) {
		std::thread t1([msg]()	{
			QMessageBox MsgBox(QMessageBox::Critical, "Error", msg);
			MsgBox.exec();
		});
		t1.detach();
	});
#endif
}

#ifdef OCT_NIRF
void QStreamTab::setNirfAcquisitionCallback()
{
#if ALAZAR_ENABLE
#ifndef ALAZAR_NIRF_ACQUISITION
	NirfEmission* pNirfEmission = m_pMainWnd->m_pDeviceControlTab->getNirfEmission();

	pNirfEmission->DidAcquireData += [&](int frame_count, const double* data) {
#else
	m_pDataAcq->ConnectDaqNirfAcquiredData([&](int frame_count, const np::Array<uint16_t, 2>& frame) {
#endif		

#ifdef ALAZAR_NIRF_ACQUISITION
		// To see raw pulse waveform
		///if (!(frame_count % RENEWAL_COUNT))
		///{
		///	QFile *pFile = new QFile("temp.nirf");
		///	if (pFile->open(QIODevice::WriteOnly))
		///		pFile->write(reinterpret_cast<const char*>(frame.raw_ptr()), sizeof(uint16_t) * frame.length());
		///	delete pFile;
		///}

#ifndef TWO_CHANNEL_NIRF
		// Scaling & averaging
		np::FloatArray2 data32f(NIRF_SCANS, m_pConfig->nAlines);
		np::DoubleArray2 data64f(NIRF_SCANS, m_pConfig->nAlines);
		ippsConvert_16u32f(frame, data32f, NIRF_SCANS * m_pConfig->nAlines);
		ippsConvert_32f64f(data32f, data64f, NIRF_SCANS * m_pConfig->nAlines);
		ippsSubC_64f_I(32752.0, data64f, NIRF_SCANS * m_pConfig->nAlines);
		ippsDivC_64f_I(8192.0, data64f, NIRF_SCANS * m_pConfig->nAlines);

		np::DoubleArray data(m_pConfig->nAlines);
		for (int i = 0; i < m_pConfig->nAlines; i++)
			ippsMean_64f(&data64f(0, i), NIRF_SCANS, &data(i));
#else
		// Scaling & de-interleaving
		np::FloatArray2 frame32(4 * NIRF_SCANS, m_pConfig->nAlines / 2);		
		ippsConvert_16u32f(frame, frame32, 2 * NIRF_SCANS * m_pConfig->nAlines);
		ippsSubC_32f_I(32752.0, frame32, 2 * NIRF_SCANS * m_pConfig->nAlines);
		ippsDivC_32f_I(8192.0, frame32, 2 * NIRF_SCANS * m_pConfig->nAlines);		
		
		// Double-precision converting
		np::DoubleArray2 data(4 * NIRF_SCANS, m_pConfig->nAlines / 2);
		ippsConvert_32f64f(frame32, data, 2 * NIRF_SCANS * m_pConfig->nAlines);		
#endif
#endif

		// Data transfer		
		if (!(frame_count % RENEWAL_COUNT))
		{
			// Get buffer from threading queue
			double* nirf_ptr = nullptr;
			{
				std::unique_lock<std::mutex> lock(m_syncNirfVisualization.mtx);

				if (!m_syncNirfVisualization.queue_buffer.empty())
				{
					nirf_ptr = m_syncNirfVisualization.queue_buffer.front();
					m_syncNirfVisualization.queue_buffer.pop();
				}
			}

			if (nirf_ptr != nullptr)
			{
				// Body
#ifndef TWO_CHANNEL_NIRF
				memcpy(nirf_ptr, data, sizeof(double) * m_pConfig->nAlines);
#else
				memcpy(nirf_ptr, data, sizeof(double) * 2 * NIRF_SCANS * m_pConfig->nAlines);
#endif

				// Visualization
				if (m_pNirfEmissionProfileDlg)
				{
#ifndef TWO_CHANNEL_NIRF
					emit plotNirf((double*)data);
#else
					np::DoubleArray data1(NIRF_SCANS * MODULATION_FREQ * 2), data2(NIRF_SCANS * MODULATION_FREQ * 2);
					ippsCplxToReal_64fc((const Ipp64fc*)data.raw_ptr(), data1, data2, NIRF_SCANS * MODULATION_FREQ * 2);

					////emit plotNirf((double*)data64f1);
					emit plotNirf((double*)data1, (double*)data2);
#endif
				}

				// Push the buffer to sync Queue
				m_syncNirfVisualization.Queue_sync.push(nirf_ptr);
			}
		}

		// Buffering (When recording)
		///static QFile* pFile = nullptr; 
		///static bool is_opened = false; 

		if (m_pMemBuff->m_bIsRecording)
		{
			if (m_pMemBuff->m_nRecordedFrames < WRITING_BUFFER_SIZE)
			{
				// Get buffer from writing queue
				double* nirf_data = nullptr;
				{
					std::unique_lock<std::mutex> lock(m_pMemBuff->m_syncBufferingNirf.mtx);

					if (!m_pMemBuff->m_syncBufferingNirf.queue_buffer.empty())
					{
						nirf_data = m_pMemBuff->m_syncBufferingNirf.queue_buffer.front();
						m_pMemBuff->m_syncBufferingNirf.queue_buffer.pop();
					}
				}

				if (nirf_data != nullptr)
				{
					// Body (Copying the frame data)
#ifndef TWO_CHANNEL_NIRF
					memcpy(nirf_data, data, sizeof(double) * m_pConfig->nAlines);
#else
					memcpy(nirf_data, data, sizeof(double) * 2 * NIRF_SCANS * m_pConfig->nAlines);
#endif

					// Push to the copy queue for copying transfered data in copy thread
					m_pMemBuff->m_syncBufferingNirf.Queue_sync.push(nirf_data);
				}

				///if (!is_opened) 
				///{ 
				///	pFile = new QFile("temp.nirf"); 
				///	if (pFile->open(QIODevice::WriteOnly)) 
				///		is_opened = true; 
				///	else 
				///		delete pFile; 
				///} 
				///
				///if (is_opened) 
				///	pFile->write(reinterpret_cast<const char*>(data), sizeof(double) * m_pConfig->nAlines);  				
			}
			///else 
			///{ 
			///	if (is_opened) 
			///	{ 
			///		pFile->close(); 
			///		is_opened = false; 
			///	} 
			///} 
		}
#ifndef ALAZAR_NIRF_ACQUISITION
	};
#else
	});
#endif

#ifndef ALAZAR_NIRF_ACQUISITION
	pNirfEmission->DidStopData += [&]() {
		// None
	};

	pNirfEmission->SendStatusMessage += [&](const char * msg) {
		QMessageBox MsgBox(QMessageBox::Critical, "Error", msg);
		MsgBox.exec();
	};
#else
	m_pDataAcq->ConnectDaqNirfStopData([&]() {
		// None
	});

	m_pDataAcq->ConnectDaqNirfSendStatusMessage([&](const char * msg) {
		QMessageBox MsgBox(QMessageBox::Critical, "Error", msg);
		MsgBox.exec();
	});
#endif
#endif
}
#endif

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

				if (m_pConfig->nChannels == 2)
				{
#if PX14_ENABLE
					ippsCplxToReal_16sc((Ipp16sc *)fulldata_ptr, (Ipp16s *)ch1_ptr, (Ipp16s *)ch2_ptr, frame_length);

#elif ALAZAR_ENABLE
					memcpy(ch1_ptr, fulldata_ptr, sizeof(uint16_t) * frame_length);
					memcpy(ch2_ptr, fulldata_ptr + frame_length, sizeof(uint16_t) * frame_length);
#endif
				}
				else
					memcpy(ch1_ptr, fulldata_ptr, sizeof(uint16_t) * frame_length);
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
                ippsSub_32f(m_visFringeBg1.raw_ptr(), m_visFringe1.raw_ptr(), m_visFringe1Rm.raw_ptr(), m_visFringe1Rm.length());
				
				ippsConvert_16u32f(ch2_ptr, m_visFringe2.raw_ptr(), m_visFringe2.length());
                ippsSub_32f(m_visFringeBg2.raw_ptr(), m_visFringe2.raw_ptr(), m_visFringe2Rm.raw_ptr(), m_visFringe2Rm.length());

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
#ifdef DUAL_CHANNEL
				(*m_pOCT2)(res_ptr, ch2_data);
#endif
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
#ifdef OCT_NIRF
		double* res3_data = m_syncNirfVisualization.Queue_sync.pop();
#endif

		if ((res1_data != nullptr) && (res2_data != nullptr)
#ifdef OCT_NIRF
			&& (res3_data != nullptr)
#endif
			)
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

				// Frame mirroring
#if defined(OCT_VERTICAL_MIRRORING)
				ippiMirror_32f_C1IR((Ipp32f*)m_visImage1.raw_ptr(), sizeof(float) * m_visImage1.size(0), { m_visImage1.size(0), m_visImage1.size(1) }, ippAxsVertical);
				ippiMirror_32f_C1IR((Ipp32f*)m_visImage2.raw_ptr(), sizeof(float) * m_visImage2.size(0), { m_visImage2.size(0), m_visImage2.size(1) }, ippAxsVertical);
#endif

				emit plotAline(&m_visImage1(0, m_pSlider_SelectAline->value()), &m_visImage2(0, m_pSlider_SelectAline->value()));

#ifndef OCT_NIRF
                // Draw Images
                visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr());
#else
				// Draw Images
#ifndef TWO_CHANNEL_NIRF
				m_visNirf = np::DoubleArray2(res3_data, 1, m_pConfig->nAlines);
#else
				np::DoubleArray2 NirfData1(MODULATION_FREQ * NIRF_SCANS, m_pConfig->nAlines / MODULATION_FREQ), NirfData2(MODULATION_FREQ * NIRF_SCANS, m_pConfig->nAlines / MODULATION_FREQ);
				m_visNirf = np::DoubleArray2(2, m_pConfig->nAlines);

				// Deinterleaving
				ippsCplxToReal_64fc((const Ipp64fc*)res3_data, NirfData1, NirfData2, NIRF_SCANS * m_pConfig->nAlines);

				// Averaging				
				tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)(m_pConfig->nAlines / MODULATION_FREQ)),
					[&](const tbb::blocked_range<size_t>& r) {
					for (size_t i = r.begin(); i != r.end(); ++i)
					{
						// Gating�� result tab����...
						Ipp64f m1, m2;
						ippsMean_64f(&NirfData1(m_pConfig->nirfIntegWindow[0].min, i), m_pConfig->nirfIntegWindow[0].max - m_pConfig->nirfIntegWindow[0].min, &m1);
						ippsMean_64f(&NirfData2(m_pConfig->nirfIntegWindow[1].max, i), m_pConfig->nirfIntegWindow[1].max - m_pConfig->nirfIntegWindow[1].min, &m2);

						for (int j = 0; j < MODULATION_FREQ; j++)
						{
							m_visNirf(0, MODULATION_FREQ * i + j) = m1;
							m_visNirf(1, MODULATION_FREQ * i + j) = m2;
						}
					}
				});
#endif
				visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr(), m_visNirf.raw_ptr());
#endif
				// Find peak decibel
				if (frame_count % 4 == 0)
				{
					np::FloatArray max_val(m_pConfig->nAlines);
					tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)m_pConfig->nAlines),
						[&](const tbb::blocked_range<size_t>& r) {
						for (size_t i = r.begin(); i != r.end(); ++i)
						{
							ippsMax_32f(&m_visImage1(0, (int)i), m_pConfig->n2ScansFFT, &max_val[(int)i]);
						}
					});

					float avg_max_val;
					ippsMean_32f(max_val, max_val.length(), &avg_max_val, ippAlgHintFast);
					m_pLabel_AvgMaxVal->setText(QString("%1 dB").arg(avg_max_val, 4, 'f', 3));
				}
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
#ifdef OCT_NIRF
			{
				std::unique_lock<std::mutex> lock(m_syncNirfVisualization.mtx);
				m_syncNirfVisualization.queue_buffer.push(res3_data);
			}
#endif
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
			if (res2_data != nullptr)
			{
				float* res2_temp = res2_data;
				do
				{
					m_syncCh2Visualization.queue_buffer.push(res2_temp);
					res2_temp = m_syncCh2Visualization.Queue_sync.pop();
				} while (res2_temp != nullptr);
			}
#ifdef OCT_NIRF
			if (res3_data != nullptr)
			{
				double* res3_temp = res3_data;
				do
				{
					m_syncNirfVisualization.queue_buffer.push(res3_temp);
					res3_temp = m_syncNirfVisualization.Queue_sync.pop();
				} while (res3_temp != nullptr);
			}
#endif
			m_pThreadVisualization->_running = false;

            (void)frame_count;
		}
	};

	m_pThreadVisualization->DidStopData += [&]() {
		m_syncCh1Visualization.Queue_sync.push(nullptr);
		m_syncCh2Visualization.Queue_sync.push(nullptr);
#ifdef OCT_NIRF
		m_syncNirfVisualization.Queue_sync.push(nullptr);
#endif
	};

	m_pThreadVisualization->SendStatusMessage += [&](const char* msg) {
		QMessageBox MsgBox(QMessageBox::Critical, "Error", msg);
		MsgBox.exec();
	};
}


void QStreamTab::setCh1ScopeVoltRange(int idx)
{
#if PX14_ENABLE
	double voltage = DIGITIZER_VOLTAGE;
	for (int i = 0; i < idx; i++)
		voltage *= DIGITIZER_VOLTAGE_RATIO;
#elif ALAZAR_ENABLE
    double voltage = m_pConfig->voltRange[idx + 1];
#else
	double voltage = 0.0 * idx;
#endif
		
	m_pScope_OctFringe->resetAxis({ 0, (double)m_pConfig->nScans }, { -POWER_2(15), POWER_2(15) }, 1, voltage / (double)POWER_2(16), 0, 0, "", "V");
}

#ifdef OCT_FLIM
void QStreamTab::setCh2ScopeVoltRange(int idx)
{
#if PX14_ENABLE
	double voltage = DIGITIZER_VOLTAGE;
	for (int i = 0; i < idx; i++)
		voltage *= DIGITIZER_VOLTAGE_RATIO;
#elif ALAZAR_ENABLE
    double voltage = m_pConfig->voltRange[idx + 1];
#else
	double voltage = 0.0 * idx;
#endif

	m_pScope_FlimPulse->resetAxis({ 0, N_VIS_SAMPS_FLIM }, { 0, POWER_2(16) }, 1, voltage / (double)POWER_2(16), 0, -voltage / 2, "", "V");
}
#endif

void QStreamTab::resetObjectsForAline(int nAlines) // need modification
{	
	// Create data process object
#ifdef OCT_FLIM
	if (m_pOCT)
	{
#ifndef CUDA_ENABLED
		delete m_pOCT;
		m_pOCT = new OCTProcess(m_pConfig->nScans, nAlines);
		m_pOCT->loadCalibration();
		m_pOCT->changeDiscomValue(m_pConfig->octDiscomVal);
#else
		delete m_pOCT;
		m_pOCT = new CudaOCTProcess(m_pConfig->nScans, nAlines);
		m_pOCT->loadCalibration();
		m_pOCT->changeDiscomValue(m_pConfig->octDiscomVal);
		m_pOCT->initialize();
#endif
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
#ifndef CUDA_ENABLED
		delete m_pOCT1;
		m_pOCT1 = new OCTProcess(m_pConfig->nScans, nAlines);
#ifndef K_CLOCKING
		m_pOCT1->loadCalibration(CH_1);
#endif
		m_pOCT1->changeDiscomValue(m_pConfig->octDiscomVal);
#else
		delete m_pOCT1;
		m_pOCT1 = new CudaOCTProcess(m_pConfig->nScans, nAlines);
#ifndef K_CLOCKING
		m_pOCT1->loadCalibration(CH_1);
#endif
		m_pOCT1->changeDiscomValue(m_pConfig->octDiscomVal);
		m_pOCT1->initialize();
#endif
	}
#ifdef DUAL_CHANNEL
	if (m_pOCT2)
	{
#ifndef CUDA_ENABLED
		delete m_pOCT2;
		m_pOCT2 = new OCTProcess(m_pConfig->nScans, nAlines);
		m_pOCT2->loadCalibration(CH_2);
		m_pOCT2->changeDiscomValue(m_pConfig->octDiscomVal);
#else
		delete m_pOCT2;
		m_pOCT2 = new CudaOCTProcess(m_pConfig->nScans, nAlines);
		m_pOCT2->loadCalibration(CH_2);
		m_pOCT2->changeDiscomValue(m_pConfig->octDiscomVal);
		m_pOCT2->initialize();
#endif
	}
#endif
#endif

	// Create buffers for threading operation
	m_pMemBuff->m_syncBuffering.deallocate_queue_buffer();
#ifdef OCT_NIRF
	m_pMemBuff->m_syncBufferingNirf.deallocate_queue_buffer();
#endif
	m_syncDeinterleaving.deallocate_queue_buffer();
	m_syncCh1Processing.deallocate_queue_buffer();
	m_syncCh1Visualization.deallocate_queue_buffer();
	m_syncCh2Processing.deallocate_queue_buffer();
	m_syncCh2Visualization.deallocate_queue_buffer();
#ifdef OCT_NIRF
	m_syncNirfVisualization.deallocate_queue_buffer();
#endif

    m_pMemBuff->m_syncBuffering.allocate_queue_buffer(m_pConfig->nChannels * m_pConfig->nScans, nAlines, PROCESSING_BUFFER_SIZE);
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	m_pMemBuff->m_syncBufferingNirf.allocate_queue_buffer(1, nAlines, PROCESSING_BUFFER_SIZE);
#else
	m_pMemBuff->m_syncBufferingNirf.allocate_queue_buffer(1, 2 * nAlines, PROCESSING_BUFFER_SIZE);
#endif
#endif
    m_syncDeinterleaving.allocate_queue_buffer(m_pConfig->nChannels * m_pConfig->nScans, nAlines, PROCESSING_BUFFER_SIZE);
    m_syncCh1Processing.allocate_queue_buffer(m_pConfig->nScans, nAlines, PROCESSING_BUFFER_SIZE);
    m_syncCh1Visualization.allocate_queue_buffer(m_pConfig->n2ScansFFT, nAlines, PROCESSING_BUFFER_SIZE);
#ifdef OCT_FLIM
    m_syncCh2Processing.allocate_queue_buffer(m_pConfig->fnScans, nAlines / 4, PROCESSING_BUFFER_SIZE);
    m_syncCh2Visualization.allocate_queue_buffer(11, nAlines / 4, PROCESSING_BUFFER_SIZE);
#elif defined (STANDALONE_OCT)
    m_syncCh2Processing.allocate_queue_buffer(m_pConfig->nScans, nAlines, PROCESSING_BUFFER_SIZE);
    m_syncCh2Visualization.allocate_queue_buffer(m_pConfig->n2ScansFFT, nAlines, PROCESSING_BUFFER_SIZE);
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	m_syncNirfVisualization.allocate_queue_buffer(1, nAlines, PROCESSING_BUFFER_SIZE); // NIRF Visualization
#else
	m_syncNirfVisualization.allocate_queue_buffer(1, 2 * nAlines, PROCESSING_BUFFER_SIZE); // NIRF Visualization
#endif
#endif
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
	m_visIntensity = np::FloatArray2(nAlines / 4, 4); memset(m_visIntensity.raw_ptr(), 0, sizeof(float) * m_visIntensity.length());
	m_visMeanDelay = np::FloatArray2(nAlines / 4, 4); memset(m_visMeanDelay.raw_ptr(), 0, sizeof(float) * m_visMeanDelay.length());
	m_visLifetime = np::FloatArray2(nAlines / 4, 3); memset(m_visLifetime.raw_ptr(), 0, sizeof(float) * m_visLifetime.length());

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
	memset(m_visFringeBg2, 0, sizeof(float) * m_visFringeBg2.length());
	m_visFringe2Rm = np::FloatArray2(m_pConfig->nScans, nAlines);
	m_visImage2 = np::FloatArray2(m_pConfig->n2ScansFFT, nAlines);

#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	m_visNirf = np::DoubleArray2(1, nAlines);
#else
	m_visNirf = np::DoubleArray2(2, nAlines);
#endif
#endif
#endif 

	// Create image visualization buffers
	ColorTable temp_ctable;

	if (m_pImgObjRectImage) delete m_pImgObjRectImage;
	m_pImgObjRectImage = new ImageObject(nAlines, m_pConfig->n2ScansFFT, temp_ctable.m_colorTableVector.at(m_pComboBox_OctColorTable->currentIndex()));
#ifdef OCT_FLIM
	if (m_pImgObjIntensity) delete m_pImgObjIntensity;
	m_pImgObjIntensity = new ImageObject(nAlines / 4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
	if (m_pImgObjLifetime) delete m_pImgObjLifetime;
	m_pImgObjLifetime = new ImageObject(nAlines / 4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(m_pComboBox_LifetimeColorTable->currentIndex()));
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	if (m_pImgObjNirf) delete m_pImgObjNirf;
	m_pImgObjNirf = new ImageObject(nAlines, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ColorTable::colortable(NIRF_COLORTABLE1)));
#else
	if (m_pImgObjNirf1) delete m_pImgObjNirf1;
	m_pImgObjNirf1 = new ImageObject(nAlines, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ColorTable::colortable(NIRF_COLORTABLE1)));
	if (m_pImgObjNirf2) delete m_pImgObjNirf2;
	m_pImgObjNirf2 = new ImageObject(nAlines, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ColorTable::colortable(NIRF_COLORTABLE2)));
#endif
#endif
#endif
	
	// Create circularize object
	if (m_pCirc)
	{
		delete m_pCirc;
#ifndef CUDA_ENABLED
		m_pCirc = new circularize(m_pConfig->circRadius, nAlines, false);
#else
		m_pCirc = new CudaCircularize(m_pConfig->circRadius, nAlines, m_pConfig->n2ScansFFT);
#endif
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

#ifdef OCT_NIRF
	// Reset NIRF emission profile dialog	
    if (m_pNirfEmissionProfileDlg)
#ifndef TWO_CHANNEL_NIRF
		m_pNirfEmissionProfileDlg->getScope()->resetAxis({ 0, (double)nAlines }, { m_pConfig->nirfRange.min, m_pConfig->nirfRange.max });		
#else
        m_pNirfEmissionProfileDlg->getScope()->resetAxis({ 0, (double)NIRF_SCANS * MODULATION_FREQ * 2 }, { std::min(m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[1].min),
																				   std::max(m_pConfig->nirfRange[0].max, m_pConfig->nirfRange[1].max) });
#endif
#endif
}

#ifdef OCT_FLIM
void QStreamTab::visualizeImage(float* res1, float* res2, float* res3, float* res4) // OCT-FLIM
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
void QStreamTab::visualizeImage(float* res1, float* res2) // Standalone OCT
#else
void QStreamTab::visualizeImage(float* res1, float* res2, double* res3) // OCT-NIRF
#endif
#endif
{
	IppiSize roi_oct = { m_pConfig->n2ScansFFT, m_pConfig->nAlines };
	
	// OCT Visualization
	np::Uint8Array2 scale_temp(roi_oct.width, roi_oct.height);
	ippiScale_32f8u_C1R(res1, roi_oct.width * sizeof(float), scale_temp.raw_ptr(), roi_oct.width * sizeof(uint8_t), roi_oct, m_pConfig->octDbRange.min, m_pConfig->octDbRange.max);
	ippiTranspose_8u_C1R(scale_temp.raw_ptr(), roi_oct.width * sizeof(uint8_t), m_pImgObjRectImage->arr.raw_ptr(), roi_oct.height * sizeof(uint8_t), roi_oct);
#ifdef GALVANO_MIRROR
	if (m_pConfig->galvoHorizontalShift)
	{
		for (int i = 0; i < m_pConfig->n2ScansFFT; i++)
		{
			uint8_t* pImg = m_pImgObjRectImage->arr.raw_ptr() + i * m_pConfig->nAlines;
			std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + m_pConfig->nAlines);
		}
	}
#endif
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

	for (int i = 1; i < m_pConfig->ringThickness; i++)
	{
		memcpy(&m_pImgObjIntensity->arr(0, i), rectIntensity, sizeof(uint8_t) * roi_flim.width);
		memcpy(&m_pImgObjLifetime->arr(0, i), rectLifetime, sizeof(uint8_t) * roi_flim.width);
	}
	
	emit makeRgb(m_pImgObjRectImage, m_pImgObjCircImage, m_pImgObjIntensity, m_pImgObjLifetime);	

    (void)res3;
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
	if (!m_pCheckBox_CircularizeImage->isChecked()) // rect image 
	{        
		emit paintRectImage(m_pImgObjRectImage->arr.raw_ptr());
	}
	else // circ image
    {
#ifndef CUDA_ENABLED
		(*m_pCirc)(m_pImgObjRectImage->arr, m_pImgObjCircImage->arr.raw_ptr(), "vertical", m_pConfig->circCenter);
#else
		(*m_pCirc)(m_pImgObjRectImage->arr.raw_ptr(), m_pImgObjCircImage->arr.raw_ptr(), m_pConfig->circCenter);
#endif
		emit paintCircImage(m_pImgObjCircImage->qindeximg.bits());
	}

    (void)res2;
#else
	// NIRF Visualization
	IppiSize roi_nirf = { m_pConfig->nAlines, 1 };

#ifndef TWO_CHANNEL_NIRF
	np::FloatArray scanNirf(roi_nirf.width);
	ippsConvert_64f32f(res3, scanNirf, roi_nirf.width);
	uint8_t* rectNirf = m_pImgObjNirf->arr.raw_ptr();	
	ippiScale_32f8u_C1R(scanNirf, roi_nirf.width * sizeof(double), rectNirf, roi_nirf.width * sizeof(uint8_t), roi_nirf, m_pConfig->nirfRange.min, m_pConfig->nirfRange.max);
	
	for (int i = 1; i < m_pConfig->ringThickness; i++)
		memcpy(&m_pImgObjNirf->arr(0, i), rectNirf, sizeof(uint8_t) * roi_nirf.width);

	emit makeRgb(m_pImgObjRectImage, m_pImgObjCircImage, m_pImgObjNirf);
#else
	// Deinterleaving process
	np::DoubleArray2 scanNirf_itlv64(res3, 2, m_pConfig->nAlines);
	np::FloatArray2 scanNirf_itlv(2, m_pConfig->nAlines);
	np::FloatArray2 scanNirf(m_pConfig->nAlines, 2);
	ippsConvert_64f32f(scanNirf_itlv64, scanNirf_itlv, scanNirf_itlv.length());
	ippiTranspose_32f_C1R(scanNirf_itlv, sizeof(float) * 2, scanNirf, sizeof(float) * m_pConfig->nAlines, { 2, m_pConfig->nAlines });

	///np::FloatArray scanNirf1(roi_nirf.width);
	///ippsConvert_64f32f(res3, scanNirf1, roi_nirf.width);
	uint8_t* rectNirf1 = m_pImgObjNirf1->arr.raw_ptr();
	ippiScale_32f8u_C1R(&scanNirf(0, 0), roi_nirf.width * sizeof(double), rectNirf1, roi_nirf.width * sizeof(uint8_t), roi_nirf, m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[0].max);

	for (int i = 1; i < m_pConfig->ringThickness; i++)
		memcpy(&m_pImgObjNirf1->arr(0, i), rectNirf1, sizeof(uint8_t) * roi_nirf.width);
	
#if defined(CH_DIVIDING_LINE)
	np::Uint8Array boundary(roi_nirf.width);
	ippsSet_8u(255, boundary.raw_ptr(), roi_nirf.width);

	memcpy(&m_pImgObjNirf1->arr(0, 0), boundary.raw_ptr(), sizeof(uint8_t) * roi_nirf.width);
	memcpy(&m_pImgObjNirf1->arr(0, m_pConfig->ringThickness - 1), boundary.raw_ptr(), sizeof(uint8_t) * roi_nirf.width);
#endif

	///np::FloatArray scanNirf2(roi_nirf.width);
	///ippsConvert_64f32f(res3 + roi_nirf.width, scanNirf2, roi_nirf.width);
	uint8_t* rectNirf2 = m_pImgObjNirf2->arr.raw_ptr();
	ippiScale_32f8u_C1R(&scanNirf(0, 1), roi_nirf.width * sizeof(double), rectNirf2, roi_nirf.width * sizeof(uint8_t), roi_nirf, m_pConfig->nirfRange[1].min, m_pConfig->nirfRange[1].max);

	for (int i = 1; i < m_pConfig->ringThickness; i++)
		memcpy(&m_pImgObjNirf2->arr(0, i), rectNirf2, sizeof(uint8_t) * roi_nirf.width);

#if defined(CH_DIVIDING_LINE)
	memcpy(&m_pImgObjNirf2->arr(0, m_pConfig->ringThickness - 1), boundary.raw_ptr(), sizeof(uint8_t) * roi_nirf.width);
#endif
	
	emit makeRgb(m_pImgObjRectImage, m_pImgObjCircImage, m_pImgObjNirf1, m_pImgObjNirf2);
#endif

	(void)res2;
#endif
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
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (rectObj->arr.size(1) - 2 * m_pConfig->ringThickness - 1), intObj->qrgbimg.bits(), intObj->qrgbimg.byteCount());
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (rectObj->arr.size(1) - 1 * m_pConfig->ringThickness - 1), lftObj->qrgbimg.bits(), lftObj->qrgbimg.byteCount());

		// Scan adjust
#ifdef GALVANO_MIRROR
		for (int i = 0; i < m_pConfig->n2ScansFFT; i++)
		{
			uint8_t* pImg = rectObj->qrgbimg.bits() + 3 * i * m_pConfig->nAlines;
			std::rotate(pImg, pImg + 3 * m_pMainWnd->m_pDeviceControlTab->getScrollBarValue(), pImg + 3 * m_pConfig->nAlines);
		}
#endif

		// Draw image
		m_pImageView_RectImage->drawImage(rectObj->qrgbimg.bits());
	}
	// Circ View
	else
	{
		// Paste FLIM color ring to RGB rect image
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (m_pConfig->circCenter + m_pConfig->circRadius - 2 * m_pConfig->ringThickness), intObj->qrgbimg.bits(), intObj->qrgbimg.byteCount());
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (m_pConfig->circCenter + m_pConfig->circRadius - 1 * m_pConfig->ringThickness), lftObj->qrgbimg.bits(), lftObj->qrgbimg.byteCount());

		np::Uint8Array2 rect_temp(rectObj->qrgbimg.bits(), 3 * rectObj->arr.size(0), rectObj->arr.size(1));
#ifndef CUDA_ENABLED
		(*m_pCirc)(rect_temp, circObj->qrgbimg.bits(), "vertical", "rgb", m_pConfig->circCenter);
#else
		(*m_pCirc)(rect_temp.raw_ptr(), circObj->qrgbimg.bits(), "rgb", m_pConfig->circCenter);
#endif

		// Draw image  
		m_pImageView_CircImage->drawImage(circObj->qrgbimg.bits());
	}
}
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
void QStreamTab::constructRgbImage(ImageObject *rectObj, ImageObject *circObj, ImageObject *nirfObj)
#else
void QStreamTab::constructRgbImage(ImageObject *rectObj, ImageObject *circObj, ImageObject *nirfObj1, ImageObject *nirfObj2)
#endif
{
	// Convert RGB
	rectObj->convertRgb();
#ifndef TWO_CHANNEL_NIRF
	nirfObj->convertRgb();
#else
	nirfObj1->convertRgb();
	nirfObj2->convertRgb();
#endif

	// Rect View
	if (!m_pCheckBox_CircularizeImage->isChecked())
	{
		// Paste FLIM color ring to RGB rect image
#ifndef TWO_CHANNEL_NIRF
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (rectObj->arr.size(1) - m_pConfig->ringThickness - 1), nirfObj->qrgbimg.bits(), nirfObj->qrgbimg.byteCount());
#else
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (rectObj->arr.size(1) - 2 * m_pConfig->ringThickness - 1), nirfObj1->qrgbimg.bits(), nirfObj1->qrgbimg.byteCount());
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (rectObj->arr.size(1) - 1 * m_pConfig->ringThickness - 1), nirfObj2->qrgbimg.bits(), nirfObj2->qrgbimg.byteCount());
#endif

		// Scan adjust
#ifdef GALVANO_MIRROR
		for (int i = 0; i < m_pConfig->n2ScansFFT; i++)
		{
			uint8_t* pImg = rectObj->qrgbimg.bits() + 3 * i * m_pConfig->nAlines;
			std::rotate(pImg, pImg + 3 * m_pMainWnd->m_pDeviceControlTab->getScrollBarValue(), pImg + 3 * m_pConfig->nAlines);
		}
#endif

		// Draw image
		m_pImageView_RectImage->drawImage(rectObj->qrgbimg.bits());
	}
	// Circ View
	else
	{
		// Paste FLIM color ring to RGB rect image
#ifndef TWO_CHANNEL_NIRF
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (m_pConfig->circCenter + m_pConfig->circRadius - m_pConfig->ringThickness), nirfObj->qrgbimg.bits(), nirfObj->qrgbimg.byteCount());
#else
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (m_pConfig->circCenter + m_pConfig->circRadius - 2 * m_pConfig->ringThickness), nirfObj1->qrgbimg.bits(), nirfObj1->qrgbimg.byteCount());
		memcpy(rectObj->qrgbimg.bits() + 3 * rectObj->arr.size(0) * (m_pConfig->circCenter + m_pConfig->circRadius - 1 * m_pConfig->ringThickness), nirfObj2->qrgbimg.bits(), nirfObj2->qrgbimg.byteCount());
#endif

		np::Uint8Array2 rect_temp(rectObj->qrgbimg.bits(), 3 * rectObj->arr.size(0), rectObj->arr.size(1));
#ifndef CUDA_ENABLED
		(*m_pCirc)(rect_temp, circObj->qrgbimg.bits(), "vertical", "rgb", m_pConfig->circCenter);
#else
		(*m_pCirc)(rect_temp.raw_ptr(), circObj->qrgbimg.bits(), "rgb", m_pConfig->circCenter);
#endif

		// Draw image  
		m_pImageView_CircImage->drawImage(circObj->qrgbimg.bits());
	}
}
#endif
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
	m_pImgObjLifetime = new ImageObject(m_pConfig->n4Alines, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ctable_ind));

	if (!m_pOperationTab->isAcquisitionButtonToggled())
	{
		visualizeImage(m_visImage.raw_ptr(), m_visIntensity.raw_ptr(), m_visMeanDelay.raw_ptr(), m_visLifetime.raw_ptr());
		if (m_pFlimCalibDlg)
			emit m_pFlimCalibDlg->plotRoiPulse(m_pFLIM, m_pSlider_SelectAline->value() / 4);
	}
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
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
void QStreamTab::adjustNirfContrast()
{
	m_pConfig->nirfRange.min = m_pLineEdit_NirfEmissionMin->text().toFloat();
	m_pConfig->nirfRange.max = m_pLineEdit_NirfEmissionMax->text().toFloat();
		
	if (!m_pOperationTab->isAcquisitionButtonToggled())	
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr(), m_visNirf.raw_ptr());
	
	if (m_pNirfEmissionProfileDlg)
        m_pNirfEmissionProfileDlg->getScope()->resetAxis({ 0, (double)m_pConfig->nAlines },	{ m_pConfig->nirfRange.min, m_pConfig->nirfRange.max });		
}
#else
void QStreamTab::adjustNirfContrast1()
{
	m_pConfig->nirfRange[0].min = m_pLineEdit_NirfEmissionMin[0]->text().toFloat();
	m_pConfig->nirfRange[0].max = m_pLineEdit_NirfEmissionMax[0]->text().toFloat();

	if (!m_pOperationTab->isAcquisitionButtonToggled())
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr(), m_visNirf.raw_ptr());
	
	if (m_pNirfEmissionProfileDlg)
		m_pNirfEmissionProfileDlg->getScope()->resetAxis({ 0, (double)NIRF_SCANS * MODULATION_FREQ * 2 }, { std::min(m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[1].min),
																						      std::max(m_pConfig->nirfRange[0].max, m_pConfig->nirfRange[1].max) });
}

void QStreamTab::adjustNirfContrast2()
{
	m_pConfig->nirfRange[1].min = m_pLineEdit_NirfEmissionMin[1]->text().toFloat();
	m_pConfig->nirfRange[1].max = m_pLineEdit_NirfEmissionMax[1]->text().toFloat();

	if (!m_pOperationTab->isAcquisitionButtonToggled())
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr(), m_visNirf.raw_ptr());

	if (m_pNirfEmissionProfileDlg)
		m_pNirfEmissionProfileDlg->getScope()->resetAxis({ 0, (double)NIRF_SCANS * MODULATION_FREQ * 2 }, { std::min(m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[1].min),
																					          std::max(m_pConfig->nirfRange[0].max, m_pConfig->nirfRange[1].max) });
}
#endif

void QStreamTab::createNirfEmissionProfileDlg()
{
	if (m_pNirfEmissionProfileDlg == nullptr)
	{
        m_pNirfEmissionProfileDlg = new NirfEmissionProfileDlg(true, this);
		connect(m_pNirfEmissionProfileDlg, SIGNAL(finished(int)), this, SLOT(deleteNirfEmissionProfileDlg()));
#ifndef TWO_CHANNEL_NIRF
		connect(this, SIGNAL(plotNirf(void*)), m_pNirfEmissionProfileDlg, SLOT(drawData(void*)));
#else
		connect(this, SIGNAL(plotNirf(void*, void*)), m_pNirfEmissionProfileDlg, SLOT(drawData(void*, void*)));
#endif
		m_pNirfEmissionProfileDlg->show();
	}
	m_pNirfEmissionProfileDlg->raise();
	m_pNirfEmissionProfileDlg->activateWindow();
}

void QStreamTab::deleteNirfEmissionProfileDlg()
{
	m_pNirfEmissionProfileDlg->deleteLater();
	m_pNirfEmissionProfileDlg = nullptr;
}
#endif
#endif


void QStreamTab::changeVisImage(bool toggled)
{	
	if (toggled)
	{
		m_pImageView_CircImage->show(); // setVisible(toggled);
		m_pImageView_RectImage->hide(); // setVisible(!toggled);
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
#ifndef OCT_NIRF
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr());
#else
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr(), m_visNirf.raw_ptr());
#endif
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
#ifndef DUAL_CHANNEL
	float* bg2 = (toggled) ? m_pOCT1->getBg() : m_pOCT1->getBg0();
#else
	float* bg2 = (toggled) ? m_pOCT2->getBg() : m_pOCT2->getBg0();	
#endif
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
	if (circCenter + m_pConfig->circRadius + 1 > m_pConfig->n2ScansFFT)
	{
		circCenter = m_pConfig->n2ScansFFT - m_pConfig->circRadius - 1;
		m_pLineEdit_CircCenter->setText(QString::number(circCenter));
	}
	if (circCenter < 0)
	{
		circCenter = 0;
		m_pLineEdit_CircCenter->setText(QString::number(circCenter));
	}
	m_pConfig->circCenter = circCenter;

	if (!m_pOperationTab->isAcquisitionButtonToggled())
	{
#ifdef OCT_FLIM
		visualizeImage(m_visImage.raw_ptr(), m_visIntensity.raw_ptr(), m_visMeanDelay.raw_ptr(), m_visLifetime.raw_ptr());
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr());
#else
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr(), m_visNirf.raw_ptr());
#endif
#endif
	}
}

void QStreamTab::checkCircRadius(const QString &str)
{
	// Range and value test
	int circRadius = str.toInt();
	if (circRadius + m_pConfig->circCenter + 1 > m_pConfig->n2ScansFFT)
	{
		circRadius = m_pConfig->n2ScansFFT - m_pConfig->circCenter - 1;
		circRadius = (circRadius % 2) ? circRadius - 1 : circRadius;
		m_pLineEdit_CircRadius->setText(QString::number(circRadius));
	}

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
	m_pConfig->circRadius = circRadius;
		
	// Reset rect image size
	m_pImageView_CircImage->resetSize(2 * circRadius, 2 * circRadius);
	
	// Create image visualization buffers
	ColorTable temp_ctable;

	if (m_pImgObjCircImage) delete m_pImgObjCircImage;
	m_pImgObjCircImage = new ImageObject(2 * circRadius, 2 * circRadius, temp_ctable.m_colorTableVector.at(m_pComboBox_OctColorTable->currentIndex()));

	// Create circ object
	if (m_pCirc)
	{
		delete m_pCirc;
#ifndef CUDA_ENABLED
		m_pCirc = new circularize(circRadius, m_pConfig->nAlines, false);
#else
		m_pCirc = new CudaCircularize(circRadius, m_pConfig->nAlines, m_pConfig->n2ScansFFT);
#endif
	}

	// Renewal
	if (!m_pOperationTab->isAcquisitionButtonToggled())
	{
#ifdef OCT_FLIM
		visualizeImage(m_visImage.raw_ptr(), m_visIntensity.raw_ptr(), m_visMeanDelay.raw_ptr(), m_visLifetime.raw_ptr());
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr());
#else
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr(), m_visNirf.raw_ptr());
#endif
#endif
	}
}

#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
void QStreamTab::checkRingThickness(const QString &str)
{
	// Range and value test
	int ringThickness = str.toInt();
	if (ringThickness > m_pConfig->circRadius - 2)
	{
		ringThickness = m_pConfig->circRadius - 2;
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
	m_pImgObjIntensity = new ImageObject(m_pConfig->nAlines / 4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
	if (m_pImgObjLifetime) delete m_pImgObjLifetime;
	m_pImgObjLifetime = new ImageObject(m_pConfig->nAlines / 4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(m_pComboBox_LifetimeColorTable->currentIndex()));
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	if (m_pImgObjNirf) delete m_pImgObjNirf;
	m_pImgObjNirf = new ImageObject(m_pConfig->nAlines, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ColorTable::colortable(NIRF_COLORTABLE1)));
#else
	if (m_pImgObjNirf1) delete m_pImgObjNirf1;
	m_pImgObjNirf1 = new ImageObject(m_pConfig->nAlines, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ColorTable::colortable(NIRF_COLORTABLE1)));
	if (m_pImgObjNirf2) delete m_pImgObjNirf2;
	m_pImgObjNirf2 = new ImageObject(m_pConfig->nAlines, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(ColorTable::colortable(NIRF_COLORTABLE2)));
#endif
#endif
#endif
		
	// Renewal
	if (!m_pOperationTab->isAcquisitionButtonToggled())
	{
#ifdef OCT_FLIM
		visualizeImage(m_visImage.raw_ptr(), m_visIntensity.raw_ptr(), m_visMeanDelay.raw_ptr(), m_visLifetime.raw_ptr());
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr());
#else
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr(), m_visNirf.raw_ptr());
#endif
#endif
	}
}
#endif

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
#ifndef OCT_NIRF
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr());
#else
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr(), m_visNirf.raw_ptr());
#endif
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
#ifndef OCT_NIRF
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr());
#else
		visualizeImage(m_visImage1.raw_ptr(), m_visImage2.raw_ptr(), m_visNirf.raw_ptr());
#endif
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
