
#include "SaveResultDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QResultTab.h>
#include <Havana2/Viewer/QImageView.h>

#include <Havana2/Dialog/LongitudinalViewDlg.h>
#ifdef OCT_NIRF
#include <Havana2/Dialog/NirfDistCompDlg.h>
#endif

#include <Common/Array.h>

#include <ippi.h>
#include <ippcc.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <thread>
#include <chrono>
#include <utility>


SaveResultDlg::SaveResultDlg(QWidget *parent) :
    QDialog(parent), m_defaultTransformation(Qt::FastTransformation)
{
    // Set default size & frame
#ifdef OCT_FLIM
    setFixedSize(390, 160);
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
	setFixedSize(420, 100);
#else
    setFixedSize(420, 145);
#endif
#endif
    setWindowFlags(Qt::Tool);
	setWindowTitle("Save Result");

    // Set main window objects
    m_pResultTab = (QResultTab*)parent;
    m_pMainWnd = m_pResultTab->getMainWnd();
	m_pConfig = m_pMainWnd->m_pConfiguration;

    // Create widgets for saving results (save Ccoss-sections)
	m_pPushButton_SaveCrossSections = new QPushButton(this);
	m_pPushButton_SaveCrossSections->setText("Save Cross-Sections");
    m_pPushButton_SaveCrossSections->setFixedWidth(110);
	
	m_pCheckBox_RectImage = new QCheckBox(this);
	m_pCheckBox_RectImage->setText("Rect Image  ");
	m_pCheckBox_RectImage->setChecked(true);
	m_pCheckBox_CircImage = new QCheckBox(this);
	m_pCheckBox_CircImage->setText("Circ Image   ");
	m_pCheckBox_CircImage->setChecked(true);
	m_pCheckBox_LongiImage = new QCheckBox(this);
	m_pCheckBox_LongiImage->setText("Longi Image  ");
	m_pCheckBox_LongiImage->setChecked(false);

	m_pCheckBox_ResizeRectImage = new QCheckBox(this);
	m_pCheckBox_ResizeRectImage->setText("Resize (w x h)");
	m_pCheckBox_ResizeCircImage = new QCheckBox(this);
	m_pCheckBox_ResizeCircImage->setText("Resize (diameter)");
	m_pCheckBox_ResizeLongiImage = new QCheckBox(this);
	m_pCheckBox_ResizeLongiImage->setText("Resize (w x h)");

	m_pLineEdit_RectWidth = new QLineEdit(this);
	m_pLineEdit_RectWidth->setFixedWidth(35);
    m_pLineEdit_RectWidth->setText(QString::number(m_pResultTab->getRectImageView()->getRender()->m_pImage->width()));
	m_pLineEdit_RectWidth->setAlignment(Qt::AlignCenter);
	m_pLineEdit_RectWidth->setDisabled(true);
	m_pLineEdit_RectHeight = new QLineEdit(this);
	m_pLineEdit_RectHeight->setFixedWidth(35);
    m_pLineEdit_RectHeight->setText(QString::number(m_pResultTab->getRectImageView()->getRender()->m_pImage->height()));
	m_pLineEdit_RectHeight->setAlignment(Qt::AlignCenter);
	m_pLineEdit_RectHeight->setDisabled(true);
	m_pLineEdit_CircDiameter = new QLineEdit(this);
	m_pLineEdit_CircDiameter->setFixedWidth(35);
	m_pLineEdit_CircDiameter->setText(QString::number(2 * m_pResultTab->getConfigTemp()->circRadius));
	m_pLineEdit_CircDiameter->setAlignment(Qt::AlignCenter);
	m_pLineEdit_CircDiameter->setDisabled(true);
	m_pLineEdit_LongiWidth = new QLineEdit(this);
	m_pLineEdit_LongiWidth->setFixedWidth(35);
	m_pLineEdit_LongiWidth->setText(QString::number(m_pResultTab->getConfigTemp()->nFrames));
	m_pLineEdit_LongiWidth->setAlignment(Qt::AlignCenter);
	m_pLineEdit_LongiWidth->setDisabled(true);
	m_pLineEdit_LongiHeight = new QLineEdit(this);
	m_pLineEdit_LongiHeight->setFixedWidth(35);
	m_pLineEdit_LongiHeight->setText(QString::number(2 * m_pResultTab->getConfigTemp()->circRadius));
	m_pLineEdit_LongiHeight->setAlignment(Qt::AlignCenter);
	m_pLineEdit_LongiHeight->setDisabled(true);

#ifdef OCT_FLIM
	m_pCheckBox_CrossSectionCh1 = new QCheckBox(this);
	m_pCheckBox_CrossSectionCh1->setText("Channel 1");
	m_pCheckBox_CrossSectionCh1->setChecked(true);
	m_pCheckBox_CrossSectionCh2 = new QCheckBox(this);
	m_pCheckBox_CrossSectionCh2->setText("Channel 2");
	m_pCheckBox_CrossSectionCh2->setChecked(true);
	m_pCheckBox_CrossSectionCh3 = new QCheckBox(this);
	m_pCheckBox_CrossSectionCh3->setText("Channel 3");
	m_pCheckBox_CrossSectionCh3->setChecked(false);

	m_pCheckBox_Multichannel = new QCheckBox(this);
	m_pCheckBox_Multichannel->setText("Multichannel Visualization in a Single Image");
	m_pCheckBox_Multichannel->setChecked(false);
#endif

#ifdef OCT_NIRF
    m_pCheckBox_CrossSectionNirf = new QCheckBox(this);
    m_pCheckBox_CrossSectionNirf->setText("NIRF (Current Setting)");
    m_pCheckBox_CrossSectionNirf->setChecked(true);
#endif

    // Set Range
    m_pLabel_Range = new QLabel("Range  ", this);
    m_pLineEdit_RangeStart = new QLineEdit(this);
    m_pLineEdit_RangeStart->setFixedWidth(30);
    m_pLineEdit_RangeStart->setText(QString::number(1));
    m_pLineEdit_RangeStart->setAlignment(Qt::AlignCenter);
    m_pLineEdit_RangeEnd = new QLineEdit(this);
    m_pLineEdit_RangeEnd->setFixedWidth(30);
    m_pLineEdit_RangeEnd->setText(QString::number(m_pResultTab->m_vectorOctImage.size()));
    m_pLineEdit_RangeEnd->setAlignment(Qt::AlignCenter);

	// Save En Face Maps
	m_pPushButton_SaveEnFaceMaps = new QPushButton(this);
	m_pPushButton_SaveEnFaceMaps->setText("Save En Face Maps");
    m_pPushButton_SaveEnFaceMaps->setFixedWidth(110);

	m_pCheckBox_RawData = new QCheckBox(this);
	m_pCheckBox_RawData->setText("Raw Data   ");
	m_pCheckBox_RawData->setChecked(false);
	m_pCheckBox_ScaledImage = new QCheckBox(this);
	m_pCheckBox_ScaledImage->setText("Scaled Image");
	m_pCheckBox_ScaledImage->setChecked(true);

#ifdef OCT_FLIM
	m_pCheckBox_EnFaceCh1 = new QCheckBox(this);
	m_pCheckBox_EnFaceCh1->setText("Channel 1");
	m_pCheckBox_EnFaceCh1->setChecked(true);
	m_pCheckBox_EnFaceCh2 = new QCheckBox(this);
	m_pCheckBox_EnFaceCh2->setText("Channel 2");
	m_pCheckBox_EnFaceCh2->setChecked(true);
	m_pCheckBox_EnFaceCh3 = new QCheckBox(this);
	m_pCheckBox_EnFaceCh3->setText("Channel 3");
	m_pCheckBox_EnFaceCh3->setChecked(false);
#endif

#ifdef OCT_NIRF
    m_pCheckBox_EnFaceNirf = new QCheckBox(this);
    m_pCheckBox_EnFaceNirf->setText("NIRF (Current Setting)");
    m_pCheckBox_EnFaceNirf->setChecked(true);

	m_pCheckBox_NirfRingOnly = new QCheckBox(this);
	m_pCheckBox_NirfRingOnly->setText("NIRF Ring Only");
	m_pCheckBox_NirfRingOnly->setChecked(false);
#endif

	m_pCheckBox_OctMaxProjection = new QCheckBox(this);
    m_pCheckBox_OctMaxProjection->setText("OCT Max Projection");
    m_pCheckBox_OctMaxProjection->setChecked(true);

    // Set layout
	QGridLayout *pGridLayout = new QGridLayout;
	pGridLayout->setSpacing(3);

    QHBoxLayout *pHBoxLayout_Range = new QHBoxLayout;
    pHBoxLayout_Range->addWidget(m_pLabel_Range);
    pHBoxLayout_Range->addWidget(m_pLineEdit_RangeStart);
    pHBoxLayout_Range->addWidget(m_pLineEdit_RangeEnd);
    pHBoxLayout_Range->addStretch(1);


    pGridLayout->addWidget(m_pPushButton_SaveCrossSections, 0, 0);
	
	QHBoxLayout *pHBoxLayout_RectResize = new QHBoxLayout;
	pHBoxLayout_RectResize->addWidget(m_pCheckBox_RectImage);
	pHBoxLayout_RectResize->addWidget(m_pCheckBox_ResizeRectImage);
	pHBoxLayout_RectResize->addWidget(m_pLineEdit_RectWidth);
	pHBoxLayout_RectResize->addWidget(m_pLineEdit_RectHeight);
    pHBoxLayout_RectResize->addStretch(1);
    //pHBoxLayout_RectResize->addItem(new QSpacerItem(0, 0, QSizePolicy::Preferred, QSizePolicy::Fixed));

    pGridLayout->addItem(pHBoxLayout_RectResize, 0, 1, 1, 3);
    //pGridLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 3);

	QHBoxLayout *pHBoxLayout_CircResize = new QHBoxLayout;
	pHBoxLayout_CircResize->addWidget(m_pCheckBox_CircImage);
	pHBoxLayout_CircResize->addWidget(m_pCheckBox_ResizeCircImage);
	pHBoxLayout_CircResize->addWidget(m_pLineEdit_CircDiameter);
    pHBoxLayout_CircResize->addStretch(1);
    //pHBoxLayout_CircResize->addItem(new QSpacerItem(0, 0, QSizePolicy::Preferred, QSizePolicy::Fixed));

    pGridLayout->addItem(pHBoxLayout_CircResize, 1, 1, 1, 3);

	QHBoxLayout *pHBoxLayout_LongiResize = new QHBoxLayout;
	pHBoxLayout_LongiResize->addWidget(m_pCheckBox_LongiImage);
	pHBoxLayout_LongiResize->addWidget(m_pCheckBox_ResizeLongiImage);
	pHBoxLayout_LongiResize->addWidget(m_pLineEdit_LongiWidth);
	pHBoxLayout_LongiResize->addWidget(m_pLineEdit_LongiHeight);
	pHBoxLayout_LongiResize->addStretch(1);
	//pHBoxLayout_RectResize->addItem(new QSpacerItem(0, 0, QSizePolicy::Preferred, QSizePolicy::Fixed));

	pGridLayout->addItem(pHBoxLayout_LongiResize, 2, 1, 1, 3);

#ifdef OCT_FLIM
    pGridLayout->addWidget(m_pCheckBox_CrossSectionCh1, 3, 1);
    pGridLayout->addWidget(m_pCheckBox_CrossSectionCh2, 3, 2);
    pGridLayout->addWidget(m_pCheckBox_CrossSectionCh3, 3, 3);

    pGridLayout->addWidget(m_pCheckBox_Multichannel, 4, 1, 1, 3);
#endif
#ifdef OCT_NIRF
	QHBoxLayout *pHBoxLayout_NirfOption = new QHBoxLayout;
	pHBoxLayout_NirfOption->addWidget(m_pCheckBox_CrossSectionNirf);
	pHBoxLayout_NirfOption->addWidget(m_pCheckBox_NirfRingOnly);
	pHBoxLayout_NirfOption->addStretch(1);

    pGridLayout->addItem(pHBoxLayout_NirfOption, 3, 1, 1, 2);
#endif

    pGridLayout->addItem(pHBoxLayout_Range, 1, 0);


    pGridLayout->addWidget(m_pPushButton_SaveEnFaceMaps, 5, 0);

	QHBoxLayout *pHBoxLayout_DataOption = new QHBoxLayout;
	pHBoxLayout_DataOption->addWidget(m_pCheckBox_RawData);
	pHBoxLayout_DataOption->addWidget(m_pCheckBox_ScaledImage);
	pHBoxLayout_DataOption->addStretch(1);

    pGridLayout->addItem(pHBoxLayout_DataOption, 5, 1, 1, 2);

#ifdef OCT_FLIM
    pGridLayout->addWidget(m_pCheckBox_EnFaceCh1, 6, 1);
    pGridLayout->addWidget(m_pCheckBox_EnFaceCh2, 6, 2);
    pGridLayout->addWidget(m_pCheckBox_EnFaceCh3, 6, 3);
#endif

#ifdef OCT_NIRF
    pGridLayout->addWidget(m_pCheckBox_EnFaceNirf, 6, 1, 1, 2);
#endif

    pGridLayout->addWidget(m_pCheckBox_OctMaxProjection, 7, 1, 1, 2);
	
    setLayout(pGridLayout);

    // Connect
	connect(m_pPushButton_SaveCrossSections, SIGNAL(clicked(bool)), this, SLOT(saveCrossSections()));
	connect(m_pPushButton_SaveEnFaceMaps, SIGNAL(clicked(bool)), this, SLOT(saveEnFaceMaps()));

    connect(m_pLineEdit_RangeStart, SIGNAL(textChanged(const QString &)), this, SLOT(setRange()));
    connect(m_pLineEdit_RangeEnd, SIGNAL(textChanged(const QString &)), this, SLOT(setRange()));

	connect(m_pCheckBox_ResizeRectImage, SIGNAL(toggled(bool)), SLOT(enableRectResize(bool)));
	connect(m_pCheckBox_ResizeCircImage, SIGNAL(toggled(bool)), SLOT(enableCircResize(bool)));
	connect(m_pCheckBox_ResizeLongiImage, SIGNAL(toggled(bool)), SLOT(enableLongiResize(bool)));

	connect(m_pResultTab, SIGNAL(setWidgets(bool)), m_pResultTab, SLOT(setWidgetsEnabled(bool)));
	connect(this, SIGNAL(setWidgets(bool)), this, SLOT(setWidgetsEnabled(bool)));
	connect(this, SIGNAL(savedSingleFrame(int)), m_pResultTab->getProgressBar(), SLOT(setValue(int)));
}

SaveResultDlg::~SaveResultDlg()
{
}

void SaveResultDlg::closeEvent(QCloseEvent *e)
{
	if (!m_pPushButton_SaveCrossSections->isEnabled())
		e->ignore();
	else
		finished(0);
}

void SaveResultDlg::keyPressEvent(QKeyEvent *e)
{
	if (e->key() != Qt::Key_Escape)
		QDialog::keyPressEvent(e);
}

void SaveResultDlg::setCircRadius(int circ_radius)
{
	m_pLineEdit_CircDiameter->setText(QString::number(2 * circ_radius));
	m_pLineEdit_LongiHeight->setText(QString::number(2 * circ_radius));
}


void SaveResultDlg::saveCrossSections()
{
	std::thread t1([&]() {

		// Check Status /////////////////////////////////////////////////////////////////////////////
		CrossSectionCheckList checkList;
		checkList.bRect = m_pCheckBox_RectImage->isChecked();
		checkList.bCirc = m_pCheckBox_CircImage->isChecked();
		checkList.bLongi = m_pCheckBox_LongiImage->isChecked();
		checkList.bRectResize = m_pCheckBox_ResizeRectImage->isChecked();
		checkList.bCircResize = m_pCheckBox_ResizeCircImage->isChecked();
		checkList.bLongiResize = m_pCheckBox_ResizeLongiImage->isChecked();
		checkList.nRectWidth = m_pLineEdit_RectWidth->text().toInt();
		checkList.nRectHeight = m_pLineEdit_RectHeight->text().toInt();
		checkList.nCircDiameter = m_pLineEdit_CircDiameter->text().toInt();
		checkList.nLongiWidth = m_pLineEdit_LongiWidth->text().toInt();
		checkList.nLongiHeight = m_pLineEdit_LongiHeight->text().toInt();
#ifdef OCT_FLIM
		checkList.bCh[0] = m_pCheckBox_CrossSectionCh1->isChecked();
		checkList.bCh[1] = m_pCheckBox_CrossSectionCh2->isChecked();
		checkList.bCh[2] = m_pCheckBox_CrossSectionCh3->isChecked();
		checkList.bMulti = m_pCheckBox_Multichannel->isChecked();
#endif
#ifdef OCT_NIRF
		checkList.bNirf = m_pCheckBox_CrossSectionNirf->isChecked();
		checkList.bNirfRingOnly = m_pCheckBox_NirfRingOnly->isChecked();
#endif

#ifdef OCT_FLIM
		// Median filtering for FLIM maps ///////////////////////////////////////////////////////////
		IppiSize roi_flimproj = { m_pResultTab->m_intensityMap.at(0).size(0), m_pResultTab->m_intensityMap.at(0).size(1) };

		std::vector<np::Uint8Array2> intensityMap;
		std::vector<np::Uint8Array2> lifetimeMap;
		for (int i = 0; i < 3; i++)
		{
			np::Uint8Array2 intensity = np::Uint8Array2(roi_flimproj.width, roi_flimproj.height);
			np::Uint8Array2 lifetime = np::Uint8Array2(roi_flimproj.width, roi_flimproj.height);

			ippiScale_32f8u_C1R(m_pResultTab->m_intensityMap.at(i), sizeof(float) * roi_flimproj.width,
				intensity.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width,
				roi_flimproj, m_pConfig->flimIntensityRange.min, m_pConfig->flimIntensityRange.max);
#ifdef GALVANO_MIRROR
			if (m_pConfig->galvoHorizontalShift)
			{
				for (int i = 0; i < roi_flimproj.height; i++)
				{
					uint8_t* pImg = intensity.raw_ptr() + i * roi_flimproj.width;
					std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift / 4, pImg + roi_flimproj.width);
				}
			}
#endif
			(*m_pResultTab->m_pMedfiltIntensityMap)(intensity.raw_ptr());

			ippiScale_32f8u_C1R(m_pResultTab->m_lifetimeMap.at(i), sizeof(float) * roi_flimproj.width,
				lifetime.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width,
				roi_flimproj, m_pConfig->flimLifetimeRange.min, m_pConfig->flimLifetimeRange.max);
#ifdef GALVANO_MIRROR
			if (m_pConfig->galvoHorizontalShift)
			{
				for (int i = 0; i < roi_flimproj.height; i++)
				{
					uint8_t* pImg = lifetime.raw_ptr() + i * roi_flimproj.width;
					std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift / 4, pImg + roi_flimproj.width);
				}
			}
#endif
			(*m_pResultTab->m_pMedfiltLifetimeMap)(lifetime.raw_ptr());

			intensityMap.push_back(intensity);
			lifetimeMap.push_back(lifetime);
		}
#endif

#ifdef OCT_NIRF
		// NIRF maps ////////////////////////////////////////////////////////////////////////////////
#ifndef TWO_CHANNEL_NIRF
		IppiSize roi_nirf = { m_pResultTab->m_nirfMap0.size(0), m_pResultTab->m_nirfMap0.size(1) };
#else
		IppiSize roi_nirf = { m_pResultTab->m_nirfMap1_0.size(0), m_pResultTab->m_nirfMap1_0.size(1) };
#endif

#ifndef TWO_CHANNEL_NIRF
		np::Uint8Array2 nirfMap(roi_nirf.width, roi_nirf.height);
		ippiScale_32f8u_C1R(m_pResultTab->m_nirfMap0.raw_ptr(), sizeof(float) * roi_nirf.width, nirfMap.raw_ptr(), sizeof(uint8_t) * roi_nirf.width,
			roi_nirf, m_pConfig->nirfRange.min, m_pConfig->nirfRange.max);
#else
		np::Uint8Array2 nirfMap1(roi_nirf.width, roi_nirf.height);
		ippiScale_32f8u_C1R(m_pResultTab->m_nirfMap1_0.raw_ptr(), sizeof(float) * roi_nirf.width, nirfMap1.raw_ptr(), sizeof(uint8_t) * roi_nirf.width,
			roi_nirf, m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[0].max);

		np::Uint8Array2 nirfMap2(roi_nirf.width, roi_nirf.height);
		ippiScale_32f8u_C1R(m_pResultTab->m_nirfMap2_0.raw_ptr(), sizeof(float) * roi_nirf.width, nirfMap2.raw_ptr(), sizeof(uint8_t) * roi_nirf.width,
			roi_nirf, m_pConfig->nirfRange[1].min, m_pConfig->nirfRange[1].max);
#endif
#ifdef GALVANO_MIRROR
		if (m_pConfig->galvoHorizontalShift)
		{
			int roi_nirf_width_non4 = m_pResultTab->getRectImageView()->getRender()->m_pImage->width();
			for (int i = 0; i < roi_nirf.height; i++)
			{
#ifndef TWO_CHANNEL_NIRF
				uint8_t* pImg = nirfMap.raw_ptr() + i * roi_nirf.width;
				std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_nirf_width_non4);
#else
				uint8_t* pImg1 = nirfMap1.raw_ptr() + i * roi_nirf.width;
				std::rotate(pImg1, pImg1 + m_pConfig->galvoHorizontalShift, pImg1 + roi_nirf_width_non4);

				uint8_t* pImg2 = nirfMap2.raw_ptr() + i * roi_nirf.width;
				std::rotate(pImg2, pImg2 + m_pConfig->galvoHorizontalShift, pImg2 + roi_nirf_width_non4);
#endif
			}
		}
#endif
		if (m_pResultTab->getNirfDistCompDlg())
		{
			if (m_pResultTab->getNirfDistCompDlg()->isFiltered())
			{
#ifndef TWO_CHANNEL_NIRF
				(*m_pResultTab->m_pMedfiltNirf)(nirfMap.raw_ptr());
#else			
				(*m_pResultTab->m_pMedfiltNirf)(nirfMap1.raw_ptr());
				(*m_pResultTab->m_pMedfiltNirf)(nirfMap2.raw_ptr());
#endif
			}
		}
#endif
		// Set Widgets //////////////////////////////////////////////////////////////////////////////
		emit setWidgets(false);
		emit m_pResultTab->setWidgets(false);
#ifdef OCT_NIRF
		if (m_pResultTab->getNirfDistCompDlg())
		{
			emit m_pResultTab->getNirfDistCompDlg()->setWidgets(false);
			m_pResultTab->getNirfDistCompDlg()->setClosed(false);
		}
#endif
		if (m_pResultTab->getLongitudinalViewDlg())
		{
			emit m_pResultTab->getLongitudinalViewDlg()->setWidgets(false);
			m_pResultTab->getLongitudinalViewDlg()->setClosed(false);
		}
		m_nSavedFrames = 0;

		// Scaling Images ///////////////////////////////////////////////////////////////////////////
#ifdef OCT_FLIM
		std::thread scaleImages([&]() { scaling(m_pResultTab->m_vectorOctImage, intensityMap, lifetimeMap, checkList); });
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
        std::thread scaleImages([&]() { scaling(m_pResultTab->m_vectorOctImage); });
#else
#ifndef TWO_CHANNEL_NIRF
        std::thread scaleImages([&]() { scaling(m_pResultTab->m_vectorOctImage, nirfMap, checkList); });
#else
		std::thread scaleImages([&]() { scaling(m_pResultTab->m_vectorOctImage, nirfMap1, nirfMap2, checkList); });
#endif
#endif
#endif
#if defined(OCT_FLIM) || defined(OCT_NIRF)
		// Converting Images ////////////////////////////////////////////////////////////////////////
		std::thread convertImages([&]() { converting(checkList); });
#endif
		// Rect Writing /////////////////////////////////////////////////////////////////////////////
		std::thread writeRectImages([&]() { rectWriting(checkList); });

		// Circularizing ////////////////////////////////////////////////////////////////////////////		
		std::thread circularizeImages([&]() { circularizing(checkList); });

		// Circ Writing /////////////////////////////////////////////////////////////////////////////
		std::thread writeCircImages([&]() { circWriting(checkList); });

		// Wait for threads end /////////////////////////////////////////////////////////////////////
		scaleImages.join();
#if defined(OCT_FLIM) || defined(OCT_NIRF)
		convertImages.join();
#endif
		writeRectImages.join();
		circularizeImages.join();
		writeCircImages.join();		

		// Reset Widgets ////////////////////////////////////////////////////////////////////////////
		emit setWidgets(true);
		emit m_pResultTab->setWidgets(true);
#ifdef OCT_NIRF
		if (m_pResultTab->getNirfDistCompDlg())
		{
			emit m_pResultTab->getNirfDistCompDlg()->setWidgets(true);
			m_pResultTab->getNirfDistCompDlg()->setClosed(true);
		}
#endif
		if (m_pResultTab->getLongitudinalViewDlg())
		{
			emit m_pResultTab->getLongitudinalViewDlg()->setWidgets(true);
			m_pResultTab->getLongitudinalViewDlg()->setClosed(true);
		}

#ifdef OCT_FLIM
		std::vector<np::Uint8Array2> clear_vector1;
		clear_vector1.swap(intensityMap);
		std::vector<np::Uint8Array2> clear_vector2;
		clear_vector2.swap(lifetimeMap);
#endif
	});

	t1.detach();
}

void SaveResultDlg::saveEnFaceMaps()
{
	std::thread t1([&]() {

		// Check Status /////////////////////////////////////////////////////////////////////////////
		EnFaceCheckList checkList;
		checkList.bRaw = m_pCheckBox_RawData->isChecked();
		checkList.bScaled = m_pCheckBox_ScaledImage->isChecked();
#ifdef OCT_FLIM
		checkList.bCh[0] = m_pCheckBox_EnFaceCh1->isChecked();
		checkList.bCh[1] = m_pCheckBox_EnFaceCh2->isChecked();
		checkList.bCh[2] = m_pCheckBox_EnFaceCh3->isChecked();
#endif
#ifdef OCT_NIRF
        checkList.bNirf = m_pCheckBox_EnFaceNirf->isChecked();
#endif
		checkList.bOctProj = m_pCheckBox_OctMaxProjection->isChecked();

        int start = m_pLineEdit_RangeStart->text().toInt();
        int end = m_pLineEdit_RangeEnd->text().toInt();
		int width_non4 = m_pResultTab->getRectImageView()->getRender()->m_pImage->width();

		// Set Widgets //////////////////////////////////////////////////////////////////////////////
		emit setWidgets(false);
		emit m_pResultTab->setWidgets(false);
#ifdef OCT_NIRF
		if (m_pResultTab->getNirfDistCompDlg())
		{
			emit m_pResultTab->getNirfDistCompDlg()->setWidgets(false);
			m_pResultTab->getNirfDistCompDlg()->setClosed(false);
		}
#endif
		if (m_pResultTab->getLongitudinalViewDlg())
		{
			emit m_pResultTab->getLongitudinalViewDlg()->setWidgets(false);
			m_pResultTab->getLongitudinalViewDlg()->setClosed(false);
		}

		// Create directory for saving en face maps /////////////////////////////////////////////////
		QString enFacePath = m_pResultTab->m_path + "/en_face/";
		if (checkList.bRaw || checkList.bScaled) QDir().mkdir(enFacePath);

		// Raw en face map writing //////////////////////////////////////////////////////////////////
		if (checkList.bRaw)
		{
#ifdef OCT_FLIM
			for (int i = 0; i < 3; i++)
			{
				if (checkList.bCh[i])
				{
                    QFile fileIntensity(enFacePath + QString("intensity_range[%1 %2]_ch%3.enface").arg(start).arg(end).arg(i + 1));
					if (false != fileIntensity.open(QIODevice::WriteOnly))
					{
						for (int j = 0; j < (end - start + 1); j++)
							fileIntensity.write(reinterpret_cast<char*>(&m_pResultTab->m_intensityMap.at(i)(0, start - 1 + j)), sizeof(float) * width_non4 / 4);
						fileIntensity.close();
					}

                    QFile fileLifetime(enFacePath + QString("lifetime_range[%1 %2]_ch%3.enface").arg(start).arg(end).arg(i + 1));
					if (false != fileLifetime.open(QIODevice::WriteOnly))
					{
						for (int j = 0; j < (end - start + 1); j++)
							fileLifetime.write(reinterpret_cast<char*>(&m_pResultTab->m_lifetimeMap.at(i)(0, start - 1 + j)), sizeof(float) * width_non4 / 4);
						fileLifetime.close();
					}
				}
			}
#endif
#ifdef OCT_NIRF
            if (checkList.bNirf)
            {
#ifndef TWO_CHANNEL_NIRF
                QString nirfName;
                if (m_pResultTab->getNirfDistCompDlg())
                {                    
                    if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
                    {
                        if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
                           nirfName = enFacePath + QString("tbr_nirf_map_range[%1 %2].enface").arg(start).arg(end);
                        else
                           nirfName = enFacePath + QString("comp_nirf_map_range[%1 %2].enface").arg(start).arg(end);
                    }
                    else
                        nirfName = enFacePath + QString("bg_sub_nirf_map_range[%1 %2].enface").arg(start).arg(end);
                }
                else
                {
                    nirfName = enFacePath + QString("raw_nirf_map_range[%1 %2].enface").arg(start).arg(end);
                }

                QFile fileNirfMap(nirfName);
                if (false != fileNirfMap.open(QIODevice::WriteOnly))
                {
                    IppiSize roi_nirf = { m_pResultTab->m_nirfMap.size(0), m_pResultTab->m_nirfMap.size(1) };

                    np::FloatArray2 nirfMap(roi_nirf.width, roi_nirf.height);
                    ippiCopy_32f_C1R(m_pResultTab->m_nirfMap0.raw_ptr(), sizeof(float) * m_pResultTab->m_nirfMap0.size(0),
                                     nirfMap.raw_ptr(), sizeof(float) * nirfMap.size(0), roi_nirf);
#ifdef GALVANO_MIRROR
					if (m_pConfig->galvoHorizontalShift)
					{
						for (int i = 0; i < roi_nirf.height; i++)
						{
							float* pImg = nirfMap.raw_ptr() + i * roi_nirf.width;
							std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + width_non4);
						}
					}
#endif
                    fileNirfMap.write(reinterpret_cast<char*>(&nirfMap(0, start - 1)), sizeof(float) * nirfMap.size(0) * (end - start + 1));
                    fileNirfMap.close();
                }                

				if (m_pResultTab->getNirfDistCompDlg())
					if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
						m_pResultTab->getNirfDistCompDlg()->setCompInfo(nirfName.replace("enface", "log"));
#else
				QString nirfName[2];
				for (int i = 0; i < 2; i++)
				{
					if (m_pResultTab->getNirfDistCompDlg())
					{
						if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
						{
							if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
								nirfName[i] = enFacePath + QString("tbr_nirf_map_ch%1_range[%2 %3].enface").arg(i + 1).arg(start).arg(end);
							else
								nirfName[i] = enFacePath + QString("comp_nirf_map_ch%1_range[%2 %3].enface").arg(i + 1).arg(start).arg(end);
						}
						else
							nirfName[i] = enFacePath + QString("bg_sub_nirf_map_ch%1_range[%2 %3].enface").arg(i + 1).arg(start).arg(end);
					}
					else
					{
						nirfName[i] = enFacePath + QString("raw_nirf_map_ch%1_range[%2 %3].enface").arg(i + 1).arg(start).arg(end);
					}
				}

				for (int i = 0; i < 2 ; i++)
				{
					QFile fileNirfMap(nirfName[i]);
					if (false != fileNirfMap.open(QIODevice::WriteOnly))
					{
						auto pNirfMap = (i == 0) ? &m_pResultTab->m_nirfMap1 : &m_pResultTab->m_nirfMap2;
						auto pNirfMap0 = (i == 0) ? &m_pResultTab->m_nirfMap1_0 : &m_pResultTab->m_nirfMap2_0;
						IppiSize roi_nirf = { pNirfMap->size(0), pNirfMap->size(1) };

						np::FloatArray2 nirfMap(roi_nirf.width, roi_nirf.height);
						ippiCopy_32f_C1R(pNirfMap0->raw_ptr(), sizeof(float) * pNirfMap0->size(0),
							nirfMap.raw_ptr(), sizeof(float) * nirfMap.size(0), roi_nirf);
#ifdef GALVANO_MIRROR
						if (m_pConfig->galvoHorizontalShift)
						{
							int roi_nirf_width_non4 = m_pResultTab->getRectImageView()->getRender()->m_pImage->width();
							for (int i = 0; i < roi_nirf.height; i++)
							{
								float* pImg = nirfMap.raw_ptr() + i * roi_nirf.width;
								std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_nirf_width_non4);
							}
						}
#endif
						fileNirfMap.write(reinterpret_cast<char*>(&nirfMap(0, start - 1)), sizeof(float) * nirfMap.size(0) * (end - start + 1));
						fileNirfMap.close();
					}

					if (m_pResultTab->getNirfDistCompDlg())
						if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
							m_pResultTab->getNirfDistCompDlg()->setCompInfo(nirfName[i].replace("enface", "log"));
				}
#endif
            }
#endif
			if (checkList.bOctProj)
			{
                QFile fileOctMaxProj(enFacePath + QString("oct_max_projection_range[%1 %2].enface").arg(start).arg(end));
				if (false != fileOctMaxProj.open(QIODevice::WriteOnly))
				{
                    IppiSize roi_proj = { width_non4, m_pResultTab->m_octProjection.size(1) };

                    np::FloatArray2 octProj(roi_proj.width, roi_proj.height);
                    ippiCopy_32f_C1R(m_pResultTab->m_octProjection.raw_ptr(), sizeof(float) * m_pResultTab->m_octProjection.size(0),
                                     octProj.raw_ptr(), sizeof(float) * octProj.size(0), roi_proj);
#ifdef GALVANO_MIRROR
					if (m_pConfig->galvoHorizontalShift)
					{
						for (int i = 0; i < roi_proj.height; i++)
						{
							float* pImg = octProj.raw_ptr() + i * roi_proj.width;
							std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_proj.width);
						}
					}
#endif
                    fileOctMaxProj.write(reinterpret_cast<char*>(&octProj(0, start - 1)), sizeof(float) * octProj.size(0) * (end - start + 1));
					fileOctMaxProj.close();
				}
			}
		}

		// Scaled en face map writing ///////////////////////////////////////////////////////////////
		if (checkList.bScaled)
		{
			ColorTable temp_ctable;
#ifdef OCT_FLIM
			IppiSize roi_flimproj = { m_pResultTab->m_intensityMap.at(0).size(0), m_pResultTab->m_intensityMap.at(0).size(1) };

			for (int i = 0; i < 3; i++)
			{
				if (checkList.bCh[i])
				{
					ImageObject imgObjIntensity(roi_flimproj.width, roi_flimproj.height, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
					ImageObject imgObjLifetime(roi_flimproj.width, roi_flimproj.height, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentLifetimeColorTable()));
					ImageObject imgObjTemp(roi_flimproj.width, roi_flimproj.height, temp_ctable.m_colorTableVector.at(ColorTable::hsv));
					ImageObject imgObjHsv(roi_flimproj.width, roi_flimproj.height, temp_ctable.m_colorTableVector.at(ColorTable::hsv));

					// Intensity map
					ippiScale_32f8u_C1R(m_pResultTab->m_intensityMap.at(i), sizeof(float) * roi_flimproj.width,
						imgObjIntensity.arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width,
						roi_flimproj, m_pConfig->flimIntensityRange.min, m_pConfig->flimIntensityRange.max);
					ippiMirror_8u_C1IR(imgObjIntensity.arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width, roi_flimproj, ippAxsHorizontal);
#ifdef GALVANO_MIRROR
					if (m_pConfig->galvoHorizontalShift)
					{
						for (int i = 0; i < roi_flimproj.height; i++)
						{
							uint8_t* pImg = imgObjIntensity.arr.raw_ptr() + i * roi_flimproj.width;
							std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift / 4, pImg + roi_flimproj.width);
						}
					}
#endif
					IppiSize roi_flimfilt = { width_non4 / 4, roi_flimproj.height };
					np::Uint8Array2 temp_intensity(roi_flimfilt.width, roi_flimfilt.height);
					ippiCopy_8u_C1R(imgObjIntensity.arr.raw_ptr(), roi_flimproj.width, temp_intensity.raw_ptr(), roi_flimfilt.width, roi_flimfilt);
					(*m_pResultTab->m_pMedfiltIntensityMap)(temp_intensity);
					ippiCopy_8u_C1R(temp_intensity.raw_ptr(), roi_flimfilt.width, imgObjIntensity.arr.raw_ptr(), roi_flimproj.width, roi_flimproj);

                    imgObjIntensity.qindeximg.copy(0, roi_flimproj.height - end, width_non4 / 4, end - start + 1)
                        .save(enFacePath + QString("intensity_range[%1 %2]_ch%3_[%4 %5].bmp").arg(start).arg(end).arg(i + 1)
						.arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1), "bmp");

					// Lifetime map
					ippiScale_32f8u_C1R(m_pResultTab->m_lifetimeMap.at(i), sizeof(float) * roi_flimproj.width,
						imgObjLifetime.arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width,
						roi_flimproj, m_pConfig->flimLifetimeRange.min, m_pConfig->flimLifetimeRange.max);
					ippiMirror_8u_C1IR(imgObjLifetime.arr.raw_ptr(), sizeof(uint8_t) * roi_flimproj.width, roi_flimproj, ippAxsHorizontal);
#ifdef GALVANO_MIRROR
					if (m_pConfig->galvoHorizontalShift)
					{
						for (int i = 0; i < roi_flimproj.height; i++)
						{
							uint8_t* pImg = imgObjLifetime.arr.raw_ptr() + i * roi_flimproj.width;
							std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift / 4, pImg + roi_flimproj.width);
						}
					}
#endif					
					np::Uint8Array2 temp_lifetime(roi_flimfilt.width, roi_flimfilt.height);
					ippiCopy_8u_C1R(imgObjLifetime.arr.raw_ptr(), roi_flimproj.width, temp_lifetime.raw_ptr(), roi_flimfilt.width, roi_flimfilt);
					(*m_pResultTab->m_pMedfiltLifetimeMap)(temp_lifetime);
					ippiCopy_8u_C1R(temp_lifetime.raw_ptr(), roi_flimfilt.width, imgObjLifetime.arr.raw_ptr(), roi_flimproj.width, roi_flimproj);

                    imgObjLifetime.qindeximg.copy(0, roi_flimproj.height - end, width_non4 / 4, end - start + 1)
                        .save(enFacePath + QString("lifetime_range[%1 %2]_ch%3_[%4 %5].bmp").arg(start).arg(end).arg(i + 1)
						.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1), "bmp");

#if (LIFETIME_COLORTABLE == 7)
					// HSV enhanced map
					memset(imgObjTemp.qrgbimg.bits(), 255, imgObjTemp.qrgbimg.byteCount()); // Saturation is set to be 255.
					imgObjTemp.setRgbChannelData(imgObjLifetime.qindeximg.bits(), 0); // Hue	
					uint8_t *pIntensity = new uint8_t[imgObjIntensity.qindeximg.byteCount()];
					memcpy(pIntensity, imgObjIntensity.qindeximg.bits(), imgObjIntensity.qindeximg.byteCount());
					ippsMulC_8u_ISfs(1.0, pIntensity, imgObjIntensity.qindeximg.byteCount(), 0);
					imgObjTemp.setRgbChannelData(pIntensity, 2); // Value
					delete[] pIntensity;

					ippiHSVToRGB_8u_C3R(imgObjTemp.qrgbimg.bits(), 3 * roi_flimproj.width, imgObjHsv.qrgbimg.bits(), 3 * roi_flimproj.width, roi_flimproj);
#else
					// Non HSV intensity-weight map
					ImageObject tempImgObj(roi_flimproj.width, roi_flimproj.height, temp_ctable.m_colorTableVector.at(ColorTable::gray));

					imgObjLifetime.convertRgb();
					memcpy(tempImgObj.qindeximg.bits(), imgObjIntensity.arr.raw_ptr(), tempImgObj.qindeximg.byteCount());
					tempImgObj.convertRgb();

					ippsMul_8u_Sfs(imgObjLifetime.qrgbimg.bits(), tempImgObj.qrgbimg.bits(), imgObjHsv.qrgbimg.bits(), imgObjHsv.qrgbimg.byteCount(), 8);
#endif
                    imgObjHsv.qrgbimg.copy(0, roi_flimproj.height - end, width_non4 / 4, end - start + 1)
                        .save(enFacePath + QString("flim_map_range[%1 %2]_ch%3_i[%4 %5]_t[%6 %7].bmp").arg(start).arg(end).arg(i + 1)
						.arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1)
						.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1), "bmp");
				}
			}
#endif

#ifdef OCT_NIRF
            if (checkList.bNirf)
            {
                // NIRF maps ////////////////////////////////////////////////////////////////////////////////
#ifndef TWO_CHANNEL_NIRF
                IppiSize roi_nirf = { m_pResultTab->m_nirfMap0.size(0), m_pResultTab->m_nirfMap0.size(1) };
                ImageObject imgObjNirfMap(roi_nirf.width, roi_nirf.height, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));

                ippiScale_32f8u_C1R(m_pResultTab->m_nirfMap0.raw_ptr(), sizeof(float) * roi_nirf.width, imgObjNirfMap.arr.raw_ptr(), sizeof(uint8_t) * roi_nirf.width, roi_nirf, m_pConfig->nirfRange.min, m_pConfig->nirfRange.max);
                ippiMirror_8u_C1IR(imgObjNirfMap.arr.raw_ptr(), sizeof(uint8_t) * roi_nirf.width, roi_nirf, ippAxsHorizontal);
#ifdef GALVANO_MIRROR
                if (m_pConfig->galvoHorizontalShift)
                {
                    int roi_nirf_width_non4 = m_pResultTab->getRectImageView()->getRender()->m_pImage->width();
                    for (int i = 0; i < roi_nirf.height; i++)
                    {
                        uint8_t* pImg = imgObjNirfMap.arr.raw_ptr() + i * roi_nirf.width;
                        std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_nirf_width_non4);
                    }
                }
#endif
                if (m_pResultTab->getNirfDistCompDlg())
                    if (m_pResultTab->getNirfDistCompDlg()->isFiltered())
                        (*m_pResultTab->m_pMedfiltNirf)(imgObjNirfMap.arr.raw_ptr());

                QString nirfName;
                if (m_pResultTab->getNirfDistCompDlg())
                {
                    if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
                    {
                        if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
                            nirfName = enFacePath + QString("tbr_nirf_map_range[%1 %2]_i[%3 %4].bmp");
                        else
                            nirfName = enFacePath + QString("comp_nirf_map_range[%1 %2]_i[%3 %4].bmp");
                    }
                    else
                        nirfName = enFacePath + QString("bg_sub_nirf_map_range[%1 %2]_i[%3 %4].bmp");
                }
                else
                {
                    nirfName = enFacePath + QString("raw_nirf_map_range[%1 %2]_i[%3 %4].bmp");
                }

                imgObjNirfMap.qindeximg.copy(0, roi_nirf.height - end, m_pResultTab->m_nirfMap.size(0), end - start + 1).
                        save(nirfName.arg(start).arg(end).arg(m_pConfig->nirfRange.min, 2, 'f', 2).arg(m_pConfig->nirfRange.max, 2, 'f', 2), "bmp");

                if (m_pResultTab->getNirfDistCompDlg())
                    if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
						m_pResultTab->getNirfDistCompDlg()->setCompInfo(nirfName.arg(start).arg(end).arg(m_pConfig->nirfRange.min, 2, 'f', 2).arg(m_pConfig->nirfRange.max, 2, 'f', 2).replace("bmp", "log"));
#else
				for (int i = 0; i < 2; i++)
				{
					auto pNirfMap = (i == 0) ? &m_pResultTab->m_nirfMap1 : &m_pResultTab->m_nirfMap2;
					auto pNirfMap0 = (i == 0) ? &m_pResultTab->m_nirfMap1_0 : &m_pResultTab->m_nirfMap2_0;

					IppiSize roi_nirf = { pNirfMap0->size(0), pNirfMap0->size(1) };
					int table = (i == 0) ? NIRF_COLORTABLE1 : NIRF_COLORTABLE2;
					ImageObject imgObjNirfMap(roi_nirf.width, roi_nirf.height, temp_ctable.m_colorTableVector.at(table));

					ippiScale_32f8u_C1R(pNirfMap0->raw_ptr(), sizeof(float) * roi_nirf.width, imgObjNirfMap.arr.raw_ptr(), sizeof(uint8_t) * roi_nirf.width, 
						roi_nirf, m_pConfig->nirfRange[i].min, m_pConfig->nirfRange[i].max);
					ippiMirror_8u_C1IR(imgObjNirfMap.arr.raw_ptr(), sizeof(uint8_t) * roi_nirf.width, roi_nirf, ippAxsHorizontal);
#ifdef GALVANO_MIRROR
					if (m_pConfig->galvoHorizontalShift)
					{
						int roi_nirf_width_non4 = m_pResultTab->getRectImageView()->getRender()->m_pImage->width();
						for (int i = 0; i < roi_nirf.height; i++)
						{
							uint8_t* pImg = imgObjNirfMap.arr.raw_ptr() + i * roi_nirf.width;
							std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_nirf_width_non4);
						}
					}
#endif
					if (m_pResultTab->getNirfDistCompDlg())
						if (m_pResultTab->getNirfDistCompDlg()->isFiltered())
							(*m_pResultTab->m_pMedfiltNirf)(imgObjNirfMap.arr.raw_ptr());

					QString nirfName;
					if (m_pResultTab->getNirfDistCompDlg())
					{
						if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
						{
							if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
								nirfName = enFacePath + QString("tbr_nirf_map_ch%1_range[%2 %3]_i[%4 %5].bmp");
							else
								nirfName = enFacePath + QString("comp_nirf_map_ch%1_range[%2 %3]_i[%4 %5].bmp");
						}
						else
							nirfName = enFacePath + QString("bg_sub_nirf_map_ch%1_range[%2 %3]_i[%4 %5].bmp");
					}
					else
					{
						nirfName = enFacePath + QString("raw_nirf_map_ch%1_range[%2 %3]_i[%4 %5].bmp");
					}

					imgObjNirfMap.qindeximg.copy(0, roi_nirf.height - end, pNirfMap->size(0), end - start + 1).
						save(nirfName.arg(i + 1).arg(start).arg(end).arg(m_pConfig->nirfRange[i].min, 2, 'f', 2).arg(m_pConfig->nirfRange[i].max, 2, 'f', 2), "bmp");

					if (m_pResultTab->getNirfDistCompDlg())
						if (m_pResultTab->getNirfDistCompDlg()->isCompensating())

							m_pResultTab->getNirfDistCompDlg()->setCompInfo(nirfName.arg(i + 1).arg(start).arg(end).arg(m_pConfig->nirfRange[i].min, 2, 'f', 2).arg(m_pConfig->nirfRange[i].max, 2, 'f', 2).replace("bmp", "log"));
				}
#endif
            }
#endif
			if (checkList.bOctProj)
			{
				IppiSize roi_proj = { m_pResultTab->m_octProjection.size(0), m_pResultTab->m_octProjection.size(1) };
				ImageObject imgObjOctMaxProj(roi_proj.width, roi_proj.height, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()));

				ippiScale_32f8u_C1R(m_pResultTab->m_octProjection, sizeof(float) * roi_proj.width, imgObjOctMaxProj.arr.raw_ptr(), sizeof(uint8_t) * roi_proj.width, roi_proj, m_pConfig->octDbRange.min, m_pConfig->octDbRange.max);
				ippiMirror_8u_C1IR(imgObjOctMaxProj.arr.raw_ptr(), sizeof(uint8_t) * roi_proj.width, roi_proj, ippAxsHorizontal);
#ifdef GALVANO_MIRROR
				if (m_pConfig->galvoHorizontalShift)
				{
					for (int i = 0; i < roi_proj.height; i++)
					{
                        uint8_t* pImg = imgObjOctMaxProj.arr.raw_ptr() + i * roi_proj.width;
                        std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + width_non4);
					}
				}
#endif
                imgObjOctMaxProj.qindeximg.copy(0, roi_proj.height - end, width_non4, end - start + 1).
                        save(enFacePath + QString("oct_max_projection_range[%1 %2]_dB[%3 %4 g%5].bmp").arg(start).arg(end).arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2), "bmp");
			}
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(500));

		// Scaling MATLAB Script ////////////////////////////////////////////////////////////////////
		if (false == QFile::copy("scale_indicator.m", enFacePath + "scale_indicator.m"))
			printf("Error occurred while copying matlab sciprt.\n");

		// Reset Widgets ////////////////////////////////////////////////////////////////////////////
		emit setWidgets(true);
		emit m_pResultTab->setWidgets(true);
#ifdef OCT_NIRF
		if (m_pResultTab->getNirfDistCompDlg())
		{
			emit m_pResultTab->getNirfDistCompDlg()->setWidgets(true);
			m_pResultTab->getNirfDistCompDlg()->setClosed(true);
		}
#endif
		if (m_pResultTab->getLongitudinalViewDlg())
		{
			emit m_pResultTab->getLongitudinalViewDlg()->setWidgets(true);
			m_pResultTab->getLongitudinalViewDlg()->setClosed(true);
		}
	});

	t1.detach();
}

void SaveResultDlg::setRange()
{
    int start = m_pLineEdit_RangeStart->text().toInt();
    int end = m_pLineEdit_RangeEnd->text().toInt();

    if (start < 1)
        m_pLineEdit_RangeStart->setText(QString::number(1));
    if (end > m_pResultTab->m_vectorOctImage.size())
        m_pLineEdit_RangeEnd->setText(QString::number(m_pResultTab->m_vectorOctImage.size()));
}

void SaveResultDlg::enableRectResize(bool checked)
{
	m_pLineEdit_RectWidth->setEnabled(checked);
	m_pLineEdit_RectHeight->setEnabled(checked);
}

void SaveResultDlg::enableCircResize(bool checked)
{
	m_pLineEdit_CircDiameter->setEnabled(checked);
}

void SaveResultDlg::enableLongiResize(bool checked)
{
	m_pLineEdit_LongiWidth->setEnabled(checked);
	m_pLineEdit_LongiHeight->setEnabled(checked);
}

void SaveResultDlg::setWidgetsEnabled(bool enabled)
{
	// Save Cross-sections
	m_pPushButton_SaveCrossSections->setEnabled(enabled);
	m_pCheckBox_RectImage->setEnabled(enabled);
	m_pCheckBox_CircImage->setEnabled(enabled);
	m_pCheckBox_LongiImage->setEnabled(enabled);

	m_pCheckBox_ResizeRectImage->setEnabled(enabled);
	m_pCheckBox_ResizeCircImage->setEnabled(enabled);
	m_pCheckBox_ResizeLongiImage->setEnabled(enabled);

	m_pLineEdit_RectWidth->setEnabled(enabled);
	m_pLineEdit_RectHeight->setEnabled(enabled);
	m_pLineEdit_CircDiameter->setEnabled(enabled);
	m_pLineEdit_LongiWidth->setEnabled(enabled);
	m_pLineEdit_LongiHeight->setEnabled(enabled);
	if (enabled)
	{
		if (!m_pCheckBox_ResizeRectImage->isChecked())
		{
			m_pLineEdit_RectWidth->setEnabled(false);
			m_pLineEdit_RectHeight->setEnabled(false);
		}
		if (!m_pCheckBox_ResizeCircImage->isChecked())
			m_pLineEdit_CircDiameter->setEnabled(false);
		if (!m_pCheckBox_ResizeLongiImage->isChecked())
		{
			m_pLineEdit_LongiWidth->setEnabled(false);
			m_pLineEdit_LongiHeight->setEnabled(false);
		}
	}	

#ifdef OCT_FLIM
	m_pCheckBox_CrossSectionCh1->setEnabled(enabled);
	m_pCheckBox_CrossSectionCh2->setEnabled(enabled);
	m_pCheckBox_CrossSectionCh3->setEnabled(enabled);
	m_pCheckBox_Multichannel->setEnabled(enabled);
#endif

#ifdef OCT_NIRF
    m_pCheckBox_CrossSectionNirf->setEnabled(enabled);
	m_pCheckBox_NirfRingOnly->setEnabled(enabled);
#endif

	// Set Range
	m_pLabel_Range->setEnabled(enabled);
	m_pLineEdit_RangeStart->setEnabled(enabled);
	m_pLineEdit_RangeEnd->setEnabled(enabled);

	// Save En Face Maps
	m_pPushButton_SaveEnFaceMaps->setEnabled(enabled);
	m_pCheckBox_RawData->setEnabled(enabled);
	m_pCheckBox_ScaledImage->setEnabled(enabled);
#ifdef OCT_FLIM
	m_pCheckBox_EnFaceCh1->setEnabled(enabled);
	m_pCheckBox_EnFaceCh2->setEnabled(enabled);
	m_pCheckBox_EnFaceCh3->setEnabled(enabled);
#endif
#ifdef OCT_NIRF
    m_pCheckBox_EnFaceNirf->setEnabled(enabled);
#endif
	m_pCheckBox_OctMaxProjection->setEnabled(enabled);
}


#ifdef OCT_FLIM
void SaveResultDlg::scaling(std::vector<np::FloatArray2>& vectorOctImage, 
	std::vector<np::Uint8Array2>& intensityMap, std::vector<np::Uint8Array2>& lifetimeMap, CrossSectionCheckList checkList)
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
void SaveResultDlg::scaling(std::vector<np::FloatArray2>& vectorOctImage)
#else
#ifndef TWO_CHANNEL_NIRF
void SaveResultDlg::scaling(std::vector<np::FloatArray2>& vectorOctImage, np::Uint8Array2& nirfMap, CrossSectionCheckList checkList)
#else
void SaveResultDlg::scaling(std::vector<np::FloatArray2>& vectorOctImage, np::Uint8Array2& nirfMap1, np::Uint8Array2& nirfMap2, CrossSectionCheckList checkList)
#endif
#endif
#endif
{
	int nTotalFrame = (int)vectorOctImage.size();
	ColorTable temp_ctable;
	
	int frameCount = 0;
	while (frameCount < nTotalFrame)
	{
		// Create Image Object Array for threading operation
        IppiSize roi_oct = { vectorOctImage.at(0).size(0), vectorOctImage.at(0).size(1) };

		ImgObjVector* pImgObjVec = new ImgObjVector;

		// Image objects for OCT Images
		pImgObjVec->push_back(new ImageObject(roi_oct.height, roi_oct.width, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()), m_pConfig->octDbGamma));

#ifdef OCT_FLIM
		// Image objects for Ch1 FLIM
		pImgObjVec->push_back(new ImageObject(roi_oct.height / 4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE)));
		pImgObjVec->push_back(new ImageObject(roi_oct.height / 4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentLifetimeColorTable())));
		// Image objects for Ch2 FLIM
		pImgObjVec->push_back(new ImageObject(roi_oct.height / 4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE)));
		pImgObjVec->push_back(new ImageObject(roi_oct.height / 4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentLifetimeColorTable())));
		// Image objects for Ch3 FLIM
		pImgObjVec->push_back(new ImageObject(roi_oct.height / 4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE)));
		pImgObjVec->push_back(new ImageObject(roi_oct.height / 4, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentLifetimeColorTable())));
#endif

#ifdef OCT_NIRF
        // Image objects for NIRF
		if (!checkList.bNirfRingOnly)
		{
#ifndef TWO_CHANNEL_NIRF
			pImgObjVec->push_back(new ImageObject(roi_oct.height, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1)));
#else
			pImgObjVec->push_back(new ImageObject(roi_oct.height, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1)));
			pImgObjVec->push_back(new ImageObject(roi_oct.height, m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE2)));
#endif
		}
		else
		{
#ifndef TWO_CHANNEL_NIRF
			pImgObjVec->push_back(new ImageObject(roi_oct.height, roi_oct.width, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1)));
#else
			pImgObjVec->push_back(new ImageObject(roi_oct.height, roi_oct.width, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1)));
			pImgObjVec->push_back(new ImageObject(roi_oct.height, roi_oct.width, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE2)));
#endif
		}
#endif

		// OCT Visualization
		np::Uint8Array2 scale_temp(roi_oct.width, roi_oct.height);
		ippiScale_32f8u_C1R(vectorOctImage.at(frameCount), roi_oct.width * sizeof(float),
			scale_temp.raw_ptr(), roi_oct.width * sizeof(uint8_t), roi_oct, m_pConfig->octDbRange.min, m_pConfig->octDbRange.max);
		ippiTranspose_8u_C1R(scale_temp.raw_ptr(), roi_oct.width * sizeof(uint8_t), pImgObjVec->at(0)->arr.raw_ptr(), roi_oct.height * sizeof(uint8_t), roi_oct);
#ifdef GALVANO_MIRROR
		if (m_pConfig->galvoHorizontalShift)
		{
            int roi_oct_height_non4 = m_pResultTab->getRectImageView()->getRender()->m_pImage->width();
			for (int i = 0; i < roi_oct.width; i++)
			{
                uint8_t* pImg = pImgObjVec->at(0)->arr.raw_ptr() + i * roi_oct.height;
                std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_oct_height_non4);
			}
		}
#endif
		(*m_pResultTab->m_pMedfiltRect)(pImgObjVec->at(0)->arr.raw_ptr());

#ifdef OCT_FLIM
		// FLIM Visualization		
		IppiSize roi_flim = { roi_oct.height / 4, 1 };
		for (int i = 0; i < 3; i++)
		{
			if (checkList.bCh[i])
			{
				uint8_t* rectIntensity = &intensityMap.at(i)(0, frameCount);
				uint8_t* rectLifetime = &lifetimeMap.at(i)(0, frameCount);
				for (int j = 0; j < m_pConfig->ringThickness; j++)
				{
					memcpy(&pImgObjVec->at(1 + 2 * i)->arr(0, j), rectIntensity, sizeof(uint8_t) * roi_flim.width);
					memcpy(&pImgObjVec->at(2 + 2 * i)->arr(0, j), rectLifetime, sizeof(uint8_t) * roi_flim.width);
				}
			}
		}
#endif
#ifdef OCT_NIRF
        // NIRF Visualization
        IppiSize roi_nirf = { roi_oct.height, 1 };
		if (checkList.bNirf || checkList.bNirfRingOnly)
		{
#ifndef TWO_CHANNEL_NIRF
			uint8_t* rectNirf = &nirfMap(0, frameCount);
			for (int j = 0; j < m_pConfig->ringThickness; j++)
				memcpy(&pImgObjVec->at(1)->arr(0, j), rectNirf, sizeof(uint8_t) * roi_nirf.width);

			if (checkList.bNirfRingOnly)
			{
				ippiMirror_8u_C1IR(pImgObjVec->at(1)->arr.raw_ptr(), sizeof(uint8_t) * pImgObjVec->at(1)->arr.size(0), { pImgObjVec->at(1)->arr.size(0), pImgObjVec->at(1)->arr.size(1) }, ippAxsHorizontal);
				memset(&pImgObjVec->at(0)->arr(0, pImgObjVec->at(0)->arr.size(1) - m_pConfig->ringThickness), 0, sizeof(uint8_t) * m_pConfig->ringThickness * pImgObjVec->at(0)->arr.size(0));
			}
#else
			uint8_t* rectNirf1 = &nirfMap1(0, frameCount);
			uint8_t* rectNirf2 = &nirfMap2(0, frameCount);			
			int offset = checkList.bNirfRingOnly ? m_pConfig->ringThickness : 0;
			for (int j = 0; j < m_pConfig->ringThickness; j++)
			{
				memcpy(&pImgObjVec->at(1)->arr(0, offset + j), rectNirf1, sizeof(uint8_t) * roi_nirf.width);
				memcpy(&pImgObjVec->at(2)->arr(0, j), rectNirf2, sizeof(uint8_t) * roi_nirf.width);
			}

#ifdef CH_DIVIDING_LINE
			np::Uint8Array boundary(pImgObjVec->at(1)->arr.size(0));
			ippsSet_8u(255, boundary.raw_ptr(), pImgObjVec->at(1)->arr.size(0));

			memcpy(&pImgObjVec->at(1)->arr(0, 0), boundary.raw_ptr(), sizeof(uint8_t) * pImgObjVec->at(1)->arr.size(0));
			memcpy(&pImgObjVec->at(1)->arr(0, m_pConfig->ringThickness - 1), boundary.raw_ptr(), sizeof(uint8_t) * pImgObjVec->at(1)->arr.size(0));
			memcpy(&pImgObjVec->at(2)->arr(0, m_pConfig->ringThickness - 1), boundary.raw_ptr(), sizeof(uint8_t) * pImgObjVec->at(2)->arr.size(0));
#endif

			if (checkList.bNirfRingOnly)
			{
				ippiMirror_8u_C1IR(pImgObjVec->at(1)->arr.raw_ptr(), sizeof(uint8_t) * pImgObjVec->at(1)->arr.size(0), { pImgObjVec->at(1)->arr.size(0), pImgObjVec->at(1)->arr.size(1) }, ippAxsHorizontal);
				ippiMirror_8u_C1IR(pImgObjVec->at(2)->arr.raw_ptr(), sizeof(uint8_t) * pImgObjVec->at(2)->arr.size(0), { pImgObjVec->at(2)->arr.size(0), pImgObjVec->at(2)->arr.size(1) }, ippAxsHorizontal);
				memset(&pImgObjVec->at(0)->arr(0, pImgObjVec->at(0)->arr.size(1) - 2 * m_pConfig->ringThickness), 0, sizeof(uint8_t) * 2 * m_pConfig->ringThickness * pImgObjVec->at(0)->arr.size(0));
			}
#endif
        }
#endif
		frameCount++;

		// Push the buffers to sync Queues
#ifdef OCT_FLIM
		m_syncQueueConverting.push(pImgObjVec);
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
		m_syncQueueRectWriting.push(pImgObjVec);
#else
        m_syncQueueConverting.push(pImgObjVec);
#endif
#endif
	}
}

#if defined(OCT_FLIM) || defined(OCT_NIRF)
void SaveResultDlg::converting(CrossSectionCheckList checkList)
{
	int nTotalFrame = (int)m_pResultTab->m_vectorOctImage.size();

	int frameCount = 0;
	while (frameCount < nTotalFrame)
	{
		// Get the buffer from the previous sync Queue
		ImgObjVector *pImgObjVec = m_syncQueueConverting.pop();

		// Converting RGB
#ifdef OCT_FLIM
		if (checkList.bCh[0] || checkList.bCh[1] || checkList.bCh[2])
			pImgObjVec->at(0)->convertRgb();

		tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)3),
			[&](const tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				if (checkList.bCh[i])
				{
					pImgObjVec->at(1 + 2 * i)->convertScaledRgb(); // intensity
					pImgObjVec->at(2 + 2 * i)->convertScaledRgb(); // lifetime

					if (checkList.bMulti)
					{
						pImgObjVec->at(1 + 2 * i)->scaled4(); // intensity
						pImgObjVec->at(2 + 2 * i)->scaled4(); // lifetime
					}
				}
			}
		});
#elif defined(OCT_NIRF)
        if (checkList.bNirf)
        {
			if (!checkList.bNirfRingOnly)
			{
				pImgObjVec->at(0)->convertRgb();
				pImgObjVec->at(1)->convertNonScaledRgb(); // nirf
#ifdef TWO_CHANNEL_NIRF
				pImgObjVec->at(2)->convertNonScaledRgb(); // nirf
#endif
			}
        }
#endif
		frameCount++;

		// Push the buffers to sync Queues
		m_syncQueueRectWriting.push(pImgObjVec);		
	}
}
#endif

void SaveResultDlg::rectWriting(CrossSectionCheckList checkList)
{
	int nTotalFrame = (int)m_pResultTab->m_vectorOctImage.size();
	QString folderName;
	for (int i = 0; i < m_pResultTab->m_path.length(); i++)
		if (m_pResultTab->m_path.at(i) == QChar('/')) folderName = m_pResultTab->m_path.right(m_pResultTab->m_path.length() - i - 1);

    int start = m_pLineEdit_RangeStart->text().toInt();
    int end = m_pLineEdit_RangeEnd->text().toInt();

#ifdef OCT_FLIM
	if (!checkList.bCh[0] && !checkList.bCh[1] && !checkList.bCh[2])
	{
#elif defined (OCT_NIRF)
    if (!checkList.bNirf || checkList.bNirfRingOnly)
    {
#endif
		QString rectPath;

#ifndef TWO_CHANNEL_NIRF
		QString rectNirfPath;
#else
		QString rectNirfPath[2];
#endif

#ifndef OCT_NIRF
		rectPath = m_pResultTab->m_path + QString("/rect_image_dB[%1 %2 g%d]/").arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2);
#else
        rectPath = m_pResultTab->m_path + QString("/rect_image_dB[%1 %2 g%3]%4/").arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2).arg(checkList.bNirfRingOnly ? "_ring-masked" : "");
		if (checkList.bNirfRingOnly)
		{
#ifndef TWO_CHANNEL_NIRF
			if (m_pResultTab->getNirfDistCompDlg())
			{
				if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
				{
					if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
						rectNirfPath = QString("/rect_tbr_nirf_i[%1 %2]/");
					else
						rectNirfPath = QString("/rect_comp_nirf_i[%1 %2]/");
				}
				else
					rectNirfPath = QString("/rect_bg_sub_nirf_i[%1 %2]/");
			}
			else
			{
				rectNirfPath = QString("/rect_raw_nirf_i[%1 %2]/");
			}
			rectNirfPath = m_pResultTab->m_path + rectNirfPath.arg(m_pConfig->nirfRange.min, 2, 'f', 2).arg(m_pConfig->nirfRange.max, 2, 'f', 2);
#else
			for (int i = 0; i < 2; i++)
			{
				if (m_pResultTab->getNirfDistCompDlg())
				{
					if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
					{
						if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
							rectNirfPath[i] = QString("/rect_tbr_nirf_i%1[%2 %3]/");
						else
							rectNirfPath[i] = QString("/rect_comp_nirf_i%1[%2 %3]/");
					}
					else
						rectNirfPath[i] = QString("/rect_bg_sub_nirf_i%1[%2 %3]/");
				}
				else
				{
					rectNirfPath[i] = QString("/rect_raw_nirf_i%1[%2 %3]/");
				}
				rectNirfPath[i] = m_pResultTab->m_path + rectNirfPath[i].arg(i + 1).arg(m_pConfig->nirfRange[i].min, 2, 'f', 2).arg(m_pConfig->nirfRange[i].max, 2, 'f', 2);
			}
#endif
		}
#endif

		if (checkList.bRect)
		{
			QDir().mkdir(rectPath);
#ifdef OCT_NIRF
			if (checkList.bNirfRingOnly)
			{
#ifndef TWO_CHANNEL_NIRF
				QDir().mkdir(rectNirfPath);
#else
				QDir().mkdir(rectNirfPath[0]);
				QDir().mkdir(rectNirfPath[1]);
#endif
			}
#endif
		}

		int frameCount = 0;
		while (frameCount < nTotalFrame)
		{
			// Get the buffer from the previous sync Queue
			ImgObjVector *pImgObjVec = m_syncQueueRectWriting.pop();	

            // Range test
            if (((frameCount + 1) >= start) && ((frameCount + 1) <= end))
            {
                // Write rect images
                if (checkList.bRect)
                {
                    int original_nAlines = m_pResultTab->getRectImageView()->getRender()->m_pImage->width();
					
                    if (original_nAlines == pImgObjVec->at(0)->getWidth())
                    {
                        if (!checkList.bRectResize)
                            pImgObjVec->at(0)->qindeximg.save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
						else
						{
							cv::Mat src(pImgObjVec->at(0)->getHeight(), pImgObjVec->at(0)->getWidth(), CV_8UC1, pImgObjVec->at(0)->qindeximg.bits());
							IplImage src_img(src);
							IplImage *p_dst_img = cvCreateImage(cv::Size(checkList.nRectWidth, checkList.nRectHeight), IPL_DEPTH_8U, 1);

							cvResize(&src_img, p_dst_img, CV_INTER_AREA);
							
							QImage scaled_img((uchar*)p_dst_img->imageData, checkList.nRectWidth, checkList.nRectHeight, QImage::Format_Indexed8);
							scaled_img.setColorCount(256);
							scaled_img.setColorTable(pImgObjVec->at(0)->getColorTable());

							scaled_img.save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
						}
                    }
                    else
                    {
                        // Cut useless A-lines
                        if (!checkList.bRectResize)
                            pImgObjVec->at(0)->qindeximg.copy(0, 0, original_nAlines, pImgObjVec->at(0)->getHeight()).
                                    save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                        else
						{
							cv::Mat src(pImgObjVec->at(0)->getHeight(), pImgObjVec->at(0)->getWidth(), CV_8UC1, pImgObjVec->at(0)->qindeximg.bits());
							cv::Mat src_crop = src(cv::Rect(0, 0, original_nAlines, pImgObjVec->at(0)->getHeight()));
							IplImage src_img(src_crop);
							IplImage *p_dst_img = cvCreateImage(cv::Size(checkList.nRectWidth, checkList.nRectHeight), IPL_DEPTH_8U, 1);

							cvResize(&src_img, p_dst_img, CV_INTER_AREA);

							QImage scaled_img((uchar*)p_dst_img->imageData, checkList.nRectWidth, checkList.nRectHeight, QImage::Format_Indexed8);
							scaled_img.setColorCount(256);
							scaled_img.setColorTable(pImgObjVec->at(0)->getColorTable());

							scaled_img.save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
						}
                    }

#ifdef OCT_NIRF
					// NIRF ring here
					if (checkList.bNirfRingOnly)
					{
#ifndef TWO_CHANNEL_NIRF
						if (original_nAlines == pImgObjVec->at(0)->getWidth())
						{
							if (!checkList.bRectResize)
								pImgObjVec->at(1)->qindeximg.save(rectNirfPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
							else
							{
								cv::Mat src(pImgObjVec->at(1)->getHeight(), pImgObjVec->at(1)->getWidth(), CV_8UC1, pImgObjVec->at(1)->qindeximg.bits());
								IplImage src_img(src);
								IplImage *p_dst_img = cvCreateImage(cv::Size(checkList.nRectWidth, checkList.nRectHeight), IPL_DEPTH_8U, 1);

								cvResize(&src_img, p_dst_img, CV_INTER_AREA);

								QImage scaled_img((uchar*)p_dst_img->imageData, checkList.nRectWidth, checkList.nRectHeight, QImage::Format_Indexed8);
								scaled_img.setColorCount(256);
								scaled_img.setColorTable(pImgObjVec->at(1)->getColorTable());

								scaled_img.save(rectNirfPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
							}
						}
						else
						{
							// Cut useless A-lines
							if (!checkList.bRectResize)
								pImgObjVec->at(1)->qindeximg.copy(0, 0, original_nAlines, pImgObjVec->at(0)->getHeight()).
								save(rectNirfPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
							else
							{
								cv::Mat src(pImgObjVec->at(1)->getHeight(), pImgObjVec->at(1)->getWidth(), CV_8UC1, pImgObjVec->at(1)->qindeximg.bits());
								cv::Mat src_crop = src(cv::Rect(0, 0, original_nAlines, pImgObjVec->at(1)->getHeight()));
								IplImage src_img(src_crop);
								IplImage *p_dst_img = cvCreateImage(cv::Size(checkList.nRectWidth, checkList.nRectHeight), IPL_DEPTH_8U, 1);

								cvResize(&src_img, p_dst_img, CV_INTER_AREA);

								QImage scaled_img((uchar*)p_dst_img->imageData, checkList.nRectWidth, checkList.nRectHeight, QImage::Format_Indexed8);
								scaled_img.setColorCount(256);
								scaled_img.setColorTable(pImgObjVec->at(1)->getColorTable());

								scaled_img.save(rectNirfPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
							}
						}
#else
						for (int i = 1; i <= 2; i++)
						{
							if (original_nAlines == pImgObjVec->at(0)->getWidth())
							{
								if (!checkList.bRectResize)
									pImgObjVec->at(i)->qindeximg.save(rectNirfPath[i - 1] + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
								else
								{
									cv::Mat src(pImgObjVec->at(i)->getHeight(), pImgObjVec->at(i)->getWidth(), CV_8UC1, pImgObjVec->at(i)->qindeximg.bits());
									IplImage src_img(src);
									IplImage *p_dst_img = cvCreateImage(cv::Size(checkList.nRectWidth, checkList.nRectHeight), IPL_DEPTH_8U, 1);

									cvResize(&src_img, p_dst_img, CV_INTER_AREA);

									QImage scaled_img((uchar*)p_dst_img->imageData, checkList.nRectWidth, checkList.nRectHeight, QImage::Format_Indexed8);
									scaled_img.setColorCount(256);
									scaled_img.setColorTable(pImgObjVec->at(i)->getColorTable());

									scaled_img.save(rectNirfPath[i - 1] + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
								}
							}
							else
							{
								// Cut useless A-lines
								if (!checkList.bRectResize)
									pImgObjVec->at(i)->qindeximg.copy(0, 0, original_nAlines, pImgObjVec->at(0)->getHeight()).
									save(rectNirfPath[i - 1] + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
								else
								{
									cv::Mat src(pImgObjVec->at(i)->getHeight(), pImgObjVec->at(i)->getWidth(), CV_8UC1, pImgObjVec->at(i)->qindeximg.bits());
									cv::Mat src_crop = src(cv::Rect(0, 0, original_nAlines, pImgObjVec->at(i)->getHeight()));
									IplImage src_img(src_crop);
									IplImage *p_dst_img = cvCreateImage(cv::Size(checkList.nRectWidth, checkList.nRectHeight), IPL_DEPTH_8U, 1);

									cvResize(&src_img, p_dst_img, CV_INTER_AREA);

									QImage scaled_img((uchar*)p_dst_img->imageData, checkList.nRectWidth, checkList.nRectHeight, QImage::Format_Indexed8);
									scaled_img.setColorCount(256);
									scaled_img.setColorTable(pImgObjVec->at(i)->getColorTable());

									scaled_img.save(rectNirfPath[i - 1] + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
								}
							}
						}
#endif
					}
#endif
                }
            }
		
			frameCount++;

			// Push the buffers to sync Queues
			m_syncQueueCircularizing.push(pImgObjVec);
		}

#ifdef OCT_NIRF
		if (checkList.bNirfRingOnly)
			if (checkList.bRect)
				if (m_pResultTab->getNirfDistCompDlg())
					if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
#ifndef TWO_CHANNEL_NIRF
						m_pResultTab->getNirfDistCompDlg()->setCompInfo(rectNirfPath + "dist_comp_info.log");
#else
						for (int i = 0; i < 2; i++)
							m_pResultTab->getNirfDistCompDlg()->setCompInfo(rectNirfPath[i] + "dist_comp_info.log");
#endif
#endif

#ifdef OCT_FLIM
	}
	else
	{
		QString rectPath[3];
		if (!checkList.bMulti)
		{
			for (int i = 0; i < 3; i++)
			{
				if (checkList.bCh[i])
				{
                    rectPath[i] = m_pResultTab->m_path + QString("/rect_image_dB[%1 %2 g%3]_ch%4_i[%5 %6]_t[%7 %8]/")
                        .arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2)
						.arg(i + 1).arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1)
						.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1);
					if (checkList.bRect) QDir().mkdir(rectPath[i]);
				}
			}
		}
		else
		{
            rectPath[0] = m_pResultTab->m_path + QString("/rect_merged_dB[%1 %2 g%3]_ch%4%5%6_i[%7 %8]_t[%9 %10]/")
                .arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2)
				.arg(checkList.bCh[0] ? "1" : "").arg(checkList.bCh[1] ? "2" : "").arg(checkList.bCh[2] ? "3" : "")
				.arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1)
				.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1);
			if (checkList.bRect) QDir().mkdir(rectPath[0]);
		}

		int frameCount = 0;
		while (frameCount < nTotalFrame)
		{
			// Get the buffer from the previous sync Queue
			ImgObjVector *pImgObjVec = m_syncQueueRectWriting.pop();
			
            // Range test
            if (((frameCount + 1) >= start) && ((frameCount + 1) <= end))
            {
                if (!checkList.bMulti)
                {
                    for (int i = 0; i < 3; i++)
                    {
                        if (checkList.bCh[i])
                        {
                            // Paste FLIM color ring to RGB rect image
							for (int j = 0; j < m_pConfig->ringThickness; j++)
							{
								memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - 2 * m_pConfig->ringThickness + j),
									pImgObjVec->at(1 + 2 * i)->qrgbimg.bits(), 3 * pImgObjVec->at(0)->arr.size(0)); // Intensity
								memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - 1 * m_pConfig->ringThickness + j),
									pImgObjVec->at(2 + 2 * i)->qrgbimg.bits(), 3 * pImgObjVec->at(0)->arr.size(0)); // Lifetime
							}

                            if (checkList.bRect)
                            {
                                // Write rect images
                                if (!checkList.bRectResize)
                                    pImgObjVec->at(0)->qrgbimg.save(rectPath[i] + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                                else
                                    pImgObjVec->at(0)->qrgbimg.scaled(checkList.nRectWidth, checkList.nRectHeight, Qt::IgnoreAspectRatio, m_defaultTransformation).
                                    save(rectPath[i] + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                            }
                        }
                    }
                }
                else
                {
                    int nCh = (checkList.bCh[0] ? 1 : 0) + (checkList.bCh[1] ? 1 : 0) + (checkList.bCh[2] ? 1 : 0);
                    int n = 0;

                    for (int i = 0; i < 3; i++)
                    {
                        if (checkList.bCh[i])
                        {
#if (LIFETIME_COLORTABLE == 7)
                            // HSV channel setting
                            ImageObject imgObjTemp(pImgObjVec->at(1 + 2 * i)->qrgbimg.width(), pImgObjVec->at(1 + 2 * i)->qrgbimg.height(), pImgObjVec->at(1 + 2 * i)->getColorTable());
                            ImageObject imgObjHsv( pImgObjVec->at(1 + 2 * i)->qrgbimg.width(), pImgObjVec->at(1 + 2 * i)->qrgbimg.height(), pImgObjVec->at(1 + 2 * i)->getColorTable());

                            IppiSize roi_flimproj = { pImgObjVec->at(1 + 2 * i)->qrgbimg.width(), pImgObjVec->at(1 + 2 * i)->qrgbimg.height() };

                            memset(imgObjTemp.qrgbimg.bits(), 255, imgObjTemp.qrgbimg.byteCount()); // Saturation is set to be 255.
                            imgObjTemp.setRgbChannelData(pImgObjVec->at(2 + 2 * i)->qindeximg.bits(), 0); // Hue
                            uint8_t *pIntensity = new uint8_t[pImgObjVec->at(1 + 2 * i)->qindeximg.byteCount()];
                            memcpy(pIntensity, pImgObjVec->at(1 + 2 * i)->qindeximg.bits(), pImgObjVec->at(1 + 2 * i)->qindeximg.byteCount());
                            ippsMulC_8u_ISfs(1.0, pIntensity, pImgObjVec->at(1 + 2 * i)->qindeximg.byteCount(), 0);
                            imgObjTemp.setRgbChannelData(pIntensity, 2); // Value
                            delete[] pIntensity;

                            ippiHSVToRGB_8u_C3R(imgObjTemp.qrgbimg.bits(), 3 * roi_flimproj.width, imgObjHsv.qrgbimg.bits(), 3 * roi_flimproj.width, roi_flimproj);
                            *pImgObjVec->at(1 + 2 * i) = std::move(imgObjHsv);
#else
                            // Non HSV intensity-weight map
                            ColorTable temp_ctable;
                            ImageObject imgObjTemp(pImgObjVec->at(1 + 2 * i)->qrgbimg.width(), pImgObjVec->at(1 + 2 * i)->qrgbimg.height(), temp_ctable.m_colorTableVector.at(ColorTable::gray));
                            ImageObject imgObjHsv (pImgObjVec->at(1 + 2 * i)->qrgbimg.width(), pImgObjVec->at(1 + 2 * i)->qrgbimg.height(), pImgObjVec->at(1 + 2 * i)->getColorTable());

                            memcpy(imgObjTemp.qindeximg.bits(), pImgObjVec->at(1 + 2 * i)->qindeximg.bits(), imgObjTemp.qindeximg.byteCount());
                            imgObjTemp.convertRgb();

                            ippsMul_8u_Sfs(pImgObjVec->at(2 + 2 * i)->qrgbimg.bits(), imgObjTemp.qrgbimg.bits(), imgObjHsv.qrgbimg.bits(), imgObjTemp.qrgbimg.byteCount(), 8);
                            pImgObjVec->at(1 + 2 * i)->qrgbimg = std::move(imgObjHsv.qrgbimg);
#endif
                            // Paste FLIM color ring to RGB rect image
							for (int j = 0; j < m_pConfig->ringThickness; j++)
							{
								memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - (nCh - n) * m_pConfig->ringThickness + j),
									pImgObjVec->at(1 + 2 * i)->qrgbimg.bits(), 3 * pImgObjVec->at(0)->arr.size(0));
							}
							n++;
                        }

                        if (checkList.bRect)
                        {
                            // Write rect images
                            if (!checkList.bRectResize)
                                pImgObjVec->at(0)->qrgbimg.save(rectPath[0] + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                            else
                                pImgObjVec->at(0)->qrgbimg.scaled(checkList.nRectWidth, checkList.nRectHeight, Qt::IgnoreAspectRatio, m_defaultTransformation).
                                save(rectPath[0] + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                        }
                    }
                }
            }

			frameCount++;

			// Push the buffers to sync Queues
			m_syncQueueCircularizing.push(pImgObjVec);
		}
	}
#elif defined(OCT_NIRF)
    }
    else
    {
        QString nirfName;
#ifndef TWO_CHANNEL_NIRF
        if (m_pResultTab->getNirfDistCompDlg())
        {
            if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
            {
                if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
                    nirfName = QString("/rect_image_dB[%1 %2 g%3]_tbr_nirf_i[%4 %5]/");
                else
                    nirfName = QString("/rect_image_dB[%1 %2 g%3]_comp_nirf_i[%4 %5]/");
            }
            else
                nirfName = QString("/rect_image_dB[%1 %2 g%3]_bg_sub_nirf_i[%4 %5]/");
        }
        else
        {
            nirfName = QString("/rect_image_dB[%1 %2 g%3]_raw_nirf_i[%4 %5]/");
        }
        QString rectPath = m_pResultTab->m_path + nirfName.arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2)
                .arg(m_pConfig->nirfRange.min, 2, 'f', 2).arg(m_pConfig->nirfRange.max, 2, 'f', 2);
#else
		if (m_pResultTab->getNirfDistCompDlg())
		{
			if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
			{
				if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
					nirfName = QString("/rect_image_dB[%1 %2 g%3]_tbr_nirf_i1[%4 %5]_i2[%6 %7]/");
				else
					nirfName = QString("/rect_image_dB[%1 %2 g%3]_comp_nirf_i1[%4 %5]_i2[%6 %7]/");
			}
			else
				nirfName = QString("/rect_image_dB[%1 %2 g%3]_bg_sub_nirf_i1[%4 %5]_i2[%6 %7]/");
		}
		else
		{
			nirfName = QString("/rect_image_dB[%1 %2 g%3]_raw_nirf_i1[%4 %5]_i2[%6 %7]/");
		}
		QString rectPath = m_pResultTab->m_path + nirfName.arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2)
			.arg(m_pConfig->nirfRange[0].min, 2, 'f', 2).arg(m_pConfig->nirfRange[0].max, 2, 'f', 2)
			.arg(m_pConfig->nirfRange[1].min, 2, 'f', 2).arg(m_pConfig->nirfRange[1].max, 2, 'f', 2);
#endif
        if (checkList.bRect) QDir().mkdir(rectPath);

        int frameCount = 0;
        while (frameCount < nTotalFrame)
        {
            // Get the buffer from the previous sync Queue
            ImgObjVector *pImgObjVec = m_syncQueueRectWriting.pop();

            // Range test
            if (((frameCount + 1) >= start) && ((frameCount + 1) <= end))
            {
                // Paste FLIM color ring to RGB rect image
#ifndef TWO_CHANNEL_NIRF
                memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - 1 * m_pConfig->ringThickness),
                    pImgObjVec->at(1)->qrgbimg.bits(), pImgObjVec->at(1)->qrgbimg.byteCount()); // Nirf
#else
				memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - 2 * m_pConfig->ringThickness),
					pImgObjVec->at(1)->qrgbimg.bits(), pImgObjVec->at(1)->qrgbimg.byteCount()); // Nirf
				memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - 1 * m_pConfig->ringThickness),
					pImgObjVec->at(2)->qrgbimg.bits(), pImgObjVec->at(2)->qrgbimg.byteCount()); // Nirf
#endif

                if (checkList.bRect)
                {
                    int original_nAlines = m_pResultTab->getRectImageView()->getRender()->m_pImage->width();

                    if (original_nAlines == pImgObjVec->at(0)->getWidth())
                    {
                        // Write rect images
                        if (!checkList.bRectResize)
                            pImgObjVec->at(0)->qrgbimg.save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                        else
                            pImgObjVec->at(0)->qrgbimg.scaled(checkList.nRectWidth, checkList.nRectHeight, Qt::IgnoreAspectRatio, m_defaultTransformation).
                                    save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                    }
                    else
                    {
                        // Write rect images
                        if (!checkList.bRectResize)
                            pImgObjVec->at(0)->qrgbimg.copy(0, 0, original_nAlines, pImgObjVec->at(0)->getHeight()).
                                    save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                        else
                            pImgObjVec->at(0)->qrgbimg.copy(0, 0, original_nAlines, pImgObjVec->at(0)->getHeight()).
									scaled(checkList.nRectWidth, checkList.nRectHeight, Qt::IgnoreAspectRatio, m_defaultTransformation).
                                    save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                    }
                }
            }

            frameCount++;

            // Push the buffers to sync Queues
            m_syncQueueCircularizing.push(pImgObjVec);
        }        

		if (checkList.bRect)
			if (m_pResultTab->getNirfDistCompDlg())
				if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
					m_pResultTab->getNirfDistCompDlg()->setCompInfo(rectPath + "dist_comp_info.log");
    }
#endif
}

void SaveResultDlg::circularizing(CrossSectionCheckList checkList) // with longitudinal making
{
	int nTotalFrame = (int)m_pResultTab->m_vectorOctImage.size();
	int nTotalFrame4 = ((nTotalFrame + 3) >> 2) << 2;
	ColorTable temp_ctable;
	
	// Image objects for longitudinal image
#ifdef OCT_FLIM
	ImgObjVector *pImgObjVecLongi[3];
	if (checkList.bLongi)
	{
		for (int i = 0; i < 3; i++)
		{
			pImgObjVecLongi [i] = new ImgObjVector;
			for (int j = 0; j < (int)(m_pResultTab->m_vectorOctImage.at(0).size(1) / 2); j++)
			{
				ImageObject *pLongiImgObj = new ImageObject(nTotalFrame4, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()), m_pConfig->octDbGamma);
				pImgObjVecLongi[i]->push_back(pLongiImgObj);
			}
		}
	}
#else
	ImgObjVector *pImgObjVecLongi = new ImgObjVector;
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	ImgObjVector *pImgObjVecLongiRing = nullptr;
#else
	ImgObjVector *pImgObjVecLongiRing[2] = { nullptr, nullptr };
#endif
#endif
	if (checkList.bLongi)
	{
		for (int i = 0; i < (int)(m_pResultTab->m_vectorOctImage.at(0).size(1) / 2); i++)
		{
			ImageObject *pLongiImgObj = new ImageObject(nTotalFrame4, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()), m_pConfig->octDbGamma);
			pImgObjVecLongi->push_back(pLongiImgObj);
		}

#ifdef OCT_NIRF
		if (checkList.bNirfRingOnly)
		{
#ifndef TWO_CHANNEL_NIRF
			pImgObjVecLongiRing = new ImgObjVector;
			for (int i = 0; i < (int)(m_pResultTab->m_vectorOctImage.at(0).size(1) / 2); i++)
			{
				ImageObject *pLongiRingImgObj = new ImageObject(nTotalFrame4, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
				pImgObjVecLongiRing->push_back(pLongiRingImgObj);
			}
#else
			pImgObjVecLongiRing[0] = new ImgObjVector;
			pImgObjVecLongiRing[1] = new ImgObjVector;
			for (int i = 0; i < (int)(m_pResultTab->m_vectorOctImage.at(0).size(1) / 2); i++)
			{
				ImageObject *pLongiRingImgObj1 = new ImageObject(nTotalFrame4, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
				pImgObjVecLongiRing[0]->push_back(pLongiRingImgObj1);

				ImageObject *pLongiRingImgObj2 = new ImageObject(nTotalFrame4, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE2));
				pImgObjVecLongiRing[1]->push_back(pLongiRingImgObj2);
			}
#endif
		}
#endif

		m_pResultTab->getProgressBar()->setRange(0, (int)m_pResultTab->m_vectorOctImage.size() * 2 + m_pResultTab->m_vectorOctImage.at(0).size(1) / 2 - 1);
	}
#endif

	int frameCount = 0;
	while (frameCount < nTotalFrame)
	{
		// Get the buffer from the previous sync Queue
		ImgObjVector *pImgObjVec = m_syncQueueCircularizing.pop();
		ImgObjVector *pImgObjVecCirc = new ImgObjVector;

#ifdef OCT_FLIM
		if (!checkList.bCh[0] && !checkList.bCh[1] && !checkList.bCh[2])
		{
#elif defined (OCT_NIRF)
		if (!checkList.bNirf || checkList.bNirfRingOnly)
		{
#endif
			// ImageObject for circ writing
			ImageObject *pCircImgObj = new ImageObject(2 * m_pResultTab->getConfigTemp()->circRadius, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()), m_pConfig->octDbGamma);
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
			ImageObject *pCircNirfObj = nullptr;
#else
			ImageObject *pCircNirfObj1 = nullptr;
			ImageObject *pCircNirfObj2 = nullptr;
#endif
			if (checkList.bNirfRingOnly)
			{
#ifndef TWO_CHANNEL_NIRF
				pCircNirfObj = new ImageObject(2 * m_pResultTab->getConfigTemp()->circRadius, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
#else
				pCircNirfObj1 = new ImageObject(2 * m_pResultTab->getConfigTemp()->circRadius, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
				pCircNirfObj2 = new ImageObject(2 * m_pResultTab->getConfigTemp()->circRadius, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE2));
#endif
			}
#endif
			// Buffer & center
			np::Uint8Array2 rect_temp(pImgObjVec->at(0)->qindeximg.bits(), pImgObjVec->at(0)->arr.size(0), pImgObjVec->at(0)->arr.size(1));
			int center = (!m_pResultTab->getPolishedSurfaceFindingStatus()) ? m_pConfig->circCenter :
				(m_pResultTab->getConfigTemp()->n2ScansFFT / 2 - m_pResultTab->getConfigTemp()->nScans / 4) + m_pResultTab->m_polishedSurface(frameCount) - m_pConfig->ballRadius;
			
#ifdef OCT_NIRF
			if (checkList.bNirfRingOnly)
#ifndef TWO_CHANNEL_NIRF
				memset(&rect_temp(0, center + m_pResultTab->getConfigTemp()->circRadius - 1 * m_pConfig->ringThickness), 0, sizeof(uint8_t) * 1 * m_pConfig->ringThickness * pImgObjVec->at(0)->arr.size(0));
#else
				memset(&rect_temp(0, center + m_pResultTab->getConfigTemp()->circRadius - 2 * m_pConfig->ringThickness), 0, sizeof(uint8_t) * 2 * m_pConfig->ringThickness * pImgObjVec->at(0)->arr.size(0));
#endif
#endif

			// Circularize
			if (checkList.bCirc)
				(*m_pResultTab->m_pCirc)(rect_temp, pCircImgObj->qindeximg.bits(), "vertical", center);

#ifdef OCT_NIRF
			// NIRF ring only here
#ifndef TWO_CHANNEL_NIRF
			np::Uint8Array2 rect_nirf_temp;
#else
			np::Uint8Array2 rect_nirf_temp1, rect_nirf_temp2;
#endif
			if (checkList.bNirfRingOnly)
			{
#ifndef TWO_CHANNEL_NIRF
				rect_nirf_temp = np::Uint8Array2(pImgObjVec->at(1)->qindeximg.bits(), pImgObjVec->at(1)->arr.size(0), pImgObjVec->at(1)->arr.size(1));
				if (checkList.bCirc)
					(*m_pResultTab->m_pCirc)(rect_nirf_temp, pCircNirfObj->qindeximg.bits(), "vertical", pImgObjVec->at(1)->arr.size(1) - m_pResultTab->getConfigTemp()->circRadius);
#else
				rect_nirf_temp1 = np::Uint8Array2(pImgObjVec->at(1)->qindeximg.bits(), pImgObjVec->at(1)->arr.size(0), pImgObjVec->at(1)->arr.size(1));
				rect_nirf_temp2 = np::Uint8Array2(pImgObjVec->at(2)->qindeximg.bits(), pImgObjVec->at(2)->arr.size(0), pImgObjVec->at(2)->arr.size(1));
				if (checkList.bCirc)
				{
					(*m_pResultTab->m_pCirc)(rect_nirf_temp1, pCircNirfObj1->qindeximg.bits(), "vertical", pImgObjVec->at(1)->arr.size(1) - m_pResultTab->getConfigTemp()->circRadius);
					(*m_pResultTab->m_pCirc)(rect_nirf_temp2, pCircNirfObj2->qindeximg.bits(), "vertical", pImgObjVec->at(2)->arr.size(1) - m_pResultTab->getConfigTemp()->circRadius);
				}
#endif
			}
#endif

			// Longitudinal
			if (checkList.bLongi)
			{
				IppiSize roi_longi = { 1, m_pResultTab->getConfigTemp()->circRadius };
				int n2Alines = m_pResultTab->m_vectorOctImage.at(0).size(1) / 2;

				tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)n2Alines),
					[&](const tbb::blocked_range<size_t>& r) {
					for (size_t i = r.begin(); i != r.end(); ++i)
					{
#ifdef OCT_FLIM
						ippiCopy_8u_C1R(&rect_temp((int)i, center), sizeof(uint8_t) * 2 * n2Alines,
							pImgObjVecLongi[0]->at((int)i)->qindeximg.bits() + frameCount, sizeof(uint8_t) * nTotalFrame4, roi_longi);
						ippiCopy_8u_C1R(&rect_temp((int)i + n2Alines, center), sizeof(uint8_t) * 2 * n2Alines,
							pImgObjVecLongi[0]->at((int)i)->qindeximg.bits() + m_pResultTab->getConfigTemp()->circRadius * nTotalFrame4 + frameCount, sizeof(uint8_t) * nTotalFrame4, roi_longi);
#else
						ippiCopy_8u_C1R(&rect_temp((int)i, center), sizeof(uint8_t) * 2 * n2Alines,
							pImgObjVecLongi->at((int)i)->qindeximg.bits() + frameCount, sizeof(uint8_t) * nTotalFrame4, roi_longi);
						ippiCopy_8u_C1R(&rect_temp((int)i + n2Alines, center), sizeof(uint8_t) * 2 * n2Alines,
							pImgObjVecLongi->at((int)i)->qindeximg.bits() + m_pResultTab->getConfigTemp()->circRadius * nTotalFrame4 + frameCount, sizeof(uint8_t) * nTotalFrame4, roi_longi);

#ifdef OCT_NIRF
						if (checkList.bNirfRingOnly)
						{
#ifndef TWO_CHANNEL_NIRF
							ippiCopy_8u_C1R(&rect_nirf_temp((int)i, pImgObjVec->at(1)->arr.size(1) - m_pResultTab->getConfigTemp()->circRadius), sizeof(uint8_t) * 2 * n2Alines,
								pImgObjVecLongiRing->at((int)i)->qindeximg.bits() + frameCount, sizeof(uint8_t) * nTotalFrame4, roi_longi);
							ippiCopy_8u_C1R(&rect_nirf_temp((int)i + n2Alines, pImgObjVec->at(1)->arr.size(1) - m_pResultTab->getConfigTemp()->circRadius), sizeof(uint8_t) * 2 * n2Alines,
								pImgObjVecLongiRing->at((int)i)->qindeximg.bits() + m_pResultTab->getConfigTemp()->circRadius * nTotalFrame4 + frameCount, sizeof(uint8_t) * nTotalFrame4, roi_longi);
#else
							ippiCopy_8u_C1R(&rect_nirf_temp1((int)i, pImgObjVec->at(1)->arr.size(1) - m_pResultTab->getConfigTemp()->circRadius), sizeof(uint8_t) * 2 * n2Alines,
								pImgObjVecLongiRing[0]->at((int)i)->qindeximg.bits() + frameCount, sizeof(uint8_t) * nTotalFrame4, roi_longi);
							ippiCopy_8u_C1R(&rect_nirf_temp1((int)i + n2Alines, pImgObjVec->at(1)->arr.size(1) - m_pResultTab->getConfigTemp()->circRadius), sizeof(uint8_t) * 2 * n2Alines,
								pImgObjVecLongiRing[0]->at((int)i)->qindeximg.bits() + m_pResultTab->getConfigTemp()->circRadius * nTotalFrame4 + frameCount, sizeof(uint8_t) * nTotalFrame4, roi_longi);

							ippiCopy_8u_C1R(&rect_nirf_temp2((int)i, pImgObjVec->at(2)->arr.size(1) - m_pResultTab->getConfigTemp()->circRadius), sizeof(uint8_t) * 2 * n2Alines,
								pImgObjVecLongiRing[1]->at((int)i)->qindeximg.bits() + frameCount, sizeof(uint8_t) * nTotalFrame4, roi_longi);
							ippiCopy_8u_C1R(&rect_nirf_temp2((int)i + n2Alines, pImgObjVec->at(2)->arr.size(1) - m_pResultTab->getConfigTemp()->circRadius), sizeof(uint8_t) * 2 * n2Alines,
								pImgObjVecLongiRing[1]->at((int)i)->qindeximg.bits() + m_pResultTab->getConfigTemp()->circRadius * nTotalFrame4 + frameCount, sizeof(uint8_t) * nTotalFrame4, roi_longi);
#endif
						}
#endif
#endif
					}
				});
			}

			// Vector pushing back
			pImgObjVecCirc->push_back(pCircImgObj);
#ifdef OCT_NIRF
			if (checkList.bNirfRingOnly)
			{
#ifndef TWO_CHANNEL_NIRF
				pImgObjVecCirc->push_back(pCircNirfObj);
#else
				pImgObjVecCirc->push_back(pCircNirfObj1);
				pImgObjVecCirc->push_back(pCircNirfObj2);
#endif
			}
#endif
#ifdef OCT_FLIM
		}
		else
		{
			ImageObject *pCircImgObj[3];
			if (!checkList.bMulti)
			{
				for (int i = 0; i < 3; i++)
				{
					// ImageObject for circ writing
					pCircImgObj[i] = new ImageObject(2 * m_pResultTab->getConfigTemp()->circRadius, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()), m_pConfig->octDbGamma);

					// Buffer & center
					int center = (!m_pResultTab->getPolishedSurfaceFindingStatus()) ? m_pConfig->circCenter :
						(m_pResultTab->getConfigTemp()->n2ScansFFT / 2 - m_pResultTab->getConfigTemp()->nScans / 4) + m_pResultTab->m_polishedSurface(frameCount) - m_pConfig->ballRadius;
					np::Uint8Array2 rect_temp(pImgObjVec->at(0)->qrgbimg.bits(), 3 * pImgObjVec->at(0)->arr.size(0), pImgObjVec->at(0)->arr.size(1));

					if (checkList.bCh[i] && (checkList.bCirc || checkList.bLongi))
					{
						// Paste FLIM color ring to RGB rect image
						for (int j = 0; j < m_pConfig->ringThickness; j++)
						{
							memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (center + m_pResultTab->getConfigTemp()->circRadius - 2 * m_pConfig->ringThickness + j),
								pImgObjVec->at(1 + 2 * i)->qrgbimg.bits(), 3 * pImgObjVec->at(0)->arr.size(0)); // Intensity
							memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (center + m_pResultTab->getConfigTemp()->circRadius - 1 * m_pConfig->ringThickness + j),
								pImgObjVec->at(2 + 2 * i)->qrgbimg.bits(), 3 * pImgObjVec->at(0)->arr.size(0)); // Lifetime
						}
						// Circularize
						(*m_pResultTab->m_pCirc)(rect_temp, pCircImgObj[i]->qrgbimg.bits(), "vertical", "rgb", center);
					}

					// Longitudinal
					if (checkList.bLongi)
					{
						IppiSize roi_longi = { 3, m_pResultTab->getConfigTemp()->circRadius };
						int n2Alines = m_pResultTab->m_vectorOctImage.at(0).size(1) / 2;

						tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)n2Alines),
							[&](const tbb::blocked_range<size_t>& r) {
							for (size_t j = r.begin(); j != r.end(); ++j)
							{
								ippiCopy_8u_C1R(&rect_temp(3 * (int)j, center), sizeof(uint8_t) * 2 * 3 * n2Alines,
									pImgObjVecLongi[i]->at((int)j)->qrgbimg.bits() + 3 * frameCount, sizeof(uint8_t) * 3 * nTotalFrame4, roi_longi);
								ippiCopy_8u_C1R(&rect_temp(3 * ((int)j + n2Alines), center), sizeof(uint8_t) * 2 * 3 * n2Alines,
									pImgObjVecLongi[i]->at((int)j)->qrgbimg.bits() + m_pResultTab->getConfigTemp()->circRadius * 3 * nTotalFrame4 + 3 * frameCount, sizeof(uint8_t) * 3 * nTotalFrame4, roi_longi);
							}
						});
					}

					// Vector pushing back
					pImgObjVecCirc->push_back(pCircImgObj[i]);
				}
			}
			else
			{
				int nCh = (checkList.bCh[0] ? 1 : 0) + (checkList.bCh[1] ? 1 : 0) + (checkList.bCh[2] ? 1 : 0);
				int n = 0;

				// ImageObject for circ writing
				pCircImgObj[0] = new ImageObject(2 * m_pResultTab->getConfigTemp()->circRadius, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()), m_pConfig->octDbGamma);

				// Buffer & center
				int center = (!m_pResultTab->getPolishedSurfaceFindingStatus()) ? m_pConfig->circCenter :
					(m_pResultTab->getConfigTemp()->n2ScansFFT / 2 - m_pResultTab->getConfigTemp()->nScans / 4) + m_pResultTab->m_polishedSurface(frameCount) - m_pConfig->ballRadius;
				np::Uint8Array2 rect_temp(pImgObjVec->at(0)->qrgbimg.bits(), 3 * pImgObjVec->at(0)->arr.size(0), pImgObjVec->at(0)->arr.size(1));

				// Paste FLIM color ring to RGB rect image
				for (int i = 0; i < 3; i++)
				{
					if (checkList.bCh[i] && (checkList.bCirc || checkList.bLongi))
					{
						for (int j = 0; j < m_pConfig->ringThickness; j++)
						{
							memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (center + m_pResultTab->getConfigTemp()->circRadius - (nCh - n) * m_pConfig->ringThickness + j),
								pImgObjVec->at(1 + 2 * i)->qrgbimg.bits(), 3 * pImgObjVec->at(0)->arr.size(0));
						}
						n++;
					}
				}

				// Circularize
				if (checkList.bCirc)
					(*m_pResultTab->m_pCirc)(rect_temp, pCircImgObj[0]->qrgbimg.bits(), "vertical", "rgb", center);

				// Longitudinal
				if (checkList.bLongi)
				{
					IppiSize roi_longi = { 3, m_pResultTab->getConfigTemp()->circRadius };
					int n2Alines = m_pResultTab->m_vectorOctImage.at(0).size(1) / 2;

					tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)n2Alines),
						[&](const tbb::blocked_range<size_t>& r) {
						for (size_t j = r.begin(); j != r.end(); ++j)
						{
							ippiCopy_8u_C1R(&rect_temp(3 * (int)j, center), sizeof(uint8_t) * 2 * 3 * n2Alines,
								pImgObjVecLongi[0]->at((int)j)->qrgbimg.bits() + 3 * frameCount, sizeof(uint8_t) * 3 * nTotalFrame4, roi_longi);
							ippiCopy_8u_C1R(&rect_temp(3 * ((int)j + n2Alines), center), sizeof(uint8_t) * 2 * 3 * n2Alines,
								pImgObjVecLongi[0]->at((int)j)->qrgbimg.bits() + m_pResultTab->getConfigTemp()->circRadius * 3 * nTotalFrame4 + 3 * frameCount, sizeof(uint8_t) * 3 * nTotalFrame4, roi_longi);
						}
					});
				}

				// Vector pushing back
				pImgObjVecCirc->push_back(pCircImgObj[0]);
			}
		}
#elif defined (OCT_NIRF)
		}
		else
		{
			// ImageObject for circ writing
			ImageObject *pCircImgObj = new ImageObject(2 * m_pResultTab->getConfigTemp()->circRadius, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()), m_pConfig->octDbGamma);

			// Buffer & center
			np::Uint8Array2 rect_temp(pImgObjVec->at(0)->qrgbimg.bits(), 3 * pImgObjVec->at(0)->arr.size(0), pImgObjVec->at(0)->arr.size(1));
			int center = (!m_pResultTab->getPolishedSurfaceFindingStatus()) ? m_pConfig->circCenter :
				(m_pResultTab->getConfigTemp()->n2ScansFFT / 2 - m_pResultTab->getConfigTemp()->nScans / 4) + m_pResultTab->m_polishedSurface(frameCount) - m_pConfig->ballRadius;

			if (checkList.bCirc || checkList.bLongi)
			{
				// Paste FLIM color ring to RGB rect image
#ifndef TWO_CHANNEL_NIRF
				memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (center + m_pResultTab->getConfigTemp()->circRadius - 1 * m_pConfig->ringThickness),
					pImgObjVec->at(1)->qrgbimg.bits(), pImgObjVec->at(1)->qrgbimg.byteCount()); // Nirf
#else
				memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (center + m_pResultTab->getConfigTemp()->circRadius - 2 * m_pConfig->ringThickness),
					pImgObjVec->at(1)->qrgbimg.bits(), pImgObjVec->at(1)->qrgbimg.byteCount()); // Nirf
				memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (center + m_pResultTab->getConfigTemp()->circRadius - 1 * m_pConfig->ringThickness),
					pImgObjVec->at(2)->qrgbimg.bits(), pImgObjVec->at(2)->qrgbimg.byteCount()); // Nirf
#endif
			}

			// Circularize
			if (checkList.bCirc)
				(*m_pResultTab->m_pCirc)(rect_temp, pCircImgObj->qrgbimg.bits(), "vertical", "rgb", center);
			
			// Longitudinal
			if (checkList.bLongi)
			{
				IppiSize roi_longi = { 3, m_pResultTab->getConfigTemp()->circRadius };
				int n2Alines = m_pResultTab->m_vectorOctImage.at(0).size(1) / 2;

				tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)n2Alines),
					[&](const tbb::blocked_range<size_t>& r) {
					for (size_t i = r.begin(); i != r.end(); ++i)
					{
						ippiCopy_8u_C1R(&rect_temp(3 * (int)i, center), sizeof(uint8_t) * 2 * 3 * n2Alines,
							pImgObjVecLongi->at((int)i)->qrgbimg.bits() + 3 * frameCount, sizeof(uint8_t) * 3 * nTotalFrame4, roi_longi);
						ippiCopy_8u_C1R(&rect_temp(3 * ((int)i + n2Alines), center), sizeof(uint8_t) * 2 * 3 * n2Alines,
							pImgObjVecLongi->at((int)i)->qrgbimg.bits() + m_pResultTab->getConfigTemp()->circRadius * 3 * nTotalFrame4 + 3 * frameCount, sizeof(uint8_t) * 3 * nTotalFrame4, roi_longi);
					}
				});
			}

			// Vector pushing back
			pImgObjVecCirc->push_back(pCircImgObj);
		}
#endif;
		frameCount++;
		emit savedSingleFrame(m_nSavedFrames++);

		// Push the buffers to sync Queues
		m_syncQueueCircWriting.push(pImgObjVecCirc);

		// Delete ImageObjects
#ifdef OCT_FLIM
		for (int i = 0; i < 7; i++)
			delete pImgObjVec->at(i);
#elif defined (STANDALONE_OCT)
		delete pImgObjVec->at(0);
#ifdef OCT_NIRF
		delete pImgObjVec->at(1);
#ifdef TWO_CHANNEL_NIRF
		delete pImgObjVec->at(2);
#endif
#endif
#endif
		delete pImgObjVec;
	}

	// Write longtiduinal images
	if (checkList.bLongi)
	{
		QString folderName;
		for (int i = 0; i < m_pResultTab->m_path.length(); i++)
			if (m_pResultTab->m_path.at(i) == QChar('/')) folderName = m_pResultTab->m_path.right(m_pResultTab->m_path.length() - i - 1);

		int start = m_pLineEdit_RangeStart->text().toInt();
		int end = m_pLineEdit_RangeEnd->text().toInt();

#ifdef OCT_FLIM
		if (!checkList.bCh[0] && !checkList.bCh[1] && !checkList.bCh[2])
		{
#elif defined (OCT_NIRF)
		if (!checkList.bNirf || checkList.bNirfRingOnly)
		{
#endif
			QString longiPath;

#ifndef TWO_CHANNEL_NIRF
			QString longiNirfPath;
#else
			QString longiNirfPath[2];
#endif

#ifndef OCT_NIRF
			longiPath = m_pResultTab->m_path + QString("/longi_image[%1 %2]_dB[%3 %4 g%5]/").arg(start).arg(end).arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2);
#else
			longiPath = m_pResultTab->m_path + QString("/longi_image[%1 %2]_dB[%3 %4 g%5]%6/").arg(start).arg(end).arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2).arg(checkList.bNirfRingOnly ? "_ring-masked" : "");
			if (checkList.bNirfRingOnly)
			{
#ifndef TWO_CHANNEL_NIRF
				if (m_pResultTab->getNirfDistCompDlg())
				{
					if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
					{
						if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
							longiNirfPath = QString("/longi_tbr_nirf[%1 %2]_i[%3 %4]/");
						else
							longiNirfPath = QString("/longi_comp_nirf[%1 %2]_i[%3 %4]/");
					}
					else
						longiNirfPath = QString("/longi_bg_sub_nirf[%1 %2]_i[%3 %4]/");
				}
				else
				{
					longiNirfPath = QString("/longi_raw_nirf[%1 %2]_i[%3 %4]/");
				}
				longiNirfPath = m_pResultTab->m_path + longiNirfPath.arg(start).arg(end).arg(m_pConfig->nirfRange.min, 2, 'f', 2).arg(m_pConfig->nirfRange.max, 2, 'f', 2);
#else
				for (int i = 0; i < 2; i++)
				{
					if (m_pResultTab->getNirfDistCompDlg())
					{
						if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
						{
							if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
								longiNirfPath[i] = QString("/longi_tbr_nirf[%1 %2]_i%3[%4 %5]/");
							else
								longiNirfPath[i] = QString("/longi_comp_nirf[%1 %2]_i%3[%4 %5]/");
						}
						else
							longiNirfPath[i] = QString("/longi_bg_sub_nirf[%1 %2]_i%3[%4 %5]/");
					}
					else
					{
						longiNirfPath[i] = QString("/longi_raw_nirf[%1 %2]_i%3[%4 %5]/");
					}
					longiNirfPath[i] = m_pResultTab->m_path + longiNirfPath[i].arg(start).arg(end).arg(i + 1).arg(m_pConfig->nirfRange[i].min, 2, 'f', 2).arg(m_pConfig->nirfRange[i].max, 2, 'f', 2);
				}
#endif
			}
#endif
			
			QDir().mkdir(longiPath);
#ifdef OCT_NIRF
			if (checkList.bNirfRingOnly)
			{
#ifndef TWO_CHANNEL_NIRF
				QDir().mkdir(longiNirfPath);
#else
				QDir().mkdir(longiNirfPath[0]);
				QDir().mkdir(longiNirfPath[1]);
#endif
			}
#endif

			int alineCount = 0;
			while (alineCount < m_pResultTab->m_vectorOctImage.at(0).size(1) / 2)
			{
#ifdef OCT_FLIM
				// Write longi images
				ippiMirror_8u_C1IR(pImgObjVecLongi[0]->at(alineCount)->qindeximg.bits(), sizeof(uint8_t) * nTotalFrame4, { nTotalFrame4, m_pResultTab->getConfigTemp()->circRadius }, ippAxsHorizontal);
				if (!checkList.bLongiResize)
					pImgObjVecLongi[0]->at(alineCount)->qindeximg.copy(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius).
						save(longiPath + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");
				else
					pImgObjVecLongi[0]->at(alineCount)->qindeximg.copy(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius).
						scaled(checkList.nLongiWidth, checkList.nLongiHeight, Qt::IgnoreAspectRatio, m_defaultTransformation).
						save(longiPath + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");

				// Delete ImageObjects
				delete pImgObjVecLongi[0]->at(alineCount);
				delete pImgObjVecLongi[1]->at(alineCount);
				delete pImgObjVecLongi[2]->at(alineCount++);
#else
				// Write longi images
				ippiMirror_8u_C1IR(pImgObjVecLongi->at(alineCount)->qindeximg.bits(), sizeof(uint8_t) * nTotalFrame4, { nTotalFrame4, m_pResultTab->getConfigTemp()->circRadius }, ippAxsHorizontal);
				if (!checkList.bLongiResize)
					pImgObjVecLongi->at(alineCount)->qindeximg.copy(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius).
						save(longiPath + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");
				else
				{
					cv::Mat src(pImgObjVecLongi->at(alineCount)->getHeight(), pImgObjVecLongi->at(alineCount)->getWidth(), CV_8UC1, pImgObjVecLongi->at(alineCount)->qindeximg.bits());
					cv::Mat src_crop = src(cv::Rect(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius));
					IplImage src_img(src_crop);
					IplImage *p_dst_img = cvCreateImage(cv::Size(checkList.nLongiWidth, checkList.nLongiHeight), IPL_DEPTH_8U, 1);

					cvResize(&src_img, p_dst_img, CV_INTER_CUBIC);

					QImage scaled_img((uchar*)p_dst_img->imageData, checkList.nLongiWidth, checkList.nLongiHeight, QImage::Format_Indexed8);
					scaled_img.setColorCount(256);
					scaled_img.setColorTable(pImgObjVecLongi->at(alineCount)->getColorTable());

					scaled_img.save(longiPath + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");
				}

#ifdef OCT_NIRF
				// NIRF ring here
				if (checkList.bNirfRingOnly)
				{
#ifndef TWO_CHANNEL_NIRF
					ippiMirror_8u_C1IR(pImgObjVecLongiRing->at(alineCount)->qindeximg.bits(), sizeof(uint8_t) * nTotalFrame4, { nTotalFrame4, m_pResultTab->getConfigTemp()->circRadius }, ippAxsHorizontal);
					if (!checkList.bLongiResize)
						pImgObjVecLongiRing->at(alineCount)->qindeximg.copy(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius).
							save(longiNirfPath + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");
					else
					{
						cv::Mat src(pImgObjVecLongiRing->at(alineCount)->getHeight(), pImgObjVecLongiRing->at(alineCount)->getWidth(), CV_8UC1, pImgObjVecLongiRing->at(alineCount)->qindeximg.bits());
						cv::Mat src_crop = src(cv::Rect(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius));
						IplImage src_img(src_crop);
						IplImage *p_dst_img = cvCreateImage(cv::Size(checkList.nLongiWidth, checkList.nLongiHeight), IPL_DEPTH_8U, 1);

						cvResize(&src_img, p_dst_img, CV_INTER_CUBIC);

						QImage scaled_img((uchar*)p_dst_img->imageData, checkList.nLongiWidth, checkList.nLongiHeight, QImage::Format_Indexed8);
						scaled_img.setColorCount(256);
						scaled_img.setColorTable(pImgObjVecLongiRing->at(alineCount)->getColorTable());

						scaled_img.save(longiNirfPath + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");
					}
#else
					for (int i = 0; i < 2; i++)
					{
						ippiMirror_8u_C1IR(pImgObjVecLongiRing[i]->at(alineCount)->qindeximg.bits(), sizeof(uint8_t) * nTotalFrame4, { nTotalFrame4, m_pResultTab->getConfigTemp()->circRadius }, ippAxsHorizontal);
						if (!checkList.bLongiResize)
							pImgObjVecLongiRing[i]->at(alineCount)->qindeximg.copy(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius).
								save(longiNirfPath[i] + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");
						else
						{
							cv::Mat src(pImgObjVecLongiRing[i]->at(alineCount)->getHeight(), pImgObjVecLongiRing[i]->at(alineCount)->getWidth(), CV_8UC1, pImgObjVecLongiRing[i]->at(alineCount)->qindeximg.bits());
							cv::Mat src_crop = src(cv::Rect(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius));
							IplImage src_img(src_crop);
							IplImage *p_dst_img = cvCreateImage(cv::Size(checkList.nLongiWidth, checkList.nLongiHeight), IPL_DEPTH_8U, 1);

							cvResize(&src_img, p_dst_img, CV_INTER_AREA);

							QImage scaled_img((uchar*)p_dst_img->imageData, checkList.nLongiWidth, checkList.nLongiHeight, QImage::Format_Indexed8);
							scaled_img.setColorCount(256);
							scaled_img.setColorTable(pImgObjVecLongiRing[i]->at(alineCount)->getColorTable());

							scaled_img.save(longiNirfPath[i] + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");							
						}
					}
#endif
				}
#endif

				// Delete ImageObjects
				delete pImgObjVecLongi->at(alineCount++);
#ifdef OCT_NIRF
				if (checkList.bNirfRingOnly)
				{
#ifndef TWO_CHANNEL_NIRF
					delete pImgObjVecLongiRing->at(alineCount - 1);
#else
					for (int i = 0; i < 2; i++)
						delete pImgObjVecLongiRing[i]->at(alineCount - 1);
#endif
					if (checkList.bLongi)
						if (m_pResultTab->getNirfDistCompDlg())
							if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
#ifndef TWO_CHANNEL_NIRF
								m_pResultTab->getNirfDistCompDlg()->setCompInfo(longiNirfPath + "dist_comp_info.log");
#else
								for (int i = 0; i < 2; i++)
									m_pResultTab->getNirfDistCompDlg()->setCompInfo(longiNirfPath[i] + "dist_comp_info.log");
#endif
				}
#endif

				emit savedSingleFrame(m_nSavedFrames++);
#endif
			}
#ifdef OCT_FLIM
		}
		else
		{
			QString longiPath[3];
			if (!checkList.bMulti)
			{
				for (int i = 0; i < 3; i++)
				{
					if (checkList.bCh[i])
					{
						longiPath[i] = m_pResultTab->m_path + QString("/longi_image[%1 %2]_dB[%3 %4 g%5]_ch%6_i[%7 %8]_t[%9 %10]/")
							.arg(start).arg(end).arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2)
							.arg(i + 1).arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1)
							.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1);
						if (checkList.bLongi) QDir().mkdir(longiPath[i]);
					}
				}
			}
			else
			{
				longiPath[0] = m_pResultTab->m_path + QString("/longi_merged[%1 %2]_dB[%3 %4 g%5]_ch%6%7%8_i[%9 %10]_t[%11 %12]/")
					.arg(start).arg(end).arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2)
					.arg(checkList.bCh[0] ? "1" : "").arg(checkList.bCh[1] ? "2" : "").arg(checkList.bCh[2] ? "3" : "")
					.arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1)
					.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1);
				if (checkList.bLongi) QDir().mkdir(longiPath[0]);
			}

			int alineCount = 0;
			while (alineCount < m_pResultTab->m_vectorOctImage.at(0).size(1) / 2)
			{				
				if (checkList.bLongi)
				{
					if (!checkList.bMulti)
					{
						for (int i = 0; i < 3; i++)
						{
							if (checkList.bCh[i])
							{
								// Write longi images
								ippiMirror_8u_C1IR(pImgObjVecLongi[i]->at(alineCount)->qrgbimg.bits(), sizeof(uint8_t) * 3 * nTotalFrame4, { 3 * nTotalFrame4, m_pResultTab->getConfigTemp()->circRadius }, ippAxsHorizontal);
								if (!checkList.bLongiResize)
									pImgObjVecLongi[i]->at(alineCount)->qrgbimg.copy(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius).
										save(longiPath[i] + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");
								else
									pImgObjVecLongi[i]->at(alineCount)->qrgbimg.copy(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius).
										scaled(checkList.nLongiWidth, checkList.nLongiHeight, Qt::IgnoreAspectRatio, m_defaultTransformation).
										save(longiPath[i] + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");
							}
						}
					}
					else
					{
						// Write longi images
						ippiMirror_8u_C1IR(pImgObjVecLongi[0]->at(alineCount)->qrgbimg.bits(), sizeof(uint8_t) * 3 * nTotalFrame4, { 3 * nTotalFrame4, m_pResultTab->getConfigTemp()->circRadius }, ippAxsHorizontal);
						if (!checkList.bLongiResize)
							pImgObjVecLongi[0]->at(alineCount)->qrgbimg.copy(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius).
								save(longiPath[0] + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");
						else
							pImgObjVecLongi[0]->at(alineCount)->qrgbimg.copy(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius).
								scaled(checkList.nLongiWidth, checkList.nLongiHeight, Qt::IgnoreAspectRatio, m_defaultTransformation).
								save(longiPath[0] + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");
					}
				}
			
				// Delete ImageObjects
				delete pImgObjVecLongi[0]->at(alineCount);
				delete pImgObjVecLongi[1]->at(alineCount);
				delete pImgObjVecLongi[2]->at(alineCount++);			
			}
		}
#elif defined (OCT_NIRF)
		}
		else
		{
			QString nirfName;
#ifndef TWO_CHANNEL_NIRF
			if (m_pResultTab->getNirfDistCompDlg())
			{
				if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
				{
					if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
						nirfName = QString("/longi_image[%1 %2]_dB[%3 %4 g%5]_tbr_nirf_i[%6 %7]/");
					else
						nirfName = QString("/longi_image[%1 %2]_dB[%3 %4 g%5]_comp_nirf_i[%6 %7]/");
				}
				else
					nirfName = QString("/longi_image[%1 %2]_dB[%3 %4 g%5]_bg_sub_nirf_i[%6 %7]/");
			}
			else
			{
				nirfName = QString("/longi_image[%1 %2]_dB[%3 %4 g%5]_raw_nirf_i[%6 %7]/");
			}
			QString longiPath = m_pResultTab->m_path + nirfName.arg(start).arg(end).arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2)
				.arg(m_pConfig->nirfRange.min, 2, 'f', 2).arg(m_pConfig->nirfRange.max, 2, 'f', 2);
#else
			if (m_pResultTab->getNirfDistCompDlg())
			{
				if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
				{
					if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
						nirfName = QString("/longi_image[%1 %2]_dB[%3 %4 g%5]_tbr_nirf_i1[%6 %7]_i2[%8 %9]/");
					else
						nirfName = QString("/longi_image[%1 %2]_dB[%3 %4 g%5]_comp_nirf_i1[%6 %7]_i2[%8 %9]/");
				}
				else
					nirfName = QString("/longi_image[%1 %2]_dB[%3 %4 g%5]_bg_sub_nirf_i1[%6 %7]_i2[%8 %9]/");
			}
			else
			{
				nirfName = QString("/longi_image[%1 %2]_dB[%3 %4 g%5]_raw_nirf_i1[%6 %7]_i2[%8 %9]/");
			}
			QString longiPath = m_pResultTab->m_path + nirfName.arg(start).arg(end).arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2)
				.arg(m_pConfig->nirfRange[0].min, 2, 'f', 2).arg(m_pConfig->nirfRange[0].max, 2, 'f', 2)
				.arg(m_pConfig->nirfRange[1].min, 2, 'f', 2).arg(m_pConfig->nirfRange[1].max, 2, 'f', 2);
#endif
			QDir().mkdir(longiPath);

			int alineCount = 0;
			while (alineCount < m_pResultTab->m_vectorOctImage.at(0).size(1) / 2)
			{
				// Write longi images
				ippiMirror_8u_C1IR(pImgObjVecLongi->at(alineCount)->qrgbimg.bits(), sizeof(uint8_t) * 3 * nTotalFrame4, { 3 * nTotalFrame4, m_pResultTab->getConfigTemp()->circRadius }, ippAxsHorizontal);
				if (!checkList.bLongiResize)
					pImgObjVecLongi->at(alineCount)->qrgbimg.copy(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius).
						save(longiPath + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");
				else
					pImgObjVecLongi->at(alineCount)->qrgbimg.copy(start - 1, 0, end - start + 1, 2 * m_pResultTab->getConfigTemp()->circRadius).
						scaled(checkList.nLongiWidth, checkList.nLongiHeight, Qt::IgnoreAspectRatio, m_defaultTransformation).
						save(longiPath + QString("longi_%1_%2.bmp").arg(folderName).arg(alineCount + 1, 4, 10, (QChar)'0'), "bmp");

				// Delete ImageObjects
				delete pImgObjVecLongi->at(alineCount++);

				emit savedSingleFrame(m_nSavedFrames++);
			}

			if (checkList.bLongi)
				if (m_pResultTab->getNirfDistCompDlg())
					if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
						m_pResultTab->getNirfDistCompDlg()->setCompInfo(longiPath + "dist_comp_info.log");
		}
#endif
	}

#ifdef OCT_FLIM
	if (checkList.bLongi)
		for (int i = 0; i < 3; i++)
			delete pImgObjVecLongi[i];
#else
	// Delete ImageObjects
	delete pImgObjVecLongi;
#endif
}

void SaveResultDlg::circWriting(CrossSectionCheckList checkList)
{
	int nTotalFrame = (int)m_pResultTab->m_vectorOctImage.size();
	QString folderName;
	for (int i = 0; i < m_pResultTab->m_path.length(); i++)
		if (m_pResultTab->m_path.at(i) == QChar('/')) folderName = m_pResultTab->m_path.right(m_pResultTab->m_path.length() - i - 1);

    int start = m_pLineEdit_RangeStart->text().toInt();
    int end = m_pLineEdit_RangeEnd->text().toInt();

#ifdef OCT_FLIM
	if (!checkList.bCh[0] && !checkList.bCh[1] && !checkList.bCh[2])
	{
#elif defined (OCT_NIRF)
    if (!checkList.bNirf || checkList.bNirfRingOnly)
    {
#endif
		QString circPath;

#ifndef TWO_CHANNEL_NIRF
		QString circNirfPath;
#else
		QString circNirfPath[2];
#endif

#ifndef OCT_NIRF
        circPath = m_pResultTab->m_path + QString("/circ_image_dB[%1 %2 g%3]/").arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2);
#else
        circPath = m_pResultTab->m_path + QString("/circ_image_dB[%1 %2 g%3]%4/").arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2).arg(checkList.bNirfRingOnly ? "_ring-masked" : "");
		if (checkList.bNirfRingOnly)
		{
#ifndef TWO_CHANNEL_NIRF
			if (m_pResultTab->getNirfDistCompDlg())
			{
				if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
				{
					if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
						circNirfPath = QString("/circ_tbr_nirf_i[%1 %2]/");
					else
						circNirfPath = QString("/circ_comp_nirf_i[%1 %2]/");
				}
				else
					circNirfPath = QString("/circ_bg_sub_nirf_i[%1 %2]/");
			}
			else
			{
				circNirfPath = QString("/circ_raw_nirf_i[%1 %2]/");
			}
			circNirfPath = m_pResultTab->m_path + circNirfPath.arg(m_pConfig->nirfRange.min, 2, 'f', 2).arg(m_pConfig->nirfRange.max, 2, 'f', 2);
#else
			for (int i = 0; i < 2; i++)
			{
				if (m_pResultTab->getNirfDistCompDlg())
				{
					if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
					{
						if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
							circNirfPath[i] = QString("/circ_tbr_nirf_i%1[%2 %3]/");
						else
							circNirfPath[i] = QString("/circ_comp_nirf_i%1[%2 %3]/");
					}
					else
						circNirfPath[i] = QString("/circ_bg_sub_nirf_i%1[%2 %3]/");
				}
				else
				{
					circNirfPath[i] = QString("/circ_raw_nirf_i%1[%2 %3]/");
				}
				circNirfPath[i] = m_pResultTab->m_path + circNirfPath[i].arg(i + 1).arg(m_pConfig->nirfRange[i].min, 2, 'f', 2).arg(m_pConfig->nirfRange[i].max, 2, 'f', 2);
			}
#endif
		}
#endif

		if (checkList.bCirc)
		{
			QDir().mkdir(circPath);
#ifdef OCT_NIRF
			if (checkList.bNirfRingOnly)
			{
#ifndef TWO_CHANNEL_NIRF
				QDir().mkdir(circNirfPath);
#else
				QDir().mkdir(circNirfPath[0]);
				QDir().mkdir(circNirfPath[1]);
#endif
			}
#endif
		}

		int frameCount = 0;
		while (frameCount < nTotalFrame)
		{
			// Get the buffer from the previous sync Queue
			ImgObjVector *pImgObjVecCirc = m_syncQueueCircWriting.pop();

            // Range test
            if (((frameCount + 1) >= start) && ((frameCount + 1) <= end))
            {
                // Write circ images
                if (checkList.bCirc)
                {
                    if (!checkList.bCircResize)
                        pImgObjVecCirc->at(0)->qindeximg.save(circPath + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
					else
					{
						cv::Mat src(pImgObjVecCirc->at(0)->getHeight(), pImgObjVecCirc->at(0)->getWidth(), CV_8UC1, pImgObjVecCirc->at(0)->qindeximg.bits());
						IplImage src_img(src);
						IplImage *p_dst_img = cvCreateImage(cv::Size(checkList.nCircDiameter, checkList.nCircDiameter), IPL_DEPTH_8U, 1);

						cvResize(&src_img, p_dst_img, CV_INTER_AREA);

						QImage scaled_img((uchar*)p_dst_img->imageData, checkList.nCircDiameter, checkList.nCircDiameter, QImage::Format_Indexed8);
						scaled_img.setColorCount(256);
						scaled_img.setColorTable(pImgObjVecCirc->at(0)->getColorTable());

						scaled_img.save(circPath + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
					}

#ifdef OCT_NIRF
					// NIRF ring here
					if (checkList.bNirfRingOnly)
					{
#ifndef TWO_CHANNEL_NIRF						
						if (!checkList.bCircResize)
							pImgObjVecCirc->at(1)->qindeximg.save(circNirfPath + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
						else
						{
							cv::Mat src(pImgObjVecCirc->at(1)->getHeight(), pImgObjVecCirc->at(1)->getWidth(), CV_8UC1, pImgObjVecCirc->at(1)->qindeximg.bits());
							IplImage src_img(src);
							IplImage *p_dst_img = cvCreateImage(cv::Size(checkList.nCircDiameter, checkList.nCircDiameter), IPL_DEPTH_8U, 1);

							cvResize(&src_img, p_dst_img, CV_INTER_AREA);

							QImage scaled_img((uchar*)p_dst_img->imageData, checkList.nCircDiameter, checkList.nCircDiameter, QImage::Format_Indexed8);
							scaled_img.setColorCount(256);
							scaled_img.setColorTable(pImgObjVecCirc->at(1)->getColorTable());

							scaled_img.save(circNirfPath + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
						}
#else
						for (int i = 1; i <= 2; i++)
						{
							if (!checkList.bCircResize)
								pImgObjVecCirc->at(i)->qindeximg.save(circNirfPath[i - 1] + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
							else
							{
								cv::Mat src(pImgObjVecCirc->at(i)->getHeight(), pImgObjVecCirc->at(i)->getWidth(), CV_8UC1, pImgObjVecCirc->at(i)->qindeximg.bits());
								IplImage src_img(src);
								IplImage *p_dst_img = cvCreateImage(cv::Size(checkList.nCircDiameter, checkList.nCircDiameter), IPL_DEPTH_8U, 1);

								cvResize(&src_img, p_dst_img, CV_INTER_AREA);

								QImage scaled_img((uchar*)p_dst_img->imageData, checkList.nCircDiameter, checkList.nCircDiameter, QImage::Format_Indexed8);
								scaled_img.setColorCount(256);
								scaled_img.setColorTable(pImgObjVecCirc->at(i)->getColorTable());

								scaled_img.save(circNirfPath[i - 1] + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
							}
						}
#endif
					}
#endif
                }
            }
		
			frameCount++;
			emit savedSingleFrame(m_nSavedFrames++);

			// Delete ImageObjects
			delete pImgObjVecCirc->at(0);
#ifdef OCT_NIRF
			if (checkList.bNirfRingOnly)
			{
#ifndef TWO_CHANNEL_NIRF
				delete pImgObjVecCirc->at(1);
#else
				delete pImgObjVecCirc->at(1);
				delete pImgObjVecCirc->at(2);
#endif
				if (checkList.bCirc)
					if (m_pResultTab->getNirfDistCompDlg())
						if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
#ifndef TWO_CHANNEL_NIRF
							m_pResultTab->getNirfDistCompDlg()->setCompInfo(circNirfPath + "dist_comp_info.log");
#else
							for (int i = 0; i < 2; i++)
								m_pResultTab->getNirfDistCompDlg()->setCompInfo(circNirfPath[i] + "dist_comp_info.log");
#endif
			}
#endif
			delete pImgObjVecCirc;				
		}
#ifdef OCT_FLIM
	}
	else
	{
		QString circPath[3];
		if (!checkList.bMulti)
		{
			for (int i = 0; i < 3; i++)
			{
				if (checkList.bCh[i])
				{
                    circPath[i] = m_pResultTab->m_path + QString("/circ_image_dB[%1 %2 g%3]_ch%4_i[%5 %6]_t[%7 %8]/")
                        .arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2)
						.arg(i + 1).arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1)
						.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1);
					if (checkList.bCirc) QDir().mkdir(circPath[i]);
				}
			}
		}
		else
		{
            circPath[0] = m_pResultTab->m_path + QString("/circ_merged_dB[%1 %2 g%3]_ch%4%5%6_i[%7 %8]_t[%9 %10]/")
                .arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2)
				.arg(checkList.bCh[0] ? "1" : "").arg(checkList.bCh[1] ? "2" : "").arg(checkList.bCh[2] ? "3" : "")
				.arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1)
				.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1);
			if (checkList.bCirc) QDir().mkdir(circPath[0]);
		}

		int frameCount = 0;
		while (frameCount < nTotalFrame)
		{
			// Get the buffer from the previous sync Queue
			ImgObjVector *pImgObjVecCirc = m_syncQueueCircWriting.pop();

            // Range test
            if (((frameCount + 1) >= start) && ((frameCount + 1) <= end))
            {
                if (checkList.bCirc)
                {
                    if (!checkList.bMulti)
                    {
                        for (int i = 0; i < 3; i++)
                        {
                            if (checkList.bCh[i])
                            {
                                // Write circ images
                                if (!checkList.bCircResize)
                                    pImgObjVecCirc->at(i)->qrgbimg.save(circPath[i] + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                                else
                                    pImgObjVecCirc->at(i)->qrgbimg.scaled(checkList.nCircDiameter, checkList.nCircDiameter, Qt::IgnoreAspectRatio, m_defaultTransformation).
                                        save(circPath[i] + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                            }
                        }
                    }
                    else
                    {
                        // Write circ images
                        if (!checkList.bCircResize)
                            pImgObjVecCirc->at(0)->qrgbimg.save(circPath[0] + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                        else
                            pImgObjVecCirc->at(0)->qrgbimg.scaled(checkList.nCircDiameter, checkList.nCircDiameter, Qt::IgnoreAspectRatio, m_defaultTransformation).
                                save(circPath[0] + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                    }
                }
            }

			frameCount++;
			emit savedSingleFrame(m_nSavedFrames++);

			// Delete ImageObjects
			if (!checkList.bMulti)
			{
				for (int i = 0; i < 3; i++)
					delete pImgObjVecCirc->at(i);
			}
			else
				delete pImgObjVecCirc->at(0);
			delete pImgObjVecCirc;
		}
	}
#elif defined (OCT_NIRF)
    }
    else
    {
        QString nirfName;
#ifndef TWO_CHANNEL_NIRF
        if (m_pResultTab->getNirfDistCompDlg())
        {
            if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
            {
                if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
                    nirfName = QString("/circ_image_dB[%1 %2 g%3]_tbr_nirf_i[%4 %5]/");
                else
                    nirfName = QString("/circ_image_dB[%1 %2 g%3]_comp_nirf_i[%4 %5]/");
            }
            else
                nirfName = QString("/circ_image_dB[%1 %2 g%3]_bg_sub_nirf_i[%4 %5]/");
        }
        else
        {
            nirfName = QString("/circ_image_dB[%1 %2 g%3]_raw_nirf_i[%4 %5]/");
        }
        QString circPath = m_pResultTab->m_path + nirfName.arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2)
                .arg(m_pConfig->nirfRange.min, 2, 'f', 2).arg(m_pConfig->nirfRange.max, 2, 'f', 2);
#else
		if (m_pResultTab->getNirfDistCompDlg())
		{
			if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
			{
				if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
					nirfName = QString("/circ_image_dB[%1 %2 g%3]_tbr_nirf_i1[%4 %5]_i2[%6 %7]/");
				else
					nirfName = QString("/circ_image_dB[%1 %2 g%3]_comp_nirf_i1[%4 %5]_i2[%6 %7]/");
			}
			else
				nirfName = QString("/circ_image_dB[%1 %2 g%3]_bg_sub_nirf_i1[%4 %5]_i2[%6 %7]/");
		}
		else
		{
			nirfName = QString("/circ_image_dB[%1 %2 g%3]_raw_nirf_i1[%4 %5]_i2[%6 %7]/");
		}
		QString circPath = m_pResultTab->m_path + nirfName.arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).arg(m_pConfig->octDbGamma, 3, 'f', 2)
			.arg(m_pConfig->nirfRange[0].min, 2, 'f', 2).arg(m_pConfig->nirfRange[0].max, 2, 'f', 2)
			.arg(m_pConfig->nirfRange[1].min, 2, 'f', 2).arg(m_pConfig->nirfRange[1].max, 2, 'f', 2);
#endif
        if (checkList.bCirc) QDir().mkdir(circPath);

        int frameCount = 0;
        while (frameCount < nTotalFrame)
        {
            // Get the buffer from the previous sync Queue
            ImgObjVector *pImgObjVecCirc = m_syncQueueCircWriting.pop();

            // Range test
            if (((frameCount + 1) >= start) && ((frameCount + 1) <= end))
            {
                if (checkList.bCirc)
                {
                    // Write circ images
                    if (!checkList.bCircResize)
                        pImgObjVecCirc->at(0)->qrgbimg.save(circPath + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                    else
                        pImgObjVecCirc->at(0)->qrgbimg.scaled(checkList.nCircDiameter, checkList.nCircDiameter, Qt::IgnoreAspectRatio, m_defaultTransformation).
                            save(circPath + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                }
            }

            frameCount++;
			emit savedSingleFrame(m_nSavedFrames++);

            // Delete ImageObjects
            delete pImgObjVecCirc->at(0);
            delete pImgObjVecCirc;
        }

		if (checkList.bCirc)
			if (m_pResultTab->getNirfDistCompDlg())
				if (m_pResultTab->getNirfDistCompDlg()->isCompensating())		
					m_pResultTab->getNirfDistCompDlg()->setCompInfo(circPath + "dist_comp_info.log");
    }
#endif
}
