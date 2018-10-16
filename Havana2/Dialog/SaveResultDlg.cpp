
#include "SaveResultDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QResultTab.h>
#include <Havana2/Viewer/QImageView.h>

#ifdef OCT_NIRF
#include <Havana2/Dialog/NirfDistCompDlg.h>
#endif

#include <ippi.h>
#include <ippcc.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <iostream>
#include <thread>
#include <chrono>
#include <utility>


SaveResultDlg::SaveResultDlg(QWidget *parent) :
    QDialog(parent)
{
    // Set default size & frame
#ifdef OCT_FLIM
    setFixedSize(420, 150);
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
	setFixedSize(420, 100);
#else
    setFixedSize(420, 120);
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
	m_pCheckBox_RectImage->setText("Rect Image");
	m_pCheckBox_RectImage->setChecked(true);
	m_pCheckBox_CircImage = new QCheckBox(this);
	m_pCheckBox_CircImage->setText("Circ Image");
	m_pCheckBox_CircImage->setChecked(true);

	m_pCheckBox_ResizeRectImage = new QCheckBox(this);
	m_pCheckBox_ResizeRectImage->setText("Resize (w x h)");
	m_pCheckBox_ResizeCircImage = new QCheckBox(this);
	m_pCheckBox_ResizeCircImage->setText("Resize (diameter)");

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
	m_pLineEdit_CircDiameter->setText(QString::number(2 * CIRC_RADIUS));
	m_pLineEdit_CircDiameter->setAlignment(Qt::AlignCenter);
	m_pLineEdit_CircDiameter->setDisabled(true);

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
	m_pCheckBox_RawData->setText("Raw Data");
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
	pHBoxLayout_RectResize->addWidget(m_pCheckBox_ResizeRectImage);
	pHBoxLayout_RectResize->addWidget(m_pLineEdit_RectWidth);
	pHBoxLayout_RectResize->addWidget(m_pLineEdit_RectHeight);
    pHBoxLayout_RectResize->addStretch(1);
    //pHBoxLayout_RectResize->addItem(new QSpacerItem(0, 0, QSizePolicy::Preferred, QSizePolicy::Fixed));

    pGridLayout->addWidget(m_pCheckBox_RectImage, 0, 1);
    pGridLayout->addItem(pHBoxLayout_RectResize, 0, 2, 1, 2);
    pGridLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 4);

	QHBoxLayout *pHBoxLayout_CircResize = new QHBoxLayout;
	pHBoxLayout_CircResize->addWidget(m_pCheckBox_ResizeCircImage);
	pHBoxLayout_CircResize->addWidget(m_pLineEdit_CircDiameter);
    pHBoxLayout_CircResize->addStretch(1);
    //pHBoxLayout_CircResize->addItem(new QSpacerItem(0, 0, QSizePolicy::Preferred, QSizePolicy::Fixed));

    pGridLayout->addWidget(m_pCheckBox_CircImage, 1, 1);
    pGridLayout->addItem(pHBoxLayout_CircResize, 1, 2, 1, 2);

#ifdef OCT_FLIM
    pGridLayout->addWidget(m_pCheckBox_CrossSectionCh1, 2, 1);
    pGridLayout->addWidget(m_pCheckBox_CrossSectionCh2, 2, 2);
    pGridLayout->addWidget(m_pCheckBox_CrossSectionCh3, 2, 3);

    pGridLayout->addWidget(m_pCheckBox_Multichannel, 3, 1, 1, 3);
#endif
#ifdef OCT_NIRF
    pGridLayout->addWidget(m_pCheckBox_CrossSectionNirf, 2, 1, 1, 2);
#endif

    pGridLayout->addItem(pHBoxLayout_Range, 1, 0);


    pGridLayout->addWidget(m_pPushButton_SaveEnFaceMaps, 4, 0);

    pGridLayout->addWidget(m_pCheckBox_RawData, 4, 1);
    pGridLayout->addWidget(m_pCheckBox_ScaledImage, 4, 2);

#ifdef OCT_FLIM
    pGridLayout->addWidget(m_pCheckBox_EnFaceCh1, 5, 1);
    pGridLayout->addWidget(m_pCheckBox_EnFaceCh2, 5, 2);
    pGridLayout->addWidget(m_pCheckBox_EnFaceCh3, 5, 3);
#endif

#ifdef OCT_NIRF
    pGridLayout->addWidget(m_pCheckBox_EnFaceNirf, 5, 1, 1, 2);
#endif

    pGridLayout->addWidget(m_pCheckBox_OctMaxProjection, 6, 1, 1, 2);
	
    setLayout(pGridLayout);

    // Connect
	connect(m_pPushButton_SaveCrossSections, SIGNAL(clicked(bool)), this, SLOT(saveCrossSections()));
	connect(m_pPushButton_SaveEnFaceMaps, SIGNAL(clicked(bool)), this, SLOT(saveEnFaceMaps()));

    connect(m_pLineEdit_RangeStart, SIGNAL(textChanged(const QString &)), this, SLOT(setRange()));
    connect(m_pLineEdit_RangeEnd, SIGNAL(textChanged(const QString &)), this, SLOT(setRange()));

	connect(m_pCheckBox_ResizeRectImage, SIGNAL(toggled(bool)), SLOT(enableRectResize(bool)));
	connect(m_pCheckBox_ResizeCircImage, SIGNAL(toggled(bool)), SLOT(enableCircResize(bool)));

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


void SaveResultDlg::saveCrossSections()
{
	std::thread t1([&]() {
		
		// Check Status /////////////////////////////////////////////////////////////////////////////
		CrossSectionCheckList checkList;
		checkList.bRect = m_pCheckBox_RectImage->isChecked();
		checkList.bCirc = m_pCheckBox_CircImage->isChecked();
		checkList.bRectResize = m_pCheckBox_ResizeRectImage->isChecked();
		checkList.bCircResize = m_pCheckBox_ResizeCircImage->isChecked();
		checkList.nRectWidth = m_pLineEdit_RectWidth->text().toInt();
		checkList.nRectHeight = m_pLineEdit_RectHeight->text().toInt();
		checkList.nCircDiameter = m_pLineEdit_CircDiameter->text().toInt();
#ifdef OCT_FLIM
		checkList.bCh[0] = m_pCheckBox_CrossSectionCh1->isChecked();
		checkList.bCh[1] = m_pCheckBox_CrossSectionCh2->isChecked();
		checkList.bCh[2] = m_pCheckBox_CrossSectionCh3->isChecked();
		checkList.bMulti = m_pCheckBox_Multichannel->isChecked();
#endif
#ifdef OCT_NIRF
        checkList.bNirf = m_pCheckBox_CrossSectionNirf->isChecked();
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
			roi_nirf, m_pConfig->nirfRange.min, m_pConfig->nirfRange.max);

		np::Uint8Array2 nirfMap2(roi_nirf.width, roi_nirf.height);
		ippiScale_32f8u_C1R(m_pResultTab->m_nirfMap2_0.raw_ptr(), sizeof(float) * roi_nirf.width, nirfMap2.raw_ptr(), sizeof(uint8_t) * roi_nirf.width,
			roi_nirf, m_pConfig->nirfRange.min, m_pConfig->nirfRange.max);
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

		// Set Widgets //////////////////////////////////////////////////////////////////////////////
		emit setWidgets(false);

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
                        fileIntensity.write(reinterpret_cast<char*>(&m_pResultTab->m_intensityMap.at(i)(0, start - 1)), sizeof(float) * m_pResultTab->m_intensityMap.at(i).size(0) * (end - start + 1));
						fileIntensity.close();
					}

                    QFile fileLifetime(enFacePath + QString("lifetime_range[%1 %2]_ch%3.enface").arg(start).arg(end).arg(i + 1));
					if (false != fileLifetime.open(QIODevice::WriteOnly))
					{
                        fileLifetime.write(reinterpret_cast<char*>(&m_pResultTab->m_lifetimeMap.at(i)(0, start - 1)), sizeof(float) * m_pResultTab->m_lifetimeMap.at(i).size(0) * (end - start + 1));
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
                        saveCompDetailsLog(nirfName.replace("enface", "log"));
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
							saveCompDetailsLog(nirfName[i].replace("enface", "log"));
				}
#endif
            }
#endif
			if (checkList.bOctProj)
			{
                QFile fileOctMaxProj(enFacePath + QString("oct_max_projection_range[%1 %2].enface").arg(start).arg(end));
				if (false != fileOctMaxProj.open(QIODevice::WriteOnly))
				{
                    IppiSize roi_proj = { m_pResultTab->getRectImageView()->getRender()->m_pImage->width(), m_pResultTab->m_octProjection.size(1) };

                    np::FloatArray2 octProj(roi_proj.width, roi_proj.height);
                    ippiCopy_32f_C1R(m_pResultTab->m_octProjection.raw_ptr(), sizeof(float) * m_pResultTab->m_octProjection.size(0),
                                     octProj.raw_ptr(), sizeof(float) * octProj.size(0), roi_proj);
#ifdef GALVANO_MIRROR
					if (m_pConfig->galvoHorizontalShift)
					{
						int roi_proj_width_non4 = m_pResultTab->getRectImageView()->getRender()->m_pImage->width();
						for (int i = 0; i < roi_proj.height; i++)
						{
							float* pImg = octProj.raw_ptr() + i * roi_proj.width;
							std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_proj_width_non4);
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
					(*m_pResultTab->m_pMedfiltIntensityMap)(imgObjIntensity.arr.raw_ptr());
                    imgObjIntensity.qindeximg.copy(0, roi_flimproj.height - end, roi_flimproj.width, end - start + 1)
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
					(*m_pResultTab->m_pMedfiltLifetimeMap)(imgObjLifetime.arr.raw_ptr());
                    imgObjLifetime.qindeximg.copy(0, roi_flimproj.height - end, roi_flimproj.width, end - start + 1)
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
                    imgObjHsv.qrgbimg.copy(0, roi_flimproj.height - end, roi_flimproj.width, end - start + 1)
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
                ImageObject imgObjNirfMap(roi_nirf.width, roi_nirf.height, temp_ctable.m_colorTableVector.at(ColorTable::hot));

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
                        save(nirfName.arg(start).arg(end).arg(m_pConfig->nirfRange.min, 2, 'f', 1).arg(m_pConfig->nirfRange.max, 2, 'f', 1), "bmp");

                if (m_pResultTab->getNirfDistCompDlg())
                    if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
                        saveCompDetailsLog(nirfName.arg(start).arg(end).arg(m_pConfig->nirfRange.min, 2, 'f', 1).arg(m_pConfig->nirfRange.max, 2, 'f', 1).replace("bmp", "log"));
#else
				for (int i = 0; i < 2; i++)
				{
					auto pNirfMap = (i == 0) ? &m_pResultTab->m_nirfMap1 : &m_pResultTab->m_nirfMap2;
					auto pNirfMap0 = (i == 0) ? &m_pResultTab->m_nirfMap1_0 : &m_pResultTab->m_nirfMap2_0;

					IppiSize roi_nirf = { pNirfMap0->size(0), pNirfMap0->size(1) };
					ImageObject imgObjNirfMap(roi_nirf.width, roi_nirf.height, temp_ctable.m_colorTableVector.at(ColorTable::hot));

					ippiScale_32f8u_C1R(pNirfMap0->raw_ptr(), sizeof(float) * roi_nirf.width, imgObjNirfMap.arr.raw_ptr(), sizeof(uint8_t) * roi_nirf.width, roi_nirf, m_pConfig->nirfRange.min, m_pConfig->nirfRange.max);
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
						save(nirfName.arg(i + 1).arg(start).arg(end).arg(m_pConfig->nirfRange.min, 2, 'f', 1).arg(m_pConfig->nirfRange.max, 2, 'f', 1), "bmp");

					if (m_pResultTab->getNirfDistCompDlg())
						if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
							saveCompDetailsLog(nirfName.arg(i + 1).arg(start).arg(end).arg(m_pConfig->nirfRange.min, 2, 'f', 1).arg(m_pConfig->nirfRange.max, 2, 'f', 1).replace("bmp", "log"));
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
                    int roi_proj_width_non4 = m_pResultTab->getRectImageView()->getRender()->m_pImage->width();
					for (int i = 0; i < roi_proj.height; i++)
					{
                        uint8_t* pImg = imgObjOctMaxProj.arr.raw_ptr() + i * roi_proj.width;
                        std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_proj_width_non4);
					}
				}
#endif
                imgObjOctMaxProj.qindeximg.copy(0, roi_proj.height - end, m_pResultTab->getRectImageView()->getRender()->m_pImage->width(), end - start + 1).
                        save(enFacePath + QString("oct_max_projection_range[%1 %2]_dB[%3 %4].bmp").arg(start).arg(end).arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max), "bmp");
			}
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(500));

		// Scaling MATLAB Script ////////////////////////////////////////////////////////////////////
		if (false == QFile::copy("scale_indicator.m", enFacePath + "scale_indicator.m"))
			printf("Error occurred while copying matlab sciprt.\n");

		// Reset Widgets ////////////////////////////////////////////////////////////////////////////
		emit setWidgets(true);
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

void SaveResultDlg::setWidgetsEnabled(bool enabled)
{
	// Save Cross-sections
	m_pPushButton_SaveCrossSections->setEnabled(enabled);
	m_pCheckBox_RectImage->setEnabled(enabled);
	m_pCheckBox_CircImage->setEnabled(enabled);

	m_pCheckBox_ResizeRectImage->setEnabled(enabled);
	m_pCheckBox_ResizeCircImage->setEnabled(enabled);

	m_pLineEdit_RectWidth->setEnabled(enabled);
	m_pLineEdit_RectHeight->setEnabled(enabled);
	m_pLineEdit_CircDiameter->setEnabled(enabled);
	if (enabled)
	{
		if (!m_pCheckBox_ResizeRectImage->isChecked())
		{
			m_pLineEdit_RectWidth->setEnabled(false);
			m_pLineEdit_RectHeight->setEnabled(false);
		}
		if (!m_pCheckBox_ResizeCircImage->isChecked())
			m_pLineEdit_CircDiameter->setEnabled(false);
	}	

#ifdef OCT_FLIM
	m_pCheckBox_CrossSectionCh1->setEnabled(enabled);
	m_pCheckBox_CrossSectionCh2->setEnabled(enabled);
	m_pCheckBox_CrossSectionCh3->setEnabled(enabled);
	m_pCheckBox_Multichannel->setEnabled(enabled);
#endif

#ifdef OCT_NIRF
    m_pCheckBox_CrossSectionNirf->setEnabled(enabled);
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
		pImgObjVec->push_back(new ImageObject(roi_oct.height, roi_oct.width, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable())));

#ifdef OCT_FLIM
		// Image objects for Ch1 FLIM
		pImgObjVec->push_back(new ImageObject(roi_oct.height / 4, RING_THICKNESS, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE)));
		pImgObjVec->push_back(new ImageObject(roi_oct.height / 4, RING_THICKNESS, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentLifetimeColorTable())));
		// Image objects for Ch2 FLIM
		pImgObjVec->push_back(new ImageObject(roi_oct.height / 4, RING_THICKNESS, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE)));
		pImgObjVec->push_back(new ImageObject(roi_oct.height / 4, RING_THICKNESS, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentLifetimeColorTable())));
		// Image objects for Ch3 FLIM
		pImgObjVec->push_back(new ImageObject(roi_oct.height / 4, RING_THICKNESS, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE)));
		pImgObjVec->push_back(new ImageObject(roi_oct.height / 4, RING_THICKNESS, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentLifetimeColorTable())));
#endif

#ifdef OCT_NIRF
       // Image objects for NIRF
#ifndef TWO_CHANNEL_NIRF
        pImgObjVec->push_back(new ImageObject(roi_oct.height, RING_THICKNESS, temp_ctable.m_colorTableVector.at(ColorTable::hot)));
#else
		pImgObjVec->push_back(new ImageObject(roi_oct.height, 2 * RING_THICKNESS, temp_ctable.m_colorTableVector.at(ColorTable::hot)));
#endif
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
				for (int j = 0; j < RING_THICKNESS; j++)
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
        if (checkList.bNirf)
        {
#ifndef TWO_CHANNEL_NIRF
            uint8_t* rectNirf = &nirfMap(0, frameCount);
            for (int j = 0; j < RING_THICKNESS; j++)
                memcpy(&pImgObjVec->at(1)->arr(0, j), rectNirf, sizeof(uint8_t) * roi_nirf.width);
#else
			uint8_t* rectNirf1 = &nirfMap1(0, frameCount);
			for (int j = 0; j < RING_THICKNESS; j++)
				memcpy(&pImgObjVec->at(1)->arr(0, j), rectNirf1, sizeof(uint8_t) * roi_nirf.width);

			uint8_t* rectNirf2 = &nirfMap2(0, frameCount);
			for (int j = 0; j < RING_THICKNESS; j++)
				memcpy(&pImgObjVec->at(1)->arr(0, j + RING_THICKNESS), rectNirf2, sizeof(uint8_t) * roi_nirf.width);
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
            pImgObjVec->at(0)->convertRgb();
            pImgObjVec->at(1)->convertNonScaledRgb(); // nirf
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
    if (!checkList.bNirf)
    {
#endif
		QString rectPath;
        rectPath = m_pResultTab->m_path + QString("/rect_image_dB[%1 %2]/").arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max);
		if (checkList.bRect) QDir().mkdir(rectPath);

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
                            pImgObjVec->at(0)->qindeximg.scaled(checkList.nRectWidth, checkList.nRectHeight).
                                save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                    }
                    else
                    {
                        // Cut useless A-lines
                        if (!checkList.bRectResize)
                            pImgObjVec->at(0)->qindeximg.copy(0, 0, original_nAlines, pImgObjVec->at(0)->getHeight()).
                                    save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                        else
                            pImgObjVec->at(0)->qindeximg.copy(0, 0, original_nAlines, pImgObjVec->at(0)->getHeight()).scaled(checkList.nRectWidth, checkList.nRectHeight).
                                save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                    }
                }
            }
		
			frameCount++;
			emit savedSingleFrame(m_nSavedFrames++);

			// Push the buffers to sync Queues
			m_syncQueueCircularizing.push(pImgObjVec);
		}
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
                    rectPath[i] = m_pResultTab->m_path + QString("/rect_image_dB[%1 %2]_ch%3_i[%4 %5]_t[%6 %7]/")
                        .arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).
						.arg(i + 1).arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1)
						.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1);
					if (checkList.bRect) QDir().mkdir(rectPath[i]);
				}
			}
		}
		else
		{
            rectPath[0] = m_pResultTab->m_path + QString("/rect_merged_dB[%1 %2]_ch%3%4%5_i[%6 %7]_t[%8 %9]/")
                .arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max).
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
                            memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - 2 * RING_THICKNESS),
                                pImgObjVec->at(1 + 2 * i)->qrgbimg.bits(), pImgObjVec->at(1 + 2 * i)->qrgbimg.byteCount()); // Intensity
                            memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - 1 * RING_THICKNESS),
                                pImgObjVec->at(2 + 2 * i)->qrgbimg.bits(), pImgObjVec->at(2 + 2 * i)->qrgbimg.byteCount()); // Lifetime

                            if (checkList.bRect)
                            {
                                // Write rect images
                                if (!checkList.bRectResize)
                                    pImgObjVec->at(0)->qrgbimg.save(rectPath[i] + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                                else
                                    pImgObjVec->at(0)->qrgbimg.scaled(checkList.nRectWidth, checkList.nRectHeight).
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
                            memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - (nCh - n++) * RING_THICKNESS),
                                pImgObjVec->at(1 + 2 * i)->qrgbimg.bits(), pImgObjVec->at(1 + 2 * i)->qrgbimg.byteCount());
                        }

                        if (checkList.bRect)
                        {
                            // Write rect images
                            if (!checkList.bRectResize)
                                pImgObjVec->at(0)->qrgbimg.save(rectPath[0] + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                            else
                                pImgObjVec->at(0)->qrgbimg.scaled(checkList.nRectWidth, checkList.nRectHeight).
                                save(rectPath[0] + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                        }
                    }
                }
            }

			frameCount++;
			emit savedSingleFrame(m_nSavedFrames++);

			// Push the buffers to sync Queues
			m_syncQueueCircularizing.push(pImgObjVec);
		}
	}
#elif defined(OCT_NIRF)
    }
    else
    {
        QString nirfName;
        if (m_pResultTab->getNirfDistCompDlg())
        {
            if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
            {
                if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
                    nirfName = QString("/rect_image_dB[%1 %2]_tbr_nirf_i[%3 %4]/");
                else
                    nirfName = QString("/rect_image_dB[%1 %2]_comp_nirf_i[%3 %4]/");
            }
            else
                nirfName = QString("/rect_image_dB[%1 %2]_bg_sub_nirf_i[%3 %4]/");
        }
        else
        {
            nirfName = QString("/rect_image_dB[%1 %2]_raw_nirf_i[%3 %4]/");
        }
        QString rectPath = m_pResultTab->m_path + nirfName.arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max)
                .arg(m_pConfig->nirfRange.min, 2, 'f', 1).arg(m_pConfig->nirfRange.max, 2, 'f', 1);
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
                memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - 1 * RING_THICKNESS),
                    pImgObjVec->at(1)->qrgbimg.bits(), pImgObjVec->at(1)->qrgbimg.byteCount()); // Nirf
#else
				memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - 2 * RING_THICKNESS),
					pImgObjVec->at(1)->qrgbimg.bits(), pImgObjVec->at(1)->qrgbimg.byteCount()); // Nirf
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
                            pImgObjVec->at(0)->qrgbimg.scaled(checkList.nRectWidth, checkList.nRectHeight).
                                    save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                    }
                    else
                    {
                        // Write rect images
                        if (!checkList.bRectResize)
                            pImgObjVec->at(0)->qrgbimg.copy(0, 0, original_nAlines, pImgObjVec->at(0)->getHeight()).
                                    save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                        else
                            pImgObjVec->at(0)->qrgbimg.copy(0, 0, original_nAlines, pImgObjVec->at(0)->getHeight()).scaled(checkList.nRectWidth, checkList.nRectHeight).
                                    save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                    }
                }
            }

            frameCount++;
            emit savedSingleFrame(m_nSavedFrames++);

            // Push the buffers to sync Queues
            m_syncQueueCircularizing.push(pImgObjVec);
        }        

		if (checkList.bRect)
			if (m_pResultTab->getNirfDistCompDlg())
				if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
					saveCompDetailsLog(rectPath + "dist_comp_details.log");
    }
#endif
}

void SaveResultDlg::circularizing(CrossSectionCheckList checkList)
{
	int nTotalFrame = (int)m_pResultTab->m_vectorOctImage.size();
	ColorTable temp_ctable;

	int frameCount = 0;
	while (frameCount < nTotalFrame)
	{
		// Get the buffer from the previous sync Queue
		ImgObjVector *pImgObjVec = m_syncQueueCircularizing.pop();
		ImgObjVector *pImgObjVecCirc = new ImgObjVector;
		//int polishedSurface = 0;

#ifdef OCT_FLIM
		if (!checkList.bCh[0] && !checkList.bCh[1] && !checkList.bCh[2])
		{
#elif defined (OCT_NIRF)
        if (!checkList.bNirf)
        {
#endif
			// ImageObject for circ writing
			ImageObject *pCircImgObj = new ImageObject(2 * CIRC_RADIUS, 2 * CIRC_RADIUS, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()));

			// Circularize
			if (checkList.bCirc)
			{
				np::Uint8Array2 rect_temp(pImgObjVec->at(0)->qindeximg.bits(), pImgObjVec->at(0)->arr.size(0), pImgObjVec->at(0)->arr.size(1));

				// Ball lens polished surface detection				
				//np::DoubleArray mean_profile(PROJECTION_OFFSET);
				//tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)PROJECTION_OFFSET),
				//	[&](const tbb::blocked_range<size_t>& r) {
				//	for (size_t i = r.begin(); i != r.end(); ++i)
				//	{
				//		uint8_t *pLine = &rect_temp(0, m_pConfig->circCenter + (int)i);
				//		ippiMean_8u_C1R(pLine, rect_temp.size(0), { rect_temp.size(0), 1 }, &mean_profile((int)i));
				//	}
				//});

				//np::DoubleArray drv_profile(PROJECTION_OFFSET - 1);
				//memset(drv_profile, 0, sizeof(double) * drv_profile.length());
				//tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)(PROJECTION_OFFSET - 1)),
				//	[&](const tbb::blocked_range<size_t>& r) {
				//	for (size_t i = r.begin(); i != r.end(); ++i)
				//	{
				//		drv_profile((int)i) = mean_profile((int)i + 1) - mean_profile((int)i);
				//	}
				//});

				//for (int i = 0; i < PROJECTION_OFFSET - 2; i++)
				//{
				//	bool det = (drv_profile(i + 1) * drv_profile(i) < 0) ? true : false;
				//	if (det && (mean_profile(i) > 30))
				//	{
				//		polishedSurface = i + 1;
				//		printf("%d\n", polishedSurface);
				//		break;
				//	}
				//}
				//polishedSurface = BALL_SIZE;
				
				int center = m_pConfig->circCenter; // +polishedSurface - BALL_SIZE;
				(*m_pResultTab->m_pCirc)(rect_temp, pCircImgObj->qindeximg.bits(), "vertical", center);
			}

			// Vector pushing back
			pImgObjVecCirc->push_back(pCircImgObj);
#ifdef OCT_FLIM
		}
		else
		{
			// ImageObject for circ writing
			ImageObject *pCircImgObj[3];
			if (!checkList.bMulti)
			{
				for (int i = 0; i < 3; i++)
				{
					pCircImgObj[i] = new ImageObject(2 * CIRC_RADIUS, 2 * CIRC_RADIUS, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()));

					if (checkList.bCh[i] && checkList.bCirc)
					{
						// Paste FLIM color ring to RGB rect image
						int center = m_pConfig->circCenter; // +polishedSurface - BALL_SIZE;
						memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (center + CIRC_RADIUS - 2 * RING_THICKNESS),
							pImgObjVec->at(1 + 2 * i)->qrgbimg.bits(), pImgObjVec->at(1 + 2 * i)->qrgbimg.byteCount()); // Intensity
						memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (center + CIRC_RADIUS - 1 * RING_THICKNESS),
							pImgObjVec->at(2 + 2 * i)->qrgbimg.bits(), pImgObjVec->at(2 + 2 * i)->qrgbimg.byteCount()); // Lifetime

						// Circularize
						np::Uint8Array2 rect_temp(pImgObjVec->at(0)->qrgbimg.bits(), 3 * pImgObjVec->at(0)->arr.size(0), pImgObjVec->at(0)->arr.size(1));
						(*m_pResultTab->m_pCirc)(rect_temp, pCircImgObj[i]->qrgbimg.bits(), "vertical", "rgb", m_pConfig->circCenter);
					}

					// Vector pushing back
					pImgObjVecCirc->push_back(pCircImgObj[i]);
				}
			}
			else
			{
				int nCh = (checkList.bCh[0] ? 1 : 0) + (checkList.bCh[1] ? 1 : 0) + (checkList.bCh[2] ? 1 : 0);
				int n = 0;

				pCircImgObj[0] = new ImageObject(2 * CIRC_RADIUS, 2 * CIRC_RADIUS, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()));

				for (int i = 0; i < 3; i++)
				{
					if (checkList.bCh[i])
					{
						int center = m_pConfig->circCenter; // +polishedSurface - BALL_SIZE;
						memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (center + CIRC_RADIUS - (nCh - n++) * RING_THICKNESS),
							pImgObjVec->at(1 + 2 * i)->qrgbimg.bits(), pImgObjVec->at(1 + 2 * i)->qrgbimg.byteCount());
					}
				}

				// Circularize
				np::Uint8Array2 rect_temp(pImgObjVec->at(0)->qrgbimg.bits(), 3 * pImgObjVec->at(0)->arr.size(0), pImgObjVec->at(0)->arr.size(1));
				(*m_pResultTab->m_pCirc)(rect_temp, pCircImgObj[0]->qrgbimg.bits(), "vertical", "rgb", m_pConfig->circCenter);

				// Vector pushing back
				pImgObjVecCirc->push_back(pCircImgObj[0]);
			}
		}	
#elif defined (OCT_NIRF)
        }
        else
        {
            // ImageObject for circ writing
            ImageObject *pCircImgObj = new ImageObject(2 * CIRC_RADIUS, 2 * CIRC_RADIUS, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()));

            if (checkList.bCirc)
            {
                np::Uint8Array2 rect_temp(pImgObjVec->at(0)->qrgbimg.bits(), 3 * pImgObjVec->at(0)->arr.size(0), pImgObjVec->at(0)->arr.size(1));

				//// Ball lens polished surface detection				
				//np::DoubleArray mean_profile(PROJECTION_OFFSET);
				//tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)PROJECTION_OFFSET),
				//	[&](const tbb::blocked_range<size_t>& r) {
				//	for (size_t i = r.begin(); i != r.end(); ++i)
				//	{
				//		uint8_t *pLine = &rect_temp(0, m_pConfig->circCenter + (int)i);
				//		ippiMean_8u_C1R(pLine, rect_temp.size(0), { rect_temp.size(0), 1 }, &mean_profile((int)i));
				//	}
				//});

				//np::DoubleArray drv_profile(PROJECTION_OFFSET - 1);
				//memset(drv_profile, 0, sizeof(double) * drv_profile.length());
				//tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)(PROJECTION_OFFSET - 1)),
				//	[&](const tbb::blocked_range<size_t>& r) {
				//	for (size_t i = r.begin(); i != r.end(); ++i)
				//	{
				//		drv_profile((int)i) = mean_profile((int)i + 1) - mean_profile((int)i);
				//	}
				//});

				//for (int i = 0; i < PROJECTION_OFFSET - 2; i++)
				//{
				//	bool det = (drv_profile(i + 1) * drv_profile(i) < 0) ? true : false;
				//	if (det && (mean_profile(i) > 30))
				//	{
				//		polishedSurface = i + 1;
				//		//printf("%d\n", polishedSurface);
				//		break;
				//	}
				//}

				int center = m_pConfig->circCenter; // +polishedSurface - BALL_SIZE;

				// Paste FLIM color ring to RGB rect image
#ifndef TWO_CHANNEL_NIRF
				memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (center + CIRC_RADIUS - 1 * RING_THICKNESS),
					pImgObjVec->at(1)->qrgbimg.bits(), pImgObjVec->at(1)->qrgbimg.byteCount()); // Nirf
#else
				memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (center + CIRC_RADIUS - 2 * RING_THICKNESS),
					pImgObjVec->at(1)->qrgbimg.bits(), pImgObjVec->at(1)->qrgbimg.byteCount()); // Nirf
#endif

				// Circularize
                (*m_pResultTab->m_pCirc)(rect_temp, pCircImgObj->qrgbimg.bits(), "vertical", "rgb", center);
            }

            // Vector pushing back
            pImgObjVecCirc->push_back(pCircImgObj);
        }
#endif
		frameCount++;

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
#endif
#endif
		delete pImgObjVec;
	}
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
    if (!checkList.bNirf)
    {
#endif
		QString circPath;
        circPath = m_pResultTab->m_path + QString("/circ_image_dB[%1 %2]/").arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max);
		if (checkList.bCirc) QDir().mkdir(circPath);

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
                        pImgObjVecCirc->at(0)->qindeximg.scaled(checkList.nCircDiameter, checkList.nCircDiameter).
                            save(circPath + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
                }
            }
		
			frameCount++;
			emit savedSingleFrame(m_nSavedFrames++);

			// Delete ImageObjects
			delete pImgObjVecCirc->at(0);
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
                    circPath[i] = m_pResultTab->m_path + QString("/circ_image_dB[%1 %2]_ch%3_i[%4 %5]_t[%6 %7]/")
                        .arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max)
						.arg(i + 1).arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1)
						.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1);
					if (checkList.bCirc) QDir().mkdir(circPath[i]);
				}
			}
		}
		else
		{
            circPath[0] = m_pResultTab->m_path + QString("/circ_merged_dB[%1 %2]_ch%3%4%5_i[%6 %7]_t[%8 %9]/")
                .arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max)
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
                                    pImgObjVecCirc->at(i)->qrgbimg.scaled(checkList.nCircDiameter, checkList.nCircDiameter).
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
                            pImgObjVecCirc->at(0)->qrgbimg.scaled(checkList.nCircDiameter, checkList.nCircDiameter).
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
        if (m_pResultTab->getNirfDistCompDlg())
        {
            if (m_pResultTab->getNirfDistCompDlg()->isCompensating())
            {
                if (m_pResultTab->getNirfDistCompDlg()->isTBRMode())
                    nirfName = QString("/circ_image_dB[%1 %2]_tbr_nirf_i[%3 %4]/");
                else
                    nirfName = QString("/circ_image_dB[%1 %2]_comp_nirf_i[%3 %4]/");
            }
            else
                nirfName = QString("/circ_image_dB[%1 %2]_bg_sub_nirf_i[%3 %4]/");
        }
        else
        {
            nirfName = QString("/circ_image_dB[%1 %2]_raw_nirf_i[%3 %4]/");
        }
        QString circPath = m_pResultTab->m_path + nirfName.arg(m_pConfig->octDbRange.min).arg(m_pConfig->octDbRange.max)
                .arg(m_pConfig->nirfRange.min, 2, 'f', 1).arg(m_pConfig->nirfRange.max, 2, 'f', 1);
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
                        pImgObjVecCirc->at(0)->qrgbimg.scaled(checkList.nCircDiameter, checkList.nCircDiameter).
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
	                saveCompDetailsLog(circPath + "dist_comp_details.log");
    }
#endif
}

#ifdef OCT_NIRF
void SaveResultDlg::saveCompDetailsLog(const QString savepath)
{
    QFile file(savepath);
    file.open(QIODevice::WriteOnly | QIODevice::Text);
    QTextStream out(&file);

    QDate date = QDate::currentDate();
    QTime time = QTime::currentTime();

    out << QString("@ Created: %1-%2-%3 %4-%5-%6\n\n")
        .arg(date.year()).arg(date.month(), 2, 10, (QChar)'0').arg(date.day(), 2, 10, (QChar)'0')
        .arg(time.hour(), 2, 10, (QChar)'0').arg(time.minute(), 2, 10, (QChar)'0').arg(time.second(), 2, 10, (QChar)'0');

    out << QString("@ Compensation Curve: (a*exp(b*x)+c*exp(d*x))/(a+c)\n");
    out << QString("@ a: %1\n").arg(QString::number(m_pConfig->nirfCompCoeffs[0]));
    out << QString("@ b: %1\n").arg(QString::number(m_pConfig->nirfCompCoeffs[1]));
    out << QString("@ c: %1\n").arg(QString::number(m_pConfig->nirfCompCoeffs[2]));
    out << QString("@ d: %1\n\n").arg(QString::number(m_pConfig->nirfCompCoeffs[3]));

    out << QString("@ Factor Threshold: %1\n").arg(QString::number(m_pConfig->nirfFactorThres));
    out << QString("@ Factor Proportional Constant: %1\n").arg(QString::number(m_pConfig->nirfFactorPropConst));
    out << QString("@ Distance Proportional Constant: %1\n\n").arg(QString::number(m_pConfig->nirfDistPropConst));

    out << QString("@ NIRF Offset: %1\n").arg(QString::number(m_pResultTab->getCurrentNirfOffset()));
    out << QString("@ Lumen Contour Offset: %1\n").arg(QString::number(m_pConfig->nirfLumContourOffset));
    out << QString("@ Outer Sheath Position: %1\n").arg(QString::number(m_pConfig->nirfOuterSheathPos));
#ifdef GALVANO_MIRROR
    out << QString("@ Fast Scan Adjustment: %1\n\n").arg(QString::number(m_pConfig->galvoHorizontalShift));
#endif

    out << QString("@ NIRF Background: %1\n").arg(QString::number(m_pResultTab->getNirfDistCompDlg()->nirfBg));
    out << QString("@ NIRF Background for TBR: %1\n").arg(QString::number(m_pResultTab->getNirfDistCompDlg()->nirfBackgroundLevel));

    file.close();
}
#endif
