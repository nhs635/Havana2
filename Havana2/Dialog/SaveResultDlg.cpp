
#include "SaveResultDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QResultTab.h>
#include <Havana2/Viewer/QImageView.h>

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
	setFixedSize(420, 100);
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
	m_pLineEdit_RectWidth->setText(QString::number(m_pResultTab->m_vectorOctImage.at(0).size(0)));
	m_pLineEdit_RectWidth->setAlignment(Qt::AlignCenter);
	m_pLineEdit_RectWidth->setDisabled(true);
	m_pLineEdit_RectHeight = new QLineEdit(this);
	m_pLineEdit_RectHeight->setFixedWidth(35);
	m_pLineEdit_RectHeight->setText(QString::number(m_pResultTab->m_vectorOctImage.at(0).size(1)));
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

	// Save En Face Maps
	m_pPushButton_SaveEnFaceMaps = new QPushButton(this);
	m_pPushButton_SaveEnFaceMaps->setText("Save En Face Maps");

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

	m_pCheckBox_OctMaxProjection = new QCheckBox(this);
	m_pCheckBox_OctMaxProjection->setText("OCT Max Projection");
#ifdef OCT_FLIM
	m_pCheckBox_OctMaxProjection->setChecked(false);
#elif defined (STANDALONE_OCT)
	m_pCheckBox_OctMaxProjection->setChecked(true);
#endif

	// Set layout
	QGridLayout *pGridLayout = new QGridLayout;
	pGridLayout->setSpacing(3);

	pGridLayout->addWidget(m_pPushButton_SaveCrossSections, 0, 0);
	
	QHBoxLayout *pHBoxLayout_RectResize = new QHBoxLayout;
	pHBoxLayout_RectResize->addWidget(m_pCheckBox_ResizeRectImage);
	pHBoxLayout_RectResize->addWidget(m_pLineEdit_RectWidth);
	pHBoxLayout_RectResize->addWidget(m_pLineEdit_RectHeight);
	pHBoxLayout_RectResize->addStretch(1);
	//pHBoxLayout_RectResize->addItem(new QSpacerItem(0, 0, QSizePolicy::Preferred, QSizePolicy::Fixed));

	pGridLayout->addWidget(m_pCheckBox_RectImage, 0, 1);
	pGridLayout->addItem(pHBoxLayout_RectResize, 0, 2, 1, 2);

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

	pGridLayout->addWidget(m_pPushButton_SaveEnFaceMaps, 4, 0);

	pGridLayout->addWidget(m_pCheckBox_RawData, 4, 1);
	pGridLayout->addWidget(m_pCheckBox_ScaledImage, 4, 2);

#ifdef OCT_FLIM
	pGridLayout->addWidget(m_pCheckBox_EnFaceCh1, 5, 1);
	pGridLayout->addWidget(m_pCheckBox_EnFaceCh2, 5, 2);
	pGridLayout->addWidget(m_pCheckBox_EnFaceCh3, 5, 3);
#endif

	pGridLayout->addWidget(m_pCheckBox_OctMaxProjection, 6, 1, 1, 2);
	
    setLayout(pGridLayout);

    // Connect
	connect(m_pPushButton_SaveCrossSections, SIGNAL(clicked(bool)), this, SLOT(saveCrossSections()));
	connect(m_pPushButton_SaveEnFaceMaps, SIGNAL(clicked(bool)), this, SLOT(saveEnFaceMaps()));

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
		// Set Widgets //////////////////////////////////////////////////////////////////////////////
		emit setWidgets(false);
		emit m_pResultTab->setWidgets(false);
		m_nSavedFrames = 0;

		// Scaling Images ///////////////////////////////////////////////////////////////////////////
#ifdef OCT_FLIM
		std::thread scaleImages([&]() { scaling(m_pResultTab->m_vectorOctImage, intensityMap, lifetimeMap, checkList); });
#elif defined (STANDALONE_OCT)
		std::thread scaleImages([&]() { scaling(m_pResultTab->m_vectorOctImage); });
#endif
#ifdef OCT_FLIM
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
#ifdef OCT_FLIM
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
		checkList.bOctProj = m_pCheckBox_OctMaxProjection->isChecked();

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
					QFile fileIntensity(enFacePath + QString("intensity_ch%1.enface").arg(i + 1));
					if (false != fileIntensity.open(QIODevice::WriteOnly))
					{
						fileIntensity.write(reinterpret_cast<char*>(m_pResultTab->m_intensityMap.at(i).raw_ptr()), sizeof(float) * m_pResultTab->m_intensityMap.at(i).length());
						fileIntensity.close();
					}

					QFile fileLifetime(enFacePath + QString("lifetime_ch%1.enface").arg(i + 1));
					if (false != fileLifetime.open(QIODevice::WriteOnly))
					{
						fileLifetime.write(reinterpret_cast<char*>(m_pResultTab->m_lifetimeMap.at(i).raw_ptr()), sizeof(float) * m_pResultTab->m_lifetimeMap.at(i).length());
						fileLifetime.close();
					}
				}
			}
#endif
			if (checkList.bOctProj)
			{
				QFile fileOctMaxProj(enFacePath + "oct_max_projection.enface");
				if (false != fileOctMaxProj.open(QIODevice::WriteOnly))
				{
					fileOctMaxProj.write(reinterpret_cast<char*>(m_pResultTab->m_octProjection.raw_ptr()), sizeof(float) * m_pResultTab->m_octProjection.length());
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
					imgObjIntensity.qindeximg.save(enFacePath + QString("intensity_ch%1_[%2 %3].bmp").arg(i + 1)
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
					imgObjLifetime.qindeximg.save(enFacePath + QString("lifetime_ch%1_[%2 %3].bmp").arg(i + 1)
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
					imgObjHsv.qrgbimg.save(enFacePath + QString("flim_map_ch%1_i[%2 %3]_t[%4 %5].bmp").arg(i + 1)
						.arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1)
						.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1), "bmp");
				}
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
						std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_proj.width);
					}
				}
#endif
				imgObjOctMaxProj.qindeximg.save(enFacePath + "oct_max_projection.bmp", "bmp");
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

	// Save En Face Maps
	m_pPushButton_SaveEnFaceMaps->setEnabled(enabled);
	m_pCheckBox_RawData->setEnabled(enabled);
	m_pCheckBox_ScaledImage->setEnabled(enabled);
#ifdef OCT_FLIM
	m_pCheckBox_EnFaceCh1->setEnabled(enabled);
	m_pCheckBox_EnFaceCh2->setEnabled(enabled);
	m_pCheckBox_EnFaceCh3->setEnabled(enabled);
#endif
	m_pCheckBox_OctMaxProjection->setEnabled(enabled);
}


#ifdef OCT_FLIM
void SaveResultDlg::scaling(std::vector<np::FloatArray2>& vectorOctImage, 
	std::vector<np::Uint8Array2>& intensityMap, std::vector<np::Uint8Array2>& lifetimeMap, CrossSectionCheckList checkList)
#elif defined (STANDALONE_OCT)
void SaveResultDlg::scaling(std::vector<np::FloatArray2>& vectorOctImage)
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

		// OCT Visualization
		np::Uint8Array2 scale_temp(roi_oct.width, roi_oct.height);
		ippiScale_32f8u_C1R(vectorOctImage.at(frameCount), roi_oct.width * sizeof(float),
			scale_temp.raw_ptr(), roi_oct.width * sizeof(uint8_t), roi_oct, m_pConfig->octDbRange.min, m_pConfig->octDbRange.max);
		ippiTranspose_8u_C1R(scale_temp.raw_ptr(), roi_oct.width * sizeof(uint8_t), pImgObjVec->at(0)->arr.raw_ptr(), roi_oct.height * sizeof(uint8_t), roi_oct);
#ifdef GALVANO_MIRROR
		if (m_pConfig->galvoHorizontalShift)
		{
			for (int i = 0; i < roi_oct.width; i++)
			{
				uint8_t* pImg = pImgObjVec->at(0)->arr.raw_ptr() + i * roi_oct.height;
				std::rotate(pImg, pImg + m_pConfig->galvoHorizontalShift, pImg + roi_oct.height);
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
		frameCount++;

		// Push the buffers to sync Queues
#ifdef OCT_FLIM
		m_syncQueueConverting.push(pImgObjVec);
#elif defined (STANDALONE_OCT)
		m_syncQueueRectWriting.push(pImgObjVec);
#endif
	}
}

#ifdef OCT_FLIM
void SaveResultDlg::converting(CrossSectionCheckList checkList)
{
	int nTotalFrame = (int)m_pResultTab->m_vectorOctImage.size();

	int frameCount = 0;
	while (frameCount < nTotalFrame)
	{
		// Get the buffer from the previous sync Queue
		ImgObjVector *pImgObjVec = m_syncQueueConverting.pop();

		// Converting RGB
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

#ifdef OCT_FLIM
	if (!checkList.bCh[0] && !checkList.bCh[1] && !checkList.bCh[2])
	{
#endif
		QString rectPath;
		rectPath = m_pResultTab->m_path + "/rect_image/";
		if (checkList.bRect) QDir().mkdir(rectPath);

		int frameCount = 0;
		while (frameCount < nTotalFrame)
		{
			// Get the buffer from the previous sync Queue
			ImgObjVector *pImgObjVec = m_syncQueueRectWriting.pop();	

			// Write rect images
			if (checkList.bRect)
			{			
				if (!checkList.bRectResize)
					pImgObjVec->at(0)->qindeximg.save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
				else
					pImgObjVec->at(0)->qindeximg.scaled(checkList.nRectWidth, checkList.nRectHeight).
						save(rectPath + QString("rect_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
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
					rectPath[i] = m_pResultTab->m_path + QString("/rect_image_ch%1_i[%2 %3]_t[%4 %5]/")
						.arg(i + 1).arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1)
						.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1);
					if (checkList.bRect) QDir().mkdir(rectPath[i]);
				}
			}
		}
		else
		{
			rectPath[0] = m_pResultTab->m_path + QString("/rect_merged_ch%1%2%3_i[%4 %5]_t[%6 %7]/")
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
			
			if (!checkList.bMulti)
			{
				for (int i = 0; i < 3; i++)
				{
					if (checkList.bCh[i])
					{
						// Paste FLIM color ring to RGB rect image
						memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - 2 * RING_THICKNESS - 1),
							pImgObjVec->at(1 + 2 * i)->qrgbimg.bits(), pImgObjVec->at(1 + 2 * i)->qrgbimg.byteCount()); // Intensity
						memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - 1 * RING_THICKNESS - 1),
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
						memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (pImgObjVec->at(0)->arr.size(1) - (nCh - n++) * RING_THICKNESS - 1),
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

			frameCount++;
			emit savedSingleFrame(m_nSavedFrames++);

			// Push the buffers to sync Queues
			m_syncQueueCircularizing.push(pImgObjVec);
		}
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

#ifdef OCT_FLIM
		if (!checkList.bCh[0] && !checkList.bCh[1] && !checkList.bCh[2])
		{
#endif
			// ImageObject for circ writing
			ImageObject *pCircImgObj = new ImageObject(2 * CIRC_RADIUS, 2 * CIRC_RADIUS, temp_ctable.m_colorTableVector.at(m_pResultTab->getCurrentOctColorTable()));

			// Circularize
			if (checkList.bCirc)
			{
				np::Uint8Array2 rect_temp(pImgObjVec->at(0)->qindeximg.bits(), pImgObjVec->at(0)->arr.size(0), pImgObjVec->at(0)->arr.size(1));
				(*m_pResultTab->m_pCirc)(rect_temp, pCircImgObj->qindeximg.bits(), "vertical", m_pConfig->circCenter);
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
						memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (m_pConfig->circCenter + CIRC_RADIUS - 2 * RING_THICKNESS),
							pImgObjVec->at(1 + 2 * i)->qrgbimg.bits(), pImgObjVec->at(1 + 2 * i)->qrgbimg.byteCount()); // Intensity
						memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (m_pConfig->circCenter + CIRC_RADIUS - 1 * RING_THICKNESS),
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
						memcpy(pImgObjVec->at(0)->qrgbimg.bits() + 3 * pImgObjVec->at(0)->arr.size(0) * (m_pConfig->circCenter + CIRC_RADIUS - (nCh - n++) * RING_THICKNESS),
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

#ifdef OCT_FLIM
	if (!checkList.bCh[0] && !checkList.bCh[1] && !checkList.bCh[2])
	{
#endif
		QString circPath;
		circPath = m_pResultTab->m_path + "/circ_image/";
		if (checkList.bCirc) QDir().mkdir(circPath);

		int frameCount = 0;
		while (frameCount < nTotalFrame)
		{
			// Get the buffer from the previous sync Queue
			ImgObjVector *pImgObjVecCirc = m_syncQueueCircWriting.pop();

			// Write circ images
			if (checkList.bCirc)
			{
				if (!checkList.bCircResize)
					pImgObjVecCirc->at(0)->qindeximg.save(circPath + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
				else
					pImgObjVecCirc->at(0)->qindeximg.scaled(checkList.nCircDiameter, checkList.nCircDiameter).
						save(circPath + QString("circ_%1_%2.bmp").arg(folderName).arg(frameCount + 1, 3, 10, (QChar)'0'), "bmp");
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
					circPath[i] = m_pResultTab->m_path + QString("/circ_image_ch%1_i[%2 %3]_t[%4 %5]/")
						.arg(i + 1).arg(m_pConfig->flimIntensityRange.min, 2, 'f', 1).arg(m_pConfig->flimIntensityRange.max, 2, 'f', 1)
						.arg(m_pConfig->flimLifetimeRange.min, 2, 'f', 1).arg(m_pConfig->flimLifetimeRange.max, 2, 'f', 1);
					if (checkList.bCirc) QDir().mkdir(circPath[i]);
				}
			}
		}
		else
		{
			circPath[0] = m_pResultTab->m_path + QString("/circ_merged_ch%1%2%3_i[%4 %5]_t[%6 %7]/")
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
#endif
}