
#include "LongitudinalViewDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QResultTab.h>

#include <ipps.h>
#include <ippi.h>
#include <ippcore.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>


LongitudinalViewDlg::LongitudinalViewDlg(QWidget *parent) :
    QDialog(parent), 
	m_bCanBeClosed(true),
	m_pImgObjOctLongiImage(nullptr)
#ifdef OCT_FLIM
	, m_pImgObjIntensity(nullptr), m_pImgObjLifetime(nullptr)
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	, m_pImgObjNirf(nullptr)
#else
	, m_pImgObjNirf1(nullptr), m_pImgObjNirf2(nullptr)
#endif	
#endif
#endif
	, m_pMedfilt(nullptr)
{
    // Set default size & frame
    setMinimumSize(700, 300);
    setWindowFlags(Qt::Tool);
	setWindowTitle("Longitudinal View");

    // Set main window objects
    m_pResultTab = (QResultTab*)parent;
    m_pMainWnd = m_pResultTab->getMainWnd();
	m_pConfig = m_pMainWnd->m_pConfiguration;
	
	// Create image view buffers
	ColorTable temp_ctable;	
	m_pImgObjOctLongiImage = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pResultTab->getConfigTemp()->circRadius, temp_ctable.m_colorTableVector.at(m_pConfig->octColorTable));
	
#ifdef OCT_FLIM
	m_pImgObjIntensity = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
	m_pImgObjLifetime = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(m_pConfig->flimLifetimeColorTable));
	m_pImgObjHsvEnhanced = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(m_pConfig->flimLifetimeColorTable));
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	m_pImgObjNirf = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
#else
	m_pImgObjNirf1 = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
	m_pImgObjNirf2 = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE2));
#endif
#endif

	// Create medfilt
	m_pMedfilt = new medfilt(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pResultTab->getConfigTemp()->circRadius, 3, 3); // median filtering kernel


	// Create widgets
#ifdef OCT_FLIM
	bool rgb_used = true;
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
	bool rgb_used = false;
#else
	bool rgb_used = true;
#endif
#endif
	m_pImageView_LongitudinalView = new QImageView(ColorTable::colortable(m_pConfig->octColorTable), m_pResultTab->getConfigTemp()->nFrames, 2 * m_pResultTab->getConfigTemp()->circRadius, rgb_used);
	m_pImageView_LongitudinalView->setMinimumWidth(600);
	m_pImageView_LongitudinalView->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
	m_pImageView_LongitudinalView->setVLineChangeCallback([&](int frame) { m_pResultTab->setCurrentFrame(frame); });
	m_pImageView_LongitudinalView->setVerticalLine(1, m_pResultTab->getCurrentFrame());
	m_pImageView_LongitudinalView->getRender()->update();

	m_pLabel_CurrentAline = new QLabel(this);
	m_pLabel_CurrentAline->setFixedWidth(250);
	QString str; str.sprintf("Current A-line : (%4d, %4d) / %4d (%3.2f deg)  ", 
		1, m_pResultTab->getConfigTemp()->nAlines / 2 + 1, m_pResultTab->getConfigTemp()->nAlines, 0);
	m_pLabel_CurrentAline->setText(str);

	m_pSlider_CurrentAline = new QSlider(this);
	m_pSlider_CurrentAline->setOrientation(Qt::Horizontal);
	m_pSlider_CurrentAline->setRange(0, m_pResultTab->getConfigTemp()->nAlines / 2 - 1);
	m_pSlider_CurrentAline->setValue(0);
	
	// Create layout
	QGridLayout *pGridLayout = new QGridLayout;
	pGridLayout->setSpacing(3);

	pGridLayout->addWidget(m_pImageView_LongitudinalView, 0, 0, 1, 2);
	pGridLayout->addWidget(m_pLabel_CurrentAline, 1, 0);
	pGridLayout->addWidget(m_pSlider_CurrentAline, 1, 1);

	// Set layout
	this->setLayout(pGridLayout);

	// Connect
	connect(m_pSlider_CurrentAline, SIGNAL(valueChanged(int)), this, SLOT(drawLongitudinalImage(int)));

#ifdef OCT_FLIM
	connect(this, SIGNAL(paintLongiImage(uint8_t*)), m_pImageView_LongitudinalView, SLOT(drawRgbImage(uint8_t*)));
#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	connect(this, SIGNAL(paintLongiImage(uint8_t*)), m_pImageView_LongitudinalView, SLOT(drawRgbImage(uint8_t*)));
#else
	connect(this, SIGNAL(paintLongiImage(uint8_t*)), m_pImageView_LongitudinalView, SLOT(drawImage(uint8_t*)));
#endif
#endif
	connect(this, SIGNAL(setWidgets(bool)), this, SLOT(setWidgetEnabled(bool)));
}

LongitudinalViewDlg::~LongitudinalViewDlg()
{
	if (m_pImgObjOctLongiImage) delete m_pImgObjOctLongiImage;
#ifdef OCT_FLIM
	if (m_pImgObjIntensity) delete m_pImgObjIntensity;
	if (m_pImgObjLifetime) delete m_pImgObjLifetime;
	if (m_pImgObjHsvEnhanced) delete m_pImgObjHsvEnhanced;
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
}

void LongitudinalViewDlg::closeEvent(QCloseEvent * e)
{
	if (!m_bCanBeClosed)
		e->ignore();
	else
		finished(0);
}

void LongitudinalViewDlg::keyPressEvent(QKeyEvent *e)
{
	if (e->key() != Qt::Key_Escape)
		QDialog::keyPressEvent(e);
}


void LongitudinalViewDlg::setWidgetEnabled(bool enabled)
{
	m_pImageView_LongitudinalView->setEnabled(enabled);
	m_pLabel_CurrentAline->setEnabled(enabled);
	m_pSlider_CurrentAline->setEnabled(enabled);
}

void LongitudinalViewDlg::setLongiRadius(int circ_radius)
{
	m_pImageView_LongitudinalView->resetSize(m_pResultTab->getConfigTemp()->nFrames, 2 * circ_radius);

	// Create image view buffers
	ColorTable temp_ctable;

	if (m_pImgObjOctLongiImage)
	{
		delete m_pImgObjOctLongiImage;
		m_pImgObjOctLongiImage = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * circ_radius, temp_ctable.m_colorTableVector.at(m_pConfig->octColorTable));
	}

	// Create medfilt
	if (m_pMedfilt)
	{
		delete m_pMedfilt;
		m_pMedfilt = new medfilt(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * circ_radius, 3, 3); // median filtering kernel
	}

	drawLongitudinalImage(getCurrentAline());
}

#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
void LongitudinalViewDlg::setLongiRingThickness(int ring_thickness)
{
	// Create image view buffers
	ColorTable temp_ctable;

#ifdef OCT_FLIM
	if (m_pImgObjIntensity)
	{
		delete m_pImgObjIntensity;
		m_pImgObjIntensity = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(INTENSITY_COLORTABLE));
	}
	if (m_pImgObjLifetime)
	{
		delete m_pImgObjLifetime;
		m_pImgObjLifetime = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(m_pConfig->flimLifetimeColorTable));
	}
	if (m_pImgObjHsvEnhanced)
	{
		delete m_pImgObjHsvEnhanced;
		m_pImgObjHsvEnhanced = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(m_pConfig->flimLifetimeColorTable));
	}
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	if (m_pImgObjNirf)
	{
		delete m_pImgObjNirf;
		m_pImgObjNirf = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
	}
#else
	if (m_pImgObjNirf1)
	{
		delete m_pImgObjNirf1;
		m_pImgObjNirf1 = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE1));
	}
	if (m_pImgObjNirf2)
	{
		delete m_pImgObjNirf2;
		m_pImgObjNirf2 = new ImageObject(((m_pResultTab->getConfigTemp()->nFrames + 3) >> 2) << 2, 2 * m_pConfig->ringThickness, temp_ctable.m_colorTableVector.at(NIRF_COLORTABLE2));
	}
#endif
#endif

	drawLongitudinalImage(getCurrentAline());

	(void)ring_thickness;
}
#endif

void LongitudinalViewDlg::drawLongitudinalImage(int aline)
{	
	// Make longitudinal - OCT
	IppiSize roi_longi = { m_pImgObjOctLongiImage->getHeight(), m_pImgObjOctLongiImage->getWidth() };

	np::FloatArray2 longi_temp(roi_longi.width, roi_longi.height);
	np::Uint8Array2 scale_temp(roi_longi.width, roi_longi.height);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)(m_pResultTab->getConfigTemp()->nFrames)),
		[&](const tbb::blocked_range<size_t>& r) {
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			int center = (!m_pResultTab->getPolishedSurfaceFindingStatus()) ? m_pConfig->circCenter :
				(m_pResultTab->getConfigTemp()->n2ScansFFT / 2 - m_pResultTab->getConfigTemp()->nScans / 4) + m_pResultTab->m_polishedSurface((int)i) - m_pConfig->ballRadius;
			memcpy(&longi_temp(0, (int)i), &m_pResultTab->m_vectorOctImage.at((int)i)(center, aline), sizeof(float) * m_pResultTab->getConfigTemp()->circRadius);
			memcpy(&longi_temp(m_pResultTab->getConfigTemp()->circRadius, (int)i), &m_pResultTab->m_vectorOctImage.at((int)i)(center, m_pResultTab->getConfigTemp()->nAlines / 2 + aline), sizeof(float) * m_pResultTab->getConfigTemp()->circRadius);
			ippsFlip_32f_I(&longi_temp(0, (int)i), m_pResultTab->getConfigTemp()->circRadius);
		}
	});
	ippiScale_32f8u_C1R(longi_temp.raw_ptr(), roi_longi.width * sizeof(float),
		scale_temp.raw_ptr(), roi_longi.width * sizeof(uint8_t), roi_longi, m_pConfig->octDbRange.min, m_pConfig->octDbRange.max);
	ippiTranspose_8u_C1R(scale_temp.raw_ptr(), roi_longi.width * sizeof(uint8_t), m_pImgObjOctLongiImage->arr.raw_ptr(), roi_longi.height * sizeof(uint8_t), roi_longi);
	(*m_pMedfilt)(m_pImgObjOctLongiImage->arr.raw_ptr());
	
#ifdef OCT_FLIM		
	m_pImgObjOctLongiImage->convertRgb();

	// Make longitudinal - FLIM
	tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)(m_pResultTab->getConfigTemp()->nFrames)),
		[&](const tbb::blocked_range<size_t>& r) {
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			m_pImgObjIntensity->arr((int)i, 0) = m_pResultTab->m_pImgObjIntensityMap->arr(aline / 4, (int)(m_pResultTab->getConfigTemp()->nFrames - i - 1));
			m_pImgObjIntensity->arr((int)i, m_pConfig->ringThickness) = m_pResultTab->m_pImgObjIntensityMap->arr(m_pResultTab->getConfigTemp()->n4Alines / 2 + aline / 4, (int)(m_pResultTab->getConfigTemp()->nFrames - i - 1));
			m_pImgObjLifetime->arr((int)i, 0) = m_pResultTab->m_pImgObjLifetimeMap->arr(aline / 4, (int)(m_pResultTab->getConfigTemp()->nFrames - i - 1));
			m_pImgObjLifetime->arr((int)i, m_pConfig->ringThickness) = m_pResultTab->m_pImgObjLifetimeMap->arr(m_pResultTab->getConfigTemp()->n4Alines / 2 + aline / 4, (int)(m_pResultTab->getConfigTemp()->nFrames - i - 1));
		}
	});
	tbb::parallel_for(tbb::blocked_range<size_t>(1, (size_t)(m_pConfig->ringThickness)),
		[&](const tbb::blocked_range<size_t>& r) {
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			memcpy(&m_pImgObjIntensity->arr(0, (int)i), &m_pImgObjIntensity->arr(0, 0), sizeof(uint8_t) * m_pImgObjIntensity->arr.size(0));
			memcpy(&m_pImgObjIntensity->arr(0, (int)(i + m_pConfig->ringThickness)), &m_pImgObjIntensity->arr(0, m_pConfig->ringThickness), sizeof(uint8_t) * m_pImgObjIntensity->arr.size(0));
			memcpy(&m_pImgObjLifetime->arr(0, (int)i), &m_pImgObjLifetime->arr(0, 0), sizeof(uint8_t) * m_pImgObjLifetime->arr.size(0));
			memcpy(&m_pImgObjLifetime->arr(0, (int)(i + m_pConfig->ringThickness)), &m_pImgObjLifetime->arr(0, m_pConfig->ringThickness), sizeof(uint8_t) * m_pImgObjLifetime->arr.size(0));
		}
	});
	
	if (!m_pResultTab->isHsvEnhanced())
	{
		m_pImgObjIntensity->convertNonScaledRgb();
		m_pImgObjLifetime->convertNonScaledRgb();

		memcpy(m_pImgObjOctLongiImage->qrgbimg.bits(), m_pImgObjLifetime->qrgbimg.bits(), m_pImgObjLifetime->qrgbimg.byteCount() / 2);
		memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * m_pConfig->ringThickness, m_pImgObjIntensity->qrgbimg.bits(), m_pImgObjIntensity->qrgbimg.byteCount() / 2);

		memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * (m_pImgObjOctLongiImage->arr.size(1) - 2 * m_pConfig->ringThickness),
			m_pImgObjIntensity->qrgbimg.bits() + m_pImgObjIntensity->qrgbimg.byteCount() / 2, m_pImgObjIntensity->qrgbimg.byteCount() / 2);
		memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * (m_pImgObjOctLongiImage->arr.size(1) - 1 * m_pConfig->ringThickness),
			m_pImgObjLifetime->qrgbimg.bits() + m_pImgObjLifetime->qrgbimg.byteCount() / 2, m_pImgObjLifetime->qrgbimg.byteCount() / 2);
	}
	else
	{
		// Non HSV intensity-weight map
		ColorTable temp_ctable;
		ImageObject tempImgObj(m_pImgObjHsvEnhanced->getWidth(), m_pImgObjHsvEnhanced->getHeight(), temp_ctable.m_colorTableVector.at(ColorTable::gray));

		m_pImgObjLifetime->convertRgb();
		memcpy(tempImgObj.qindeximg.bits(), m_pImgObjIntensity->arr.raw_ptr(), tempImgObj.qindeximg.byteCount());
		tempImgObj.convertRgb();

		ippsMul_8u_Sfs(m_pImgObjLifetime->qrgbimg.bits(), tempImgObj.qrgbimg.bits(), m_pImgObjHsvEnhanced->qrgbimg.bits(), tempImgObj.qrgbimg.byteCount(), 8);
		
		memcpy(m_pImgObjOctLongiImage->qrgbimg.bits(), m_pImgObjHsvEnhanced->qrgbimg.bits(), m_pImgObjHsvEnhanced->qrgbimg.byteCount() / 2);
		memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * (m_pImgObjOctLongiImage->arr.size(1) - 1 * m_pConfig->ringThickness),
			m_pImgObjHsvEnhanced->qrgbimg.bits() + m_pImgObjHsvEnhanced->qrgbimg.byteCount() / 2, m_pImgObjHsvEnhanced->qrgbimg.byteCount() / 2);
	}

	emit paintLongiImage(m_pImgObjOctLongiImage->qrgbimg.bits());

#elif defined (STANDALONE_OCT)
#ifdef OCT_NIRF
	m_pImgObjOctLongiImage->convertRgb();

	// Make longitudinla - NIRF
#ifndef TWO_CHANNEL_NIRF
	tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)(m_pResultTab->getConfigTemp()->nFrames)),
		[&](const tbb::blocked_range<size_t>& r) {
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			m_pImgObjNirf->arr((int)i,              0) = m_pResultTab->m_pImgObjNirfMap->arr(aline, (int)(m_pResultTab->getConfigTemp()->nFrames - i - 1));
			m_pImgObjNirf->arr((int)i, m_pConfig->ringThickness) = m_pResultTab->m_pImgObjNirfMap->arr(m_pResultTab->getConfigTemp()->nAlines / 2 + aline, (int)(m_pResultTab->getConfigTemp()->nFrames - i - 1));
		}
	});	
	tbb::parallel_for(tbb::blocked_range<size_t>(1, (size_t)(m_pConfig->ringThickness)),
		[&](const tbb::blocked_range<size_t>& r) {
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			memcpy(&m_pImgObjNirf->arr(0, (int)i), &m_pImgObjNirf->arr(0, 0), sizeof(uint8_t) * m_pImgObjNirf->arr.size(0));
			memcpy(&m_pImgObjNirf->arr(0, (int)(i + m_pConfig->ringThickness)), &m_pImgObjNirf->arr(0, m_pConfig->ringThickness), sizeof(uint8_t) * m_pImgObjNirf->arr.size(0));
		}
	});

	m_pImgObjNirf->convertNonScaledRgb();

	memcpy(m_pImgObjOctLongiImage->qrgbimg.bits(), m_pImgObjNirf->qrgbimg.bits(), m_pImgObjNirf->qrgbimg.byteCount() / 2);
	memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * (m_pImgObjOctLongiImage->arr.size(1) - 1 * m_pConfig->ringThickness),
		m_pImgObjNirf->qrgbimg.bits() + m_pImgObjNirf->qrgbimg.byteCount() / 2, m_pImgObjNirf->qrgbimg.byteCount() / 2);
#else
	tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)(m_pResultTab->getConfigTemp()->nFrames)),
		[&](const tbb::blocked_range<size_t>& r) {
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			m_pImgObjNirf1->arr((int)i, 0) = m_pResultTab->m_pImgObjNirfMap1->arr(aline, (int)(m_pResultTab->getConfigTemp()->nFrames - i - 1));
			m_pImgObjNirf1->arr((int)i, m_pConfig->ringThickness) = m_pResultTab->m_pImgObjNirfMap1->arr(m_pResultTab->getConfigTemp()->nAlines / 2 + aline, (int)(m_pResultTab->getConfigTemp()->nFrames - i - 1));
			m_pImgObjNirf2->arr((int)i, 0) = m_pResultTab->m_pImgObjNirfMap2->arr(aline, (int)(m_pResultTab->getConfigTemp()->nFrames - i - 1));
			m_pImgObjNirf2->arr((int)i, m_pConfig->ringThickness) = m_pResultTab->m_pImgObjNirfMap2->arr(m_pResultTab->getConfigTemp()->nAlines / 2 + aline, (int)(m_pResultTab->getConfigTemp()->nFrames - i - 1));
		}
	});
	tbb::parallel_for(tbb::blocked_range<size_t>(1, (size_t)(m_pConfig->ringThickness)),
		[&](const tbb::blocked_range<size_t>& r) {
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			memcpy(&m_pImgObjNirf1->arr(0, (int)i), &m_pImgObjNirf1->arr(0, 0), sizeof(uint8_t) * m_pImgObjNirf1->arr.size(0));
			memcpy(&m_pImgObjNirf1->arr(0, (int)(i + m_pConfig->ringThickness)), &m_pImgObjNirf1->arr(0, m_pConfig->ringThickness), sizeof(uint8_t) * m_pImgObjNirf1->arr.size(0));
			memcpy(&m_pImgObjNirf2->arr(0, (int)i), &m_pImgObjNirf2->arr(0, 0), sizeof(uint8_t) * m_pImgObjNirf2->arr.size(0));
			memcpy(&m_pImgObjNirf2->arr(0, (int)(i + m_pConfig->ringThickness)), &m_pImgObjNirf2->arr(0, m_pConfig->ringThickness), sizeof(uint8_t) * m_pImgObjNirf2->arr.size(0));
		}
	});

	m_pImgObjNirf1->convertNonScaledRgb();
	m_pImgObjNirf2->convertNonScaledRgb();

	memcpy(m_pImgObjOctLongiImage->qrgbimg.bits(), m_pImgObjNirf2->qrgbimg.bits(), m_pImgObjNirf2->qrgbimg.byteCount() / 2);
	memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * m_pConfig->ringThickness, m_pImgObjNirf1->qrgbimg.bits(), m_pImgObjNirf1->qrgbimg.byteCount() / 2);

	memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * (m_pImgObjOctLongiImage->arr.size(1) - 2 * m_pConfig->ringThickness),
		m_pImgObjNirf1->qrgbimg.bits() + m_pImgObjNirf1->qrgbimg.byteCount() / 2, m_pImgObjNirf1->qrgbimg.byteCount() / 2);
	memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * (m_pImgObjOctLongiImage->arr.size(1) - 1 * m_pConfig->ringThickness),
		m_pImgObjNirf2->qrgbimg.bits() + m_pImgObjNirf2->qrgbimg.byteCount() / 2, m_pImgObjNirf2->qrgbimg.byteCount() / 2);

#ifdef CH_DIVIDING_LINE
	np::Uint8Array boundary(3 * m_pImgObjNirf1->arr.size(0));
	ippsSet_8u(255, boundary.raw_ptr(), boundary.length());

	memcpy(m_pImgObjOctLongiImage->qrgbimg.bits(), boundary.raw_ptr(), sizeof(uint8_t) * 3 * m_pImgObjNirf1->arr.size(0));
	memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * m_pConfig->ringThickness, boundary.raw_ptr(), sizeof(uint8_t) * 3 * m_pImgObjNirf1->arr.size(0));
	memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * (2 * m_pConfig->ringThickness - 1), boundary.raw_ptr(), sizeof(uint8_t) * 3 * m_pImgObjNirf1->arr.size(0));
	
	memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * (m_pImgObjOctLongiImage->arr.size(1) - 2 * m_pConfig->ringThickness), boundary.raw_ptr(), sizeof(uint8_t) * 3 * m_pImgObjNirf1->arr.size(0));
	memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * (m_pImgObjOctLongiImage->arr.size(1) - 1 * m_pConfig->ringThickness), boundary.raw_ptr(), sizeof(uint8_t) * 3 * m_pImgObjNirf1->arr.size(0));
	memcpy(m_pImgObjOctLongiImage->qrgbimg.bits() + 3 * m_pImgObjOctLongiImage->arr.size(0) * (m_pImgObjOctLongiImage->arr.size(1) - 1), boundary.raw_ptr(), sizeof(uint8_t) * 3 * m_pImgObjNirf1->arr.size(0));
#endif
#endif

	emit paintLongiImage(m_pImgObjOctLongiImage->qrgbimg.bits());
#else
	emit paintLongiImage(m_pImgObjOctLongiImage->qindeximg.bits());
#endif
#endif
	
	// Widgets updates
	QString str; str.sprintf("Current A-line : (%4d, %4d) / %4d (%3.2f deg)  ",
		aline + 1, aline + m_pResultTab->getConfigTemp()->nAlines / 2 + 1, m_pResultTab->getConfigTemp()->nAlines,
		360.0 * (double)aline / (double)m_pResultTab->getConfigTemp()->nAlines);
	m_pLabel_CurrentAline->setText(str);

	if (m_pResultTab->getRectImageView()->isVisible())
	{
		m_pResultTab->getRectImageView()->setVerticalLine(1, aline); 
		m_pResultTab->getRectImageView()->getRender()->update();
	}
	else
	{
		m_pResultTab->getCircImageView()->setVerticalLine(1, aline);
		m_pResultTab->getCircImageView()->getRender()->update();
	}
}