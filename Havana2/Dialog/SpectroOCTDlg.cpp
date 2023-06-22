
#include "SpectroOCTDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QResultTab.h>

#ifndef CUDA_ENABLED
#include <DataProcess/OCTProcess/SOCTProcess.h>
#else
#include <DataProcess/OCTProcess/SOCTProcess.h>
#endif

#include <ipps.h>
#include <ippi.h>
#include <ippcore.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>


SpectroOCTDlg::SpectroOCTDlg(QWidget *parent) :
	QDialog(parent),
	m_bCanBeClosed(true), subject_frame(-1),
	m_pImgObjSpectroOCT(nullptr)
{
	// Set default size & frame
	setMinimumSize(600, 600);
	setWindowFlags(Qt::Tool);
	setWindowTitle("Spectroscopic OCT");

	// Set main window objects
	m_pResultTab = (QResultTab*)parent;
	m_pMainWnd = m_pResultTab->getMainWnd();
	m_pConfig = m_pMainWnd->m_pConfiguration;

	// Create image viewing widgets
	m_pImageView_SpectroOCTView = new QImageView(ColorTable::colortable::jet, 200, m_pResultTab->getConfigTemp()->n2ScansFFT);
	m_pImageView_SpectroOCTView->setMinimumWidth(400);
	m_pImageView_SpectroOCTView->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
	m_pImageView_SpectroOCTView->getRender()->update();

	// Create S-OCT relevant widgets
	m_pPushButton_Extract = new QPushButton(this);
	m_pPushButton_Extract->setText("Extract");

	m_pPushButton_Save = new QPushButton(this);
	m_pPushButton_Save->setText("Save");
	m_pPushButton_Save->setDisabled(true);

	m_pProgressBar = new QProgressBar(this);
	m_pProgressBar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	m_pProgressBar->setFormat("");
	m_pProgressBar->setValue(0);

	m_pCheckBox_AutoExtraction = new QCheckBox(this);
	m_pCheckBox_AutoExtraction->setText("Auto Extraction");
	m_pCheckBox_AutoExtraction->setDisabled(true);

	m_pLineEdit_Window = new QLineEdit(this);
	m_pLineEdit_Window->setFixedWidth(30);
	m_pLineEdit_Window->setText(QString::number(m_pConfig->spectroWindow));
	m_pLineEdit_Window->setAlignment(Qt::AlignCenter);

	m_pLabel_Window = new QLabel("Window Size ", this);
	m_pLabel_Window->setBuddy(m_pLineEdit_Window);

	m_pLineEdit_Overlap = new QLineEdit(this);
	m_pLineEdit_Overlap->setFixedWidth(30);
	m_pLineEdit_Overlap->setText(QString::number(m_pConfig->spectroOverlap));
	m_pLineEdit_Overlap->setAlignment(Qt::AlignCenter);

	m_pLabel_Overlap = new QLabel("Overlap Size ", this);
	m_pLabel_Overlap->setBuddy(m_pLineEdit_Overlap);

	m_pLineEdit_InPlane = new QLineEdit(this);
	m_pLineEdit_InPlane->setFixedWidth(30);
	m_pLineEdit_InPlane->setText(QString::number(m_pConfig->spectroInPlaneAvgSize));
	m_pLineEdit_InPlane->setAlignment(Qt::AlignCenter);

	m_pLabel_InPlane = new QLabel("In-plane Avg Size ", this);
	m_pLabel_InPlane->setBuddy(m_pLineEdit_InPlane);

	m_pLineEdit_OutOfPlane = new QLineEdit(this);
	m_pLineEdit_OutOfPlane->setFixedWidth(30);
	m_pLineEdit_OutOfPlane->setText(QString::number(m_pConfig->spectroOutOfPlaneAvgSize));
	m_pLineEdit_OutOfPlane->setAlignment(Qt::AlignCenter);

	m_pLabel_OutOfPlane = new QLabel("Out-of-plane Avg Size ", this);
	m_pLabel_OutOfPlane->setBuddy(m_pLineEdit_Overlap);

	m_pLineEdit_RoiDepth = new QLineEdit(this);
	m_pLineEdit_RoiDepth->setFixedWidth(30);
	m_pLineEdit_RoiDepth->setText(QString::number(m_pConfig->spectroRoiDepth));
	m_pLineEdit_RoiDepth->setAlignment(Qt::AlignCenter);

	m_pLabel_RoiDepth = new QLabel("ROI Depth ", this);
	m_pLabel_RoiDepth->setBuddy(m_pLineEdit_Overlap);

	// Create color bar widgets
	uint8_t color[256];
	for (int i = 0; i < 256; i++)
		color[i] = i;

	m_pLabel_SpectroDb = new QLabel("Spectrum dB  ", this);
	m_pLabel_SpectroDb->setFixedWidth(60);
	m_pLabel_SpectroDb->setDisabled(true);

	m_pLineEdit_SpectroDbMax = new QLineEdit(this);
	m_pLineEdit_SpectroDbMax->setText(QString::number(m_pConfig->spectroDbRange.max));
	m_pLineEdit_SpectroDbMax->setAlignment(Qt::AlignCenter);
	m_pLineEdit_SpectroDbMax->setFixedWidth(30);
	m_pLineEdit_SpectroDbMax->setDisabled(true);
	m_pLineEdit_SpectroDbMin = new QLineEdit(this);
	m_pLineEdit_SpectroDbMin->setText(QString::number(m_pConfig->spectroDbRange.min));
	m_pLineEdit_SpectroDbMin->setAlignment(Qt::AlignCenter);
	m_pLineEdit_SpectroDbMin->setFixedWidth(30);
	m_pLineEdit_SpectroDbMin->setDisabled(true);

	m_pImageView_Colorbar = new QImageView(ColorTable::colortable::jet, 256, 1);
	m_pImageView_Colorbar->setFixedHeight(15);
	m_pImageView_Colorbar->drawImage(color);
	m_pImageView_Colorbar->setDisabled(true);

	// Create A-line navigating widgets
	m_pLabel_CurrentAline = new QLabel(this);
	m_pLabel_CurrentAline->setFixedWidth(150);
	QString str; str.sprintf("Current A-line : %4d / %4d  ", 1, m_pResultTab->getConfigTemp()->nAlines);
	m_pLabel_CurrentAline->setText(str);
	m_pLabel_CurrentAline->setDisabled(true);

	m_pSlider_CurrentAline = new QSlider(this);
	m_pSlider_CurrentAline->setOrientation(Qt::Horizontal);
	m_pSlider_CurrentAline->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
	m_pSlider_CurrentAline->setRange(0, m_pResultTab->getConfigTemp()->nAlines - 1);
	m_pSlider_CurrentAline->setValue(0);
	m_pSlider_CurrentAline->setDisabled(true);

	// Create information label
	m_pLabel_Info = new QLabel("", this);

	// Create layout
	QGridLayout *pGridLayout = new QGridLayout;
	pGridLayout->setSpacing(3);


	QGroupBox *pGroupBox_Extract = new QGroupBox;
	pGroupBox_Extract->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
	pGroupBox_Extract->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");

	QGridLayout *pGridLayout_Extract = new QGridLayout;
	pGridLayout_Extract->setSpacing(3);

	pGridLayout_Extract->addWidget(m_pPushButton_Extract, 0, 0, 1, 2);
	pGridLayout_Extract->addWidget(m_pPushButton_Save, 1, 0, 1, 2);
	pGridLayout_Extract->addWidget(m_pProgressBar, 2, 0, 1, 2);
	pGridLayout_Extract->addWidget(m_pCheckBox_AutoExtraction, 3, 0, 1, 2);

	pGridLayout_Extract->addWidget(m_pLabel_Window, 4, 0);
	pGridLayout_Extract->addWidget(m_pLineEdit_Window, 4, 1);
	pGridLayout_Extract->addWidget(m_pLabel_Overlap, 5, 0);
	pGridLayout_Extract->addWidget(m_pLineEdit_Overlap, 5, 1);
	pGridLayout_Extract->addWidget(m_pLabel_InPlane, 6, 0);
	pGridLayout_Extract->addWidget(m_pLineEdit_InPlane, 6, 1);
	pGridLayout_Extract->addWidget(m_pLabel_OutOfPlane, 7, 0);
	pGridLayout_Extract->addWidget(m_pLineEdit_OutOfPlane, 7, 1);
	pGridLayout_Extract->addWidget(m_pLabel_RoiDepth, 8, 0);
	pGridLayout_Extract->addWidget(m_pLineEdit_RoiDepth, 8, 1);

	pGroupBox_Extract->setLayout(pGridLayout_Extract);


	QGroupBox *pGroupBox_SpectrumVisualization = new QGroupBox;
	pGroupBox_SpectrumVisualization->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
	pGroupBox_SpectrumVisualization->setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}");

	QGridLayout *pGridLayout_SpectrumVisualization = new QGridLayout;
	pGridLayout_SpectrumVisualization->setSpacing(3);

	QGridLayout *pGridLayout_SpectroDbColorbar = new QGridLayout;
	pGridLayout_SpectroDbColorbar->setSpacing(3);
	pGridLayout_SpectroDbColorbar->addWidget(m_pLabel_SpectroDb, 0, 0, 1, 3);
	pGridLayout_SpectroDbColorbar->addWidget(m_pLineEdit_SpectroDbMin, 1, 0);
	pGridLayout_SpectroDbColorbar->addWidget(m_pImageView_Colorbar, 1, 1);
	pGridLayout_SpectroDbColorbar->addWidget(m_pLineEdit_SpectroDbMax, 1, 2);

	pGridLayout_SpectrumVisualization->addItem(pGridLayout_SpectroDbColorbar, 0, 0);
	pGridLayout_SpectrumVisualization->addWidget(m_pLabel_CurrentAline, 1, 0, Qt::AlignLeft);
	pGridLayout_SpectrumVisualization->addWidget(m_pSlider_CurrentAline, 2, 0);

	pGroupBox_SpectrumVisualization->setLayout(pGridLayout_SpectrumVisualization);

	// Set layout
	pGridLayout->addWidget(m_pImageView_SpectroOCTView, 0, 0, 3, 1);
	pGridLayout->addWidget(pGroupBox_Extract, 0, 1);
	pGridLayout->addWidget(pGroupBox_SpectrumVisualization, 1, 1);
	pGridLayout->addWidget(m_pLabel_Info, 2, 1, Qt::AlignBottom | Qt::AlignLeft);

	this->setLayout(pGridLayout);

	// Connect
	connect(this, SIGNAL(paintSpectroImage(uint8_t*)), m_pImageView_SpectroOCTView, SLOT(drawImage(uint8_t*)));
	connect(m_pPushButton_Extract, SIGNAL(clicked(bool)), this, SLOT(spectrumExtract()));
	connect(m_pPushButton_Save, SIGNAL(clicked(bool)), this, SLOT(spectrumSave()));
	connect(m_pLineEdit_Window, SIGNAL(editingFinished()), this, SLOT(changeWindowSize()));
	connect(m_pLineEdit_Overlap, SIGNAL(editingFinished()), this, SLOT(changeOverlapSize()));
	connect(m_pLineEdit_InPlane, SIGNAL(textChanged(const QString&)), this, SLOT(changeInPlaneSize(const QString&)));
	connect(m_pLineEdit_OutOfPlane, SIGNAL(textChanged(const QString&)), this, SLOT(changeOutOfPlaneSize(const QString&)));
	connect(m_pLineEdit_RoiDepth, SIGNAL(textChanged(const QString&)), this, SLOT(changeRoiDepth(const QString&)));
	connect(m_pLineEdit_SpectroDbMin, SIGNAL(textChanged(const QString&)), this, SLOT(changeDbRange()));
	connect(m_pLineEdit_SpectroDbMax, SIGNAL(textChanged(const QString&)), this, SLOT(changeDbRange()));
	connect(m_pSlider_CurrentAline, SIGNAL(valueChanged(int)), this, SLOT(drawSpectroImage(int)));

	connect(this, SIGNAL(setWidgets(bool)), this, SLOT(setWidgetEnabled(bool)));
	connect(this, SIGNAL(processedSingleAline(int)), m_pProgressBar, SLOT(setValue(int)));
}

SpectroOCTDlg::~SpectroOCTDlg()
{
	if (t_spectro.joinable()) t_spectro.join();
	if (m_pImgObjSpectroOCT) delete m_pImgObjSpectroOCT;
}

void SpectroOCTDlg::closeEvent(QCloseEvent * e)
{
	if (!m_bCanBeClosed)
		e->ignore();
	else
		finished(0);
}

void SpectroOCTDlg::keyPressEvent(QKeyEvent *e)
{
	if (e->key() != Qt::Key_Escape)
		QDialog::keyPressEvent(e);
}


void SpectroOCTDlg::setWidgetEnabled(bool enabled)
{
	m_pPushButton_Extract->setEnabled(enabled);
	m_pPushButton_Save->setEnabled(enabled);
	m_pCheckBox_AutoExtraction->setEnabled(enabled);

	m_pLabel_Window->setEnabled(enabled);
	m_pLineEdit_Window->setEnabled(enabled);
	m_pLabel_Overlap->setEnabled(enabled);
	m_pLineEdit_Overlap->setEnabled(enabled);

	m_pLabel_InPlane->setEnabled(enabled);
	m_pLineEdit_InPlane->setEnabled(enabled);
	m_pLabel_OutOfPlane->setEnabled(enabled);
	m_pLineEdit_OutOfPlane->setEnabled(enabled);
	m_pLabel_RoiDepth->setEnabled(enabled);
	m_pLineEdit_RoiDepth->setEnabled(enabled);

	m_pLabel_SpectroDb->setEnabled(enabled);
	m_pImageView_Colorbar->setEnabled(enabled);
	m_pLineEdit_SpectroDbMin->setEnabled(enabled);
	m_pLineEdit_SpectroDbMax->setEnabled(enabled);

	m_pLabel_CurrentAline->setEnabled(enabled);
	m_pSlider_CurrentAline->setEnabled(enabled);

	if (!enabled)
	{
		int n_oop = 2 * m_pConfig->spectroOutOfPlaneAvgSize + 1;

		m_pProgressBar->setFormat("Extracting... %p%");
		if ((m_pConfig->spectroInPlaneAvgSize == 0) && (m_pConfig->spectroOutOfPlaneAvgSize == 0))
			m_pProgressBar->setRange(0, n_oop * m_pConfigTemp->nAlines - 1);
		else
			m_pProgressBar->setRange(0, (n_oop + 1) * m_pConfigTemp->nAlines - 1);
	}
	else
	{
		m_pProgressBar->setFormat("");
		m_pProgressBar->setValue(0);
	}

	//m_pResultTab->getRectImageView()->setUpdatesEnabled(enabled);
	//m_pResultTab->getCircImageView()->setUpdatesEnabled(enabled);

	if (enabled)
	{
		//m_pSlider_CurrentAline->setRange(m_pConfigTemp->spectroInPlaneAvgSize, m_pConfigTemp->nAlines - 1 - m_pConfigTemp->spectroInPlaneAvgSize);
		//if (getCurrentAline() < m_pSlider_CurrentAline->minimum())
		//	m_pSlider_CurrentAline->setValue(m_pSlider_CurrentAline->minimum());
		//else if (getCurrentAline() > m_pSlider_CurrentAline->maximum())
		//	m_pSlider_CurrentAline->setValue(m_pSlider_CurrentAline->maximum());

		QString str; str.sprintf("frame: %d\nnwin: %d\nnoverlap: %d\nnk: %d\nin-plane: %d\nout-of-plane: %d",
			subject_frame + 1, m_pConfig->spectroWindow, m_pConfig->spectroOverlap, m_vecSpectra.at(0).size(1),
			2 * m_pConfig->spectroInPlaneAvgSize + 1, 2 * m_pConfig->spectroOutOfPlaneAvgSize + 1);
		m_pLabel_Info->setText(str);
	}
}


void SpectroOCTDlg::drawSpectroImage(int aline)
{
	if (m_vecSpectra.size() > 0)
	{
		IppiSize roi_spectro = { m_pImgObjSpectroOCT->getHeight(), m_pImgObjSpectroOCT->getWidth() };

		np::Uint8Array2 scale_temp(roi_spectro.width, roi_spectro.height);

		if (subject_frame == m_pResultTab->getCurrentFrame())
		{
			ippiScale_32f8u_C1R(m_vecSpectra.at(aline).raw_ptr(), roi_spectro.width * sizeof(float),
				scale_temp.raw_ptr(), roi_spectro.width * sizeof(uint8_t), { roi_spectro.width, m_vecSpectra.at(aline).size(1) }, m_pConfig->spectroDbRange.min, m_pConfig->spectroDbRange.max);
#if defined(OCT_VERTICAL_MIRRORING)
			ippiMirror_8u_C1IR(scale_temp.raw_ptr(), sizeof(uint8_t) * roi_spectro.width, roi_spectro, ippAxsVertical);
#endif
		}
		else
			memset(scale_temp.raw_ptr(), 0, sizeof(uint8_t) * scale_temp.length());

		ippiTranspose_8u_C1R(scale_temp.raw_ptr(), roi_spectro.width * sizeof(uint8_t), m_pImgObjSpectroOCT->arr.raw_ptr(), roi_spectro.height * sizeof(uint8_t), roi_spectro);

		emit paintSpectroImage(m_pImgObjSpectroOCT->qindeximg.bits());

		// Widgets updates
		QString str; str.sprintf("Current A-line : %4d / %4d  ", aline + 1, m_pResultTab->getConfigTemp()->nAlines);
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
}

void SpectroOCTDlg::changeWindowSize()
{
	m_pConfig->spectroWindow = m_pLineEdit_Window->text().toInt();
	changeOverlapSize();

	subject_frame = -1;
}

void SpectroOCTDlg::changeOverlapSize()
{
	int overlap = m_pLineEdit_Overlap->text().toInt();
	if (overlap >= m_pConfig->spectroWindow)
	{
		m_pConfig->spectroOverlap = m_pConfig->spectroWindow - 1;
		m_pLineEdit_Overlap->setText(QString::number(m_pConfig->spectroOverlap));
	}
	else
		m_pConfig->spectroOverlap = overlap;

	subject_frame = -1;
}

void SpectroOCTDlg::changeInPlaneSize(const QString& str)
{
	m_pConfig->spectroInPlaneAvgSize = str.toInt();
	subject_frame = -1;
}

void SpectroOCTDlg::changeOutOfPlaneSize(const QString& str)
{
	m_pConfig->spectroOutOfPlaneAvgSize = str.toInt();
	subject_frame = -1;
}

void SpectroOCTDlg::changeRoiDepth(const QString& str)
{
	m_pConfig->spectroRoiDepth = str.toInt();
	subject_frame = -1;
}

void SpectroOCTDlg::changeDbRange()
{
	m_pConfig->spectroDbRange.min = m_pLineEdit_SpectroDbMin->text().toFloat();
	m_pConfig->spectroDbRange.max = m_pLineEdit_SpectroDbMax->text().toFloat();

	drawSpectroImage(getCurrentAline());
}

void SpectroOCTDlg::spectrumExtract()
{
	if (subject_frame == m_pResultTab->getCurrentFrame())
		return;

	if (t_spectro.joinable())
		t_spectro.join();

	if (!t_spectro.joinable())
	{
		t_spectro = std::thread([&]() {

			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

			QFile file(m_pResultTab->m_fileName);
			if (false == file.open(QFile::ReadOnly))
				printf("[ERROR] Invalid external data!\n");
			else
			{
				// Read Ini File & Initialization ///////////////////////////////////////////////////////////
				QString fileTitle;
				for (int i = 0; i < m_pResultTab->m_fileName.length(); i++)
					if (m_pResultTab->m_fileName.at(i) == QChar('.')) fileTitle = m_pResultTab->m_fileName.left(i);
				QString bgName = fileTitle + ".background";
				QString calibName = fileTitle + ".calibration";

				m_pConfigTemp = m_pResultTab->getConfigTemp();

				// Set Buffers //////////////////////////////////////////////////////////////////////////////	
				if (((m_pResultTab->getCurrentFrame() - m_pConfig->spectroOutOfPlaneAvgSize) < 0) 
					|| ((m_pResultTab->getCurrentFrame() + m_pConfig->spectroOutOfPlaneAvgSize) >= m_pConfigTemp->nFrames))
					return;
#ifndef K_CLOCKING
				int nk = (int)((m_pConfigTemp->nScans / 2 - m_pConfig->spectroOverlap) / (m_pConfig->spectroWindow - m_pConfig->spectroOverlap));
#else
				int nk = (int)((m_pConfigTemp->nScans - m_pConfig->spectroOverlap) / (m_pConfig->spectroWindow - m_pConfig->spectroOverlap));
#endif
				int n_oop = 2 * m_pConfig->spectroOutOfPlaneAvgSize + 1;

				np::Uint16Array2 fringe = np::Uint16Array2(m_pConfigTemp->nScans, n_oop * m_pConfigTemp->nAlines);

				std::vector<np::FloatArray2> clear_vector;
				clear_vector.swap(m_vecSpectra);				
				for (int i = 0; i < n_oop * m_pConfigTemp->nAlines; i++)
				{
					np::FloatArray2 spectrum(m_pConfigTemp->n2ScansFFT, nk);
					m_vecSpectra.push_back(spectrum);
				}
				
				// Set Widgets //////////////////////////////////////////////////////////////////////////////
				emit setWidgets(false);

				ColorTable temp_ctable;
				if (m_pImgObjSpectroOCT) delete m_pImgObjSpectroOCT;
				m_pImgObjSpectroOCT = new ImageObject(((nk + 3) >> 2) << 2, m_pConfigTemp->n2ScansFFT, temp_ctable.m_colorTableVector.at(ColorTable::jet));

				m_pImageView_SpectroOCTView->resetSize(nk, m_pConfigTemp->n2ScansFFT);

				// Set OCT FLIM Object //////////////////////////////////////////////////////////////////////
				int total = 0;

//#ifndef CUDA_ENABLED
				SOCTProcess* pSOCT1 = new SOCTProcess(m_pConfigTemp->nScans, n_oop * m_pConfigTemp->nAlines, m_pConfig->spectroWindow, m_pConfig->spectroOverlap);
#ifndef K_CLOCKING
				pSOCT1->loadCalibration(CH_1, calibName.toUtf8().constData(), bgName.toUtf8().constData(), m_pConfigTemp->erasmus);
#endif
				pSOCT1->changeDiscomValue(m_pConfigTemp->octDiscomVal);
				pSOCT1->DidProcess += [this, &total]() { total++; emit processedSingleAline(total); };
//#else
//				CudaOCTProcess* pOCT1 = new CudaOCTProcess(m_pConfigTemp->nScans, m_pConfigTemp->nAlines);
//#ifndef K_CLOCKING
//				pOCT1->loadCalibration(CH_1, calibName.toUtf8().constData(), bgName.toUtf8().constData(), m_pConfigTemp->erasmus);
//#endif
//				pOCT1->changeDiscomValue(m_pConfigTemp->octDiscomVal);
//				pOCT1->initialize();
//#endif
				// Check averaging process is required //////////////////////////////////////////////////////
				subject_frame = m_pResultTab->getCurrentFrame();

				np::Uint16Array contour(n_oop * m_pConfigTemp->nAlines);
				if ((m_pConfig->spectroInPlaneAvgSize != 0) || (m_pConfig->spectroOutOfPlaneAvgSize != 0))
				{
					//m_pResultTab->getAutoContouringButton()->setChecked(true);
					m_pResultTab->autoContouring(true);

					int k = 0;
					for (int i = -m_pConfig->spectroOutOfPlaneAvgSize; i <= +m_pConfig->spectroOutOfPlaneAvgSize; i++)
					{
						memcpy(&contour(k * m_pConfigTemp->nAlines), &m_pResultTab->m_contourMap(0, subject_frame + i), sizeof(uint16_t) * m_pConfigTemp->nAlines);
						ippsAddC_16u_ISfs(m_pConfig->circCenter, &contour(k++ * m_pConfigTemp->nAlines), m_pConfigTemp->nAlines, 0);
					}
				}

				// Load data ////////////////////////////////////////////////////////////////////////////////						
				int k = 0;
				for (int i = -m_pConfig->spectroOutOfPlaneAvgSize; i <= +m_pConfig->spectroOutOfPlaneAvgSize; i++)
					loadingRawFringe(&file, m_pConfigTemp, subject_frame + i,
						np::Uint16Array2(&fringe(0, k++ * m_pConfigTemp->nAlines), m_pConfigTemp->nScans, m_pConfig->nAlines));

				// SOCT Process /////////////////////////////////////////////////////////////////////////////
				(*pSOCT1)(m_vecSpectra, fringe);
				if ((m_pConfig->spectroInPlaneAvgSize != 0) || (m_pConfig->spectroOutOfPlaneAvgSize != 0))
					pSOCT1->spectrum_averaging(m_vecSpectra, contour,
						m_pConfig->spectroInPlaneAvgSize, m_pConfig->spectroOutOfPlaneAvgSize, m_pConfig->spectroRoiDepth);
				pSOCT1->db_scaling(m_vecSpectra);

				// Delete OCT FLIM Object & threading sync buffers //////////////////////////////////////////
				delete pSOCT1;

				// Reset Widgets /////////////////////////////////////////////////////////////////////////////
				emit setWidgets(true);

				// Visualization /////////////////////////////////////////////////////////////////////////////
				drawSpectroImage(getCurrentAline());
			}

			std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
			printf("elapsed time : %.2f sec\n\n", elapsed.count());
		});
	}
}

void SpectroOCTDlg::spectrumSave()
{
	if (subject_frame != m_pResultTab->getCurrentFrame())
		return;
	
	m_pProgressBar->setFormat("Saviing... %p%");		
	m_pProgressBar->setRange(0, m_pConfigTemp->nAlines - 1);
	
	QDir dir(m_pResultTab->m_path + "/spectroscopic");
	if (!dir.exists())
		dir.mkpath(m_pResultTab->m_path + "/spectroscopic");
		
	QString fname; fname.sprintf("spectra_f%03d_win%d_overlap%d_ip%d_oop%d.soct",
		subject_frame + 1, m_pConfig->spectroWindow, m_pConfig->spectroOverlap,
			2 * m_pConfig->spectroInPlaneAvgSize + 1, 2 * m_pConfig->spectroOutOfPlaneAvgSize + 1);
	
	QFile file(m_pResultTab->m_path + "/spectroscopic/" + fname);
	file.open(QIODevice::WriteOnly);
	for (int i = 0; i < m_vecSpectra.size(); i++)
	{
		file.write(reinterpret_cast<const char*>(m_vecSpectra.at(i).raw_ptr()), sizeof(float) * m_vecSpectra.at(i).length());
		emit processedSingleAline(i);
	}
	file.close();

	m_pProgressBar->setFormat("");
	m_pProgressBar->setValue(0);
}

void SpectroOCTDlg::loadingRawFringe(QFile* pFile, Configuration* pConfig, int frame, np::Uint16Array2& fringe)
{
	if (frame < pConfig->nFrames)
	{
		if (fringe.length() == pConfig->nFrameSize)
		{
			pFile->seek(frame * sizeof(uint16_t) * pConfig->nFrameSize);
			pFile->read(reinterpret_cast<char *>(fringe.raw_ptr()), sizeof(uint16_t) * pConfig->nFrameSize);
		}
		else
		{
			printf("Invalid fringe buffer.\n");
		}
	}
}