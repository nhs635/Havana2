
#include "NirfCrossTalkCompDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QResultTab.h>

#include <Havana2/Viewer/QImageView.h>


#ifdef OCT_NIRF
#ifdef TWO_CHANNEL_NIRF

#define A  6.3270
#define B  0.1714
#define C -8.2700
#define D -1.9060

NirfCrossTalkCompDlg::NirfCrossTalkCompDlg(QWidget *parent) :
    QDialog(parent),
	compensationMode(RATIO_BASED),
	nirfBg1(0.0f), nirfBg2(0.0f), gainValue1(0.0f), gainValue2(0.0f), ratio(0.0f)
{
    // Set default size & frame
    setFixedSize(360, 230);
    setWindowFlags(Qt::Tool);
	setWindowTitle("2Ch NIRF Cross Talk Correction");

    // Set main window objects
    m_pResultTab = (QResultTab*)parent;
    m_pMainWnd = m_pResultTab->getMainWnd();
	m_pConfig = m_pMainWnd->m_pConfiguration;
	m_pConfigTemp = m_pResultTab->getConfigTemp();
	
	// Create widgets for 2Ch NIRF cross-talk compensation (common parameter)
	m_pLabel_Ch1_Bg = new QLabel("Ch1 BG  ", this);
	m_pLineEdit_Ch1_Bg = new QLineEdit(this);
	m_pLineEdit_Ch1_Bg->setFixedWidth(45);
	m_pLineEdit_Ch1_Bg->setText(QString::number(0, 'f', 4));
	m_pLineEdit_Ch1_Bg->setAlignment(Qt::AlignCenter);;

	m_pLabel_Ch2_Bg = new QLabel("Ch2 BG  ", this);
	m_pLineEdit_Ch2_Bg = new QLineEdit(this);
	m_pLineEdit_Ch2_Bg->setFixedWidth(45);
	m_pLineEdit_Ch2_Bg->setText(QString::number(0, 'f', 4));
	m_pLineEdit_Ch2_Bg->setAlignment(Qt::AlignCenter);;

	m_pLabel_Ch1_GainVoltage = new QLabel("   Ch1 Gain  ", this);
	m_pLineEdit_Ch1_GainVoltage = new QLineEdit(this);
	m_pLineEdit_Ch1_GainVoltage->setFixedWidth(45);
	m_pLineEdit_Ch1_GainVoltage->setText(QString::number(0, 'f', 3));
	m_pLineEdit_Ch1_GainVoltage->setAlignment(Qt::AlignCenter);;
	m_pLabel_Ch1_GainValue = new QLabel(QString::number(0, 'f', 2));
	m_pLabel_Ch1_GainValue->setFixedWidth(60);

	m_pLabel_Ch2_GainVoltage = new QLabel("   Ch2 Gain  ", this);
	m_pLineEdit_Ch2_GainVoltage = new QLineEdit(this);
	m_pLineEdit_Ch2_GainVoltage->setFixedWidth(45);
	m_pLineEdit_Ch2_GainVoltage->setText(QString::number(0, 'f', 3));
	m_pLineEdit_Ch2_GainVoltage->setAlignment(Qt::AlignCenter);;
	m_pLabel_Ch2_GainValue = new QLabel(QString::number(0, 'f', 2));
	m_pLabel_Ch2_GainValue->setFixedWidth(60);

	// Create widgets for 2Ch NIRF cross-talk compensation (compensation mode selection)
	m_pRadioButton_RatioBased = new QRadioButton(this);
	m_pRadioButton_RatioBased->setText("Ratio-based Compensation ");
	m_pRadioButton_SpectrumBased = new QRadioButton(this);
	m_pRadioButton_SpectrumBased->setText("Spectrum-based Compensation ");
	m_pRadioButton_SpectrumBased->setDisabled(true);

	m_pButtonGroup_CompensationMode = new QButtonGroup(this);
	m_pButtonGroup_CompensationMode->addButton(m_pRadioButton_RatioBased, RATIO_BASED);
	m_pButtonGroup_CompensationMode->addButton(m_pRadioButton_SpectrumBased, SPECTRUM_BASED);
	m_pRadioButton_RatioBased->setChecked(true);

	// Ratio-based Compensation
	m_pLabel_Ratio = new QLabel("Ratio  ", this);
	m_pLineEdit_Ratio = new QLineEdit(this);
	m_pLineEdit_Ratio->setFixedWidth(50);
	m_pLineEdit_Ratio->setText(QString::number(m_pConfig->nirfCrossTalkRatio, 'f', 4));
	m_pLineEdit_Ratio->setAlignment(Qt::AlignCenter);;
	m_pToggleButton_RatioSetting = new QPushButton(this);
	m_pToggleButton_RatioSetting->setCheckable(true);
	m_pToggleButton_RatioSetting->setText("Ratio Setting");
	
	
	// Set layout
	QVBoxLayout *pVBoxLayout = new QVBoxLayout;
	pVBoxLayout->setSpacing(3);

	QHBoxLayout *pHBoxLayout_Ch1 = new QHBoxLayout;
	pHBoxLayout_Ch1->addWidget(m_pLabel_Ch1_Bg);
	pHBoxLayout_Ch1->addWidget(m_pLineEdit_Ch1_Bg);
	pHBoxLayout_Ch1->addWidget(m_pLabel_Ch1_GainVoltage);
	pHBoxLayout_Ch1->addWidget(m_pLineEdit_Ch1_GainVoltage);
	pHBoxLayout_Ch1->addWidget(new QLabel("V   ", this));
	pHBoxLayout_Ch1->addWidget(m_pLabel_Ch1_GainValue);
	pHBoxLayout_Ch1->addStretch(1);
	
	QHBoxLayout *pHBoxLayout_Ch2 = new QHBoxLayout;
	pHBoxLayout_Ch2->addWidget(m_pLabel_Ch2_Bg);
	pHBoxLayout_Ch2->addWidget(m_pLineEdit_Ch2_Bg);
	pHBoxLayout_Ch2->addWidget(m_pLabel_Ch2_GainVoltage);
	pHBoxLayout_Ch2->addWidget(m_pLineEdit_Ch2_GainVoltage);
	pHBoxLayout_Ch2->addWidget(new QLabel("V   ", this));
	pHBoxLayout_Ch2->addWidget(m_pLabel_Ch2_GainValue);
	pHBoxLayout_Ch2->addStretch(1);

	pVBoxLayout->addItem(pHBoxLayout_Ch1);
	pVBoxLayout->addItem(pHBoxLayout_Ch2);
	
	QHBoxLayout *pHBoxLayout_RatioBased = new QHBoxLayout;
	pHBoxLayout_RatioBased->addWidget(m_pRadioButton_RatioBased);
	pHBoxLayout_RatioBased->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));
	pHBoxLayout_RatioBased->addWidget(m_pLabel_Ratio);
	pHBoxLayout_RatioBased->addWidget(m_pLineEdit_Ratio);
	pHBoxLayout_RatioBased->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));
	pHBoxLayout_RatioBased->addWidget(m_pToggleButton_RatioSetting);

	pVBoxLayout->addItem(pHBoxLayout_RatioBased);
	
	pVBoxLayout->addWidget(m_pRadioButton_SpectrumBased);
	pVBoxLayout->addStretch(1);

	setLayout(pVBoxLayout);

	// Connect
	connect(m_pLineEdit_Ch1_Bg, SIGNAL(textChanged(const QString &)), this, SLOT(changeCh1NirfBackground(const QString &)));
	connect(m_pLineEdit_Ch1_GainVoltage, SIGNAL(textChanged(const QString &)), this, SLOT(changeCh1GainVoltage(const QString &)));
	connect(m_pLineEdit_Ch2_Bg, SIGNAL(textChanged(const QString &)), this, SLOT(changeCh2NirfBackground(const QString &)));
	connect(m_pLineEdit_Ch2_GainVoltage, SIGNAL(textChanged(const QString &)), this, SLOT(changeCh2GainVoltage(const QString &)));

	connect(m_pButtonGroup_CompensationMode, SIGNAL(buttonClicked(int)), this, SLOT(changeCompensationMode(int)));
	connect(m_pLineEdit_Ratio, SIGNAL(textChanged(const QString &)), this, SLOT(changeRatio(const QString &)));
	connect(m_pToggleButton_RatioSetting, SIGNAL(toggled(bool)), this, SLOT(setRatio(bool)));

	// Initialization
	loadNirfBackground();
	ratio = m_pConfig->nirfCrossTalkRatio;
#ifdef PROGRAMMATIC_GAIN_CONTROL
	m_pLineEdit_Ch1_GainVoltage->setText(QString::number(m_pConfigTemp->pmtGainVoltage[0], 'f', 3));
	m_pLineEdit_Ch2_GainVoltage->setText(QString::number(m_pConfigTemp->pmtGainVoltage[1], 'f', 3));
#endif

	// Invalidate
	m_pResultTab->invalidate();
}

NirfCrossTalkCompDlg::~NirfCrossTalkCompDlg()
{
	m_pResultTab->getNirfMap1View()->getRender()->m_bRectDrawing = false;
	m_pResultTab->getNirfMap2View()->getRender()->m_bRectDrawing = false;

	m_pResultTab->getNirfMap1View()->setReleasedMouseCallback([&](QRect rect) { (void)rect; });
	m_pResultTab->getNirfMap2View()->setReleasedMouseCallback([&](QRect rect) { (void)rect; });
}

void NirfCrossTalkCompDlg::keyPressEvent(QKeyEvent *e)
{
	if (e->key() != Qt::Key_Escape)
		QDialog::keyPressEvent(e);
}

void NirfCrossTalkCompDlg::loadNirfBackground()
{
	QString distMapName = m_pResultTab->m_path + "/nirf_bg.bin";

	QFile file(distMapName);
	if (false == file.open(QFile::ReadOnly))
		printf("[ERROR] Invalid external data or there is no such a file (nirf_bg.bin)!\n");
	else
	{
		np::DoubleArray nirfBgMap1((int)file.size() / sizeof(double) / 2);
		np::DoubleArray nirfBgMap2((int)file.size() / sizeof(double) / 2);
		
		int read = 0;
		for (int i = 0; m_pConfigTemp->nFrames; i++)
		{
			file.read(reinterpret_cast<char*>(&nirfBgMap1(read)), sizeof(double) * m_pConfigTemp->nAlines);
			file.read(reinterpret_cast<char*>(&nirfBgMap2(read)), sizeof(double) * m_pConfigTemp->nAlines);
			read += m_pConfigTemp->nAlines;
		}
		file.close();

		double temp_bg;
		ippsMean_64f(nirfBgMap1.raw_ptr(), nirfBgMap1.length(), &temp_bg);
		nirfBg1 = temp_bg;
		ippsMean_64f(nirfBgMap2.raw_ptr(), nirfBgMap2.length(), &temp_bg);
		nirfBg2 = temp_bg;

		m_pLineEdit_Ch1_Bg->setText(QString::number(nirfBg1, 'f', 4));
		m_pLineEdit_Ch2_Bg->setText(QString::number(nirfBg2, 'f', 4));
	}
}

void NirfCrossTalkCompDlg::changeCh1NirfBackground(const QString &str)
{
	nirfBg1 = str.toFloat();

	// Invalidate
	m_pResultTab->invalidate();
}

void NirfCrossTalkCompDlg::changeCh2NirfBackground(const QString &str)
{
	nirfBg2 = str.toFloat();

	// Invalidate
	m_pResultTab->invalidate();
}

void NirfCrossTalkCompDlg::changeCh1GainVoltage(const QString &str)
{
	float gainVoltage = str.toFloat();
	gainValue1 = powf(10.0, A * expf(B * gainVoltage) + C * expf(D * gainVoltage));
	m_pLabel_Ch1_GainValue->setText(QString::number(gainValue1, 'f', 2));

	// Invalidate
	m_pResultTab->invalidate();
}

void NirfCrossTalkCompDlg::changeCh2GainVoltage(const QString &str)
{
	float gainVoltage = str.toFloat();
	gainValue2 = powf(10.0, A * expf(B * gainVoltage) + C * expf(D * gainVoltage));
	m_pLabel_Ch2_GainValue->setText(QString::number(gainValue2, 'f', 2));

	// Invalidate
	m_pResultTab->invalidate();
}

void NirfCrossTalkCompDlg::changeCompensationMode(int id)
{
	switch (id)
	{
		case RATIO_BASED:
			compensationMode = RATIO_BASED;
			m_pLabel_Ratio->setEnabled(true);
			m_pLineEdit_Ratio->setEnabled(true);
			m_pToggleButton_RatioSetting->setEnabled(true);

			break;
		case SPECTRUM_BASED:
			compensationMode = SPECTRUM_BASED;
			if (m_pToggleButton_RatioSetting->isChecked())
				m_pToggleButton_RatioSetting->setChecked(false);
			m_pLabel_Ratio->setDisabled(true);
			m_pLineEdit_Ratio->setDisabled(true);
			m_pToggleButton_RatioSetting->setDisabled(true);

			break;
	}

	// Invalidate
	m_pResultTab->invalidate();
}

void NirfCrossTalkCompDlg::changeRatio(const QString &str)
{
	ratio = str.toFloat();
	m_pConfig->nirfCrossTalkRatio = ratio;

	// Invalidate
	m_pResultTab->invalidate();
}

void NirfCrossTalkCompDlg::setRatio(bool toggled)
{
	if (toggled)
	{
		m_pLineEdit_Ratio->setDisabled(true);

		m_pResultTab->getNirfMap1View()->getRender()->m_bRectDrawing = true;
		m_pResultTab->getNirfMap2View()->getRender()->m_bRectDrawing = true;
		
		m_pResultTab->getNirfMap1View()->setReleasedMouseCallback([&](QRect rect)
		{
			int l = (rect.topLeft().x() < rect.bottomRight().x()) ? rect.topLeft().x() : rect.bottomRight().x();
			int t = (int)m_pResultTab->m_vectorOctImage.size() - ((rect.topLeft().y() > rect.bottomRight().y()) ? rect.topLeft().y() : rect.bottomRight().y());
			int w = abs(rect.topLeft().x() - rect.bottomRight().x());
			int h = abs(rect.topLeft().y() - rect.bottomRight().y());
			
			IppiSize roi = { w, h };
			np::FloatArray2 overlap_region(roi.width, roi.height);

			ippiDiv_32f_C1R(&m_pResultTab->m_nirfMap1_Raw(l, t), sizeof(float) * m_pResultTab->m_nirfMap1_Raw.size(0),
				&m_pResultTab->m_nirfMap2_Raw(l, t), sizeof(float) * m_pResultTab->m_nirfMap2_Raw.size(0),
				overlap_region.raw_ptr(), sizeof(float) * roi.width, roi);
			ippsMean_32f(overlap_region.raw_ptr(), overlap_region.length(), &ratio, ippAlgHintAccurate);			

			m_pConfig->nirfCrossTalkRatio = ratio;
			m_pLineEdit_Ratio->setText(QString::number(ratio, 'f', 4));
		});
		m_pResultTab->getNirfMap2View()->setReleasedMouseCallback([&](QRect rect)
		{
			int l = (rect.topLeft().x() < rect.bottomRight().x()) ? rect.topLeft().x() : rect.bottomRight().x();
			int t = (int)m_pResultTab->m_vectorOctImage.size() - ((rect.topLeft().y() > rect.bottomRight().y()) ? rect.topLeft().y() : rect.bottomRight().y());
			int w = abs(rect.topLeft().x() - rect.bottomRight().x());
			int h = abs(rect.topLeft().y() - rect.bottomRight().y());

			IppiSize roi = { w, h };
			np::FloatArray2 overlap_region(roi.width, roi.height);

			ippiDiv_32f_C1R(&m_pResultTab->m_nirfMap1_Raw(l, t), sizeof(float) * m_pResultTab->m_nirfMap1_Raw.size(0),
				&m_pResultTab->m_nirfMap2_Raw(l, t), sizeof(float) * m_pResultTab->m_nirfMap2_Raw.size(0),
				overlap_region.raw_ptr(), sizeof(float) * roi.width, roi);
			ippsMean_32f(overlap_region.raw_ptr(), overlap_region.length(), &ratio, ippAlgHintAccurate);

			m_pConfig->nirfCrossTalkRatio = ratio;
			m_pLineEdit_Ratio->setText(QString::number(ratio, 'f', 4));
		});
	}
	else
	{
		m_pLineEdit_Ratio->setEnabled(true);

		m_pResultTab->getNirfMap1View()->getRender()->m_bRectDrawing = false;
		m_pResultTab->getNirfMap2View()->getRender()->m_bRectDrawing = false;

		m_pResultTab->getNirfMap1View()->setReleasedMouseCallback([&](QRect rect) { (void)rect; });
		m_pResultTab->getNirfMap2View()->setReleasedMouseCallback([&](QRect rect) { (void)rect; });
	}
}


#endif
#endif
