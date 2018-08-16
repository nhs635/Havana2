
#include "NirfDistCompDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QResultTab.h>

#include <iostream>
#include <thread>

#include <ipps.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>


#ifdef OCT_NIRF
NirfDistCompDlg::NirfDistCompDlg(QWidget *parent) :
    QDialog(parent), nirfBg(0), nirfBackgroundLevel(0)
{
    // Set default size & frame
    setFixedSize(360, 330);
    setWindowFlags(Qt::Tool);
	setWindowTitle("NIRF Distance Compensation");

    // Set main window objects
    m_pResultTab = (QResultTab*)parent;
    m_pMainWnd = m_pResultTab->getMainWnd();
	m_pConfig = m_pMainWnd->m_pConfiguration;

	// Set compensation data
    distDecayCurve = np::FloatArray(CIRC_RADIUS);
	compCurve = np::FloatArray(CIRC_RADIUS);
	
	// Create widgets for data loading & compensation
	m_pPushButton_LoadDistanceMap = new QPushButton(this);
    m_pPushButton_LoadDistanceMap->setText("Load Distance Map");

	m_pPushButton_LoadNirfBackground = new QPushButton(this);
    m_pPushButton_LoadNirfBackground->setText("Load NIRF Background");

	m_pToggleButton_Compensation = new QPushButton(this);
	m_pToggleButton_Compensation->setText("Compensation On");
	m_pToggleButton_Compensation->setCheckable(true);
    m_pToggleButton_Compensation->setDisabled(true);
	
	// Create widgets for compensation details
	m_pRenderArea_DistanceDecayCurve = new QRenderArea(this);
	m_pRenderArea_DistanceDecayCurve->setSize({ 0, CIRC_RADIUS }, { 0, 1 });
	m_pRenderArea_DistanceDecayCurve->setMinimumHeight(80);
	m_pRenderArea_DistanceDecayCurve->setGrid(4, 8, 1);
	
	m_pLabel_DistanceDecayCurve = new QLabel("Distance Decay Curve ([0, 1])", this);
	m_pLabel_DistanceDecayCurve->setBuddy(m_pRenderArea_DistanceDecayCurve);
	
	m_pRenderArea_CompensationCurve = new QRenderArea(this);
	m_pRenderArea_CompensationCurve->setSize({ 0, CIRC_RADIUS }, { 0, round(m_pConfig->nirfFactorThres * 1.1) });
	m_pRenderArea_CompensationCurve->setMinimumHeight(80);
	m_pRenderArea_CompensationCurve->setGrid(4, 8, 1);

	m_pLabel_CompensationCurve = new QLabel(QString("Compensation Curve ([0, %1])").arg((int)round(m_pConfig->nirfFactorThres * 1.1)), this);
	m_pLabel_CompensationCurve->setBuddy(m_pRenderArea_CompensationCurve);
		
	for (int i = 0; i < 4; i++)
	{
		m_pLineEdit_CompensationCoeff[i] = new QLineEdit(this);
		m_pLineEdit_CompensationCoeff[i]->setFixedWidth(70);
		m_pLineEdit_CompensationCoeff[i]->setText(QString::number(m_pConfig->nirfCompCoeffs[i]));
		m_pLineEdit_CompensationCoeff[i]->setAlignment(Qt::AlignCenter);
	}

	m_pLabel_CompensationCoeff = new QLabel("Compensation Coefficient\n((a*exp(b*x)+c*exp(d*x))/(a+c))");

	m_pLineEdit_FactorThreshold = new QLineEdit(this);
	m_pLineEdit_FactorThreshold->setFixedWidth(35);
	m_pLineEdit_FactorThreshold->setText(QString::number(m_pConfig->nirfFactorThres));
	m_pLineEdit_FactorThreshold->setAlignment(Qt::AlignCenter);

	m_pLabel_FactorThreshold = new QLabel("Factor Threshold");
	m_pLabel_FactorThreshold->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
	m_pLabel_FactorThreshold->setBuddy(m_pLineEdit_FactorThreshold);

	m_pLineEdit_FactorPropConst = new QLineEdit(this);
	m_pLineEdit_FactorPropConst->setFixedWidth(35);
	m_pLineEdit_FactorPropConst->setText(QString::number(m_pConfig->nirfFactorPropConst));
	m_pLineEdit_FactorPropConst->setAlignment(Qt::AlignCenter);

	m_pLabel_FactorPropConst = new QLabel("Factor Proportional Constant");
	m_pLabel_FactorPropConst->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
	m_pLabel_FactorPropConst->setBuddy(m_pLineEdit_FactorPropConst);

	m_pLineEdit_DistPropConst = new QLineEdit(this);
	m_pLineEdit_DistPropConst->setFixedWidth(35);
	m_pLineEdit_DistPropConst->setText(QString::number(m_pConfig->nirfDistPropConst));
	m_pLineEdit_DistPropConst->setAlignment(Qt::AlignCenter);

	m_pLabel_DistPropConst = new QLabel("Distance Proportional Constant  ");
	m_pLabel_DistPropConst->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
	m_pLabel_DistPropConst->setBuddy(m_pLineEdit_DistPropConst);

    // Create widgets for zero point setting
	m_pSpinBox_LumenContourOffset = new QSpinBox(this);
    m_pSpinBox_LumenContourOffset->setFixedWidth(45);
	m_pSpinBox_LumenContourOffset->setRange(0, CIRC_RADIUS);
	m_pSpinBox_LumenContourOffset->setSingleStep(1);
	m_pSpinBox_LumenContourOffset->setValue(m_pConfig->nirfLumContourOffset);
	m_pSpinBox_LumenContourOffset->setAlignment(Qt::AlignCenter);
    m_pSpinBox_LumenContourOffset->setDisabled(true);

    m_pLabel_LumenContourOffset = new QLabel("   Lumen Contour Offset  ");
	m_pLabel_LumenContourOffset->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
	m_pLabel_LumenContourOffset->setBuddy(m_pSpinBox_LumenContourOffset);
    m_pLabel_LumenContourOffset->setDisabled(true);

	m_pSpinBox_OuterSheathPosition = new QSpinBox(this);
    m_pSpinBox_OuterSheathPosition->setFixedWidth(45);
	m_pSpinBox_OuterSheathPosition->setRange(0, CIRC_RADIUS);
	m_pSpinBox_OuterSheathPosition->setSingleStep(1);
	m_pSpinBox_OuterSheathPosition->setValue(m_pConfig->nirfOuterSheathPos);
	m_pSpinBox_OuterSheathPosition->setAlignment(Qt::AlignCenter);
    m_pSpinBox_OuterSheathPosition->setDisabled(true);
	
    m_pLabel_OuterSheathPosition = new QLabel("   Outer Sheath Position");
	m_pLabel_OuterSheathPosition->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
	m_pLabel_OuterSheathPosition->setBuddy(m_pSpinBox_OuterSheathPosition);
    m_pLabel_OuterSheathPosition->setDisabled(true);

    // Create widgets for TBR mode
    m_pCheckBox_Filtering = new QCheckBox(this);
    m_pCheckBox_Filtering->setText("Filtering On");

    m_pCheckBox_TBRMode = new QCheckBox(this);
    m_pCheckBox_TBRMode->setText("TBR Converting On");
    m_pCheckBox_TBRMode->setDisabled(true);

    m_pLineEdit_Background_Level = new QLineEdit(this);
    m_pLineEdit_Background_Level->setFixedWidth(35);
    m_pLineEdit_Background_Level->setText(QString::number(nirfBackgroundLevel));
    m_pLineEdit_Background_Level->setAlignment(Qt::AlignCenter);

    m_pLabel_Background_Level =  new QLabel("Background Level   ");
    m_pLabel_Background_Level->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
    m_pLabel_Background_Level->setBuddy(m_pLineEdit_Background_Level);
    m_pLabel_Background_Level->setDisabled(true);

	
    // Initialization
    loadDistanceMap();
    loadNirfBackground();
    changeCompensationCurve();
    changeZeroPointSetting();

	// Create layout
	QGridLayout *pGridLayout_Buttons = new QGridLayout;
	pGridLayout_Buttons->setSpacing(2);

	pGridLayout_Buttons->addWidget(m_pPushButton_LoadDistanceMap, 0, 0);
	pGridLayout_Buttons->addWidget(m_pPushButton_LoadNirfBackground, 0, 1);
	pGridLayout_Buttons->addWidget(m_pToggleButton_Compensation, 1, 0, 1, 2);

	QGridLayout *pGridLayout_CompCurves = new QGridLayout;
	pGridLayout_CompCurves->setSpacing(2);

	pGridLayout_CompCurves->addWidget(m_pLabel_DistanceDecayCurve, 0, 0);
	pGridLayout_CompCurves->addWidget(m_pRenderArea_DistanceDecayCurve, 1, 0);
	pGridLayout_CompCurves->addWidget(m_pLabel_CompensationCurve, 0, 1);
	pGridLayout_CompCurves->addWidget(m_pRenderArea_CompensationCurve, 1, 1);

	QGridLayout *pGridLayout_CompCoeffs = new QGridLayout;
	pGridLayout_CompCoeffs->setSpacing(1);

	pGridLayout_CompCoeffs->addWidget(m_pLabel_CompensationCoeff, 0, 0, 1, 5);
	pGridLayout_CompCoeffs->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
	for (int i = 0; i < 4; i++)
		pGridLayout_CompCoeffs->addWidget(m_pLineEdit_CompensationCoeff[i], 1, i + 1);
	
	QGridLayout *pGridLayout_CompConsts = new QGridLayout;
	pGridLayout_CompConsts->setSpacing(1);
	
	pGridLayout_CompConsts->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 0, 0);
	pGridLayout_CompConsts->addWidget(m_pLabel_FactorThreshold, 0, 1);
	pGridLayout_CompConsts->addWidget(m_pLineEdit_FactorThreshold, 0, 2);
    pGridLayout_CompConsts->addWidget(m_pLabel_LumenContourOffset, 0, 3);
    pGridLayout_CompConsts->addWidget(m_pSpinBox_LumenContourOffset, 0, 4);

	pGridLayout_CompConsts->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
	pGridLayout_CompConsts->addWidget(m_pLabel_FactorPropConst, 1, 1);
	pGridLayout_CompConsts->addWidget(m_pLineEdit_FactorPropConst, 1, 2);
    pGridLayout_CompConsts->addWidget(m_pLabel_OuterSheathPosition, 1, 3);
    pGridLayout_CompConsts->addWidget(m_pSpinBox_OuterSheathPosition, 1, 4);

	pGridLayout_CompConsts->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 2, 0);
	pGridLayout_CompConsts->addWidget(m_pLabel_DistPropConst, 2, 1);
	pGridLayout_CompConsts->addWidget(m_pLineEdit_DistPropConst, 2, 2);

    QHBoxLayout *pHBoxLayout_TBR = new QHBoxLayout;
    pHBoxLayout_TBR->setSpacing(1);

    pHBoxLayout_TBR->addWidget(m_pCheckBox_Filtering);
    pHBoxLayout_TBR->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));
    pHBoxLayout_TBR->addWidget(m_pCheckBox_TBRMode);
    pHBoxLayout_TBR->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));
    pHBoxLayout_TBR->addWidget(m_pLabel_Background_Level);
    pHBoxLayout_TBR->addWidget(m_pLineEdit_Background_Level);


	m_pVBoxLayout = new QVBoxLayout;
	m_pVBoxLayout->setSpacing(5);

	m_pVBoxLayout->addItem(pGridLayout_Buttons);
	m_pVBoxLayout->addItem(pGridLayout_CompCurves);
	m_pVBoxLayout->addItem(pGridLayout_CompCoeffs);
    m_pVBoxLayout->addItem(pGridLayout_CompConsts);
    m_pVBoxLayout->addItem(pHBoxLayout_TBR);
	
	// Set layout
	this->setLayout(m_pVBoxLayout);

	// Connect
	connect(m_pPushButton_LoadDistanceMap, SIGNAL(clicked(bool)), this, SLOT(loadDistanceMap()));
	connect(m_pPushButton_LoadNirfBackground, SIGNAL(clicked(bool)), this, SLOT(loadNirfBackground()));
    connect(m_pToggleButton_Compensation, SIGNAL(toggled(bool)), this, SLOT(compensation(bool)));
	connect(m_pLineEdit_CompensationCoeff[0], SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	connect(m_pLineEdit_CompensationCoeff[1], SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	connect(m_pLineEdit_CompensationCoeff[2], SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	connect(m_pLineEdit_CompensationCoeff[3], SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	connect(m_pLineEdit_FactorThreshold, SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	connect(m_pLineEdit_FactorPropConst, SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	connect(m_pLineEdit_DistPropConst, SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
    connect(m_pSpinBox_LumenContourOffset, SIGNAL(valueChanged(int)), this, SLOT(changeZeroPointSetting()));
    connect(m_pSpinBox_OuterSheathPosition, SIGNAL(valueChanged(int)), this, SLOT(changeZeroPointSetting()));
    connect(m_pCheckBox_Filtering, SIGNAL(toggled(bool)), this, SLOT(filtering(bool)));
    connect(m_pCheckBox_TBRMode, SIGNAL(toggled(bool)), this, SLOT(tbrConvering(bool)));
    connect(m_pLineEdit_Background_Level, SIGNAL(textChanged(const QString &)), this, SLOT(chagneBackgroundLevel(const QString &)));
}

NirfDistCompDlg::~NirfDistCompDlg()
{
}

void NirfDistCompDlg::keyPressEvent(QKeyEvent *e)
{
	if (e->key() != Qt::Key_Escape)
		QDialog::keyPressEvent(e);
}


void NirfDistCompDlg::loadDistanceMap()
{
    QString distMapName = m_pResultTab->m_path + "/dist_map.bin";

    QFile file(distMapName);
    if (false == file.open(QFile::ReadOnly))
        printf("[ERROR] Invalid external data or there is no such a file (dist_map.bin)!\n");
    else
    {
        distMap = np::Uint16Array2(m_pResultTab->m_nirfMap.size(0), m_pResultTab->m_nirfMap.size(1));
        file.read(reinterpret_cast<char*>(distMap.raw_ptr()), sizeof(uint16_t) * distMap.length());
        file.close();

        compMap = np::FloatArray2((int)m_pResultTab->m_vectorOctImage.at(0).size(1), (int)m_pResultTab->m_vectorOctImage.size());

        // Update widgets states
        m_pSpinBox_LumenContourOffset->setEnabled(true);
        m_pLabel_LumenContourOffset->setEnabled(true);
        m_pSpinBox_OuterSheathPosition->setEnabled(true);
        m_pLabel_OuterSheathPosition->setEnabled(true);

        m_pPushButton_LoadDistanceMap->setDisabled(true);
        if (!m_pPushButton_LoadDistanceMap->isEnabled() && !m_pPushButton_LoadNirfBackground->isEnabled())
            m_pToggleButton_Compensation->setEnabled(true);
    }
}

void NirfDistCompDlg::loadNirfBackground()
{
    QString distMapName = m_pResultTab->m_path + "/NIRFbg.bin";

    QFile file(distMapName);
    if (false == file.open(QFile::ReadOnly))
        printf("[ERROR] Invalid external data or there is no such a file (NIRFbg.bin)!\n");
    else
    {
        np::DoubleArray nirfBgMap((int)file.size() / sizeof(double));
        file.read(reinterpret_cast<char*>(nirfBgMap.raw_ptr()), sizeof(double) * nirfBgMap.length());
        file.close();

        double temp_bg;
        ippsMean_64f(nirfBgMap.raw_ptr(), nirfBgMap.length(), &temp_bg);

        nirfBg = temp_bg;

        // Update widgets states
        m_pPushButton_LoadNirfBackground->setDisabled(true);
        if (!m_pPushButton_LoadDistanceMap->isEnabled() && !m_pPushButton_LoadNirfBackground->isEnabled())
            m_pToggleButton_Compensation->setEnabled(true);
    }
}

void NirfDistCompDlg::compensation(bool toggled)
{
    if (toggled)
    {
        m_pToggleButton_Compensation->setText("Compensation Off");

        m_pCheckBox_TBRMode->setEnabled(true);
        m_pLabel_Background_Level->setEnabled(true);
        m_pLineEdit_Background_Level->setEnabled(true);
    }
    else
    {
        m_pToggleButton_Compensation->setText("Compensation On");

        m_pCheckBox_TBRMode->setChecked(false);
        m_pCheckBox_TBRMode->setDisabled(true);
        m_pLabel_Background_Level->setDisabled(true);
        m_pLineEdit_Background_Level->setDisabled(true);
    }

    // Update compensation map
    if (toggled) calculateCompMap();

    // Invalidate
    m_pResultTab->invalidate();
}

void NirfDistCompDlg::changeCompensationCurve()
{
    // Get and update values
	float a = m_pLineEdit_CompensationCoeff[0]->text().toFloat();
	float b = m_pLineEdit_CompensationCoeff[1]->text().toFloat();
	float c = m_pLineEdit_CompensationCoeff[2]->text().toFloat();
	float d = m_pLineEdit_CompensationCoeff[3]->text().toFloat();

	float factor_thres = m_pLineEdit_FactorThreshold->text().toFloat();
	float factor_prop_const = m_pLineEdit_FactorPropConst->text().toFloat();
	float dist_prop_const = m_pLineEdit_DistPropConst->text().toFloat();

	m_pConfig->nirfCompCoeffs[0] = a;
	m_pConfig->nirfCompCoeffs[1] = b;
	m_pConfig->nirfCompCoeffs[2] = c;
	m_pConfig->nirfCompCoeffs[3] = d;

	m_pConfig->nirfFactorThres = factor_thres;
	m_pConfig->nirfFactorPropConst = factor_prop_const;
	m_pConfig->nirfDistPropConst = dist_prop_const;

    // Update compensation curve
	m_pRenderArea_CompensationCurve->setSize({ 0, CIRC_RADIUS }, { 0, round(factor_thres * 1.1) });
    m_pLabel_CompensationCurve->setText(QString("Compensation Curve ([0, %1])").arg((int)round(factor_thres * 1.1)));
	
	for (int i = 0; i < CIRC_RADIUS; i++)
	{
		if ((a + c) != 0)
		{
			distDecayCurve[i] = (a * expf(b * dist_prop_const * (float)i) + c * expf(d * dist_prop_const * (float)i)) / (a + c);
			compCurve[i] = (i < 20) ? 1.0f : 1 / distDecayCurve[i] / factor_prop_const;
			if (compCurve[i] < 1.0f) compCurve[i] = 1.0f;
            if (compCurve[i] > factor_thres) compCurve[i] = factor_thres;
		}
		else
		{
			distDecayCurve[i] = 0.0f;
			compCurve[i] = 1.0f;
		}
	}

	memcpy(m_pRenderArea_DistanceDecayCurve->m_pData, distDecayCurve.raw_ptr(), sizeof(float) * distDecayCurve.length());
	memcpy(m_pRenderArea_CompensationCurve->m_pData, compCurve.raw_ptr(), sizeof(float) * compCurve.length());
	m_pRenderArea_DistanceDecayCurve->update();
	m_pRenderArea_CompensationCurve->update();    

    // Update compensation map
    if (isCompensating()) calculateCompMap();

    // Invalidate
    m_pResultTab->invalidate();
}

void NirfDistCompDlg::changeZeroPointSetting()
{
    // Get and update values
    int lum_offset = m_pSpinBox_LumenContourOffset->value();
    int sheath_pos = m_pSpinBox_OuterSheathPosition->value();

    m_pConfig->nirfLumContourOffset = lum_offset;
    m_pConfig->nirfOuterSheathPos = sheath_pos;

    // Update compensation map
    if (isCompensating()) calculateCompMap();

    // Invalidate
    m_pResultTab->invalidate();
}

void NirfDistCompDlg::calculateCompMap()
{
    np::Uint16Array2 distOffsetMap(distMap.size(0), distMap.size(1));
    memcpy(distOffsetMap.raw_ptr(), distMap.raw_ptr(), sizeof(uint16_t) * distMap.length());
    ippsAddC_16u_ISfs(m_pConfig->nirfLumContourOffset, distOffsetMap.raw_ptr(), distOffsetMap.length(), 0);
    ippsSubC_16u_ISfs(m_pConfig->nirfOuterSheathPos, distOffsetMap.raw_ptr(), distOffsetMap.length(), 0);

    memset(compMap.raw_ptr(), 0, sizeof(float) * compMap.length());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)distOffsetMap.size(1)),
        [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i)
        {
            for (int j = 0; j < distOffsetMap.size(0); j++)
            {
                if (distOffsetMap(j, (int)i) > CIRC_RADIUS)
                    distOffsetMap(j, (int)i) = CIRC_RADIUS - 1;

                compMap(j, (int)i) = compCurve[distOffsetMap(j, (int)i)];
            }
        }
    });
}

void NirfDistCompDlg::filtering(bool toggled)
{
    if (toggled)
        m_pCheckBox_Filtering->setText("Filtering Off");
    else
        m_pCheckBox_Filtering->setText("Filtering On");

    // Invalidate
    m_pResultTab->invalidate();
}

void NirfDistCompDlg::tbrConvering(bool toggled)
{
    if (toggled)
        m_pCheckBox_TBRMode->setText("TBR Converting Off");
    else
        m_pCheckBox_TBRMode->setText("TBR Converting On");

    nirfBackgroundLevel = m_pLineEdit_Background_Level->text().toFloat();

    // Invalidate
    m_pResultTab->invalidate();
}

void NirfDistCompDlg::chagneBackgroundLevel(const QString & str)
{
    nirfBackgroundLevel = str.toFloat();

    // Invalidate
    m_pResultTab->invalidate();
}

#endif
