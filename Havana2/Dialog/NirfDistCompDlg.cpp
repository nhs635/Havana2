
#include "NirfDistCompDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QResultTab.h>

#include <iostream>
#include <thread>

#include <ipps.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <QFileInfo>
#include <QSettings>
#include <QDateTime>



#ifdef OCT_NIRF
NirfDistCompDlg::NirfDistCompDlg(QWidget *parent) :
    QDialog(parent), m_bCanBeClosed(true), nirfBg(0), tbrBg(0)
{
    // Set default size & frame
    setFixedSize(380, 380);
    setWindowFlags(Qt::Tool);
	setWindowTitle("NIRF Distance Compensation");

    // Set main window objects
    m_pResultTab = (QResultTab*)parent;
    m_pMainWnd = m_pResultTab->getMainWnd();
	m_pConfig = m_pMainWnd->m_pConfiguration;

	// Set compensation data
	compMap = np::FloatArray2();
    distDecayCurve = np::FloatArray(m_pConfig->circRadius);
	compCurve = np::FloatArray(m_pConfig->circRadius);
	
	// Create widgets for data loading & compensation
	m_pPushButton_LoadDistanceMap = new QPushButton(this);
    m_pPushButton_LoadDistanceMap->setText("Load Distance Map");

	m_pPushButton_LoadNirfBackground = new QPushButton(this);
    m_pPushButton_LoadNirfBackground->setText("Load NIRF Background");

	m_pToggleButton_Compensation = new QPushButton(this);
	m_pToggleButton_Compensation->setText("Compensation On");
	m_pToggleButton_Compensation->setCheckable(true);
    m_pToggleButton_Compensation->setDisabled(true);
	
	m_pToggleButton_TBRMode = new QPushButton(this);
	m_pToggleButton_TBRMode->setText("TBR Converting On");
	m_pToggleButton_TBRMode->setCheckable(true);
	m_pToggleButton_TBRMode->setDisabled(true);
	
	// Create widgets for compensation details
	m_pRenderArea_DistanceDecayCurve = new QRenderArea(this);
	m_pRenderArea_DistanceDecayCurve->setSize({ 0, (double)m_pConfig->circRadius }, { 0, 1 });
	m_pRenderArea_DistanceDecayCurve->setMinimumHeight(80);
	m_pRenderArea_DistanceDecayCurve->setGrid(4, 8, 1);
	
	m_pLabel_DistanceDecayCurve = new QLabel("Distance Decay Curve ([0, 1])", this);
	m_pLabel_DistanceDecayCurve->setBuddy(m_pRenderArea_DistanceDecayCurve);
	
	m_pRenderArea_CompensationCurve = new QRenderArea(this);
	m_pRenderArea_CompensationCurve->setSize({ 0, (double)m_pConfig->circRadius }, { 0, round(m_pConfig->nirfFactorThres * 1.1) });
	m_pRenderArea_CompensationCurve->setMinimumHeight(80);
	m_pRenderArea_CompensationCurve->setGrid(4, 8, 1);

	m_pLabel_CompensationCurve = new QLabel(QString("Compensation Curve ([0, %1])").arg((int)round(m_pConfig->nirfFactorThres * 1.1)), this);
	m_pLabel_CompensationCurve->setBuddy(m_pRenderArea_CompensationCurve);
		
	for (int i = 0; i < 4; i++)
	{
		m_pLineEdit_CompensationCoeff[i] = new QLineEdit(this);
		m_pLineEdit_CompensationCoeff[i]->setFixedWidth(70);
		m_pLineEdit_CompensationCoeff[i]->setAlignment(Qt::AlignCenter);
	}
	m_pLineEdit_CompensationCoeff[0]->setText(QString::number(m_pConfig->nirfCompCoeffs_a));
	m_pLineEdit_CompensationCoeff[1]->setText(QString::number(m_pConfig->nirfCompCoeffs_b));
	m_pLineEdit_CompensationCoeff[2]->setText(QString::number(m_pConfig->nirfCompCoeffs_c));
	m_pLineEdit_CompensationCoeff[3]->setText(QString::number(m_pConfig->nirfCompCoeffs_d));

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
	m_pSpinBox_LumenContourOffset->setRange(0, m_pConfig->circRadius);
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
	m_pSpinBox_OuterSheathPosition->setRange(0, m_pConfig->circRadius);
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
    m_pCheckBox_Filtering->setText("Filtering On  ");

	m_pCheckBox_ZeroTBRDefinition = new QCheckBox(this);
	m_pCheckBox_ZeroTBRDefinition->setText("Zero TBR Definition");
	m_pCheckBox_ZeroTBRDefinition->setChecked(true);
	m_pCheckBox_ZeroTBRDefinition->setDisabled(true);
	
	m_pLineEdit_NIRF_Background = new QLineEdit(this);
	m_pLineEdit_NIRF_Background->setFixedWidth(35);
	m_pLineEdit_NIRF_Background->setText(QString::number(nirfBg));
	m_pLineEdit_NIRF_Background->setAlignment(Qt::AlignCenter);
	m_pLineEdit_NIRF_Background->setDisabled(true);

	m_pLabel_NIRF_Background = new QLabel("NIRF Background   ");
	m_pLabel_NIRF_Background->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
	m_pLabel_NIRF_Background->setBuddy(m_pLineEdit_NIRF_Background);
	m_pLabel_NIRF_Background->setDisabled(true);
	
    m_pLineEdit_TBR_Background = new QLineEdit(this);
	m_pLineEdit_TBR_Background->setFixedWidth(35);
	m_pLineEdit_TBR_Background->setText(QString::number(tbrBg));
	m_pLineEdit_TBR_Background->setAlignment(Qt::AlignCenter);
	m_pLineEdit_TBR_Background->setDisabled(true);

    m_pLabel_TBR_Background =  new QLabel("   TBR Background   ");
	m_pLabel_TBR_Background->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
	m_pLabel_TBR_Background->setBuddy(m_pLineEdit_TBR_Background);
	m_pLabel_TBR_Background->setDisabled(true);
	
	// Create widgets for guide line indicator
	m_pCheckBox_ShowLumenContour = new QCheckBox(this);
	m_pCheckBox_ShowLumenContour->setText("Show Lumen Contour  ");
	m_pCheckBox_ShowLumenContour->setChecked(true);

	
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
	pGridLayout_Buttons->addWidget(m_pToggleButton_TBRMode, 2, 0, 1, 2);

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

    pHBoxLayout_TBR->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));
	pHBoxLayout_TBR->addWidget(m_pLabel_NIRF_Background);
	pHBoxLayout_TBR->addWidget(m_pLineEdit_NIRF_Background);
    pHBoxLayout_TBR->addWidget(m_pLabel_TBR_Background);
    pHBoxLayout_TBR->addWidget(m_pLineEdit_TBR_Background);

	QHBoxLayout *pHBoxLayout_TBR_Option = new QHBoxLayout;
	pHBoxLayout_TBR_Option->setSpacing(1);

	pHBoxLayout_TBR_Option->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));
	pHBoxLayout_TBR_Option->addWidget(m_pCheckBox_ShowLumenContour);
	pHBoxLayout_TBR_Option->addWidget(m_pCheckBox_Filtering);
	pHBoxLayout_TBR_Option->addWidget(m_pCheckBox_ZeroTBRDefinition);



	m_pVBoxLayout = new QVBoxLayout;
	m_pVBoxLayout->setSpacing(5);

	m_pVBoxLayout->addItem(pGridLayout_Buttons);
	m_pVBoxLayout->addItem(pGridLayout_CompCurves);
	m_pVBoxLayout->addItem(pGridLayout_CompCoeffs);
    m_pVBoxLayout->addItem(pGridLayout_CompConsts);
    m_pVBoxLayout->addItem(pHBoxLayout_TBR);
	m_pVBoxLayout->addItem(pHBoxLayout_TBR_Option);
	
	// Set layout
	this->setLayout(m_pVBoxLayout);

	// Connect
	connect(m_pPushButton_LoadDistanceMap, SIGNAL(clicked(bool)), this, SLOT(loadDistanceMap()));
	connect(m_pPushButton_LoadNirfBackground, SIGNAL(clicked(bool)), this, SLOT(loadNirfBackground()));
    connect(m_pToggleButton_Compensation, SIGNAL(toggled(bool)), this, SLOT(compensation(bool)));
	connect(m_pToggleButton_TBRMode, SIGNAL(toggled(bool)), this, SLOT(tbrConvering(bool)));
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
	connect(m_pCheckBox_ZeroTBRDefinition, SIGNAL(toggled(bool)), this, SLOT(tbrZeroDefinition(bool)));
	connect(m_pLineEdit_NIRF_Background, SIGNAL(textChanged(const QString &)), this, SLOT(changeNirfBackground(const QString &)));
    connect(m_pLineEdit_TBR_Background, SIGNAL(textChanged(const QString &)), this, SLOT(changeTbrBackground(const QString &)));
	connect(m_pCheckBox_ShowLumenContour, SIGNAL(toggled(bool)), this, SLOT(showLumenContour(bool)));
	connect(this, SIGNAL(setWidgets(bool)), this, SLOT(setWidgetEnabled(bool)));

	// Get Log file
	QString infopath = m_pResultTab->m_path + QString("/dist_comp_info.log");
	if (QFileInfo::exists(infopath)) getCompInfo(infopath);
}

NirfDistCompDlg::~NirfDistCompDlg()
{
}

void NirfDistCompDlg::closeEvent(QCloseEvent * e)
{
	if (!m_bCanBeClosed)
		e->ignore();
	else
		finished(0);
}

void NirfDistCompDlg::keyPressEvent(QKeyEvent *e)
{
	if (e->key() != Qt::Key_Escape)
		QDialog::keyPressEvent(e);
}


void NirfDistCompDlg::setWidgetEnabled(bool enabled)
{
	m_pPushButton_LoadDistanceMap->setEnabled(enabled && !(compMap.raw_ptr()));
	m_pPushButton_LoadNirfBackground->setEnabled(enabled && !(nirfBg > 0));

	m_pToggleButton_Compensation->setEnabled(enabled && (compMap.raw_ptr()) && (nirfBg > 0));
	m_pToggleButton_TBRMode->setEnabled(enabled && (compMap.raw_ptr()) && (nirfBg > 0));

	m_pLabel_DistanceDecayCurve->setEnabled(enabled);
	m_pLabel_CompensationCurve->setEnabled(enabled);

	m_pRenderArea_DistanceDecayCurve->setEnabled(enabled);
	m_pRenderArea_CompensationCurve->setEnabled(enabled);

	m_pLabel_CompensationCoeff->setEnabled(enabled);
	for (int i = 0; i < 4; i++)
		m_pLineEdit_CompensationCoeff[i]->setEnabled(enabled);

	m_pLabel_FactorThreshold->setEnabled(enabled);
	m_pLineEdit_FactorThreshold->setEnabled(enabled);

	m_pLabel_FactorPropConst->setEnabled(enabled);
	m_pLineEdit_FactorPropConst->setEnabled(enabled);

	m_pLabel_DistPropConst->setEnabled(enabled);
	m_pLineEdit_DistPropConst->setEnabled(enabled);
		
	m_pLabel_LumenContourOffset->setEnabled(enabled && (compMap.raw_ptr()));
	m_pSpinBox_LumenContourOffset->setEnabled(enabled && (compMap.raw_ptr()));

	m_pLabel_OuterSheathPosition->setEnabled(enabled && (compMap.raw_ptr()));
	m_pSpinBox_OuterSheathPosition->setEnabled(enabled && (compMap.raw_ptr()));

	m_pCheckBox_Filtering->setEnabled(enabled);

	m_pLabel_NIRF_Background->setEnabled(enabled && (nirfBg > 0));
	m_pLineEdit_NIRF_Background->setEnabled(enabled && (nirfBg > 0));

	m_pLabel_TBR_Background->setEnabled(enabled && (nirfBg > 0));
	m_pLineEdit_TBR_Background->setEnabled(enabled && (nirfBg > 0));

	//zero tbr ����...
}

void NirfDistCompDlg::loadDistanceMap()
{
    QString distMapName = m_pResultTab->m_path + "/dist_map.bin";

    QFile file(distMapName);
    if (false == file.open(QFile::ReadOnly))
        printf("[ERROR] Invalid external data or there is no such a file (dist_map.bin)!\n");
    else
    {
#ifndef TWO_CHANNEL_NIRF
        distMap = np::Uint16Array2(m_pResultTab->m_nirfMap.size(0), m_pResultTab->m_nirfMap.size(1));
#else
		distMap = np::Uint16Array2(m_pResultTab->m_nirfMap1.size(0), m_pResultTab->m_nirfMap1.size(1));
#endif
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
    QString distMapName = m_pResultTab->m_path + "/nirf_bg.bin";

    QFile file(distMapName);
    if (false == file.open(QFile::ReadOnly))
        printf("[ERROR] Invalid external data or there is no such a file (nirf_bg.bin)!\n");
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
		m_pLabel_NIRF_Background->setEnabled(true);
		m_pLineEdit_NIRF_Background->setEnabled(true);
		m_pLineEdit_NIRF_Background->setText(QString::number(nirfBg, 'f', 3));
        if (!m_pPushButton_LoadDistanceMap->isEnabled() && !m_pPushButton_LoadNirfBackground->isEnabled())
            m_pToggleButton_Compensation->setEnabled(true);
    }
}

void NirfDistCompDlg::compensation(bool toggled)
{
    if (toggled)
    {
        m_pToggleButton_Compensation->setText("Compensation Off");

        m_pToggleButton_TBRMode->setEnabled(true);
        m_pLabel_TBR_Background->setEnabled(true);
        m_pLineEdit_TBR_Background->setEnabled(true);
    }
    else
    {
        m_pToggleButton_Compensation->setText("Compensation On");

		m_pToggleButton_TBRMode->setChecked(false);
		m_pToggleButton_TBRMode->setDisabled(true);
        m_pLabel_TBR_Background->setDisabled(true);
        m_pLineEdit_TBR_Background->setDisabled(true);
    }

    // Update compensation map
    if (toggled) calculateCompMap();

    // Invalidate
    m_pResultTab->invalidate();
}

void NirfDistCompDlg::tbrConvering(bool toggled)
{
	if (toggled)
	{
		m_pToggleButton_TBRMode->setText("TBR Converting Off");

		m_pCheckBox_ZeroTBRDefinition->setEnabled(true);
	}
	else
	{
		m_pToggleButton_TBRMode->setText("TBR Converting On");

		m_pCheckBox_ZeroTBRDefinition->setDisabled(true);
	}

	tbrBg = m_pLineEdit_TBR_Background->text().toFloat();

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

	m_pConfig->nirfCompCoeffs_a = a;
	m_pConfig->nirfCompCoeffs_b = b;
	m_pConfig->nirfCompCoeffs_c = c;
	m_pConfig->nirfCompCoeffs_d = d;

	m_pConfig->nirfFactorThres = factor_thres;
	m_pConfig->nirfFactorPropConst = factor_prop_const;
	m_pConfig->nirfDistPropConst = dist_prop_const;

    // Update compensation curve
	m_pRenderArea_CompensationCurve->setSize({ 0, (double)m_pConfig->circRadius }, { 0, round(factor_thres * 1.1) });
    m_pLabel_CompensationCurve->setText(QString("Compensation Curve ([0, %1])").arg((int)round(factor_thres * 1.1)));
	
	for (int i = 0; i < m_pConfig->circRadius; i++)
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
                if (distOffsetMap(j, (int)i) > m_pConfig->circRadius)
                    distOffsetMap(j, (int)i) = m_pConfig->circRadius - 1;

                compMap(j, (int)i) = compCurve[distOffsetMap(j, (int)i)];
            }
        }
    });
}

void NirfDistCompDlg::filtering(bool toggled)
{
    if (toggled)
        m_pCheckBox_Filtering->setText("Filtering Off  ");
    else
        m_pCheckBox_Filtering->setText("Filtering On  ");

    // Invalidate
    m_pResultTab->invalidate();
}

void NirfDistCompDlg::tbrZeroDefinition(bool toggled)
{
	// Invalidate
	m_pResultTab->invalidate();

	(void)toggled;
}

void NirfDistCompDlg::changeNirfBackground(const QString &str)
{
	nirfBg = str.toFloat();

	// Invalidate
	m_pResultTab->invalidate();
}

void NirfDistCompDlg::changeTbrBackground(const QString &str)
{
    tbrBg = str.toFloat();

    // Invalidate
    m_pResultTab->invalidate();
}

void NirfDistCompDlg::showLumenContour(bool)
{
	// Invalidate
	m_pResultTab->invalidate();
}


void NirfDistCompDlg::getCompInfo(const QString &infopath)
{
	QSettings settings(infopath, QSettings::IniFormat);
	settings.beginGroup("distance_compensation");

	// Load parameters
	int nirfOffset = settings.value("nirfOffset").toInt();
	m_pResultTab->setNirfOffset(nirfOffset);

	m_pConfig->nirfCompCoeffs_a = settings.value("nirfCompCoeffs_a").toFloat();
	m_pConfig->nirfCompCoeffs_b = settings.value("nirfCompCoeffs_b").toFloat();
	m_pConfig->nirfCompCoeffs_c = settings.value("nirfCompCoeffs_c").toFloat();
	m_pConfig->nirfCompCoeffs_d = settings.value("nirfCompCoeffs_d").toFloat();

	m_pLineEdit_CompensationCoeff[0]->setText(QString::number(m_pConfig->nirfCompCoeffs_a));
	m_pLineEdit_CompensationCoeff[1]->setText(QString::number(m_pConfig->nirfCompCoeffs_b));
	m_pLineEdit_CompensationCoeff[2]->setText(QString::number(m_pConfig->nirfCompCoeffs_c));
	m_pLineEdit_CompensationCoeff[3]->setText(QString::number(m_pConfig->nirfCompCoeffs_d));

	m_pConfig->nirfFactorThres = settings.value("nirfFactorThres").toFloat();
	m_pLineEdit_FactorThreshold->setText(QString::number(m_pConfig->nirfFactorThres));

	m_pConfig->nirfFactorPropConst = settings.value("nirfFactorPropConst").toFloat();
	m_pLineEdit_FactorPropConst->setText(QString::number(m_pConfig->nirfFactorPropConst));

	m_pConfig->nirfDistPropConst = settings.value("nirfDistPropConst").toFloat();
	m_pLineEdit_DistPropConst->setText(QString::number(m_pConfig->nirfDistPropConst));

	m_pConfig->nirfLumContourOffset = settings.value("nirfLumContourOffset").toInt();
	m_pSpinBox_LumenContourOffset->setValue(m_pConfig->nirfLumContourOffset);

	m_pConfig->nirfOuterSheathPos = settings.value("nirfOuterSheathPos").toInt();
	m_pSpinBox_OuterSheathPosition->setValue(m_pConfig->nirfOuterSheathPos);

	calculateCompMap();

	nirfBg = settings.value("nirfBg").toFloat();;
	m_pLineEdit_NIRF_Background->setText(QString::number(nirfBg));

	tbrBg = settings.value("tbrBg").toFloat();;
	m_pLineEdit_TBR_Background->setText(QString::number(tbrBg));

	m_pCheckBox_Filtering->setChecked(true);

	m_pToggleButton_Compensation->setChecked(true);
	m_pToggleButton_TBRMode->setChecked(true);

	bool isZeroTbr = settings.value("zeroTbrDefinition").toBool();
	m_pCheckBox_ZeroTBRDefinition->setChecked(isZeroTbr);

	settings.endGroup();
}

void NirfDistCompDlg::setCompInfo(const QString &infopath)
{
	QSettings settings(infopath, QSettings::IniFormat);
	settings.beginGroup("distance_compensation");

	// Load parameters
	QDate date = QDate::currentDate();
	QTime time = QTime::currentTime();
	settings.setValue("time", QString("%1-%2-%3 %4-%5-%6")
		.arg(date.year()).arg(date.month(), 2, 10, (QChar)'0').arg(date.day(), 2, 10, (QChar)'0')
		.arg(time.hour(), 2, 10, (QChar)'0').arg(time.minute(), 2, 10, (QChar)'0').arg(time.second(), 2, 10, (QChar)'0'));

	settings.setValue("nirfOffset", m_pResultTab->getCurrentNirfOffset());
	
	settings.setValue("nirfCompCoeffs_a", QString::number(m_pConfig->nirfCompCoeffs_a, 'f', 10));
	settings.setValue("nirfCompCoeffs_b", QString::number(m_pConfig->nirfCompCoeffs_b, 'f', 10));
	settings.setValue("nirfCompCoeffs_c", QString::number(m_pConfig->nirfCompCoeffs_c, 'f', 10));
	settings.setValue("nirfCompCoeffs_d", QString::number(m_pConfig->nirfCompCoeffs_d, 'f', 10));
	
	settings.setValue("nirfFactorThres", QString::number(m_pConfig->nirfFactorThres, 'f', 1));
	settings.setValue("nirfFactorPropConst", QString::number(m_pConfig->nirfFactorPropConst, 'f', 3));
	settings.setValue("nirfDistPropConst", QString::number(m_pConfig->nirfDistPropConst, 'f', 3));

	settings.setValue("nirfLumContourOffset", m_pConfig->nirfLumContourOffset);
	settings.setValue("nirfOuterSheathPos", m_pConfig->nirfOuterSheathPos);

	settings.setValue("nirfBg", QString::number(nirfBg, 'f', 3));
	settings.setValue("tbrBg", QString::number(tbrBg, 'f', 3));

	if (!m_pResultTab->getPolishedSurfaceFindingStatus())
		settings.setValue("circCenter", QString::number(m_pConfig->circCenter));
	else
		settings.setValue("ballRadius", QString::number(m_pConfig->ballRadius));
	settings.setValue("circRadius", QString::number(m_pConfig->circRadius));
	
	settings.setValue("zeroTbrDefinition", QString::number(m_pCheckBox_ZeroTBRDefinition->isChecked()));

	settings.endGroup();
}

#endif
