
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
	QDialog(parent), m_bCanBeClosed(true), compConst(1.0)
#ifndef TWO_CHANNEL_NIRF
	, nirfBg(0.0), tbrBg(0.0)
#else
	, crossTalkRatio(0.0)
#endif
{
	// Set default size & frame
#ifndef TWO_CHANNEL_NIRF
	setFixedSize(380, 565);
#else
	setFixedSize(380, 605);
#endif
	setWindowFlags(Qt::Tool);
	setWindowTitle("NIRF Distance Compensation");

	// Set main window objects
	m_pResultTab = (QResultTab*)parent;
	m_pMainWnd = m_pResultTab->getMainWnd();
	m_pConfig = m_pMainWnd->m_pConfiguration;

	// Set compensation data
#ifndef TWO_CHANNEL_NIRF
	compMap = np::FloatArray2();
	distDecayCurve = np::FloatArray(m_pConfig->circRadius);
	compCurve = np::FloatArray(m_pConfig->circRadius);
#else
	for (int i = 0; i < 2; i++)
	{
		compMap[i] = np::FloatArray2();
		distDecayCurve[i] = np::FloatArray(m_pConfig->circRadius);
		compCurve[i] = np::FloatArray(m_pConfig->circRadius);
		nirfBg[i] = 0.0f; tbrBg[i] = 0.0f;
	}
#endif

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

	m_pPushButton_ExportTBR = new QPushButton(this);
	m_pPushButton_ExportTBR->setText("Export TBR Data");
	m_pPushButton_ExportTBR->setDisabled(true);

	// Create widgets for compensation details
#ifndef TWO_CHANNEL_NIRF
	m_pRenderArea_DistanceDecayCurve = new QRenderArea(this);
#else
	m_pRenderArea_DistanceDecayCurve = new QRenderArea2(this);
#endif
	m_pRenderArea_DistanceDecayCurve->setSize({ 0, (double)m_pConfig->circRadius }, { 0, 1 });
	m_pRenderArea_DistanceDecayCurve->setMinimumHeight(80);
	m_pRenderArea_DistanceDecayCurve->setGrid(4, 8, 1);

	m_pLabel_DistanceDecayCurve = new QLabel("Distance Decay Curve ([0, 1])", this);
	m_pLabel_DistanceDecayCurve->setBuddy(m_pRenderArea_DistanceDecayCurve);

#ifndef TWO_CHANNEL_NIRF
	m_pRenderArea_CompensationCurve = new QRenderArea(this);
#else
	m_pRenderArea_CompensationCurve = new QRenderArea2(this);
#endif
	m_pRenderArea_CompensationCurve->setSize({ 0, (double)m_pConfig->circRadius }, { 0, round(m_pConfig->nirfFactorThres * 1.1) });
	m_pRenderArea_CompensationCurve->setMinimumHeight(80);
	m_pRenderArea_CompensationCurve->setGrid(4, 8, 1);

	m_pLabel_CompensationCurve = new QLabel(QString("Compensation Curve ([0, %1])").arg((int)round(m_pConfig->nirfFactorThres * 1.1)), this);
	m_pLabel_CompensationCurve->setBuddy(m_pRenderArea_CompensationCurve);

#ifndef TWO_CHANNEL_NIRF
	m_pLineEdit_CompensationCoeff_a = new QLineEdit(this);
	m_pLineEdit_CompensationCoeff_a->setFixedWidth(70);
	m_pLineEdit_CompensationCoeff_a->setAlignment(Qt::AlignCenter);
	m_pLineEdit_CompensationCoeff_a->setText(QString::number(m_pConfig->nirfCompCoeffs_a));

	m_pLineEdit_CompensationCoeff_b = new QLineEdit(this);
	m_pLineEdit_CompensationCoeff_b->setFixedWidth(70);
	m_pLineEdit_CompensationCoeff_b->setAlignment(Qt::AlignCenter);
	m_pLineEdit_CompensationCoeff_b->setText(QString::number(m_pConfig->nirfCompCoeffs_b));

	m_pLineEdit_CompensationCoeff_c = new QLineEdit(this);
	m_pLineEdit_CompensationCoeff_c->setFixedWidth(70);
	m_pLineEdit_CompensationCoeff_c->setAlignment(Qt::AlignCenter);
	m_pLineEdit_CompensationCoeff_c->setText(QString::number(m_pConfig->nirfCompCoeffs_c));

	m_pLineEdit_CompensationCoeff_d = new QLineEdit(this);
	m_pLineEdit_CompensationCoeff_d->setFixedWidth(70);
	m_pLineEdit_CompensationCoeff_d->setAlignment(Qt::AlignCenter);
	m_pLineEdit_CompensationCoeff_d->setText(QString::number(m_pConfig->nirfCompCoeffs_d));
#else
	for (int i = 0; i < 2; i++)
	{
		m_pLineEdit_CompensationCoeff_a[i] = new QLineEdit(this);
		m_pLineEdit_CompensationCoeff_a[i]->setFixedWidth(70);
		m_pLineEdit_CompensationCoeff_a[i]->setAlignment(Qt::AlignCenter);
		m_pLineEdit_CompensationCoeff_a[i]->setText(QString::number(m_pConfig->nirfCompCoeffs_a[i]));

		m_pLineEdit_CompensationCoeff_b[i] = new QLineEdit(this);
		m_pLineEdit_CompensationCoeff_b[i]->setFixedWidth(70);
		m_pLineEdit_CompensationCoeff_b[i]->setAlignment(Qt::AlignCenter);
		m_pLineEdit_CompensationCoeff_b[i]->setText(QString::number(m_pConfig->nirfCompCoeffs_b[i]));

		m_pLineEdit_CompensationCoeff_c[i] = new QLineEdit(this);
		m_pLineEdit_CompensationCoeff_c[i]->setFixedWidth(70);
		m_pLineEdit_CompensationCoeff_c[i]->setAlignment(Qt::AlignCenter);
		m_pLineEdit_CompensationCoeff_c[i]->setText(QString::number(m_pConfig->nirfCompCoeffs_c[i]));

		m_pLineEdit_CompensationCoeff_d[i] = new QLineEdit(this);
		m_pLineEdit_CompensationCoeff_d[i]->setFixedWidth(70);
		m_pLineEdit_CompensationCoeff_d[i]->setAlignment(Qt::AlignCenter);
		m_pLineEdit_CompensationCoeff_d[i]->setText(QString::number(m_pConfig->nirfCompCoeffs_d[i]));
	}
#endif

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
	//m_pCheckBox_ZeroTBRDefinition->setChecked(true);
	m_pCheckBox_ZeroTBRDefinition->setDisabled(true);
	m_pCheckBox_ZeroTBRDefinition->setVisible(false);

	m_pCheckBox_GwMasking = new QCheckBox(this);
	m_pCheckBox_GwMasking->setText("GW Masking");
	m_pCheckBox_GwMasking->setDisabled(true);
	
#ifndef TWO_CHANNEL_NIRF
	m_pLineEdit_NIRF_Background = new QLineEdit(this);
	m_pLineEdit_NIRF_Background->setFixedWidth(45);
	m_pLineEdit_NIRF_Background->setText(QString::number(nirfBg));
	m_pLineEdit_NIRF_Background->setAlignment(Qt::AlignCenter);
	m_pLineEdit_NIRF_Background->setDisabled(true);

	m_pLineEdit_TBR_Background = new QLineEdit(this);
	m_pLineEdit_TBR_Background->setFixedWidth(45);
	m_pLineEdit_TBR_Background->setText(QString::number(tbrBg));
	m_pLineEdit_TBR_Background->setAlignment(Qt::AlignCenter);
	m_pLineEdit_TBR_Background->setDisabled(true);
#else
	for (int i = 0; i < 2; i++)
	{
		m_pLineEdit_NIRF_Background[i] = new QLineEdit(this);
		m_pLineEdit_NIRF_Background[i]->setFixedWidth(45);
		m_pLineEdit_NIRF_Background[i]->setText(QString::number(nirfBg[i]));
		m_pLineEdit_NIRF_Background[i]->setAlignment(Qt::AlignCenter);
		m_pLineEdit_NIRF_Background[i]->setDisabled(true);

		m_pLineEdit_TBR_Background[i] = new QLineEdit(this);
		m_pLineEdit_TBR_Background[i]->setFixedWidth(45);
		m_pLineEdit_TBR_Background[i]->setText(QString::number(tbrBg[i]));
		m_pLineEdit_TBR_Background[i]->setAlignment(Qt::AlignCenter);
		m_pLineEdit_TBR_Background[i]->setDisabled(true);
	}
#endif

	m_pLabel_NIRF_Background = new QLabel("NIRF BG   ");
	m_pLabel_NIRF_Background->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
	m_pLabel_NIRF_Background->setDisabled(true);

	m_pLabel_TBR_Background = new QLabel("   TBR BG   ");
	m_pLabel_TBR_Background->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
	m_pLabel_TBR_Background->setDisabled(true);

	m_pLineEdit_Compensation = new QLineEdit(this);
	m_pLineEdit_Compensation->setFixedWidth(45);
	m_pLineEdit_Compensation->setText(QString::number(compConst, 'f', 4));
	m_pLineEdit_Compensation->setAlignment(Qt::AlignCenter);
	m_pLineEdit_Compensation->setDisabled(true);

	m_pLabel_Compensation = new QLabel("   Comp Constant");
	m_pLabel_Compensation->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
	m_pLabel_Compensation->setBuddy(m_pLineEdit_Compensation);
	m_pLabel_Compensation->setDisabled(true);

#ifdef TWO_CHANNEL_NIRF	
	m_pLineEdit_CrossTalkRatio = new QLineEdit(this);
	m_pLineEdit_CrossTalkRatio->setFixedWidth(45);
	m_pLineEdit_CrossTalkRatio->setText(QString::number(crossTalkRatio, 'f', 4));
	m_pLineEdit_CrossTalkRatio->setAlignment(Qt::AlignCenter);
	m_pLineEdit_CrossTalkRatio->setDisabled(true);

	m_pLabel_CrossTalkRatio = new QLabel("   Crosstalk Ratio (Ch2 = Ch2 - r*Ch1)   ");
	m_pLabel_CrossTalkRatio->setAlignment(Qt::AlignVCenter | Qt::AlignRight);
	m_pLabel_CrossTalkRatio->setBuddy(m_pLineEdit_CrossTalkRatio);
	m_pLabel_CrossTalkRatio->setDisabled(true);
#endif

	// Create widgets for guide line indicator
	m_pCheckBox_ShowLumenContour = new QCheckBox(this);
	m_pCheckBox_ShowLumenContour->setText("Show Lumen Contour  ");
	m_pCheckBox_ShowLumenContour->setChecked(true);
	
	// Create widgets for compensation results - distance dependency check
#ifndef TWO_CHANNEL_NIRF
	m_pRenderArea_Correlation = new QRenderArea(this);
#else
	m_pRenderArea_Correlation = new QRenderArea2(this);
#endif
	m_pRenderArea_Correlation->m_bScattered = true;
	m_pRenderArea_Correlation->m_bMaskUse = true;
#ifndef TWO_CHANNEL_NIRF
	m_pRenderArea_Correlation->setSize({ 0, (double)m_pConfig->circRadius / 1.5 }, { 0, 2 * m_pConfig->nirfRange.max }, m_pResultTab->m_nirfMap.length());
#else
	m_pRenderArea_Correlation->setSize({ 0, (double)m_pConfig->circRadius / 1.5 }, { 0, 2 * max(m_pConfig->nirfRange[0].max, m_pConfig->nirfRange[1].max) }, m_pResultTab->m_nirfMap1.length());
#endif
	m_pRenderArea_Correlation->setMinimumHeight(150);
	m_pRenderArea_Correlation->setGrid(4, 8, 1);

	m_pCheckBox_Correlation = new QCheckBox(this);
	m_pCheckBox_Correlation->setText("DIST-NIRF Correlation ");
	m_pCheckBox_Correlation->setChecked(false);	

	// Create layout
	QGridLayout *pGridLayout_Buttons = new QGridLayout;
	pGridLayout_Buttons->setSpacing(2);

	pGridLayout_Buttons->addWidget(m_pPushButton_LoadDistanceMap, 0, 0);
	pGridLayout_Buttons->addWidget(m_pPushButton_LoadNirfBackground, 0, 1);
	pGridLayout_Buttons->addWidget(m_pToggleButton_Compensation, 1, 0, 1, 2);
	pGridLayout_Buttons->addWidget(m_pToggleButton_TBRMode, 2, 0, 1, 2);
	pGridLayout_Buttons->addWidget(m_pPushButton_ExportTBR, 3, 0, 1, 2);

	QGridLayout *pGridLayout_CompCurves = new QGridLayout;
	pGridLayout_CompCurves->setSpacing(2);

	pGridLayout_CompCurves->addWidget(m_pLabel_DistanceDecayCurve, 0, 0);
	pGridLayout_CompCurves->addWidget(m_pRenderArea_DistanceDecayCurve, 1, 0);
	pGridLayout_CompCurves->addWidget(m_pLabel_CompensationCurve, 0, 1);
	pGridLayout_CompCurves->addWidget(m_pRenderArea_CompensationCurve, 1, 1);

	QGridLayout *pGridLayout_CompCoeffs = new QGridLayout;
	pGridLayout_CompCoeffs->setSpacing(1);

	pGridLayout_CompCoeffs->addWidget(m_pLabel_CompensationCoeff, 0, 0, 1, 5);
#ifndef TWO_CHANNEL_NIRF
	pGridLayout_CompCoeffs->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 1, 0);
	pGridLayout_CompCoeffs->addWidget(m_pLineEdit_CompensationCoeff_a, 1, 1);
	pGridLayout_CompCoeffs->addWidget(m_pLineEdit_CompensationCoeff_b, 1, 2);
	pGridLayout_CompCoeffs->addWidget(m_pLineEdit_CompensationCoeff_c, 1, 3);
	pGridLayout_CompCoeffs->addWidget(m_pLineEdit_CompensationCoeff_d, 1, 4);
#else
	for (int i = 0; i < 2; i++)
	{
		pGridLayout_CompCoeffs->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), i + 1, 0);
		pGridLayout_CompCoeffs->addWidget(m_pLineEdit_CompensationCoeff_a[i], i + 1, 1);
		pGridLayout_CompCoeffs->addWidget(m_pLineEdit_CompensationCoeff_b[i], i + 1, 2);
		pGridLayout_CompCoeffs->addWidget(m_pLineEdit_CompensationCoeff_c[i], i + 1, 3);
		pGridLayout_CompCoeffs->addWidget(m_pLineEdit_CompensationCoeff_d[i], i + 1, 4);
	}
#endif

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
	pGridLayout_CompConsts->addWidget(m_pLabel_Compensation, 2, 3);
	pGridLayout_CompConsts->addWidget(m_pLineEdit_Compensation, 2, 4);

#ifdef TWO_CHANNEL_NIRF
	pGridLayout_CompConsts->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 3, 0);	
	pGridLayout_CompConsts->addWidget(m_pLabel_CrossTalkRatio, 3, 1, 1, 3);
	pGridLayout_CompConsts->addWidget(m_pLineEdit_CrossTalkRatio, 3, 4);
#endif

	QHBoxLayout *pHBoxLayout_TBR = new QHBoxLayout;
	pHBoxLayout_TBR->setSpacing(1);

	pHBoxLayout_TBR->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));
#ifndef TWO_CHANNEL_NIRF
	pHBoxLayout_TBR->addWidget(m_pLabel_NIRF_Background);
	pHBoxLayout_TBR->addWidget(m_pLineEdit_NIRF_Background);
	pHBoxLayout_TBR->addWidget(m_pLabel_TBR_Background);
	pHBoxLayout_TBR->addWidget(m_pLineEdit_TBR_Background);
#else
	pHBoxLayout_TBR->addWidget(m_pLabel_NIRF_Background);
	pHBoxLayout_TBR->addWidget(m_pLineEdit_NIRF_Background[0]);
	pHBoxLayout_TBR->addWidget(m_pLineEdit_NIRF_Background[1]);
	pHBoxLayout_TBR->addWidget(m_pLabel_TBR_Background);
	pHBoxLayout_TBR->addWidget(m_pLineEdit_TBR_Background[0]);
	pHBoxLayout_TBR->addWidget(m_pLineEdit_TBR_Background[1]);
#endif

	QHBoxLayout *pHBoxLayout_TBR_Option = new QHBoxLayout;
	pHBoxLayout_TBR_Option->setSpacing(1);

	pHBoxLayout_TBR_Option->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));
	pHBoxLayout_TBR_Option->addWidget(m_pCheckBox_ShowLumenContour);
	pHBoxLayout_TBR_Option->addWidget(m_pCheckBox_Filtering);
	pHBoxLayout_TBR_Option->addWidget(m_pCheckBox_GwMasking);
	///pGridLayout_TBR_Option->addWidget(m_pCheckBox_ZeroTBRDefinition, 1, 2);

	QGridLayout *pGridLayout_Correlation = new QGridLayout;
	pGridLayout_Correlation->setSpacing(2);

	pGridLayout_Correlation->addWidget(m_pCheckBox_Correlation, 0, 0, 1, 2);
	pGridLayout_Correlation->addWidget(m_pRenderArea_Correlation, 1, 0, 1, 2);
		

	m_pVBoxLayout = new QVBoxLayout;
	m_pVBoxLayout->setSpacing(5);

	m_pVBoxLayout->addItem(pGridLayout_Buttons);
	m_pVBoxLayout->addItem(pGridLayout_CompCurves);
	m_pVBoxLayout->addItem(pGridLayout_CompCoeffs);
	m_pVBoxLayout->addItem(pGridLayout_CompConsts);
	m_pVBoxLayout->addItem(pHBoxLayout_TBR);
	m_pVBoxLayout->addItem(pHBoxLayout_TBR_Option);
	m_pVBoxLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));
	m_pVBoxLayout->addItem(pGridLayout_Correlation);

	// Set layout
	this->setLayout(m_pVBoxLayout);

	// Connect
	connect(m_pPushButton_LoadDistanceMap, SIGNAL(clicked(bool)), this, SLOT(loadDistanceMap()));
	connect(m_pPushButton_LoadNirfBackground, SIGNAL(clicked(bool)), this, SLOT(loadNirfBackground()));
	connect(m_pToggleButton_Compensation, SIGNAL(toggled(bool)), this, SLOT(compensation(bool)));
	connect(m_pToggleButton_TBRMode, SIGNAL(toggled(bool)), this, SLOT(tbrConvering(bool)));
	connect(m_pPushButton_ExportTBR, SIGNAL(clicked(bool)), this, SLOT(exportTbrData()));
#ifndef TWO_CHANNEL_NIRF
	connect(m_pLineEdit_CompensationCoeff_a, SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	connect(m_pLineEdit_CompensationCoeff_b, SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	connect(m_pLineEdit_CompensationCoeff_c, SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	connect(m_pLineEdit_CompensationCoeff_d, SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
#else
	for (int i = 0; i < 2; i++)
	{
		connect(m_pLineEdit_CompensationCoeff_a[i], SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
		connect(m_pLineEdit_CompensationCoeff_b[i], SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
		connect(m_pLineEdit_CompensationCoeff_c[i], SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
		connect(m_pLineEdit_CompensationCoeff_d[i], SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	}
#endif
	connect(m_pLineEdit_FactorThreshold, SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	connect(m_pLineEdit_FactorPropConst, SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	connect(m_pLineEdit_DistPropConst, SIGNAL(textChanged(const QString &)), this, SLOT(changeCompensationCurve()));
	connect(m_pSpinBox_LumenContourOffset, SIGNAL(valueChanged(int)), this, SLOT(changeZeroPointSetting()));
	connect(m_pSpinBox_OuterSheathPosition, SIGNAL(valueChanged(int)), this, SLOT(changeZeroPointSetting()));
	connect(m_pCheckBox_Filtering, SIGNAL(toggled(bool)), this, SLOT(filtering(bool)));
	connect(m_pCheckBox_ZeroTBRDefinition, SIGNAL(toggled(bool)), this, SLOT(tbrZeroDefinition(bool)));
	connect(m_pCheckBox_GwMasking, SIGNAL(toggled(bool)), this, SLOT(gwMasking(bool)));
#ifndef TWO_CHANNEL_NIRF
	connect(m_pLineEdit_NIRF_Background, SIGNAL(textChanged(const QString &)), this, SLOT(changeNirfBackground(const QString &)));
	connect(m_pLineEdit_TBR_Background, SIGNAL(textChanged(const QString &)), this, SLOT(changeTbrBackground(const QString &)));
#else
	connect(m_pLineEdit_CrossTalkRatio, SIGNAL(textChanged(const QString &)), this, SLOT(changeCrossTalkRatio(const QString &)));
	connect(m_pLineEdit_NIRF_Background[0], SIGNAL(textChanged(const QString &)), this, SLOT(changeNirfBackground1(const QString &)));
	connect(m_pLineEdit_NIRF_Background[1], SIGNAL(textChanged(const QString &)), this, SLOT(changeNirfBackground2(const QString &)));
	connect(m_pLineEdit_TBR_Background[0], SIGNAL(textChanged(const QString &)), this, SLOT(changeTbrBackground1(const QString &)));
	connect(m_pLineEdit_TBR_Background[1], SIGNAL(textChanged(const QString &)), this, SLOT(changeTbrBackground2(const QString &)));
#endif
	connect(m_pLineEdit_Compensation, SIGNAL(textChanged(const QString &)), this, SLOT(changeCompConstant(const QString &)));
	connect(m_pCheckBox_ShowLumenContour, SIGNAL(toggled(bool)), this, SLOT(showLumenContour(bool)));
	connect(m_pCheckBox_Correlation, SIGNAL(toggled(bool)), this, SLOT(showCorrelationPlot(bool)));
	connect(this, SIGNAL(setWidgets(bool)), this, SLOT(setWidgetEnabled(bool)));

	// Initialization
	loadDistanceMap();
	loadNirfBackground();

	// Get Log file
	QString infopath = m_pResultTab->m_path + QString("/dist_comp_info.log");
	if (QFileInfo::exists(infopath))
		getCompInfo(infopath);

	// Initialization
	changeCompensationCurve();
	changeZeroPointSetting();
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
#ifndef TWO_CHANNEL_NIRF
	m_pPushButton_LoadDistanceMap->setEnabled(enabled && !(compMap.raw_ptr()));
#else
	m_pPushButton_LoadDistanceMap->setEnabled(enabled && !(compMap[0].raw_ptr()) && !(compMap[1].raw_ptr()));
#endif
	m_pPushButton_LoadNirfBackground->setEnabled(enabled && !(nirfBg > 0));
	
#ifndef TWO_CHANNEL_NIRF
	m_pToggleButton_Compensation->setEnabled(enabled && (compMap.raw_ptr()) && (nirfBg > 0));
	m_pToggleButton_TBRMode->setEnabled(enabled && (compMap.raw_ptr()) && (nirfBg > 0));

	m_pPushButton_ExportTBR->setEnabled(enabled && (compMap.raw_ptr()) && (gwMap.raw_ptr()));
#else
	m_pToggleButton_Compensation->setEnabled(enabled && (compMap[0].raw_ptr()) && (compMap[1].raw_ptr()) && (nirfBg > 0));
	m_pToggleButton_TBRMode->setEnabled(enabled && (compMap[0].raw_ptr()) && (compMap[1].raw_ptr()) && (nirfBg > 0));

	m_pPushButton_ExportTBR->setEnabled(enabled && (compMap[0].raw_ptr()) && (compMap[1].raw_ptr()) && (gwMap.raw_ptr()));
#endif

	m_pLabel_DistanceDecayCurve->setEnabled(enabled);
	m_pLabel_CompensationCurve->setEnabled(enabled);

	m_pRenderArea_DistanceDecayCurve->setEnabled(enabled);
	m_pRenderArea_CompensationCurve->setEnabled(enabled);

	m_pCheckBox_Correlation->setEnabled(enabled);
	m_pRenderArea_Correlation->setEnabled(enabled);

	m_pLabel_CompensationCoeff->setEnabled(enabled);

#ifndef TWO_CHANNEL_NIRF
	m_pLineEdit_CompensationCoeff_a->setEnabled(enabled);
	m_pLineEdit_CompensationCoeff_b->setEnabled(enabled);
	m_pLineEdit_CompensationCoeff_c->setEnabled(enabled);
	m_pLineEdit_CompensationCoeff_d->setEnabled(enabled);
#else
	for (int i = 0; i < 2; i++)
	{
		m_pLineEdit_CompensationCoeff_a[i]->setEnabled(enabled);
		m_pLineEdit_CompensationCoeff_b[i]->setEnabled(enabled);
		m_pLineEdit_CompensationCoeff_c[i]->setEnabled(enabled);
		m_pLineEdit_CompensationCoeff_d[i]->setEnabled(enabled);
	}
#endif

	m_pLabel_FactorThreshold->setEnabled(enabled);
	m_pLineEdit_FactorThreshold->setEnabled(enabled);

	m_pLabel_FactorPropConst->setEnabled(enabled);
	m_pLineEdit_FactorPropConst->setEnabled(enabled);

	m_pLabel_DistPropConst->setEnabled(enabled);
	m_pLineEdit_DistPropConst->setEnabled(enabled);
		
#ifndef TWO_CHANNEL_NIRF
	m_pLabel_LumenContourOffset->setEnabled(enabled && (compMap.raw_ptr()));
	m_pSpinBox_LumenContourOffset->setEnabled(enabled && (compMap.raw_ptr()));

	m_pLabel_OuterSheathPosition->setEnabled(enabled && (compMap.raw_ptr()));
	m_pSpinBox_OuterSheathPosition->setEnabled(enabled && (compMap.raw_ptr()));
#else
	m_pLabel_LumenContourOffset->setEnabled(enabled && (compMap[0].raw_ptr()) && (compMap[1].raw_ptr()));
	m_pSpinBox_LumenContourOffset->setEnabled(enabled && (compMap[0].raw_ptr()) && (compMap[1].raw_ptr()));

	m_pLabel_OuterSheathPosition->setEnabled(enabled && (compMap[0].raw_ptr()) && (compMap[1].raw_ptr()));
	m_pSpinBox_OuterSheathPosition->setEnabled(enabled && (compMap[0].raw_ptr()) && (compMap[1].raw_ptr()));
#endif

	m_pCheckBox_Filtering->setEnabled(enabled);
	m_pCheckBox_GwMasking->setEnabled(enabled && (gwMap.raw_ptr()));

#ifndef TWO_CHANNEL_NIRF
	m_pLabel_NIRF_Background->setEnabled(enabled && (nirfBg > 0));
	m_pLineEdit_NIRF_Background->setEnabled(enabled && (nirfBg > 0));

	m_pLabel_TBR_Background->setEnabled(enabled && (nirfBg > 0));
	m_pLineEdit_TBR_Background->setEnabled(enabled && (nirfBg > 0));

	m_pLabel_Compensation->setEnabled(enabled && (nirfBg > 0));
	m_pLineEdit_Compensation->setEnabled(enabled && (nirfBg > 0));
#else
	m_pLabel_NIRF_Background->setEnabled(enabled && (nirfBg[0] > 0) && (nirfBg[1] > 0));
	m_pLineEdit_NIRF_Background[0]->setEnabled(enabled && (nirfBg[0] > 0) && (nirfBg[1] > 0));
	m_pLineEdit_NIRF_Background[1]->setEnabled(enabled && (nirfBg[0] > 0) && (nirfBg[1] > 0));

	m_pLabel_TBR_Background->setEnabled(enabled && (nirfBg[0] > 0) && (nirfBg[1] > 0));
	m_pLineEdit_TBR_Background[0]->setEnabled(enabled && (nirfBg[0] > 0) && (nirfBg[1] > 0));
	m_pLineEdit_TBR_Background[1]->setEnabled(enabled && (nirfBg[0] > 0) && (nirfBg[1] > 0));

	m_pLabel_Compensation->setEnabled(enabled && (nirfBg[0] > 0) && (nirfBg[1] > 0));
	m_pLineEdit_Compensation->setEnabled(enabled && (nirfBg[0] > 0) && (nirfBg[1] > 0));
	
	m_pLabel_CrossTalkRatio->setEnabled(enabled && (nirfBg[0] > 0) && (nirfBg[1] > 0));
	m_pLineEdit_CrossTalkRatio->setEnabled(enabled && (nirfBg[0] > 0) && (nirfBg[1] > 0));
#endif

	m_pCheckBox_ShowLumenContour->setEnabled(enabled);
	m_pCheckBox_ZeroTBRDefinition->setEnabled(false);

	//zero tbr ฐüทร...
}

void NirfDistCompDlg::loadDistanceMap()
{
	// Get distance map
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

#ifndef TWO_CHANNEL_NIRF
        compMap = np::FloatArray2((int)m_pResultTab->m_vectorOctImage.at(0).size(1), (int)m_pResultTab->m_vectorOctImage.size());
#else
		compMap[0] = np::FloatArray2((int)m_pResultTab->m_vectorOctImage.at(0).size(1), (int)m_pResultTab->m_vectorOctImage.size());
		compMap[1] = np::FloatArray2((int)m_pResultTab->m_vectorOctImage.at(0).size(1), (int)m_pResultTab->m_vectorOctImage.size());
#endif

        // Update widgets states
        m_pSpinBox_LumenContourOffset->setEnabled(true);
        m_pLabel_LumenContourOffset->setEnabled(true);
        m_pSpinBox_OuterSheathPosition->setEnabled(true);
        m_pLabel_OuterSheathPosition->setEnabled(true);

        m_pPushButton_LoadDistanceMap->setDisabled(true);
        if (!m_pPushButton_LoadDistanceMap->isEnabled() && !m_pPushButton_LoadNirfBackground->isEnabled())
            m_pToggleButton_Compensation->setEnabled(true);

		m_pCheckBox_Correlation->setChecked(true);
    }

	// Get guide-wire map (if required)
	QString gwMapName = m_pResultTab->m_path + "/gw_map.bin";

	QFile file_gw(gwMapName);
	if (false == file_gw.open(QFile::ReadOnly))
		printf("[ERROR] Invalid external data or there is no such a file (gw_map.bin)!\n");
	else
	{
#ifndef TWO_CHANNEL_NIRF
		gwMap = np::FloatArray2(m_pResultTab->m_nirfMap.size(0), m_pResultTab->m_nirfMap.size(1));
#else
		gwMap = np::FloatArray2(m_pResultTab->m_nirfMap1.size(0), m_pResultTab->m_nirfMap1.size(1));
#endif
		file_gw.read(reinterpret_cast<char*>(gwMap.raw_ptr()), sizeof(float) * gwMap.length());
		file_gw.close();

		// Update widgets states
		m_pCheckBox_GwMasking->setEnabled(true);
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
#ifndef TWO_CHANNEL_NIRF
        np::DoubleArray nirfBgMap((int)file.size() / sizeof(double));
        file.read(reinterpret_cast<char*>(nirfBgMap.raw_ptr()), sizeof(double) * nirfBgMap.length());
        file.close();

        double temp_bg;
        ippsMean_64f(nirfBgMap.raw_ptr(), nirfBgMap.length(), &temp_bg);

        nirfBg = temp_bg;
#else
		np::DoubleArray nirfBgMap((int)file.size() / sizeof(double));
		np::DoubleArray nirfBgMap1((int)file.size() / 2 / sizeof(double));
		np::DoubleArray nirfBgMap2((int)file.size() / 2 / sizeof(double));
		file.read(reinterpret_cast<char*>(nirfBgMap.raw_ptr()), sizeof(double) * nirfBgMap.length());
		file.close();

		double temp_bg1, temp_bg2;
		ippsCplxToReal_64fc((const Ipp64fc*)nirfBgMap.raw_ptr(), nirfBgMap1, nirfBgMap2, nirfBgMap2.length());

		ippsMean_64f(nirfBgMap1.raw_ptr(), nirfBgMap1.length(), &temp_bg1);
		ippsMean_64f(nirfBgMap2.raw_ptr(), nirfBgMap2.length(), &temp_bg2);

		nirfBg[0] = temp_bg1;
		nirfBg[1] = temp_bg2;
#endif

        // Update widgets states
        m_pPushButton_LoadNirfBackground->setDisabled(true);
		m_pLabel_NIRF_Background->setEnabled(true);
#ifndef TWO_CHANNEL_NIRF
		m_pLineEdit_NIRF_Background->setEnabled(true);
		m_pLineEdit_NIRF_Background->setText(QString::number(nirfBg, 'f', 3));
#else
		m_pLineEdit_NIRF_Background[0]->setEnabled(true);
		m_pLineEdit_NIRF_Background[1]->setEnabled(true);
		m_pLineEdit_NIRF_Background[0]->setText(QString::number(nirfBg[0], 'f', 3));
		m_pLineEdit_NIRF_Background[1]->setText(QString::number(nirfBg[1], 'f', 3));
#endif
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
#ifndef TWO_CHANNEL_NIRF
        m_pLineEdit_TBR_Background->setEnabled(true);
#else
		m_pLineEdit_TBR_Background[0]->setEnabled(true);
		m_pLineEdit_TBR_Background[1]->setEnabled(true);
#endif
		m_pLabel_Compensation->setEnabled(true);
		m_pLineEdit_Compensation->setEnabled(true);
#ifdef TWO_CHANNEL_NIRF
		m_pLabel_CrossTalkRatio->setEnabled(true);
		m_pLineEdit_CrossTalkRatio->setEnabled(true);
#endif
    }
    else
    {
        m_pToggleButton_Compensation->setText("Compensation On");

		m_pToggleButton_TBRMode->setChecked(false);
		m_pToggleButton_TBRMode->setDisabled(true);
        m_pLabel_TBR_Background->setDisabled(true);
#ifndef TWO_CHANNEL_NIRF
        m_pLineEdit_TBR_Background->setDisabled(true);
#else
		m_pLineEdit_TBR_Background[0]->setDisabled(true);
		m_pLineEdit_TBR_Background[1]->setDisabled(true);
#endif
		m_pLabel_Compensation->setDisabled(true);
		m_pLineEdit_Compensation->setDisabled(true);
#ifdef TWO_CHANNEL_NIRF
		m_pLabel_CrossTalkRatio->setDisabled(true);
		m_pLineEdit_CrossTalkRatio->setDisabled(true);
#endif
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

		m_pPushButton_ExportTBR->setEnabled(true && (gwMap.raw_ptr()) && (m_pCheckBox_GwMasking->isChecked()));
		//m_pCheckBox_ZeroTBRDefinition->setEnabled(true);
	}
	else
	{
		m_pToggleButton_TBRMode->setText("TBR Converting On");

		m_pPushButton_ExportTBR->setEnabled(false);
		//m_pCheckBox_ZeroTBRDefinition->setDisabled(true);
	}

#ifndef TWO_CHANNEL_NIRF
	tbrBg = m_pLineEdit_TBR_Background->text().toFloat();
#else
	tbrBg[0] = m_pLineEdit_TBR_Background[0]->text().toFloat();
	tbrBg[1] = m_pLineEdit_TBR_Background[1]->text().toFloat();
#endif

	// Invalidate
	m_pResultTab->invalidate();
}

void NirfDistCompDlg::exportTbrData()
{
	QFile file(m_pResultTab->m_path + "/TBR_data.csv");
	if (file.open(QFile::WriteOnly))
	{
		{
			QTextStream stream(&file);
#ifndef TWO_CHANNEL_NIRF
			stream << "Frame#" << "\t" << "mTBR" << "\t" << "pTBR" << "\n";
#else
            stream << "Frame#" << "\t" << "mTBR1" << "\t" << "mTBR2" << "\t" << "pTBR1" << "\t" << "pTBR2" << "\n";
#endif
		}

#ifndef TWO_CHANNEL_NIRF
		for (int i = 0; i < m_pResultTab->m_nirfMap0.size(1); i++)
		{
			// Copy original NIRF data
			const float* data = &m_pResultTab->m_nirfMap0(0, i);
			np::FloatArray data0(m_pResultTab->m_nirfMap.size(0)), mask(m_pResultTab->m_nirfMap.size(0));
			memcpy(data0, data, sizeof(float) * data0.length());

			// Excluding GW region
			np::Uint8Array valid_region_8u(m_pResultTab->m_nirfMap.size(0));
			ippiCompareC_32f_C1R(data0.raw_ptr(), data0.size(0), 0.0f, valid_region_8u.raw_ptr(), valid_region_8u.size(0), { data0.size(0), 1 }, ippCmpEq);
			ippsConvert_8u32f(valid_region_8u, mask, mask.length());
			ippsDivC_32f_I(255.0f, mask, mask.length());
			ippsSubCRev_32f_I(1.0f, mask, mask.length());

			// Get MASK length
			Ipp32f mean, std, maxi, mask_len;
			ippsSum_32f(mask.raw_ptr(), mask.length(), &mask_len, ippAlgHintFast);

			// Masking the NIRF data
			ippsMul_32f_I(mask.raw_ptr(), data0.raw_ptr(), data0.length());
			ippsMeanStdDev_32f(data0.raw_ptr(), data0.length(), &mean, &std, ippAlgHintFast);
			ippsMax_32f(data0.raw_ptr(), data0.length(), &maxi);

			// Compensation of GW artifact
			mean = mean * data0.length() / mask_len;

			// Write to the CSV file
			QTextStream stream(&file);
			stream << i + 1 << "\t" << mean << "\t" << maxi << "\n";
		}
#else
        for (int i = 0; i < m_pResultTab->m_nirfMap1_0.size(1); i++)
        {
            // Copy original NIRF data
            const float* data1 = &m_pResultTab->m_nirfMap1_0(0, i);
            const float* data2 = &m_pResultTab->m_nirfMap2_0(0, i);
            np::FloatArray data1_0(m_pResultTab->m_nirfMap1.size(0)), mask(m_pResultTab->m_nirfMap1.size(0));
            np::FloatArray data2_0(m_pResultTab->m_nirfMap2.size(0)); // , mask2(m_pResultTab->m_nirfMap2.size(0));
            memcpy(data1_0, data1, sizeof(float) * data1_0.length());
            memcpy(data2_0, data2, sizeof(float) * data2_0.length());

            // Excluding GW region
            np::Uint8Array valid_region_8u(m_pResultTab->m_nirfMap1.size(0));
            ippiCompareC_32f_C1R(data1_0.raw_ptr(), data1_0.size(0), 0.0f, valid_region_8u.raw_ptr(), valid_region_8u.size(0), { data1_0.size(0), 1 }, ippCmpEq);
            ippsConvert_8u32f(valid_region_8u, mask, mask.length());
            ippsDivC_32f_I(255.0f, mask, mask.length());
            ippsSubCRev_32f_I(1.0f, mask, mask.length());

            // Get MASK length
            Ipp32f mean1, mean2, std, maxi1, maxi2, mask_len;
            ippsSum_32f(mask.raw_ptr(), mask.length(), &mask_len, ippAlgHintFast);

            // Masking the NIRF data
            ippsMul_32f_I(mask.raw_ptr(), data1_0.raw_ptr(), data1_0.length());
            ippsMeanStdDev_32f(data1_0.raw_ptr(), data1_0.length(), &mean1, &std, ippAlgHintFast);
            ippsMax_32f(data1_0.raw_ptr(), data1_0.length(), &maxi1);

            ippsMul_32f_I(mask.raw_ptr(), data2_0.raw_ptr(), data2_0.length());
            ippsMeanStdDev_32f(data2_0.raw_ptr(), data2_0.length(), &mean2, &std, ippAlgHintFast);
            ippsMax_32f(data2_0.raw_ptr(), data2_0.length(), &maxi2);

            // Compensation of GW artifact
            mean1 = mean1 * data1_0.length() / mask_len;
            mean2 = mean2 * data2_0.length() / mask_len;

            // Write to the CSV file
            QTextStream stream(&file);
            stream << i + 1 << "\t" << mean1 << "\t" << mean2 << "\t" << maxi1 << "\t" << maxi2 << "\n";
        }
#endif

		QDesktopServices::openUrl(QUrl("file:///" + m_pResultTab->m_path));
	}
	file.close();
}

void NirfDistCompDlg::changeCompensationCurve()
{
    // Get and update values
#ifndef TWO_CHANNEL_NIRF
	float a = m_pLineEdit_CompensationCoeff_a->text().toFloat();
	float b = m_pLineEdit_CompensationCoeff_b->text().toFloat();
	float c = m_pLineEdit_CompensationCoeff_c->text().toFloat();
	float d = m_pLineEdit_CompensationCoeff_d->text().toFloat();

	m_pConfig->nirfCompCoeffs_a = a;
	m_pConfig->nirfCompCoeffs_b = b;
	m_pConfig->nirfCompCoeffs_c = c;
	m_pConfig->nirfCompCoeffs_d = d;
#else
	for (int i = 0; i < 2; i++)
	{
		float a = m_pLineEdit_CompensationCoeff_a[i]->text().toFloat();
		float b = m_pLineEdit_CompensationCoeff_b[i]->text().toFloat();
		float c = m_pLineEdit_CompensationCoeff_c[i]->text().toFloat();
		float d = m_pLineEdit_CompensationCoeff_d[i]->text().toFloat();

		m_pConfig->nirfCompCoeffs_a[i] = a;
		m_pConfig->nirfCompCoeffs_b[i] = b;
		m_pConfig->nirfCompCoeffs_c[i] = c;
		m_pConfig->nirfCompCoeffs_d[i] = d;
	}
#endif

	float factor_thres = m_pLineEdit_FactorThreshold->text().toFloat();
	float factor_prop_const = m_pLineEdit_FactorPropConst->text().toFloat();
	float dist_prop_const = m_pLineEdit_DistPropConst->text().toFloat();

	m_pConfig->nirfFactorThres = factor_thres;
	m_pConfig->nirfFactorPropConst = factor_prop_const;
	m_pConfig->nirfDistPropConst = dist_prop_const;

    // Update compensation curve
	m_pRenderArea_CompensationCurve->setSize({ 0, (double)m_pConfig->circRadius }, { 0, round(factor_thres * 1.1) });
    m_pLabel_CompensationCurve->setText(QString("Compensation Curve ([0, %1])").arg((int)round(factor_thres * 1.1)));
	
	for (int i = 0; i < m_pConfig->circRadius; i++)
	{
#ifndef TWO_CHANNEL_NIRF
		if ((m_pConfig->nirfCompCoeffs_a + m_pConfig->nirfCompCoeffs_c) != 0)
		{
			distDecayCurve[i] = (m_pConfig->nirfCompCoeffs_a * expf(m_pConfig->nirfCompCoeffs_b * dist_prop_const * (float)i) 
									+ m_pConfig->nirfCompCoeffs_c * expf(m_pConfig->nirfCompCoeffs_d * dist_prop_const * (float)i)) / (m_pConfig->nirfCompCoeffs_a + m_pConfig->nirfCompCoeffs_c);
			compCurve[i] = (i < 20) ? 1.0f : 1 / distDecayCurve[i] / factor_prop_const;
			if (compCurve[i] < 1.0f) compCurve[i] = 1.0f;
            if (compCurve[i] > factor_thres) compCurve[i] = factor_thres;
		}
		else
		{
			distDecayCurve[i] = 0.0f;
			compCurve[i] = 1.0f;
		}
#else
		for (int j = 0; j < 2; j++)
		{
			if ((m_pConfig->nirfCompCoeffs_a[j] + m_pConfig->nirfCompCoeffs_c[j]) != 0)
			{
				distDecayCurve[j][i] = (m_pConfig->nirfCompCoeffs_a[j] * expf(m_pConfig->nirfCompCoeffs_b[j] * dist_prop_const * (float)i)
					+ m_pConfig->nirfCompCoeffs_c[j] * expf(m_pConfig->nirfCompCoeffs_d[j] * dist_prop_const * (float)i)) / (m_pConfig->nirfCompCoeffs_a[j] + m_pConfig->nirfCompCoeffs_c[j]);
				compCurve[j][i] = (i < 20) ? 1.0f : 1 / distDecayCurve[j][i] / factor_prop_const;
				if (compCurve[j][i] < 1.0f) compCurve[j][i] = 1.0f;
				if (compCurve[j][i] > factor_thres) compCurve[j][i] = factor_thres;
			}
			else
			{
				distDecayCurve[j][i] = 0.0f;
				compCurve[j][i] = 1.0f;
			}
		}
#endif
	}

#ifndef TWO_CHANNEL_NIRF
	memcpy(m_pRenderArea_DistanceDecayCurve->m_pData, distDecayCurve.raw_ptr(), sizeof(float) * distDecayCurve.length());
	memcpy(m_pRenderArea_CompensationCurve->m_pData, compCurve.raw_ptr(), sizeof(float) * compCurve.length());
#else
	memcpy(m_pRenderArea_DistanceDecayCurve->m_pData1, distDecayCurve[0].raw_ptr(), sizeof(float) * distDecayCurve[0].length());
	memcpy(m_pRenderArea_DistanceDecayCurve->m_pData2, distDecayCurve[1].raw_ptr(), sizeof(float) * distDecayCurve[1].length());
	memcpy(m_pRenderArea_CompensationCurve->m_pData1, compCurve[0].raw_ptr(), sizeof(float) * compCurve[0].length());
	memcpy(m_pRenderArea_CompensationCurve->m_pData2, compCurve[1].raw_ptr(), sizeof(float) * compCurve[1].length());
#endif
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
    distOffsetMap = np::Uint16Array2(distMap.size(0), distMap.size(1));
    memcpy(distOffsetMap.raw_ptr(), distMap.raw_ptr(), sizeof(uint16_t) * distMap.length());
    ippsAddC_16u_ISfs(m_pConfig->nirfLumContourOffset, distOffsetMap.raw_ptr(), distOffsetMap.length(), 0);
    ippsSubC_16u_ISfs(m_pConfig->nirfOuterSheathPos, distOffsetMap.raw_ptr(), distOffsetMap.length(), 0);

#ifndef TWO_CHANNEL_NIRF
    memset(compMap.raw_ptr(), 0, sizeof(float) * compMap.length());
#else
	memset(compMap[0].raw_ptr(), 0, sizeof(float) * compMap[0].length());
	memset(compMap[1].raw_ptr(), 0, sizeof(float) * compMap[1].length());
#endif
    tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)distOffsetMap.size(1)),
        [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i)
        {
            for (int j = 0; j < distOffsetMap.size(0); j++)
            {
                if (distOffsetMap(j, (int)i) > m_pConfig->circRadius)
                    distOffsetMap(j, (int)i) = m_pConfig->circRadius - 1;

#ifndef TWO_CHANNEL_NIRF
                compMap(j, (int)i) = compCurve[distOffsetMap(j, (int)i)];
#else
				compMap[0](j, (int)i) = compCurve[0][distOffsetMap(j, (int)i)];
				compMap[1](j, (int)i) = compCurve[1][distOffsetMap(j, (int)i)];
#endif
            }
        }
    });
}

void NirfDistCompDlg::updateCorrelation(int frame)
{
	if (m_pCheckBox_Correlation->isChecked())
	{
		// GW artifact region should be excluded...

		// Distance
		np::FloatArray dist_data(distOffsetMap.length());
		ippsConvert_16u32f(distOffsetMap.raw_ptr(), dist_data, distOffsetMap.length());
		float max_lim; ippsMax_32f(dist_data, dist_data.length(), &max_lim);
#ifndef TWO_CHANNEL_NIRF
		m_pRenderArea_Correlation->setSize({ 0, (double)max_lim }, { 0, 2 * m_pConfig->nirfRange.max }, m_pResultTab->m_nirfMap0.length());		
#else
		m_pRenderArea_Correlation->setSize({ 0, (double)max_lim }, { 0, 2 * max(m_pConfig->nirfRange[0].max, m_pConfig->nirfRange[1].max) }, m_pResultTab->m_nirfMap1_0.length());
#endif
		memcpy(m_pRenderArea_Correlation->m_pDataX, dist_data, sizeof(float) * dist_data.length());

		// NIRF data at current setting
#ifndef TWO_CHANNEL_NIRF
		memcpy(m_pRenderArea_Correlation->m_pData, m_pResultTab->m_nirfMap0.raw_ptr(), sizeof(float) * m_pResultTab->m_nirfMap0.length());
#else
		memcpy(m_pRenderArea_Correlation->m_pData1, m_pResultTab->m_nirfMap1_0.raw_ptr(), sizeof(float) * m_pResultTab->m_nirfMap1_0.length());
		memcpy(m_pRenderArea_Correlation->m_pData2, m_pResultTab->m_nirfMap2_0.raw_ptr(), sizeof(float) * m_pResultTab->m_nirfMap2_0.length());
#endif

		// Current frame NIRF data
		ippsSet_32f(1.0f, m_pRenderArea_Correlation->m_pMask, m_pRenderArea_Correlation->m_buff_len);
		ippsSet_32f(0.0f, &m_pRenderArea_Correlation->m_pMask[distOffsetMap.size(0) * frame], distOffsetMap.size(0));

#ifndef TWO_CHANNEL_NIRF
		// Get correlation coefficient - total
		Ipp32f x_mean, x_std;
		Ipp32f y_mean, y_std;
		ippsMeanStdDev_32f(dist_data, dist_data.length(), &x_mean, &x_std, ippAlgHintFast);
		ippsMeanStdDev_32f(m_pResultTab->m_nirfMap0, m_pResultTab->m_nirfMap0.length(), &y_mean, &y_std, ippAlgHintFast);
		
		np::FloatArray dist_temp(dist_data.length()), nirf_temp(m_pResultTab->m_nirfMap0.length());
		np::FloatArray comul_temp(dist_temp.length());
		ippsSubC_32f(dist_data, x_mean, dist_temp, dist_data.length());
		ippsSubC_32f(m_pResultTab->m_nirfMap0, y_mean, nirf_temp, m_pResultTab->m_nirfMap0.length());
		ippsMul_32f_I(nirf_temp, dist_temp, dist_temp.length());

		Ipp32f cov;
		ippsSum_32f(dist_temp, dist_temp.length(), &cov, ippAlgHintFast);
		cov = cov / (dist_temp.length() - 1);

		float r_total = cov / x_std / y_std;

		// Get correlation coefficient - frame
		ippsMeanStdDev_32f(&dist_data(distOffsetMap.size(0) * frame), distOffsetMap.size(0), &x_mean, &x_std, ippAlgHintFast);
		ippsMeanStdDev_32f(&m_pResultTab->m_nirfMap0(0, frame), m_pResultTab->m_nirfMap0.size(0), &y_mean, &y_std, ippAlgHintFast);

		np::FloatArray dist_frame(distOffsetMap.size(0)), nirf_frame(m_pResultTab->m_nirfMap0.size(0));
		np::FloatArray comul_frame(dist_frame.length());
		ippsSubC_32f(&dist_data(distOffsetMap.size(0) * frame), x_mean, dist_frame, distOffsetMap.size(0));
		ippsSubC_32f(&m_pResultTab->m_nirfMap0(0, frame), y_mean, nirf_frame, m_pResultTab->m_nirfMap0.size(0));
		ippsMul_32f_I(nirf_frame, dist_frame, dist_frame.length());

		ippsSum_32f(dist_frame, dist_frame.length(), &cov, ippAlgHintFast);
		cov = cov / (dist_frame.length() - 1);

		float r_frame = cov / x_std / y_std;

		// Update
		m_pRenderArea_Correlation->update();
		m_pCheckBox_Correlation->setText(QString("DIST-NIRF Correlation (r_total = %1 / r_frame = %2)").arg(r_total, 4, 'f', 3).arg(r_frame, 4, 'f', 3));
#else
		// Get correlation coefficient - total
		Ipp32f x_mean, x_std;
		Ipp32f y_mean[2], y_std[2];
		ippsMeanStdDev_32f(dist_data, dist_data.length(), &x_mean, &x_std, ippAlgHintFast);
		ippsMeanStdDev_32f(m_pResultTab->m_nirfMap1_0, m_pResultTab->m_nirfMap1_0.length(), &y_mean[0], &y_std[0], ippAlgHintFast);
		ippsMeanStdDev_32f(m_pResultTab->m_nirfMap2_0, m_pResultTab->m_nirfMap2_0.length(), &y_mean[1], &y_std[1], ippAlgHintFast);

		np::FloatArray dist_temp(dist_data.length()), nirf_temp(m_pResultTab->m_nirfMap1_0.length());
		np::FloatArray comul_temp(dist_temp.length());
		ippsSubC_32f(dist_data, x_mean, dist_temp, dist_data.length());
		ippsSubC_32f(m_pResultTab->m_nirfMap1_0, y_mean[0], nirf_temp, m_pResultTab->m_nirfMap1_0.length());
		ippsMul_32f_I(nirf_temp, dist_temp, dist_temp.length());

		Ipp32f cov;
		ippsSum_32f(dist_temp, dist_temp.length(), &cov, ippAlgHintFast);
		cov = cov / (dist_temp.length() - 1);

		float r_total1 = cov / x_std / y_std[0];

		ippsSubC_32f(dist_data, x_mean, dist_temp, dist_data.length());
		ippsSubC_32f(m_pResultTab->m_nirfMap2_0, y_mean[1], nirf_temp, m_pResultTab->m_nirfMap2_0.length());
		ippsMul_32f_I(nirf_temp, dist_temp, dist_temp.length());

		ippsSum_32f(dist_temp, dist_temp.length(), &cov, ippAlgHintFast);
		cov = cov / (dist_temp.length() - 1);

		float r_total2 = cov / x_std / y_std[1];		

		// Get correlation coefficient - frame
		ippsMeanStdDev_32f(&dist_data(distOffsetMap.size(0) * frame), distOffsetMap.size(0), &x_mean, &x_std, ippAlgHintFast);
		ippsMeanStdDev_32f(&m_pResultTab->m_nirfMap1_0(0, frame), m_pResultTab->m_nirfMap1_0.size(0), &y_mean[0], &y_std[0], ippAlgHintFast);
		ippsMeanStdDev_32f(&m_pResultTab->m_nirfMap2_0(0, frame), m_pResultTab->m_nirfMap2_0.size(0), &y_mean[1], &y_std[1], ippAlgHintFast);

		np::FloatArray dist_frame(distOffsetMap.size(0)), nirf_frame(m_pResultTab->m_nirfMap1_0.size(0));
		np::FloatArray comul_frame(dist_frame.length());
		ippsSubC_32f(&dist_data(distOffsetMap.size(0) * frame), x_mean, dist_frame, distOffsetMap.size(0));
		ippsSubC_32f(&m_pResultTab->m_nirfMap1_0(0, frame), y_mean[0], nirf_frame, m_pResultTab->m_nirfMap1_0.size(0));
		ippsMul_32f_I(nirf_frame, dist_frame, dist_frame.length());

		ippsSum_32f(dist_frame, dist_frame.length(), &cov, ippAlgHintFast);
		cov = cov / (dist_frame.length() - 1);

		float r_frame1 = cov / x_std / y_std[0];

		ippsSubC_32f(&dist_data(distOffsetMap.size(0) * frame), x_mean, dist_frame, distOffsetMap.size(0));
		ippsSubC_32f(&m_pResultTab->m_nirfMap2_0(0, frame), y_mean[1], nirf_frame, m_pResultTab->m_nirfMap2_0.size(0));
		ippsMul_32f_I(nirf_frame, dist_frame, dist_frame.length());

		ippsSum_32f(dist_frame, dist_frame.length(), &cov, ippAlgHintFast);
		cov = cov / (dist_frame.length() - 1);

		float r_frame2 = cov / x_std / y_std[1];

		// Update
		m_pRenderArea_Correlation->update();
		m_pCheckBox_Correlation->setText(QString("DIST-NIRF Correlation (r_t = [%1, %2] / r_f = [%3, %4])").arg(r_total1, 4, 'f', 3).arg(r_total2, 4, 'f', 3).arg(r_frame1, 4, 'f', 3).arg(r_frame2, 4, 'f', 3));
#endif
	}
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

void NirfDistCompDlg::gwMasking(bool toggled)
{
	m_pPushButton_ExportTBR->setEnabled(toggled && (gwMap.raw_ptr()) && m_pToggleButton_TBRMode->isChecked());

	// Invalidate
	m_pResultTab->invalidate();	
}

void NirfDistCompDlg::tbrZeroDefinition(bool toggled)
{
	// Invalidate
	m_pResultTab->invalidate();

	(void)toggled;
}

#ifndef TWO_CHANNEL_NIRF
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
#else
void NirfDistCompDlg::changeNirfBackground1(const QString &str)
{
	nirfBg[0] = m_pLineEdit_NIRF_Background[0]->text().toFloat();

	// Invalidate
	m_pResultTab->invalidate();
}

void NirfDistCompDlg::changeNirfBackground2(const QString &str)
{
	nirfBg[1] = m_pLineEdit_NIRF_Background[1]->text().toFloat();

	// Invalidate
	m_pResultTab->invalidate();
}

void NirfDistCompDlg::changeTbrBackground1(const QString &str)
{
	tbrBg[0] = m_pLineEdit_TBR_Background[0]->text().toFloat();

	// Invalidate
	m_pResultTab->invalidate();
}

void NirfDistCompDlg::changeTbrBackground2(const QString &str)
{
	tbrBg[1] = m_pLineEdit_TBR_Background[1]->text().toFloat();

	// Invalidate
	m_pResultTab->invalidate();
}
#endif

void NirfDistCompDlg::changeCompConstant(const QString &str)
{
	compConst = str.toFloat();

	// Invalidate
	m_pResultTab->invalidate();
}

#ifdef TWO_CHANNEL_NIRF
void NirfDistCompDlg::changeCrossTalkRatio(const QString &str)
{
	crossTalkRatio = str.toFloat();

	// Invalidate
	m_pResultTab->invalidate();
}
#endif

void NirfDistCompDlg::showLumenContour(bool)
{
	// Invalidate
	m_pResultTab->invalidate();
}

void NirfDistCompDlg::showCorrelationPlot(bool toggled)
{
	if (toggled)
		updateCorrelation(m_pResultTab->getCurrentFrame());
	else
	{
		// Distance		
		memset(m_pRenderArea_Correlation->m_pDataX, 0, sizeof(float) * distOffsetMap.length());

		// NIRF data at current setting
#ifndef TWO_CHANNEL_NIRF
		memset(m_pRenderArea_Correlation->m_pData, 0, sizeof(float) * m_pResultTab->m_nirfMap0.length());
#else
		memset(m_pRenderArea_Correlation->m_pData1, 0, sizeof(float) * m_pResultTab->m_nirfMap1_0.length());
		memset(m_pRenderArea_Correlation->m_pData2, 0, sizeof(float) * m_pResultTab->m_nirfMap2_0.length());
#endif
		// Current frame NIRF data
		ippsSet_32f(1.0f, m_pRenderArea_Correlation->m_pMask, m_pRenderArea_Correlation->m_buff_len);

		// Update
		m_pRenderArea_Correlation->update();
		m_pCheckBox_Correlation->setText("DIST-NIRF Correlation ");
	}
}


void NirfDistCompDlg::getCompInfo(const QString &infopath)
{
	QSettings settings(infopath, QSettings::IniFormat);
	settings.beginGroup("distance_compensation");

	// Load parameters
	int nirfOffset = settings.value("nirfOffset").toInt();
	m_pResultTab->setNirfOffset(nirfOffset);

#ifndef TWO_CHANNEL_NIRF
	m_pConfig->nirfCompCoeffs_a = settings.value("nirfCompCoeffs_a").toFloat();
	m_pLineEdit_CompensationCoeff_a->setText(QString::number(m_pConfig->nirfCompCoeffs_a));
	m_pConfig->nirfCompCoeffs_b = settings.value("nirfCompCoeffs_b").toFloat();
	m_pLineEdit_CompensationCoeff_b->setText(QString::number(m_pConfig->nirfCompCoeffs_b));
	m_pConfig->nirfCompCoeffs_c = settings.value("nirfCompCoeffs_c").toFloat();
	m_pLineEdit_CompensationCoeff_c->setText(QString::number(m_pConfig->nirfCompCoeffs_c));
	m_pConfig->nirfCompCoeffs_d = settings.value("nirfCompCoeffs_d").toFloat();
	m_pLineEdit_CompensationCoeff_d->setText(QString::number(m_pConfig->nirfCompCoeffs_d));
#else
	m_pConfig->nirfCompCoeffs_a[0] = settings.value("nirfCompCoeffs_a1").toFloat();
	m_pLineEdit_CompensationCoeff_a[0]->setText(QString::number(m_pConfig->nirfCompCoeffs_a[0]));
	m_pConfig->nirfCompCoeffs_b[0] = settings.value("nirfCompCoeffs_b1").toFloat();
	m_pLineEdit_CompensationCoeff_b[0]->setText(QString::number(m_pConfig->nirfCompCoeffs_b[0]));
	m_pConfig->nirfCompCoeffs_c[0] = settings.value("nirfCompCoeffs_c1").toFloat();
	m_pLineEdit_CompensationCoeff_c[0]->setText(QString::number(m_pConfig->nirfCompCoeffs_c[0]));
	m_pConfig->nirfCompCoeffs_d[0] = settings.value("nirfCompCoeffs_d1").toFloat();
	m_pLineEdit_CompensationCoeff_d[0]->setText(QString::number(m_pConfig->nirfCompCoeffs_d[0]));
	
	m_pConfig->nirfCompCoeffs_a[1] = settings.value("nirfCompCoeffs_a2").toFloat();
	m_pLineEdit_CompensationCoeff_a[1]->setText(QString::number(m_pConfig->nirfCompCoeffs_a[1]));
	m_pConfig->nirfCompCoeffs_b[1] = settings.value("nirfCompCoeffs_b2").toFloat();
	m_pLineEdit_CompensationCoeff_b[1]->setText(QString::number(m_pConfig->nirfCompCoeffs_b[1]));
	m_pConfig->nirfCompCoeffs_c[1] = settings.value("nirfCompCoeffs_c2").toFloat();
	m_pLineEdit_CompensationCoeff_c[1]->setText(QString::number(m_pConfig->nirfCompCoeffs_c[1]));
	m_pConfig->nirfCompCoeffs_d[1] = settings.value("nirfCompCoeffs_d2").toFloat();
	m_pLineEdit_CompensationCoeff_d[1]->setText(QString::number(m_pConfig->nirfCompCoeffs_d[1]));
#endif

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

#ifndef TWO_CHANNEL_NIRF
	nirfBg = settings.value("nirfBg").toFloat();;
	m_pLineEdit_NIRF_Background->setText(QString::number(nirfBg));

	tbrBg = settings.value("tbrBg").toFloat();;
	m_pLineEdit_TBR_Background->setText(QString::number(tbrBg));
#else
	crossTalkRatio = settings.value("crossTalkRatio").toFloat();
	m_pLineEdit_CrossTalkRatio->setText(QString::number(crossTalkRatio));

	nirfBg[0] = settings.value("nirfBg1").toFloat();;
	m_pLineEdit_NIRF_Background[0]->setText(QString::number(nirfBg[0]));

	tbrBg[0] = settings.value("tbrBg1").toFloat();;
	m_pLineEdit_TBR_Background[0]->setText(QString::number(tbrBg[0]));

	nirfBg[1] = settings.value("nirfBg2").toFloat();;
	m_pLineEdit_NIRF_Background[1]->setText(QString::number(nirfBg[1]));

	tbrBg[1] = settings.value("tbrBg2").toFloat();;
	m_pLineEdit_TBR_Background[1]->setText(QString::number(tbrBg[1]));
#endif

	compConst = settings.value("compConst").toFloat();
	if (compConst == 0.0f) compConst = 1.0f;
	m_pLineEdit_Compensation->setText(QString::number(compConst));

	if (distMap.length() > 0)
	{
		m_pCheckBox_Filtering->setChecked(true);

		m_pToggleButton_Compensation->setChecked(true);
		m_pToggleButton_TBRMode->setChecked(true);

		bool isZeroTbr = settings.value("zeroTbrDefinition").toBool();
		m_pCheckBox_ZeroTBRDefinition->setChecked(isZeroTbr);
	}

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
	
#ifndef TWO_CHANNEL_NIRF
	settings.setValue("nirfCompCoeffs_a", QString::number(m_pConfig->nirfCompCoeffs_a, 'f', 10));
	settings.setValue("nirfCompCoeffs_b", QString::number(m_pConfig->nirfCompCoeffs_b, 'f', 10));
	settings.setValue("nirfCompCoeffs_c", QString::number(m_pConfig->nirfCompCoeffs_c, 'f', 10));
	settings.setValue("nirfCompCoeffs_d", QString::number(m_pConfig->nirfCompCoeffs_d, 'f', 10));
#else
	settings.setValue("nirfCompCoeffs_a1", QString::number(m_pConfig->nirfCompCoeffs_a[0], 'f', 10));
	settings.setValue("nirfCompCoeffs_b1", QString::number(m_pConfig->nirfCompCoeffs_b[0], 'f', 10));
	settings.setValue("nirfCompCoeffs_c1", QString::number(m_pConfig->nirfCompCoeffs_c[0], 'f', 10));
	settings.setValue("nirfCompCoeffs_d1", QString::number(m_pConfig->nirfCompCoeffs_d[0], 'f', 10));

	settings.setValue("nirfCompCoeffs_a2", QString::number(m_pConfig->nirfCompCoeffs_a[1], 'f', 10));
	settings.setValue("nirfCompCoeffs_b2", QString::number(m_pConfig->nirfCompCoeffs_b[1], 'f', 10));
	settings.setValue("nirfCompCoeffs_c2", QString::number(m_pConfig->nirfCompCoeffs_c[1], 'f', 10));
	settings.setValue("nirfCompCoeffs_d2", QString::number(m_pConfig->nirfCompCoeffs_d[1], 'f', 10));

	settings.setValue("crossTalkRatio", QString::number(crossTalkRatio, 'f', 4));
#endif
	
	settings.setValue("nirfFactorThres", QString::number(m_pConfig->nirfFactorThres, 'f', 1));
	settings.setValue("nirfFactorPropConst", QString::number(m_pConfig->nirfFactorPropConst, 'f', 3));
	settings.setValue("nirfDistPropConst", QString::number(m_pConfig->nirfDistPropConst, 'f', 3));

	settings.setValue("nirfLumContourOffset", m_pConfig->nirfLumContourOffset);
	settings.setValue("nirfOuterSheathPos", m_pConfig->nirfOuterSheathPos);

#ifndef TWO_CHANNEL_NIRF
	settings.setValue("nirfBg", QString::number(nirfBg, 'f', 4));
	settings.setValue("tbrBg", QString::number(tbrBg, 'f', 4));
#else
	settings.setValue("nirfBg1", QString::number(nirfBg[0], 'f', 4));
	settings.setValue("tbrBg1", QString::number(tbrBg[0], 'f', 4));
	settings.setValue("nirfBg2", QString::number(nirfBg[1], 'f', 4));
	settings.setValue("tbrBg2", QString::number(tbrBg[1], 'f', 4));
#endif
	settings.setValue("compConst", QString::number(compConst, 'f', 4));

	//if (!m_pResultTab->getPolishedSurfaceFindingStatus())
		settings.setValue("circCenter", QString::number(m_pConfig->circCenter));
	//else
		//settings.setValue("ballRadius", QString::number(m_pConfig->ballRadius));
	settings.setValue("circRadius", QString::number(m_pConfig->circRadius));
	
	settings.setValue("zeroTbrDefinition", QString::number(m_pCheckBox_ZeroTBRDefinition->isChecked()));

	settings.endGroup();
}

#endif
