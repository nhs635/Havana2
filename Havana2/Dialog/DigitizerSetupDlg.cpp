
#include "DigitizerSetupDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QOperationTab.h>
#include <Havana2/QStreamTab.h>
#include <Havana2/QResultTab.h>

#include <DataAcquisition/DataAcquisition.h>
#include <MemoryBuffer/MemoryBuffer.h>


DigitizerSetupDlg::DigitizerSetupDlg(QWidget *parent) :
    QDialog(parent)
{
    // Set default size & frame
#if PX14_ENABLE
    setFixedSize(400, 220);
#elif ALAZAR_ENABLE
    setFixedSize(230, 200);
#endif
    setWindowFlags(Qt::Tool);
	setWindowTitle("Digitizer Setup");

    // Set main window objects
    m_pOperationTab = (QOperationTab*)parent;
    m_pMainWnd = m_pOperationTab->getMainWnd();
	m_pConfig = m_pMainWnd->m_pConfiguration;
	m_pStreamTab = m_pMainWnd->m_pStreamTab;
	m_pResultTab = m_pMainWnd->m_pResultTab;
	m_pDataAcq = m_pOperationTab->m_pDataAcquisition;
	m_pMemBuff = m_pOperationTab->m_pMemoryBuffer;

    // Create widgets for digitizer setup
	m_pLineEdit_SamplingRate = new QLineEdit(this);
	m_pLineEdit_SamplingRate->setFixedWidth(35);
#if PX14_ENABLE || ALAZAR_ENABLE
	m_pLineEdit_SamplingRate->setText(QString::number(ADC_RATE));
#else
	m_pLineEdit_SamplingRate->setText(QString::number(0));
#endif
	m_pLineEdit_SamplingRate->setDisabled(true);
	m_pLineEdit_SamplingRate->setAlignment(Qt::AlignCenter);

	m_pComboBox_VoltageRangeCh1 = new QComboBox(this);	
#ifdef OCT_FLIM
	m_pComboBox_VoltageRangeCh2 = new QComboBox(this);
#endif

#if PX14_ENABLE
	double voltage = DIGITIZER_VOLTAGE;
#else
	double voltage = 0.0;
#endif

#if PX14_ENABLE
	for (int i = 0; i < 25; i++)
	{
		m_pComboBox_VoltageRangeCh1->addItem(QString("%1 Vpp").arg(voltage, 0, 'f', 3, '0'));
#ifdef OCT_FLIM
		m_pComboBox_VoltageRangeCh2->addItem(QString("%1 Vpp").arg(voltage, 0, 'f', 3, '0'));
#endif
		voltage *= DIGITIZER_VOLTAGE_RATIO;
	}
#elif ALAZAR_ENABLE
    for (int i = 1; i < 14; i++)
    {
        m_pComboBox_VoltageRangeCh1->addItem(QString("%1 V").arg(m_pConfig->voltRange[i], 0, 'f', 3, '0'));
#ifdef OCT_FLIM
        m_pComboBox_VoltageRangeCh2->addItem(QString("%1 V").arg(m_pConfig->voltRange[i], 0, 'f', 3, '0'));
#endif
    }
#endif
    m_pComboBox_VoltageRangeCh1->setCurrentIndex(m_pConfig->ch1VoltageRange);
#ifdef OCT_FLIM
	m_pComboBox_VoltageRangeCh2->setCurrentIndex(m_pConfig->ch2VoltageRange);
#endif	

#if PX14_ENABLE
	m_pLineEdit_PreTrigger = new QLineEdit(this);
	m_pLineEdit_PreTrigger->setFixedWidth(35);
	m_pLineEdit_PreTrigger->setText(QString::number(m_pConfig->preTrigSamps));
	m_pLineEdit_PreTrigger->setAlignment(Qt::AlignCenter);
#elif ALAZAR_ENABLE
    m_pLineEdit_TriggerDelay = new QLineEdit(this);
    m_pLineEdit_TriggerDelay->setFixedWidth(35);
    m_pLineEdit_TriggerDelay->setText(QString::number(m_pConfig->triggerDelay));
    m_pLineEdit_TriggerDelay->setAlignment(Qt::AlignCenter);
#endif

	m_pLineEdit_nChannels = new QLineEdit(this);
	m_pLineEdit_nChannels->setFixedWidth(35);
	m_pLineEdit_nChannels->setText(QString::number(m_pConfig->nChannels));
	m_pLineEdit_nChannels->setAlignment(Qt::AlignCenter);
	m_pLineEdit_nChannels->setDisabled(true); // m_pMemBuff->m_bIsAllocatedWritingBuffer);

	m_pLineEdit_nScans = new QLineEdit(this);
	m_pLineEdit_nScans->setFixedWidth(35);
	m_pLineEdit_nScans->setText(QString::number(m_pConfig->nScans));
	m_pLineEdit_nScans->setAlignment(Qt::AlignCenter);
	m_pLineEdit_nScans->setDisabled(true); // m_pMemBuff->m_bIsAllocatedWritingBuffer);

	m_pLineEdit_nAlines = new QLineEdit(this);
	m_pLineEdit_nAlines->setFixedWidth(35);
	m_pLineEdit_nAlines->setText(QString::number(m_pConfig->nAlines));
	m_pLineEdit_nAlines->setAlignment(Qt::AlignCenter);
	m_pLineEdit_nAlines->setDisabled(m_pMemBuff->m_bIsAllocatedWritingBuffer);

#if PX14_ENABLE
	m_pLabel_BootTimeBufTitle[0] = new QLabel("Index", this);
	m_pLabel_BootTimeBufTitle[0]->setFixedWidth(30);
	m_pLabel_BootTimeBufTitle[0]->setAlignment(Qt::AlignCenter);
	m_pLabel_BootTimeBufTitle[1] = new QLabel("Buffer Size", this);
	m_pLabel_BootTimeBufTitle[1]->setFixedWidth(60);
	m_pLabel_BootTimeBufTitle[1]->setAlignment(Qt::AlignCenter);

	m_pLabel_BootTimeBufInstruction = new QLabel(this);
	m_pLabel_BootTimeBufInstruction->setText("Required Buffer Size =\n2 * nChannels * \nnScans * nAlines");
	m_pLabel_BootTimeBufInstruction->setAlignment(Qt::AlignCenter);

	m_pButtonGroup_IndexSelection = new QButtonGroup(this);
	for (int i = 0; i < 4; i++)
	{
		m_pRadioButton_BootTimeBufIdx[i] = new QRadioButton(QString("%1").arg(i), this);
		m_pRadioButton_BootTimeBufIdx[i]->setDisabled(m_pMemBuff->m_bIsAllocatedWritingBuffer);
		m_pButtonGroup_IndexSelection->addButton(m_pRadioButton_BootTimeBufIdx[i], i);
		//m_pRadioButton_BootTimeBufIdx[i]->setFixedWidth(30);
		//m_pRadioButton_BootTimeBufIdx[i]->setAlignment(Qt::AlignCenter);

		m_pLineEdit_BootTimeBufSamps[i] = new QLineEdit(this);
		m_pLineEdit_BootTimeBufSamps[i]->setFixedWidth(60);
		m_pLineEdit_BootTimeBufSamps[i]->setAlignment(Qt::AlignCenter);
	}
	m_pRadioButton_BootTimeBufIdx[m_pConfig->bootTimeBufferIndex]->setChecked(true);
	getBootTimeBufCfg();

	m_pPushButton_BootTimeBufSet = new QPushButton(this);
	m_pPushButton_BootTimeBufSet->setText("Set");
	m_pPushButton_BootTimeBufSet->setFixedWidth(40);
	m_pPushButton_BootTimeBufGet = new QPushButton(this);
	m_pPushButton_BootTimeBufGet->setText("Get");
	m_pPushButton_BootTimeBufGet->setFixedWidth(40);
#endif
    
    // Set layout for digitizer setup
	QGroupBox *pGroupBox_DigitizerSetup = new QGroupBox("Digitizer Setup");
	QGridLayout *pGridLayout_DigitizerSetup = new QGridLayout;
	pGridLayout_DigitizerSetup->setSpacing(1);

	pGridLayout_DigitizerSetup->addWidget(new QLabel("Sampling Rate", this), 0, 0);
	pGridLayout_DigitizerSetup->addWidget(m_pLineEdit_SamplingRate, 0, 1);
	pGridLayout_DigitizerSetup->addWidget(new QLabel("  MHz", this), 0, 2);
    pGridLayout_DigitizerSetup->addWidget(new QLabel("Ch1 Voltage Range ", this), 1, 0);
	pGridLayout_DigitizerSetup->addWidget(m_pComboBox_VoltageRangeCh1, 1, 1, 1, 2);
#ifdef OCT_FLIM
    pGridLayout_DigitizerSetup->addWidget(new QLabel("Ch2 Voltage Range ", this), 2, 0);
	pGridLayout_DigitizerSetup->addWidget(m_pComboBox_VoltageRangeCh2, 2, 1, 1, 2);
#endif
#if PX14_ENABLE
	pGridLayout_DigitizerSetup->addWidget(new QLabel("Pre-Trigger Samples ", this), 3, 0);
	pGridLayout_DigitizerSetup->addWidget(m_pLineEdit_PreTrigger, 3, 1, 1, 2);
#elif ALAZAR_ENABLE
    pGridLayout_DigitizerSetup->addWidget(new QLabel("Trigger-Delay Samples ", this), 3, 0);
    pGridLayout_DigitizerSetup->addWidget(m_pLineEdit_TriggerDelay, 3, 1, 1, 2);
#endif
	
	pGridLayout_DigitizerSetup->addWidget(new QLabel("nChannels ", this), 4, 0);
	pGridLayout_DigitizerSetup->addWidget(m_pLineEdit_nChannels, 4, 1, 1, 2);
	pGridLayout_DigitizerSetup->addWidget(new QLabel("nScans", this), 5, 0);
	pGridLayout_DigitizerSetup->addWidget(m_pLineEdit_nScans, 5, 1, 1, 2);
	pGridLayout_DigitizerSetup->addWidget(new QLabel("nAlines", this), 6, 0);
	pGridLayout_DigitizerSetup->addWidget(m_pLineEdit_nAlines, 6, 1, 1, 2);

	pGroupBox_DigitizerSetup->setLayout(pGridLayout_DigitizerSetup);

#if PX14_ENABLE
	// Set layout for boot-time buffer configuration
	QGroupBox *pGroupBox_BootBufConfig = new QGroupBox("Boot-time Buffer Configuration");
	QGridLayout *pGridLayout_BootBufConfig = new QGridLayout;
	pGridLayout_BootBufConfig->setSpacing(1);
	
	pGridLayout_BootBufConfig->addWidget(m_pLabel_BootTimeBufTitle[0], 0, 0);
	pGridLayout_BootBufConfig->addWidget(m_pLabel_BootTimeBufTitle[1], 0, 1);
	for (int i = 0; i < 4; i++)
	{
		pGridLayout_BootBufConfig->addWidget(m_pRadioButton_BootTimeBufIdx[i], i + 1, 0);
		pGridLayout_BootBufConfig->addWidget(m_pLineEdit_BootTimeBufSamps[i], i + 1, 1);
	}
	pGridLayout_BootBufConfig->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding), 5, 0, 1, 3);
	pGridLayout_BootBufConfig->addWidget(m_pLabel_BootTimeBufInstruction, 6, 0, 1, 3);

	pGridLayout_BootBufConfig->addWidget(m_pPushButton_BootTimeBufGet, 3, 2);
	pGridLayout_BootBufConfig->addWidget(m_pPushButton_BootTimeBufSet, 4, 2);
		
	pGroupBox_BootBufConfig->setLayout(pGridLayout_BootBufConfig);
#endif

	// Set layout
	QHBoxLayout *pHBoxLayout = new QHBoxLayout;
	pHBoxLayout->setSpacing(3);
	pHBoxLayout->addWidget(pGroupBox_DigitizerSetup);
#if PX14_ENABLE
	pHBoxLayout->addWidget(pGroupBox_BootBufConfig);
#endif

    setLayout(pHBoxLayout);

    // Connect
	connect(m_pComboBox_VoltageRangeCh1, SIGNAL(currentIndexChanged(int)), this, SLOT(changeVoltageRangeCh1(int)));
#ifdef OCT_FLIM
	connect(m_pComboBox_VoltageRangeCh2, SIGNAL(currentIndexChanged(int)), this, SLOT(changeVoltageRangeCh2(int)));
#endif
#if PX14_ENABLE
	connect(m_pLineEdit_PreTrigger, SIGNAL(textChanged(const QString &)), this, SLOT(changePreTrigger(const QString &)));
#elif ALAZAR_ENABLE
    connect(m_pLineEdit_TriggerDelay, SIGNAL(textChanged(const QString &)), this, SLOT(changeTriggerDelay(const QString &)));
#endif
	connect(m_pLineEdit_nChannels, SIGNAL(textChanged(const QString &)), this, SLOT(changeNchannels(const QString &)));
	connect(m_pLineEdit_nScans, SIGNAL(textChanged(const QString &)), this, SLOT(changeNscans(const QString &)));
	connect(m_pLineEdit_nAlines, SIGNAL(textChanged(const QString &)), this, SLOT(changeNalines(const QString &)));

#if PX14_ENABLE
	connect(m_pButtonGroup_IndexSelection, SIGNAL(buttonClicked(int)), this, SLOT(changeBootTimeBufIdx(int)));
	connect(m_pPushButton_BootTimeBufGet, SIGNAL(clicked(bool)), this, SLOT(getBootTimeBufCfg()));
	connect(m_pPushButton_BootTimeBufSet, SIGNAL(clicked(bool)), this, SLOT(setBootTimeBufCfg()));
#endif
}

DigitizerSetupDlg::~DigitizerSetupDlg()
{
}

void DigitizerSetupDlg::keyPressEvent(QKeyEvent *e)
{
	if (e->key() != Qt::Key_Escape)
		QDialog::keyPressEvent(e);
}


void DigitizerSetupDlg::changeVoltageRangeCh1(int idx)
{
    m_pConfig->ch1VoltageRange = idx;
	m_pStreamTab->setCh1ScopeVoltRange(idx);
}

#ifdef OCT_FLIM
void DigitizerSetupDlg::changeVoltageRangeCh2(int idx)
{
    m_pConfig->ch2VoltageRange = idx + 1;
	m_pStreamTab->setCh2ScopeVoltRange(idx);
}
#endif

void DigitizerSetupDlg::changePreTrigger(const QString &str)
{    
#if PX14_ENABLE
	m_pConfig->preTrigSamps = str.toInt();
#endif
}

void DigitizerSetupDlg::changeTriggerDelay(const QString &str)
{
#if ALAZAR_ENABLE
    m_pConfig->triggerDelay = str.toInt();
#endif
}

void DigitizerSetupDlg::changeNchannels(const QString &str)
{
	// Actually unused
	m_pConfig->nChannels = str.toInt();
	m_pConfig->nFrameSize = m_pConfig->nChannels * m_pConfig->nScans * m_pConfig->nAlines;
}

void DigitizerSetupDlg::changeNscans(const QString &str)
{
	// Actually unused
	m_pConfig->nScans = str.toInt();
	m_pConfig->nScansFFT = NEAR_2_POWER((double)m_pConfig->nScans);
	m_pConfig->n2ScansFFT = m_pConfig->nScansFFT / 2;
	m_pConfig->nFrameSize = m_pConfig->nChannels * m_pConfig->nScans * m_pConfig->nAlines;
}

void DigitizerSetupDlg::changeNalines(const QString &str)
{
	m_pConfig->nAlines = str.toInt();
	if ((m_pConfig->nAlines > 200) && (m_pConfig->nAlines % 4 == 0))
	{
		m_pConfig->n4Alines = m_pConfig->nAlines / 4;
		m_pConfig->nAlines4 = ((m_pConfig->nAlines + 3) >> 2) << 2;
		m_pConfig->nFrameSize = m_pConfig->nChannels * m_pConfig->nScans * m_pConfig->nAlines;

		m_pStreamTab->resetObjectsForAline(m_pConfig->nAlines);
		m_pResultTab->setUserDefinedAlines(m_pConfig->nAlines);
	}
	else
		printf("nAlines should be >200 and 4's multiple.\n");
}

#if PX14_ENABLE
void DigitizerSetupDlg::changeBootTimeBufIdx(int idx)
{
	m_pConfig->bootTimeBufferIndex = idx;
}

void DigitizerSetupDlg::getBootTimeBufCfg()
{
	int buf_size;
	for (int i = 0; i < 4; i++)
	{
		m_pDataAcq->GetBootTimeBufCfg(i, buf_size);
		m_pLineEdit_BootTimeBufSamps[i]->setText(QString::number(buf_size));
	}	
}

void DigitizerSetupDlg::setBootTimeBufCfg()
{
	int buf_size;
	for (int i = 0; i < 4; i++)
	{
		buf_size = m_pLineEdit_BootTimeBufSamps[i]->text().toInt();
		m_pDataAcq->SetBootTimeBufCfg(i, buf_size);
	}

	QMessageBox::information(this, "Setting boot-time buffer", "The new boot-time buffer will be applied after PC restart.");
}
#endif
