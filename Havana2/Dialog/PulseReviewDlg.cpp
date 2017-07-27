
#include "PulseReviewDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QResultTab.h>
#include <Havana2/Viewer/QImageView.h>

#include <DataProcess/FLIMProcess/FLIMProcess.h>


#ifdef OCT_FLIM
PulseReviewDlg::PulseReviewDlg(QWidget *parent) :
    QDialog(parent)
{
    // Set default size & frame
    setFixedSize(525, 250);
    setWindowFlags(Qt::Tool);
	setWindowTitle("Pulse Review");

    // Set main window objects
    m_pResultTab = (QResultTab*)parent;
    m_pMainWnd = m_pResultTab->getMainWnd();
	m_pConfig = m_pMainWnd->m_pConfiguration;
    m_pFLIM = m_pResultTab->m_pFLIMpost;		

	// Create widgets
	m_pScope_PulseView = new QScope({ 0, (double)m_pFLIM->_resize.crop_src.size(0) }, { -POWER_2(12), POWER_2(15) }, 2, 2, 1, 1, 0, 0, "", "", false);
	m_pScope_PulseView->setMinimumHeight(200);
	m_pScope_PulseView->getRender()->setGrid(8, 32, 1, true);
	
	m_pLabel_CurrentAline = new QLabel(this);
	QString str; str.sprintf("Current A-line : %4d / %4d   ", 4, m_pFLIM->_resize.ny * 4);
	m_pLabel_CurrentAline->setText(str);
	
	m_pSlider_CurrentAline = new QSlider(this);
	m_pSlider_CurrentAline->setOrientation(Qt::Horizontal);
	m_pSlider_CurrentAline->setRange(0, m_pFLIM->_resize.ny - 1);
	m_pSlider_CurrentAline->setValue(0);

	m_pLabel_PulseType = new QLabel(this);
	m_pLabel_PulseType->setText("Pulse Type");

	m_pComboBox_PulseType = new QComboBox(this);
	m_pComboBox_PulseType->addItem("Crop");
	m_pComboBox_PulseType->addItem("Mask");

	// Create layout
	QGridLayout *pGridLayout = new QGridLayout;
	
	pGridLayout->addWidget(m_pScope_PulseView, 0, 0, 1, 4);

	pGridLayout->addWidget(m_pLabel_CurrentAline, 1, 0);
	pGridLayout->addWidget(m_pSlider_CurrentAline, 1, 1);
	pGridLayout->addWidget(m_pLabel_PulseType, 1, 2);
	pGridLayout->addWidget(m_pComboBox_PulseType, 1, 3);

	// Set layout
	this->setLayout(pGridLayout);

	// Connect
	connect(m_pSlider_CurrentAline, SIGNAL(valueChanged(int)), this, SLOT(drawPulse(int)));
	connect(m_pComboBox_PulseType, SIGNAL(currentIndexChanged(int)), this, SLOT(changeType()));
	drawPulse(0);
}

PulseReviewDlg::~PulseReviewDlg()
{
}

void PulseReviewDlg::keyPressEvent(QKeyEvent *e)
{
	if (e->key() != Qt::Key_Escape)
		QDialog::keyPressEvent(e);
}


void PulseReviewDlg::drawPulse(int aline)
{
	auto pVector = (m_pComboBox_PulseType->currentIndex() == 0) ? &m_pResultTab->m_vectorPulseCrop : &m_pResultTab->m_vectorPulseMask;

	QString str; str.sprintf("Current A-line : %4d / %4d   ", 4 * (aline + 1), m_pFLIM->_resize.ny * 4);
	m_pLabel_CurrentAline->setText(str);

	m_pScope_PulseView->drawData(&pVector->at(m_pResultTab->getCurrentFrame())(0, aline));
	
	if (m_pResultTab->getRectImageView()->isVisible())
	{
		m_pResultTab->getRectImageView()->setVerticalLine(1, 4 * aline);
		m_pResultTab->getRectImageView()->getRender()->update();
	}
	else
	{
		m_pResultTab->getCircImageView()->setVerticalLine(1, 4 * aline);
		m_pResultTab->getCircImageView()->getRender()->update();
	}
}

void PulseReviewDlg::changeType()
{
	drawPulse(m_pSlider_CurrentAline->value());
}

#endif
