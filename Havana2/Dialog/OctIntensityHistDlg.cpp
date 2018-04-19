
#include "OctIntensityHistDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QStreamTab.h>
#include <Havana2/QResultTab.h>

#include <iostream>
#include <thread>


OctIntensityHistDlg::OctIntensityHistDlg(bool isStreamTab, QWidget *parent) :
    QDialog(parent)
{
    // Set default size & frame
    setFixedSize(370, 270);
    setWindowFlags(Qt::Tool);
	setWindowTitle("OCT Intensity Histogram");

    // Set main window objects
	if (isStreamTab)
	{
		m_pStreamTab = (QStreamTab*)parent;
		m_pMainWnd = m_pStreamTab->getMainWnd();
	}
	else
	{
		m_pResultTab = (QResultTab*)parent;
		m_pMainWnd = m_pResultTab->getMainWnd();
	}
	m_pConfig = m_pMainWnd->m_pConfiguration;

	// Create layout
	m_pVBoxLayout = new QVBoxLayout;
	m_pVBoxLayout->setSpacing(20);

	// Create widgets for histogram
	if (isStreamTab)
		createHistogram(m_pConfig->nAlines);
	else
		if (m_pResultTab->m_vectorOctImage.size() == 0 )
			createHistogram(m_pConfig->nAlines);
		else
			createHistogram(m_pResultTab->m_vectorOctImage.at(0).size(1));

	// Set layout
	this->setLayout(m_pVBoxLayout);
}

OctIntensityHistDlg::~OctIntensityHistDlg()
{
	if (m_pHistogram) delete m_pHistogram;
}

void OctIntensityHistDlg::keyPressEvent(QKeyEvent *e)
{
	if (e->key() != Qt::Key_Escape)
		QDialog::keyPressEvent(e);
}


void OctIntensityHistDlg::createHistogram(int _nAlines)
{
	// Create widgets for histogram layout
	QHBoxLayout *pHBoxLayout_Histogram = new QHBoxLayout;
	pHBoxLayout_Histogram->setSpacing(3);

	uint8_t color[256];
	for (int i = 0; i < 256; i++)
		color[i] = i;

	// Create widgets for histogram
	QGridLayout *pGridLayout_IntensityHistogram = new QGridLayout;
	pGridLayout_IntensityHistogram->setSpacing(1);

	m_pLabel_Title = new QLabel("OCT Intensity (dB)", this);
		
	int n_bins = 500;
	m_pRenderArea = new QRenderArea(this);
	m_pRenderArea->setSize({ 0, (double)n_bins }, { 0, (double)(_nAlines * m_pConfig->n2ScansFFT) / (double)n_bins });
	m_pRenderArea->setFixedSize(350, 200);
	m_pRenderArea->setGrid(4, 16, 1);

	m_pHistogram = new Histogram(n_bins, _nAlines * m_pConfig->n2ScansFFT);
	
	m_pColorbar = new QImageView(ColorTable::colortable::gray, 256, 1, false);
	m_pColorbar->drawImage(color);
	m_pColorbar->getRender()->setFixedSize(350, 10);

	m_pLabel_Min = new QLabel(this);
	m_pLabel_Min->setText(QString::number(m_pConfig->octDbRange.min));
	m_pLabel_Min->setAlignment(Qt::AlignLeft);
	m_pLabel_Max = new QLabel(this);
	m_pLabel_Max->setText(QString::number(m_pConfig->octDbRange.max));
	m_pLabel_Max->setAlignment(Qt::AlignRight);
		
	// Set layout
	pGridLayout_IntensityHistogram->addWidget(m_pLabel_Title, 0, 0, 1, 4);

	pGridLayout_IntensityHistogram->addWidget(m_pRenderArea, 1, 0, 1, 4);
	pGridLayout_IntensityHistogram->addWidget(m_pColorbar->getRender(), 2, 0, 1, 4);

	pGridLayout_IntensityHistogram->addWidget(m_pLabel_Min, 3, 0);
	pGridLayout_IntensityHistogram->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed), 3, 1, 1, 2);
	pGridLayout_IntensityHistogram->addWidget(m_pLabel_Max, 3, 3);
		
	pHBoxLayout_Histogram->addItem(pGridLayout_IntensityHistogram);
	
	m_pVBoxLayout->addItem(pHBoxLayout_Histogram);

	// Connect
	connect(this, SIGNAL(plotHistogram(float*)), this, SLOT(drawHistogram(float*)));
}


void OctIntensityHistDlg::drawHistogram(float* pImg)
{
	// Histogram
	(*m_pHistogram)(pImg, m_pRenderArea->m_pData, m_pConfig->octDbRange.min, m_pConfig->octDbRange.max);
	m_pRenderArea->update();

	m_pLabel_Min->setText(QString::number(m_pConfig->octDbRange.min));
	m_pLabel_Max->setText(QString::number(m_pConfig->octDbRange.max));

	m_pColorbar->resetColormap(ColorTable::colortable(m_pConfig->octColorTable));
}