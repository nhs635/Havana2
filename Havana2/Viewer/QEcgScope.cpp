
#include <Havana2/Configuration.h>

#include "QEcgScope.h"


QEcgScope::QEcgScope(QWidget *parent) :
	QDialog(parent)
{
    // Do not use
}

QEcgScope::QEcgScope(QRange x_range, QRange y_range,
               int num_x_ticks, int num_y_ticks,
               double x_interval, double y_interval, double x_offset, double y_offset,
               QString x_unit, QString y_unit, QWidget *parent) :
	QDialog(parent)
{
    // Set default size
    resize(400, 300);

    // Create layout
    m_pGridLayout = new QGridLayout(this);
	m_pGridLayout->setSpacing(1);

    // Create render area
    m_pRenderArea = new QRenderAreaEcg(this);
    m_pRenderArea->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);

    // Set Axis
    setAxis(x_range, y_range, num_x_ticks, num_y_ticks,
            x_interval, y_interval, x_offset, y_offset, x_unit, y_unit);

    // Set layout
    m_pGridLayout->addWidget(m_pRenderArea, 0, 1);
    setLayout(m_pGridLayout);

    // Initialization
    setUpdatesEnabled(true);
}

QEcgScope::~QEcgScope()
{
}


void QEcgScope::setAxis(QRange x_range, QRange y_range,
                     int num_x_ticks, int num_y_ticks,
                     double x_interval, double y_interval, double x_offset, double y_offset,
                     QString x_unit, QString y_unit)
{
    // Set range
    m_pRenderArea->m_xRange = x_range;
    m_pRenderArea->m_yRange = y_range;

    // Set graph size
    m_pRenderArea->m_sizeGraph = { m_pRenderArea->m_xRange.max - m_pRenderArea->m_xRange.min ,
                                   m_pRenderArea->m_yRange.max - m_pRenderArea->m_yRange.min };

    // Set x ticks
    double val; QString str;
    if (num_x_ticks > 1)
    {
        for (int i = 0; i < num_x_ticks; i++)
        {
            QLabel *xlabel = new QLabel(this);
            val = x_interval * (m_pRenderArea->m_xRange.min + i * m_pRenderArea->m_sizeGraph.width() / (num_x_ticks - 1)) + x_offset;
            xlabel->setText(QString("%1%2").arg(val, 4).arg(x_unit));

            m_pLabelVector_x.push_back(xlabel);
        }

        QHBoxLayout *pHBoxLayout = new QHBoxLayout;
        pHBoxLayout->addWidget(m_pLabelVector_x.at(0));
        for (int i = 1; i < num_x_ticks; i++)
        {
            pHBoxLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));
            pHBoxLayout->addWidget(m_pLabelVector_x.at(i));
        };
        m_pGridLayout->addItem(pHBoxLayout, 1, 1);
    }

    // Set y ticks
    if (num_y_ticks > 1)
    {
        for (int i = num_y_ticks - 1; i >= 0; i--)
        {
            QLabel *ylabel = new QLabel(this);
            val = y_interval * (m_pRenderArea->m_yRange.min + i * m_pRenderArea->m_sizeGraph.height() / (num_y_ticks - 1)) + y_offset;
            val = double(int(val * 100)) / 100;
            ylabel->setText(QString("%1%2").arg(val).arg(y_unit));
            ylabel->setAlignment(Qt::AlignRight);

            m_pLabelVector_y.push_back(ylabel);
        }

        QVBoxLayout *pVBoxLayout = new QVBoxLayout;
        pVBoxLayout->addWidget(m_pLabelVector_y.at(0));
        for (int i = 1; i < num_y_ticks; i++)
        {
            pVBoxLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));
            pVBoxLayout->addWidget(m_pLabelVector_y.at(i));
        }
        m_pGridLayout->addItem(pVBoxLayout, 0, 0);
    }

	// Allocate data buffer
	for (int i = 0; i < (int)m_pRenderArea->m_sizeGraph.width(); i++)
	{
		m_pRenderArea->m_dqData.push_back(0);
		m_pRenderArea->m_dqIsPeak.push_back(false);
	}
    m_pRenderArea->update();
}

void QEcgScope::clearDeque()
{
	for (int i = 0; i < m_pRenderArea->m_dqData.size(); i++)
	{
		m_pRenderArea->m_dqData.pop_front();
		m_pRenderArea->m_dqData.push_back(0);

		m_pRenderArea->m_dqIsPeak.pop_front();
		m_pRenderArea->m_dqIsPeak.push_back(false);
	}
}

void QEcgScope::drawData(float data, bool is_peak)
{
	static int n = 0;

	if (m_pRenderArea->m_dqData.size() > (int)m_pRenderArea->m_sizeGraph.width())
		m_pRenderArea->m_dqData.pop_front();
	m_pRenderArea->m_dqData.push_back(data);

	if (m_pRenderArea->m_dqIsPeak.size() > (int)m_pRenderArea->m_sizeGraph.width())
		m_pRenderArea->m_dqIsPeak.pop_front();
	m_pRenderArea->m_dqIsPeak.push_back(is_peak);

	if (n++ % ECG_VIEW_RENEWAL_COUNT == 0) m_pRenderArea->update();
}


QRenderAreaEcg::QRenderAreaEcg(QWidget *parent) :
	QWidget(parent)
{
    QPalette pal = this->palette();
    pal.setColor(QPalette::Background, QColor(0x282d30));
    setPalette(pal);
    setAutoFillBackground(true);
}

void QRenderAreaEcg::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
	
    // Area size
    int w = this->width();
    int h = this->height();

    // Draw grid
    painter.setPen(QColor(0x4f5555)); // Minor grid color
    for (int i = 0; i <= 16; i++)
        painter.drawLine(i * w / 16, 0, i * w / 16, h);
    painter.setPen(QColor(0x7b8585)); // Major grid color
    for (int i = 0; i <= 4; i++)
        painter.drawLine(i * w / 4, 0, i * w / 4, h);
    for (int i = 0; i <= 4; i++)
        painter.drawLine(0, i * h / 4, w, i * h / 4);
	
    // Draw graph  
    painter.setPen(QColor(0xfff65d));	
	for (int i = (int)(m_xRange.min); i < (int)(m_xRange.max - 1); i++)
	{
		QPointF x0, x1;						
		x0.setX((float)(i) / (float)m_sizeGraph.width() * w);
        x0.setY((float)(m_yRange.max - m_dqData.at(i)) * (float)h / (float)(m_yRange.max - m_yRange.min));
        x1.setX((float)(i + 1) / (float)m_sizeGraph.width() * w);
        x1.setY((float)(m_yRange.max - m_dqData.at(i + 1)) * (float)h / (float)(m_yRange.max - m_yRange.min));

		painter.drawLine(x0, x1);
    }
    
	// Draw peak position
	painter.setPen(QColor(0xff0000));
	for (int i = (int)(m_xRange.min); i < (int)(m_xRange.max - 1); i++)
	{
		if (m_dqIsPeak.at(i))
		{
			QPointF x0, x1;
			x0.setX((float)(i) / (float)m_sizeGraph.width() * w); x0.setY(0.0f);
			x1.setX((float)(i) / (float)m_sizeGraph.width() * w); x1.setY((float)h);

			painter.drawLine(x0, x1);
		}
	}
}
