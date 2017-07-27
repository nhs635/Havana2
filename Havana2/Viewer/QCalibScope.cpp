
#include "QCalibScope.h"


QCalibScope::QCalibScope(QWidget *parent) :
	QDialog(parent)
{
    // Do not use
}

QCalibScope::QCalibScope(QRange x_range, QRange y_range,
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
    m_pRenderArea = new QRenderAreaCalib(this);
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

QCalibScope::~QCalibScope()
{
	if (m_pRenderArea->m_pData)
		delete[] m_pRenderArea->m_pData;
}


void QCalibScope::setAxis(QRange x_range, QRange y_range,
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
            ylabel->setText(QString("%1%2").arg(val, 4).arg(y_unit));
            ylabel->setFixedWidth(40);
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
	m_pRenderArea->m_pData = new float[(int)m_pRenderArea->m_sizeGraph.width()];
	memset(m_pRenderArea->m_pData, 0, sizeof(float) * (int)m_pRenderArea->m_sizeGraph.width());

    m_pRenderArea->update();
}

void QCalibScope::resetAxis(QRange x_range, QRange y_range,
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
    int num_x_ticks = m_pLabelVector_x.size();
	if (num_x_ticks > 1)
	{
		for (int i = 0; i < num_x_ticks; i++)
		{
			QLabel *xlabel = m_pLabelVector_x.at(i);
			val = x_interval * (m_pRenderArea->m_xRange.min + i * m_pRenderArea->m_sizeGraph.width() / (num_x_ticks - 1)) + x_offset;
			xlabel->setText(QString("%1%2").arg(val, 4).arg(x_unit));
		}
	}

	// Set y ticks
    int num_y_ticks = m_pLabelVector_y.size();
	if (num_y_ticks > 1)
	{
		for (int i = 0; i < num_x_ticks; i++)
		{
			QLabel *ylabel = m_pLabelVector_y.at(num_x_ticks - 1 - i);
			val = y_interval * (m_pRenderArea->m_yRange.min + i * m_pRenderArea->m_sizeGraph.height() / (num_y_ticks - 1)) + y_offset;
			val = double(int(val * 100)) / 100;
			ylabel->setText(QString("%1%2").arg(val, 4).arg(y_unit));
		}
	}

	// Reallocate data buffer
	if (m_pRenderArea->m_pData)
	{
		delete[] m_pRenderArea->m_pData;
		m_pRenderArea->m_pData = nullptr;
	}
	m_pRenderArea->m_pData = new float[(int)m_pRenderArea->m_sizeGraph.width()];
	memset(m_pRenderArea->m_pData, 0, sizeof(float) * (int)m_pRenderArea->m_sizeGraph.width());

    m_pRenderArea->update();
}

void QCalibScope::drawData(float* pData)
{
	if (m_pRenderArea->m_pData != nullptr)
		memcpy(m_pRenderArea->m_pData, pData, sizeof(float) * (int)m_pRenderArea->m_sizeGraph.width());

	m_pRenderArea->update();
}


QRenderAreaCalib::QRenderAreaCalib(QWidget *parent) :
	QWidget(parent), m_bSelectionAvailable(false), m_bMousePressed(false)
{
    QPalette pal = this->palette();
    pal.setColor(QPalette::Background, QColor(0x282d30));
    setPalette(pal);
    setAutoFillBackground(true);

	memset(m_selected, 0, sizeof(int) * 2);
	memset(m_selected1, 0, sizeof(int) * 2);
}

void QRenderAreaCalib::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    // Area size
    int w = this->width();
    int h = this->height();

    // Draw grid
    painter.setPen(QColor(0x4f5555)); // Minor grid color
    for (int i = 0; i <= 64; i++)
        painter.drawLine(i * w / 64, 0, i * w / 64, h);
    painter.setPen(QColor(0x7b8585)); // Major grid color
    for (int i = 0; i <= 8; i++)
        painter.drawLine(i * w / 8, 0, i * w / 8, h);
    for (int i = 0; i <= 4; i++)
        painter.drawLine(0, i * h / 4, w, i * h / 4);
	
    // Draw graph
    if (m_pData != nullptr)
    {
		int temp;
		memcpy(m_selected1, m_selected, sizeof(int) * 2);
		// m_selected1 ==> ordered
		if (m_selected1[1] < m_selected1[0])
		{
			temp = m_selected1[1];
			m_selected1[1] = m_selected1[0];
			m_selected1[0] = temp;
		}

		for (int i = (int)(m_xRange.min); i < (int)(m_xRange.max - 1); i++)
		{
			QPointF x0, x1;
			x0.setX((float)(i) / (float)m_sizeGraph.width() * w);
			x0.setY((float)(m_yRange.max - m_pData[i]) * (float)h / (float)(m_yRange.max - m_yRange.min));
			x1.setX((float)(i + 1) / (float)m_sizeGraph.width() * w);
			x1.setY((float)(m_yRange.max - m_pData[i + 1]) * (float)h / (float)(m_yRange.max - m_yRange.min));

			if (((i >= m_selected1[0]) && (i <= m_selected1[1])) && m_bSelectionAvailable)
				painter.setPen(QColor(0xe15dff)); // pink
			else
				painter.setPen(QColor(0xfff65d)); // yellow
            
			painter.drawLine(x0, x1);			
        }
    }
}

void QRenderAreaCalib::mousePressEvent(QMouseEvent *e)
{
	if (m_bSelectionAvailable)
	{
		m_bMousePressed = true;

		QPoint p = e->pos();
		if ((p.x() > 0) && (p.y() > 0) && (p.x() < this->width()) && (p.y() < this->height()))
		{
			int x = (double)p.x() / (double)this->width() * (double)m_sizeGraph.width();
			m_selected[0] = x;
			m_selected[1] = x;
			//printf("[%d %d]\n", m_selected[0], m_selected[1]);
			this->update();
		}
	}
}

void QRenderAreaCalib::mouseMoveEvent(QMouseEvent *e)
{
	if (m_bSelectionAvailable)
	{
		if (m_bMousePressed)
		{
			QPoint p = e->pos();
			if ((p.x() > 0) && (p.y() > 0) && (p.x() < this->width()) && (p.y() < this->height()))
			{
				int x = (double)p.x() / (double)this->width() * (double)m_sizeGraph.width();
				m_selected[1] = x;
				//printf("[%d %d]\n", m_selected[0], m_selected[1]);
				this->update();
			}
		}	
	}
}

void QRenderAreaCalib::mouseReleaseEvent(QMouseEvent *e)
{
	if (m_bSelectionAvailable)
	{
		m_bMousePressed = false;

		QPoint p = e->pos();
		if ((p.x() > 0) && (p.y() > 0) && (p.x() < this->width()) && (p.y() < this->height()))
		{
			int x = (double)p.x() / (double)this->width() * (double)m_sizeGraph.width();
			m_selected[1] = x;
			//printf("[%d %d]\n", m_selected[0], m_selected[1]);
			this->update();
		}
	}
}