
#include "QScope2.h"

QScope2::QScope2(QWidget *parent) :
	QDialog(parent)
{
    // Do not use
}

QScope2::QScope2(QRange x_range, QRange y_range,
               int num_x_ticks, int num_y_ticks,
               double x_interval, double y_interval, double x_offset, double y_offset,
               QString x_unit, QString y_unit, bool _64_use, QWidget *parent) :
	QDialog(parent)
{
    // Set default size
    resize(400, 300);

    // Create layout
    m_pGridLayout = new QGridLayout(this);
	m_pGridLayout->setSpacing(1);

    // Create render area
    m_pRenderArea = new QRenderArea2(this);
	m_pRenderArea->m_b64Use = _64_use;
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

QScope2::~QScope2()
{
	if (m_pRenderArea->m_pData1)
		delete[] m_pRenderArea->m_pData1;
	if (m_pRenderArea->m_pData2)
		delete[] m_pRenderArea->m_pData2;
	if (m_pRenderArea->m_pData1_64)
		delete[] m_pRenderArea->m_pData1;
	if (m_pRenderArea->m_pData2_64)
		delete[] m_pRenderArea->m_pData2;
	if (m_pRenderArea->m_pSelectedRegion)
		delete[] m_pRenderArea->m_pSelectedRegion;
}


void QScope2::setAxis(QRange x_range, QRange y_range,
                     int num_x_ticks, int num_y_ticks,
                     double x_interval, double y_interval, double x_offset, double y_offset,
                     QString x_unit, QString y_unit)
{
	// Set size
	m_pRenderArea->setSize(x_range, y_range);

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
}

void QScope2::resetAxis(QRange x_range, QRange y_range,
                       double x_interval, double y_interval, double x_offset, double y_offset,
                       QString x_unit, QString y_unit)
{
	// Set size
	m_pRenderArea->setSize(x_range, y_range);

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
		for (int i = 0; i < num_y_ticks; i++)
		{
			QLabel *ylabel = m_pLabelVector_y.at(num_y_ticks - 1 - i);
			val = y_interval * (m_pRenderArea->m_yRange.min + i * m_pRenderArea->m_sizeGraph.height() / (num_y_ticks - 1)) + y_offset;
			val = double(int(val * 100)) / 100;
			ylabel->setText(QString("%1%2").arg(val, 4).arg(y_unit));
		}
	}
}

void QScope2::setVerticalLine(int len, ...)
{
	m_pRenderArea->m_vLineLen = len;

	va_list ap;
	va_start(ap, len);
	for (int i = 0; i < len; i++)
	{
		int n = va_arg(ap, int);
		m_pRenderArea->m_pVLineInd[i] = n;
	}
	va_end(ap);
}


void QScope2::drawData(const float* pData1, const float* pData2)
{
	if (m_pRenderArea->m_pData1 != nullptr)
		memcpy(m_pRenderArea->m_pData1, pData1, sizeof(float) * (int)m_pRenderArea->m_sizeGraph.width());
	if (m_pRenderArea->m_pData2 != nullptr)
		memcpy(m_pRenderArea->m_pData2, pData2, sizeof(float) * (int)m_pRenderArea->m_sizeGraph.width());

	m_pRenderArea->update();
}

void QScope2::drawData(const double* pData1_64, const double* pData2_64)
{
	if (m_pRenderArea->m_pData1_64 != nullptr)
		memcpy(m_pRenderArea->m_pData1_64, pData1_64, sizeof(double) * (int)m_pRenderArea->m_sizeGraph.width());
	if (m_pRenderArea->m_pData2_64 != nullptr)
		memcpy(m_pRenderArea->m_pData2_64, pData2_64, sizeof(double) * (int)m_pRenderArea->m_sizeGraph.width());

	m_pRenderArea->update();
}


QRenderArea2::QRenderArea2(QWidget *parent) :
	QWidget(parent), m_pData1(nullptr), m_pData2(nullptr), m_pDataX(nullptr), m_pMask(nullptr),
	m_pSelectedRegion(nullptr),	m_pData1_64(nullptr), m_pData2_64(nullptr), m_bSelectionAvailable(false), m_bMaskUse(false), m_b64Use(false),
	m_buff_len(0), m_vLineLen(0), m_nHMajorGrid(8), m_nHMinorGrid(64), m_nVMajorGrid(4), m_bZeroLine(false), m_bScattered(false)
{
    QPalette pal = this->palette();
    pal.setColor(QPalette::Background, QColor(0x282d30));
    setPalette(pal);
    setAutoFillBackground(true);

	m_pVLineInd = new int[10];

	m_selected[0] = -1; m_selected[1] = -1;
}

QRenderArea2::~QRenderArea2()
{
	delete[] m_pVLineInd;
}

void QRenderArea2::setSize(QRange xRange, QRange yRange, int len)
{
	// Set range
	m_xRange = xRange;
	m_yRange = yRange;

	// Set graph size
	m_sizeGraph = { m_xRange.max - m_xRange.min, m_yRange.max - m_yRange.min };

	// Buffer length
	m_buff_len = (len != 0) ? len : m_sizeGraph.width();

	// Allocate data buffer
	if (!m_b64Use)
	{
		if (m_pData1) { delete[] m_pData1; m_pData1 = nullptr; }
		m_pData1 = new float[m_buff_len];
		memset(m_pData1, 0, sizeof(float) * m_buff_len);

		if (m_pData2) { delete[] m_pData2; m_pData2 = nullptr; }
		m_pData2 = new float[m_buff_len];
		memset(m_pData2, 0, sizeof(float) * m_buff_len);

		if (m_pDataX) { delete[] m_pDataX; m_pDataX = nullptr; }
		m_pDataX = new float[m_buff_len];
		memset(m_pDataX, 0, sizeof(float) * m_buff_len);
	}
	else
	{
		if (m_pData1_64) { delete[] m_pData1_64; m_pData1_64 = nullptr; }
		m_pData1_64 = new double[m_buff_len];
		memset(m_pData1_64, 0, sizeof(double) * m_buff_len);

		if (m_pData2_64) { delete[] m_pData2_64; m_pData2_64 = nullptr; }
		m_pData2_64 = new double[m_buff_len];
		memset(m_pData2_64, 0, sizeof(double) * m_buff_len);
	}

	if (m_pSelectedRegion) { delete[] m_pSelectedRegion; m_pSelectedRegion = nullptr; }
	m_pSelectedRegion = new uint8_t[m_buff_len];
	memset(m_pSelectedRegion, 0, sizeof(uint8_t) * m_buff_len);

	if (m_bMaskUse)
	{
		if (m_pMask) { delete[] m_pMask; m_pMask = nullptr; }
		m_pMask = new float[m_buff_len];
		memset(m_pMask, 0, sizeof(float) * m_buff_len);
	}

	this->update();
}

void QRenderArea2::setGrid(int nHMajorGrid, int nHMinorGrid, int nVMajorGrid, bool zeroLine)
{
	m_nHMajorGrid = nHMajorGrid;
	m_nHMinorGrid = nHMinorGrid;
	m_nVMajorGrid = nVMajorGrid;
	m_bZeroLine = zeroLine;
}

void QRenderArea2::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    // Area size
    int w = this->width();
    int h = this->height();

	// Draw grid
	painter.setPen(QColor(0x4f5555)); // Minor grid color (horizontal)
	for (int i = 0; i <= m_nHMinorGrid; i++)
		painter.drawLine(i * w / m_nHMinorGrid, 0, i * w / m_nHMinorGrid, h);

	painter.setPen(QColor(0x7b8585)); // Major grid color (horizontal)
	for (int i = 0; i <= m_nHMajorGrid; i++)
		painter.drawLine(i * w / m_nHMajorGrid, 0, i * w / m_nHMajorGrid, h);
	for (int i = 0; i <= m_nVMajorGrid; i++) // Major grid color (vertical)
		painter.drawLine(0, i * h / m_nVMajorGrid, w, i * h / m_nVMajorGrid);

	if (m_bZeroLine) // zero line
		painter.drawLine(0, (m_yRange.max / (m_yRange.max - m_yRange.min)) * h, w, (m_yRange.max / (m_yRange.max - m_yRange.min)) * h);
	
    // Draw graph   
	if (m_pData2 != nullptr)
	{
		painter.setPen(QColor(0x5dfff2));

		for (int i = 0; i < m_buff_len - 1; i++)
		{
			QPointF x0, x1;
			if (!m_bScattered)
			{
				x0.setX((float)(i) / (float)m_sizeGraph.width() * w);
				x0.setY((float)(m_yRange.max - m_pData2[i]) * (float)h / (float)(m_yRange.max - m_yRange.min));
				x1.setX((float)(i + 1) / (float)m_sizeGraph.width() * w);
				x1.setY((float)(m_yRange.max - m_pData2[i + 1]) * (float)h / (float)(m_yRange.max - m_yRange.min));

				painter.drawLine(x0, x1);
			}
			else
			{
				if (m_pDataX != nullptr)
				{
					x0.setX((float)(m_pDataX[i]) * (float)w / (float)(m_xRange.max - m_xRange.min));
					x0.setY((float)(m_yRange.max - m_pData2[i]) * (float)h / (float)(m_yRange.max - m_yRange.min));

					painter.drawPoint(x0);
				}
			}
		}
	}
	if (m_pData1 != nullptr)
	{
		painter.setPen(QColor(0xfff65d));

		for (int i = 0; i < m_buff_len - 1; i++)
		{
			QPointF x0, x1;
			if (!m_bScattered)
			{
				x0.setX((float)(i) / (float)m_sizeGraph.width() * w);
				x0.setY((float)(m_yRange.max - m_pData1[i]) * (float)h / (float)(m_yRange.max - m_yRange.min));
				x1.setX((float)(i + 1) / (float)m_sizeGraph.width() * w);
				x1.setY((float)(m_yRange.max - m_pData1[i + 1]) * (float)h / (float)(m_yRange.max - m_yRange.min));

				painter.drawLine(x0, x1);
			}
			else
			{
				if (m_pDataX != nullptr)
				{
					x0.setX((float)(m_pDataX[i]) * (float)w / (float)(m_xRange.max - m_xRange.min));
					x0.setY((float)(m_yRange.max - m_pData1[i]) * (float)h / (float)(m_yRange.max - m_yRange.min));

					painter.drawPoint(x0);
				}
			}
		}
	}

	if (m_pData2_64 != nullptr) // double case 2
	{
		painter.setPen(QColor(0x5dfff2));
		for (int i = 0; i < m_buff_len - 1; i++)
		{
			QPointF x0, x1;
			x0.setX((double)(i) / (double)m_sizeGraph.width() * w);
			x0.setY((double)(m_yRange.max - m_pData2_64[i]) * (double)h / (double)(m_yRange.max - m_yRange.min));
			x1.setX((double)(i + 1) / (double)m_sizeGraph.width() * w);
			x1.setY((double)(m_yRange.max - m_pData2_64[i + 1]) * (double)h / (double)(m_yRange.max - m_yRange.min));

			painter.drawLine(x0, x1);
		}
	}
	if (m_pData1_64 != nullptr) // double case 1
	{
		painter.setPen(QColor(0xfff65d));
		for (int i = 0; i < m_buff_len - 1; i++)
		{
			QPointF x0, x1;
			x0.setX((double)(i) / (double)m_sizeGraph.width() * w);
			x0.setY((double)(m_yRange.max - m_pData1_64[i]) * (double)h / (double)(m_yRange.max - m_yRange.min));
			x1.setX((double)(i + 1) / (double)m_sizeGraph.width() * w);
			x1.setY((double)(m_yRange.max - m_pData1_64[i + 1]) * (double)h / (double)(m_yRange.max - m_yRange.min));

			painter.drawLine(x0, x1);
		}
	}

	if (m_bMaskUse && (m_pMask != nullptr))
	{
		{
			QPen pen(QColor(0x00ff00)); // mask region (green)
			pen.setWidth(2);
			painter.setPen(pen);
			for (int i = 0; i < m_buff_len - 1; i++)
			{
				if (m_pMask[i] == 0)
				{
					QPointF x0, x1;
					if (!m_bScattered)
					{
						x0.setX((float)(i) / (float)m_sizeGraph.width() * w);
						x0.setY((float)(m_yRange.max - m_pData2[i]) * (float)h / (float)(m_yRange.max - m_yRange.min));
						x1.setX((float)(i + 1) / (float)m_sizeGraph.width() * w);
						x1.setY((float)(m_yRange.max - m_pData2[i + 1]) * (float)h / (float)(m_yRange.max - m_yRange.min));

						painter.drawLine(x0, x1);
					}
					else
					{
						if (m_pDataX != nullptr)
						{
							x0.setX((float)(m_pDataX[i]) * (float)w / (float)(m_xRange.max - m_xRange.min));
							x0.setY((float)(m_yRange.max - m_pData2[i]) * (float)h / (float)(m_yRange.max - m_yRange.min));

							painter.drawPoint(x0);
						}
					}
				}
			}
		}
		{
			QPen pen(QColor(0xff0080)); // mask region (hot pink)
			pen.setWidth(2);
			painter.setPen(pen);
			for (int i = 0; i < m_buff_len - 1; i++)
			{
				if (m_pMask[i] == 0)
				{
					QPointF x0, x1;
					if (!m_bScattered)
					{
						x0.setX((float)(i) / (float)m_sizeGraph.width() * w);
						x0.setY((float)(m_yRange.max - m_pData1[i]) * (float)h / (float)(m_yRange.max - m_yRange.min));
						x1.setX((float)(i + 1) / (float)m_sizeGraph.width() * w);
						x1.setY((float)(m_yRange.max - m_pData1[i + 1]) * (float)h / (float)(m_yRange.max - m_yRange.min));

						painter.drawLine(x0, x1);
					}
					else
					{
						if (m_pDataX != nullptr)
						{
							x0.setX((float)(m_pDataX[i]) * (float)w / (float)(m_xRange.max - m_xRange.min));
							x0.setY((float)(m_yRange.max - m_pData1[i]) * (float)h / (float)(m_yRange.max - m_yRange.min));

							painter.drawPoint(x0);
						}
					}
				}
			}
		}
	}
	if (m_bSelectionAvailable)
	{
		{
			QPen pen(QColor(0xff00ff)); pen.setWidth(3); // selected region (magenta)					
			painter.setPen(pen);
			for (int i = 0; i < m_buff_len - 1; i++)
			{
				if (m_pSelectedRegion[i] == 1)
				{
					QPointF x0, x1;
					x0.setX((float)(i) / (float)m_sizeGraph.width() * w);
					x0.setY((float)(m_yRange.max - m_pData2[i]) * (float)h / (float)(m_yRange.max - m_yRange.min));
					x1.setX((float)(i + 1) / (float)m_sizeGraph.width() * w);
					x1.setY((float)(m_yRange.max - m_pData2[i + 1]) * (float)h / (float)(m_yRange.max - m_yRange.min));

					painter.drawLine(x0, x1);
				}
			}
		}

		{
			QPen pen(QColor(0xff6666)); pen.setWidth(3); // selected region (light pink)					
			painter.setPen(pen);
			for (int i = 0; i < m_buff_len - 1; i++)
			{
				if (m_pSelectedRegion[i] == 1)
				{
					QPointF x0, x1;
					x0.setX((float)(i) / (float)m_sizeGraph.width() * w);
					x0.setY((float)(m_yRange.max - m_pData1[i]) * (float)h / (float)(m_yRange.max - m_yRange.min));
					x1.setX((float)(i + 1) / (float)m_sizeGraph.width() * w);
					x1.setY((float)(m_yRange.max - m_pData1[i + 1]) * (float)h / (float)(m_yRange.max - m_yRange.min));

					painter.drawLine(x0, x1);
				}
			}
		}
	}

	// Draw vertical lines
	for (int i = 0; i < m_vLineLen; i++)
	{
		QPointF x0, x1;
		x0.setX((float)(m_pVLineInd[i]) / (float)m_sizeGraph.width() * w);
		x0.setY((float)0);
		x1.setX((float)(m_pVLineInd[i]) / (float)m_sizeGraph.width() * w);
		x1.setY((float)h);

		painter.setPen(QColor(0xff0000));
		painter.drawLine(x0, x1);
	}
}

void QRenderArea2::mousePressEvent(QMouseEvent *e)
{
	if (m_bSelectionAvailable)
	{
		m_bMousePressed = true;

		QPoint p = e->pos();
		auto button = e->button();
		if ((p.x() > 0) && (p.y() > 0) && (p.x() < this->width()) && (p.y() < this->height()))
		{
			int x = (double)p.x() / (double)this->width() * (double)m_sizeGraph.width();

			m_selected[0] = x;
			m_selected[1] = x;

			m_bIsLeftButton = button == Qt::LeftButton;
			if (m_bIsLeftButton)
				memset(m_pSelectedRegion, 0, sizeof(uint8_t) * (int)m_sizeGraph.width());

			memset(&m_pSelectedRegion[x], 1, 1);

			//printf("[%d %d]\n", m_selected[0], m_selected[1]);
			//this->update();
			DidMouseEvent();
		}
	}
}

void QRenderArea2::mouseMoveEvent(QMouseEvent *e)
{
	if (m_bSelectionAvailable)
	{
		if (m_bMousePressed)
		{
			QPoint p = e->pos();
			auto button = e->button();
			if ((p.x() > 0) && (p.y() > 0) && (p.x() < this->width()) && (p.y() < this->height()))
			{
				int x = (double)p.x() / (double)this->width() * (double)m_sizeGraph.width();
				m_selected[1] = x;

				// ordered
				m_start = m_selected[0], m_end = m_selected[1];
				if (m_selected[0] > m_selected[1])
				{
					m_start = m_selected[1];
					m_end = m_selected[0];
				}

				if (m_bIsLeftButton)
					memset(m_pSelectedRegion, 0, sizeof(uint8_t) * (int)m_sizeGraph.width());
				memset(&m_pSelectedRegion[m_start], 1, sizeof(uint8_t) * (m_end - m_start));

				//printf("[%d %d]\n", m_start, m_end);
				//this->update();
				DidMouseEvent();
			}
		}
	}
}

void QRenderArea2::mouseReleaseEvent(QMouseEvent *e)
{
	if (m_bSelectionAvailable)
	{
		m_bMousePressed = false;

		QPoint p = e->pos();
		auto button = e->button();
		if ((p.x() > 0) && (p.y() > 0) && (p.x() < this->width()) && (p.y() < this->height()))
		{
			int x = (double)p.x() / (double)this->width() * (double)m_sizeGraph.width();
			m_selected[1] = x;

			// ordered
			m_start = m_selected[0], m_end = m_selected[1];
			if (m_selected[0] > m_selected[1])
			{
				m_start = m_selected[1];
				m_end = m_selected[0];
			}

			if (m_bIsLeftButton)
				memset(m_pSelectedRegion, 0, sizeof(uint8_t) * (int)m_sizeGraph.width());
			memset(&m_pSelectedRegion[m_start], 1, sizeof(uint8_t) * (m_end - m_start));

			//printf("[%d %d]\n", m_selected[0], m_selected[1]);
			//this->update();
			DidMouseEvent();
		}
	}
}
