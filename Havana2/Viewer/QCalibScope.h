#ifndef QCALIBSCOPE_H
#define QCALIBSCOPE_H

#include <QDialog>
#include <QtCore>
#include <QtWidgets>

#include "QScope.h"

class QRenderAreaCalib;

class QCalibScope : public QDialog
{
    Q_OBJECT

private:
    explicit QCalibScope(QWidget *parent = 0);

public:
    explicit QCalibScope(QRange x_range, QRange y_range,
                    int num_x_ticks = 2, int num_y_ticks = 2,
                    double x_interval = 1, double y_interval = 1, double x_offset = 0, double y_offset = 0,
                    QString x_unit = "", QString y_unit = "", QWidget *parent = 0);
	virtual ~QCalibScope();

public:
	void setAxis(QRange x_range, QRange y_range,
                 int num_x_ticks = 2, int num_y_ticks = 2,
                 double x_interval = 1, double y_interval = 1, double x_offset = 0, double y_offset = 0,
                 QString x_unit = "", QString y_unit = "");
    void resetAxis(QRange x_range, QRange y_range,
                   double x_interval = 1, double y_interval = 1, double x_offset = 0, double y_offset = 0,
                   QString x_unit = "", QString y_unit = "");

public slots:
    void drawData(float* pData);
	
public:
	QRenderAreaCalib *m_pRenderArea;

private:
    QGridLayout *m_pGridLayout;
	
    QVector<QLabel *> m_pLabelVector_x;
    QVector<QLabel *> m_pLabelVector_y;
};


class QRenderAreaCalib : public QWidget
{
    Q_OBJECT

public:
    explicit QRenderAreaCalib(QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *);

	void mousePressEvent(QMouseEvent *);
	void mouseMoveEvent(QMouseEvent *);
	void mouseReleaseEvent(QMouseEvent *);
	
public:
    float* m_pData;

    QRange m_xRange;
    QRange m_yRange;
    QSizeF m_sizeGraph;

	bool m_bSelectionAvailable;
	bool m_bMousePressed;
	int m_selected[2], m_selected1[2];
};


#endif // QSCOPE_H
