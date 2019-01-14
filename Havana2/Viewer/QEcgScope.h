#ifndef QECGSCOPE_H
#define QECGSCOPE_H

#include <QDialog>
#include <QtCore>
#include <QtWidgets>

#include "QScope.h"

#include <iostream>
#include "Common/Array.h"

class QRenderAreaEcg;

class QEcgScope : public QDialog
{
    Q_OBJECT

private:
    explicit QEcgScope(QWidget *parent = 0);

public:
    explicit QEcgScope(QRange x_range, QRange y_range,
                    int num_x_ticks = 2, int num_y_ticks = 2,
                    double x_interval = 1, double y_interval = 1, double x_offset = 0, double y_offset = 0,
                    QString x_unit = "", QString y_unit = "", QWidget *parent = 0);
	virtual ~QEcgScope();

public:
	void setAxis(QRange x_range, QRange y_range,
                 int num_x_ticks = 2, int num_y_ticks = 2,
                 double x_interval = 1, double y_interval = 1, double x_offset = 0, double y_offset = 0,
                 QString x_unit = "", QString y_unit = "");
	void clearDeque();

public slots:
    void drawData(double data, bool is_peak = false);

private:
    QGridLayout *m_pGridLayout;

	QRenderAreaEcg *m_pRenderArea;

    QVector<QLabel *> m_pLabelVector_x;
    QVector<QLabel *> m_pLabelVector_y;
};

class QRenderAreaEcg : public QWidget
{
    Q_OBJECT

public:
    explicit QRenderAreaEcg(QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *);

public:
	np::FloatArray m_dqData;
	np::FloatArray m_dqIsPeak;

    QRange m_xRange;
    QRange m_yRange;
    QSizeF m_sizeGraph;
};


#endif // QECGSCOPE_H
