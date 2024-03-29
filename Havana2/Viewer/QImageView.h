#ifndef QIMAGEVIEW_H
#define QIMAGEVIEW_H

#include <Havana2/Configuration.h>

#include <QDialog>
#include <QtCore>
#include <QtWidgets>

#include <stdarg.h>

#include <Common/array.h>
#include <Common/callback.h>
using ColorTableVector = QVector<QVector<QRgb>>;

class ColorTable
{
public:
	explicit ColorTable();

public:
	enum colortable { gray = 0, inv_gray, sepia, jet, parula, hot, fire, hsv, 
		smart, blueorange, cool, gem, greenfireblue, ice, lifetime2, vessel, hsv1, magenta, blue_hot, clf, bwr }; // 새로 만든 colortable 이름 추가하기
	QVector<QString> m_cNameVector;
	ColorTableVector m_colorTableVector;
};
class QRenderImage;


class QImageView : public QDialog
{
	Q_OBJECT

private:
    explicit QImageView(QWidget *parent = 0); // Disabling default constructor

public:
    explicit QImageView(ColorTable::colortable ctable, int width, int height, float gamma = 1.0f, bool rgb = false, QWidget *parent = 0);
    virtual ~QImageView();

public:
	inline QRenderImage* getRender() { return m_pRenderImage; }

protected:
    void resizeEvent(QResizeEvent *);

public:
	void resetSize(int width, int height);
    void resetColormap(ColorTable::colortable ctable, float gamma = 1.0f);
	void setSquare(bool square) { m_bSquareConstraint = square; }
#ifdef OCT_FLIM
    void setRgbEnable(bool rgb) { m_bRgbUsed = rgb; }
#endif
	
public:
	void setHorizontalLine(int len, ...);
	void setVerticalLine(int len, ...);
	void setCircle(int len, ...);
	void setHorizontalLineColor(int len, ...);
	void setContour(int len, uint16_t* pContour, uint8_t* pSelected = nullptr);
	void setMagnDefault();

	void setHLineChangeCallback(const std::function<void(int)> &slot);
	void setVLineChangeCallback(const std::function<void(int)> &slot);
	void setRLineChangeCallback(const std::function<void(int)> &slot);
	void setMovedMouseCallback(const std::function<void(QPoint&)> &slot);
	void setDoubleClickedMouseCallback(const std::function<void(void)> &slot);
	void setReleasedMouseCallback(const std::function<void(QRect)> &slot);
	void setWheelMouseCallback(const std::function<void(void)> &slot);

public slots:
	void drawImage(uint8_t* pImage);
	void drawRgbImage(uint8_t* pImage);
	void showContextMenu(const QPoint &);
	
private:
    QHBoxLayout *m_pHBoxLayout;

	ColorTable m_colorTable;
    QRenderImage *m_pRenderImage;

private:
    int m_width;
    int m_width4;
    int m_height;

	bool m_bSquareConstraint;
    bool m_bRgbUsed;
};


class QRenderImage : public QWidget
{
    Q_OBJECT

public:
    explicit QRenderImage(QWidget *parent = 0);
	virtual ~QRenderImage();

protected:
    void paintEvent(QPaintEvent *);
	void mousePressEvent(QMouseEvent *);
	void mouseDoubleClickEvent(QMouseEvent *);
	void mouseMoveEvent(QMouseEvent *);
	void mouseReleaseEvent(QMouseEvent *);
	void wheelEvent(QWheelEvent *);

public:
    QImage *m_pImage;

    int *m_pHLineInd, *m_pVLineInd;
    int m_hLineLen, m_vLineLen;
    int m_circLen;
	bool m_bRadial;
	bool m_bDiametric;
	bool m_bCanBeMagnified;
	QRect m_rectMagnified;
	float m_fMagnLevel;
	int m_rMax;
	QColor m_colorLine;
	QColor* m_pColorHLine;

	int m_contour_offset;
    np::Uint16Array m_contour;
	uint8_t* m_pSelected;

	bool m_bRectDrawing;
	bool m_bMeasureDistance;
	int m_nClicked;
	int m_point[2][2];

	callback<int> DidChangedHLine;
	callback<int> DidChangedVLine;
	callback<int> DidChangedRLine;
	callback<void> DidDoubleClickedMouse;
	callback<QPoint&> DidMovedMouse;
	callback<QRect> DidReleasedMouse;
	callback<void> DidWheelMouse;
};


#endif // QIMAGEVIEW_H
