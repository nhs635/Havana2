#ifndef LONGITUDINALVIEWDLG_H
#define LONGITUDINALVIEWDLG_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>
#include <Havana2/Viewer/QImageView.h>

#include <Common/medfilt.h>
#include <Common/ImageObject.h>

class MainWindow;
class QResultTab;


class LongitudinalViewDlg : public QDialog
{
    Q_OBJECT

// Constructer & Destructer /////////////////////////////
public:
    explicit LongitudinalViewDlg(QWidget *parent = 0);
    virtual ~LongitudinalViewDlg();

// Methods //////////////////////////////////////////////
private:
	void keyPressEvent(QKeyEvent *e);    

public:
	inline QImageView* getImageView() const { return m_pImageView_LongitudinalView; }
	inline int getCurrentAline() const { return m_pSlider_CurrentAline->value(); }
	inline void setCurrentAline(int aline) { m_pSlider_CurrentAline->setValue(aline); }

public:
	void setWidgets(bool);

public slots : // widgets
	void drawLongitudinalImage(int);

signals:
	void paintLongiImage(uint8_t*);

// Variables ////////////////////////////////////////////
private:	
    MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
    QResultTab* m_pResultTab;

public:
	ImageObject *m_pImgObjOctLongiImage; 
#ifdef OCT_FLIM
	ImageObject *m_pImgObjIntensity;
	ImageObject *m_pImgObjLifetime;
	ImageObject *m_pImgObjHsvEnhanced;
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
	ImageObject *m_pImgObjNirf;
#else
	ImageObject *m_pImgObjNirf1;
	ImageObject *m_pImgObjNirf2;
#endif
#endif
	
private:
	medfilt* m_pMedfilt;
	
private:
	// Widgets for pulse view
	QImageView *m_pImageView_LongitudinalView;
	QLabel *m_pLabel_CurrentAline;
	QSlider *m_pSlider_CurrentAline;
};

#endif // LONGITUDINALVIEWDLG_H
