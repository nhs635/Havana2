#ifndef PULSEREVIEWDLG_H
#define PULSEREVIEWDLG_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>
#include <Havana2/Viewer/QScope.h>

#include <Common/array.h>
#include <Common/callback.h>

class MainWindow;
class QResultTab;
class FLIMProcess;

class PulseReviewDlg : public QDialog
{
    Q_OBJECT

#ifdef OCT_FLIM
// Constructer & Destructer /////////////////////////////
public:
    explicit PulseReviewDlg(QWidget *parent = 0);
    virtual ~PulseReviewDlg();

// Methods //////////////////////////////////////////////
private:
	void keyPressEvent(QKeyEvent *e);

public:
	inline int getCurrentAline() const { return m_pSlider_CurrentAline->value(); }
	inline void setCurrentAline(int aline) { m_pSlider_CurrentAline->setValue(aline); }
	
public slots : // widgets
	void drawPulse(int);
	void changeType();

signals:
	//void plotRoiPulse(FLIMProcess*, int);

// Variables ////////////////////////////////////////////
private:	
    MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
	QResultTab* m_pResultTab;
#ifdef OCT_FLIM
	FLIMProcess* m_pFLIM;
#endif
	

private:
	// Widgets for pulse view
	QScope *m_pScope_PulseView;
	
	QLabel *m_pLabel_CurrentAline;
	QSlider *m_pSlider_CurrentAline;
	QLabel *m_pLabel_PulseType;
	QComboBox *m_pComboBox_PulseType;
#endif
};

#endif // PULSEREVIEWDLG_H
