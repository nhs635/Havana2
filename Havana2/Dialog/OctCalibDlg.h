#ifndef OCTCALIBDLG_H
#define OCTCALIBDLG_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>
#include <Havana2/Viewer/QCalibScope.h>

#include <Common/array.h>
#include <Common/callback.h>

class MainWindow;
class QStreamTab;
class OCTProcess;


class OctCalibDlg : public QDialog
{
    Q_OBJECT

// Constructer & Destructer /////////////////////////////
public:
    explicit OctCalibDlg(QWidget *parent = 0);
    virtual ~OctCalibDlg();

// Methods //////////////////////////////////////////////
private:
	void closeEvent(QCloseEvent *e);
	void keyPressEvent(QKeyEvent *e);

private:
	void setOctCallback();

public slots:
	void captureBackground();
	void captureD1();
	void captureD2();
	void generateCalibration();
	void proceed();

#ifdef OCT_FLIM
	void caughtBackground(uint16_t* fringe);
	void caughtD1(uint16_t* fringe);
	void caughtD2(uint16_t* fringe);
#elif defined (STANDALONE_OCT)
	void caughtBackground(uint16_t* fringe1, uint16_t* fringe2);
	void caughtD1(uint16_t* fringe, uint16_t* fringe2);
	void caughtD2(uint16_t* fringe, uint16_t* fringe2);
#endif
	void setDiscomValue(const QString &str);

private slots:
	void enableGenerateCalibPushButton(bool);
	void enableProceedPushButton(bool);
	void setWidgetsEndCalibration(void);

signals:
#ifdef OCT_FLIM
	void catchFringe(uint16_t*);
#elif defined (STANDALONE_OCT)
	void catchFringe(uint16_t*, uint16_t*);
#endif
	void setGenerateCalibPushButton(bool);
	void setProceedPushButton(bool);
	void endCalibration(void);

// Variables ////////////////////////////////////////////
private:	
    MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
    QStreamTab* m_pStreamTab;
#ifdef OCT_FLIM
	OCTProcess* m_pOCT;
#elif defined (STANDALONE_OCT)
	OCTProcess* m_pOCT1;
	OCTProcess* m_pOCT2;
#endif

private:
	bool m_bBeingCalibrated;
	bool m_bProceed;

private:
    QPushButton *m_pPushButton_CaptureBackground;
    QPushButton *m_pPushButton_CaptureD1;
    QPushButton *m_pPushButton_CaptureD2;
    QPushButton *m_pPushButton_GenerateCalibration;

    QLabel *m_pLabel_CaptureBackground;
    QLabel *m_pLabel_CaptureD1;
    QLabel *m_pLabel_CaptureD2;
    QLabel *m_pLabel_GenerateCalibration;

	QLabel *m_pLabel_DiscomValue;
	QLineEdit *m_pLineEdit_DiscomValue;

	QLabel *m_pLabel_Title;
	QPushButton *m_pPushButton_Proceed;
	
public:
	QCalibScope *m_pScope;
};


#endif // OCTCALIBDLG_H
