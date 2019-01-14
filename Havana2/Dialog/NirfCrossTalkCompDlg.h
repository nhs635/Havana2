#ifndef NIRFCROSSTALKCOMPDLG_H
#define NIRFCROSSTALKCOMPDLG_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>
#include <Havana2/Viewer/QScope.h>

#include <Common/array.h>
#include <Common/callback.h>

class MainWindow;
class QResultTab;


class NirfCrossTalkCompDlg : public QDialog
{
    Q_OBJECT

#ifdef OCT_NIRF
#ifdef TWO_CHANNEL_NIRF
// Constructer & Destructer /////////////////////////////
public:
    explicit NirfCrossTalkCompDlg(QWidget *parent = 0);
    virtual ~NirfCrossTalkCompDlg();

// Methods //////////////////////////////////////////////
private:
	void keyPressEvent(QKeyEvent *e);    

// Variables ////////////////////////////////////////////
private:	
    MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
    QResultTab* m_pResultTab;
	
private:
	// Layout
	QVBoxLayout *m_pVBoxLayout;
#endif
#endif
};

#endif // NIRFCROSSTALKCOMPDLG_H
