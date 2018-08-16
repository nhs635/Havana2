#ifndef NIRFEMISSIONPROFILEDLG_H
#define NIRFEMISSIONPROFILEDLG_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>
#include <Havana2/Viewer/QScope.h>

#include <Common/array.h>
#include <Common/callback.h>

class MainWindow;
class QStreamTab;
class QResultTab;


class NirfEmissionProfileDlg : public QDialog
{
    Q_OBJECT

#ifdef OCT_NIRF
// Constructer & Destructer /////////////////////////////
public:
    explicit NirfEmissionProfileDlg(bool _isStreaming, QWidget *parent = 0);
    virtual ~NirfEmissionProfileDlg();

// Methods //////////////////////////////////////////////
private:
    void keyPressEvent(QKeyEvent *e);

public:
    inline QScope* getScope() const { return m_pScope; }

// Variables ////////////////////////////////////////////
private:	
    MainWindow* m_pMainWnd;
    Configuration* m_pConfig;
    QStreamTab* m_pStreamTab;
    QResultTab* m_pResultTab;

private:
    bool m_bIsStreaming;
    QScope *m_pScope;
#endif
};

#endif // NIRFEMISSIONPROFILEDLG_H
