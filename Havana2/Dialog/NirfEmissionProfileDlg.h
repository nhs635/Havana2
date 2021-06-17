#ifndef NIRFEMISSIONPROFILEDLG_H
#define NIRFEMISSIONPROFILEDLG_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>
#ifndef TWO_CHANNEL_NIRF
#include <Havana2/Viewer/QScope.h>
#else
#include <Havana2/Viewer/QScope2.h>
#endif

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
#ifndef TWO_CHANNEL_NIRF
    inline QScope* getScope() const { return m_pScope; }
#else
	inline QScope2* getScope() const { return m_pScope; }
#endif

public slots:
#ifndef TWO_CHANNEL_NIRF
	void drawData(void* data);
#else
	void drawData(void* data1, void* float2);
#endif

// Variables ////////////////////////////////////////////
private:	
    MainWindow* m_pMainWnd;
    Configuration* m_pConfig;
    QStreamTab* m_pStreamTab;
    QResultTab* m_pResultTab;

private:
    bool m_bIsStreaming;
#ifndef TWO_CHANNEL_NIRF
    QScope *m_pScope;
#else
	QScope2 *m_pScope;
#endif
#endif
};

#endif // NIRFEMISSIONPROFILEDLG_H
