#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtWidgets>
#include <QtCore>

class Configuration;

class QOperationTab;
class QDeviceControlTab;
class QStreamTab;
class QResultTab;

namespace Ui {
class MainWindow;
}


class MainWindow : public QMainWindow
{
    Q_OBJECT

// Constructer & Destructer /////////////////////////////
public:
    explicit MainWindow(QWidget *parent = 0);
    virtual ~MainWindow();
	
// Methods //////////////////////////////////////////////
protected:
	void closeEvent(QCloseEvent *e);

private slots:
	void onTimer();
	void changedTab(int);

// Variables ////////////////////////////////////////////
public:
	Configuration* m_pConfiguration;

private:
	QTimer *m_pTimer;
		
private:
    Ui::MainWindow *ui;

    // Layout
    QGridLayout *m_pGridLayout;

    // Left tabs
    QGroupBox *m_pGroupBox_OperationTab;
    QGroupBox *m_pGroupBox_DeviceControlTab;

public:
    // Right tabs
    QTabWidget *m_pTabWidget;

public:
    // Tab objects
    QOperationTab *m_pOperationTab;
    QDeviceControlTab *m_pDeviceControlTab;
    QStreamTab *m_pStreamTab;
    QResultTab *m_pResultTab;	

public:
	// Status bar
	QLabel *m_pStatusLabel_ImagePos;
};

#endif // MAINWINDOW_H
