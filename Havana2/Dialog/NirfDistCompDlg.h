#ifndef NIRFDISTCOMP_H
#define NIRFDISTCOMP_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>
#include <Havana2/Viewer/QScope.h>

#include <Common/array.h>
#include <Common/callback.h>

class MainWindow;
class QResultTab;


class NirfDistCompDlg : public QDialog
{
    Q_OBJECT

#ifdef OCT_NIRF
// Constructer & Destructer /////////////////////////////
public:
    explicit NirfDistCompDlg(QWidget *parent = 0);
    virtual ~NirfDistCompDlg();

// Methods //////////////////////////////////////////////
private:
	void closeEvent(QCloseEvent *e);
	void keyPressEvent(QKeyEvent *e);    

public:
	inline void setClosed(bool closed) { m_bCanBeClosed = closed; }
    inline bool isCompensating() const { return m_pToggleButton_Compensation->isChecked(); }
    inline bool isTBRMode() const { return m_pToggleButton_TBRMode->isChecked(); }
    inline bool isFiltered() const { return m_pCheckBox_Filtering->isChecked(); }
	
public slots:
	void setWidgetEnabled(bool enabled);

private slots : 
	void loadDistanceMap();
	void loadNirfBackground();
    void compensation(bool);
	void tbrConvering(bool);
	void changeCompensationCurve();
    void changeZeroPointSetting();
    void filtering(bool);
	void changeNirfBackground(const QString &);
    void changeTbrBackground(const QString &);

public:
	void getCompInfo(const QString &);
	void setCompInfo(const QString &);
	
signals:
	void setWidgets(bool);

private:
    void calculateCompMap();

// Variables ////////////////////////////////////////////
private:	
    MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
    QResultTab* m_pResultTab;

private:
	bool m_bCanBeClosed;
	
private:
	np::FloatArray distDecayCurve;
	np::FloatArray compCurve;

public:
    np::Uint16Array2 distMap;
    float nirfBg;
    np::FloatArray2 compMap;
    float tbrBg;

private:
	// Layout
	QVBoxLayout *m_pVBoxLayout;

	// Widgets for data loading & compensation
	QPushButton *m_pPushButton_LoadDistanceMap;
	QPushButton *m_pPushButton_LoadNirfBackground;

	QPushButton *m_pToggleButton_Compensation;
	QPushButton *m_pToggleButton_TBRMode;

	// Widgets for compensation details
	QLabel *m_pLabel_DistanceDecayCurve;
	QLabel *m_pLabel_CompensationCurve;

	QRenderArea *m_pRenderArea_DistanceDecayCurve;
	QRenderArea *m_pRenderArea_CompensationCurve;
	
	QLabel *m_pLabel_CompensationCoeff;
	QLineEdit *m_pLineEdit_CompensationCoeff[4];

	QLabel *m_pLabel_FactorThreshold;
	QLineEdit *m_pLineEdit_FactorThreshold;

	QLabel *m_pLabel_FactorPropConst;
	QLineEdit *m_pLineEdit_FactorPropConst;

	QLabel *m_pLabel_DistPropConst;
	QLineEdit *m_pLineEdit_DistPropConst;

    // Widgets for zero point setting
	QLabel *m_pLabel_LumenContourOffset;
	QSpinBox *m_pSpinBox_LumenContourOffset;

	QLabel *m_pLabel_OuterSheathPosition;
	QSpinBox *m_pSpinBox_OuterSheathPosition;

    // Widgets for TBR mode
    QCheckBox *m_pCheckBox_Filtering;

	QLabel *m_pLabel_NIRF_Background;
	QLineEdit *m_pLineEdit_NIRF_Background;

    QLabel *m_pLabel_TBR_Background;
    QLineEdit *m_pLineEdit_TBR_Background;
#endif
};

#endif // NirfDistCompDlg_H
