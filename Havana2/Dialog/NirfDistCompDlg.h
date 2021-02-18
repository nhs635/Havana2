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
	inline bool isZeroTbr() const { return m_pCheckBox_ZeroTBRDefinition->isChecked(); }
	inline bool isShowLumContour() const { return m_pCheckBox_ShowLumenContour->isChecked(); }
	inline bool isGwMasked() const { return m_pCheckBox_GwMasking->isChecked(); }
	
public slots:
	void setWidgetEnabled(bool enabled);

private slots: 
	void loadDistanceMap();
	void loadNirfBackground();
    void compensation(bool);
	void tbrConvering(bool);
	void exportTbrData();
	void changeCompensationCurve();
    void changeZeroPointSetting();
    void filtering(bool);
	void gwMasking(bool);
	void tbrZeroDefinition(bool);
	void changeNirfBackground(const QString &);
    void changeTbrBackground(const QString &);
	void changeCompConstant(const QString &);
	void showLumenContour(bool);
	void showCorrelationPlot(bool);

public:
	void getCompInfo(const QString &);
	void setCompInfo(const QString &);
	
signals:
	void setWidgets(bool);

private:
    void calculateCompMap();

public:
	void updateCorrelation(int frame);

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
	np::Uint16Array2 distOffsetMap;
	np::FloatArray2 gwMap;
    float nirfBg;
    np::FloatArray2 compMap;
    float tbrBg;
	float compConst;

private:
	// Layout
	QVBoxLayout *m_pVBoxLayout;

	// Widgets for data loading & compensation
	QPushButton *m_pPushButton_LoadDistanceMap;
	QPushButton *m_pPushButton_LoadNirfBackground;

	QPushButton *m_pToggleButton_Compensation;
	QPushButton *m_pToggleButton_TBRMode;
	QPushButton *m_pPushButton_ExportTBR;

	// Widgets for compensation details
	QLabel *m_pLabel_DistanceDecayCurve;
	QLabel *m_pLabel_CompensationCurve;

	QRenderArea *m_pRenderArea_DistanceDecayCurve;
	QRenderArea *m_pRenderArea_CompensationCurve;
	
	QLabel *m_pLabel_CompensationCoeff;
	QLineEdit *m_pLineEdit_CompensationCoeff_a;
	QLineEdit *m_pLineEdit_CompensationCoeff_b;
	QLineEdit *m_pLineEdit_CompensationCoeff_c;
	QLineEdit *m_pLineEdit_CompensationCoeff_d;

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
	QCheckBox *m_pCheckBox_ZeroTBRDefinition;
	QCheckBox *m_pCheckBox_GwMasking;

	QLabel *m_pLabel_NIRF_Background;
	QLineEdit *m_pLineEdit_NIRF_Background;

    QLabel *m_pLabel_TBR_Background;
    QLineEdit *m_pLineEdit_TBR_Background;

	QLabel *m_pLabel_Compensation;
	QLineEdit *m_pLineEdit_Compensation;

	// Widgets for indicator
	QCheckBox *m_pCheckBox_ShowLumenContour;

	// Widgets for compensation results - distance dependency check
	QCheckBox *m_pCheckBox_Correlation;
	QRenderArea *m_pRenderArea_Correlation;
#endif
};

#endif // NirfDistCompDlg_H
