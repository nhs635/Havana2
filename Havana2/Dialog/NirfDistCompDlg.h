#ifndef NIRFDISTCOMP_H
#define NIRFDISTCOMP_H

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
#ifndef TWO_CHANNEL_NIRF
	void changeNirfBackground(const QString &);
    void changeTbrBackground(const QString &);
#else
	void changeNirfBackground1(const QString &);
	void changeNirfBackground2(const QString &);
	void changeTbrBackground1(const QString &);
	void changeTbrBackground2(const QString &);
#endif
	void changeCompConstant(const QString &);
#ifdef TWO_CHANNEL_NIRF
	void changeCrossTalkRatio(const QString &);
#endif
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
#ifndef TWO_CHANNEL_NIRF
	np::FloatArray distDecayCurve;
	np::FloatArray compCurve;
#else
	np::FloatArray distDecayCurve[2]; 
	np::FloatArray compCurve[2];
#endif

public:
    np::Uint16Array2 distMap;
	np::Uint16Array2 distOffsetMap;
	np::FloatArray2 gwMap;
#ifndef TWO_CHANNEL_NIRF
    float nirfBg;
    np::FloatArray2 compMap;
    float tbrBg;
#else
	float nirfBg[2];
	np::FloatArray2 compMap[2];
	float tbrBg[2];
	float crossTalkRatio;
#endif
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

#ifndef TWO_CHANNEL_NIRF
	QRenderArea *m_pRenderArea_DistanceDecayCurve;
	QRenderArea *m_pRenderArea_CompensationCurve;
#else
	QRenderArea2 *m_pRenderArea_DistanceDecayCurve;
	QRenderArea2 *m_pRenderArea_CompensationCurve;
#endif
	
	QLabel *m_pLabel_CompensationCoeff;
#ifndef TWO_CHANNEL_NIRF
	QLineEdit *m_pLineEdit_CompensationCoeff_a;
	QLineEdit *m_pLineEdit_CompensationCoeff_b;
	QLineEdit *m_pLineEdit_CompensationCoeff_c;
	QLineEdit *m_pLineEdit_CompensationCoeff_d;
#else
	QLineEdit *m_pLineEdit_CompensationCoeff_a[2];
	QLineEdit *m_pLineEdit_CompensationCoeff_b[2];
	QLineEdit *m_pLineEdit_CompensationCoeff_c[2];
	QLineEdit *m_pLineEdit_CompensationCoeff_d[2];
#endif

	QLabel *m_pLabel_FactorThreshold;
	QLineEdit *m_pLineEdit_FactorThreshold;

	QLabel *m_pLabel_FactorPropConst;
	QLineEdit *m_pLineEdit_FactorPropConst;

	QLabel *m_pLabel_DistPropConst;
	QLineEdit *m_pLineEdit_DistPropConst;
	
#ifdef TWO_CHANNEL_NIRF
	// Widgets for cross talk correction
	QLabel *m_pLabel_CrossTalkRatio;
	QLineEdit *m_pLineEdit_CrossTalkRatio;
#endif

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
#ifndef TWO_CHANNEL_NIRF
	QLineEdit *m_pLineEdit_NIRF_Background;
#else
	QLineEdit *m_pLineEdit_NIRF_Background[2];
#endif

    QLabel *m_pLabel_TBR_Background;
#ifndef TWO_CHANNEL_NIRF
    QLineEdit *m_pLineEdit_TBR_Background;
#else
	QLineEdit *m_pLineEdit_TBR_Background[2];
#endif

	QLabel *m_pLabel_Compensation;
	QLineEdit *m_pLineEdit_Compensation;

	// Widgets for indicator
	QCheckBox *m_pCheckBox_ShowLumenContour;

	// Widgets for compensation results - distance dependency check
	QCheckBox *m_pCheckBox_Correlation;
#ifndef TWO_CHANNEL_NIRF
	QRenderArea *m_pRenderArea_Correlation;
#else
	QRenderArea2 *m_pRenderArea_Correlation;
#endif
#endif
};

#endif // NirfDistCompDlg_H
