#ifndef SPECTROOCTDLG_H
#define SPECTROOCTDLG_H

#include <QObject>
#include <QtWidgets>
#include <QtCore>

#include <Havana2/Configuration.h>
#include <Havana2/Viewer/QImageView.h>

#include <Common/medfilt.h>
#include <Common/ImageObject.h>

class MainWindow;
class QResultTab;


class SpectroOCTDlg : public QDialog
{
    Q_OBJECT

// Constructer & Destructer /////////////////////////////
public:
    explicit SpectroOCTDlg(QWidget *parent = 0);
    virtual ~SpectroOCTDlg();

// Methods //////////////////////////////////////////////
private:
	void closeEvent(QCloseEvent *e);
	void keyPressEvent(QKeyEvent *e); 

public:
	inline void setClosed(bool closed) { m_bCanBeClosed = closed; }
	inline QImageView* getImageView() const { return m_pImageView_SpectroOCTView; }
	inline bool isAutoExtraction() const { return m_pCheckBox_AutoExtraction->isChecked(); }
	inline int getCurrentAline() const { return m_pSlider_CurrentAline->value(); }
	inline void setCurrentAline(int aline) { m_pSlider_CurrentAline->setValue(aline); }

public slots:
	void setWidgetEnabled(bool);
	void drawSpectroImage(int);

private slots:
	void changeWindowSize();
	void changeOverlapSize();
	void changeInPlaneSize(const QString&);
	void changeOutOfPlaneSize(const QString&);
	void changeRoiDepth(const QString&);
	void changeDbRange();
	
public slots:
	void spectrumExtract();
	void spectrumSave();

signals:
	void setWidgets(bool);
	void processedSingleAline(int);
	void paintSpectroImage(uint8_t*);

private:
	void loadingRawFringe(QFile* pFile, Configuration* pConfig, int frame, np::Uint16Array2& fringe);

// Variables ////////////////////////////////////////////
private:	
    MainWindow* m_pMainWnd;
	Configuration* m_pConfig;
	Configuration *m_pConfigTemp;
    QResultTab* m_pResultTab;

private:
	bool m_bCanBeClosed;
	std::thread t_spectro;

public:
	ImageObject *m_pImgObjSpectroOCT; 

public:
	int subject_frame;
	std::vector<np::FloatArray2> m_vecSpectra;
		
private:
	QImageView *m_pImageView_SpectroOCTView;

	QPushButton *m_pPushButton_Extract;	
	QPushButton *m_pPushButton_Save;
	QProgressBar *m_pProgressBar;
	QCheckBox *m_pCheckBox_AutoExtraction;

	QLabel *m_pLabel_Window;
	QLineEdit *m_pLineEdit_Window;
	QLabel *m_pLabel_Overlap;
	QLineEdit *m_pLineEdit_Overlap;
	
	QLabel *m_pLabel_InPlane;
	QLineEdit *m_pLineEdit_InPlane;
	QLabel *m_pLabel_OutOfPlane;
	QLineEdit *m_pLineEdit_OutOfPlane;
	QLabel *m_pLabel_RoiDepth;
	QLineEdit *m_pLineEdit_RoiDepth;
		
	QLabel *m_pLabel_SpectroDb;
	QImageView *m_pImageView_Colorbar;
	QLineEdit *m_pLineEdit_SpectroDbMin;
	QLineEdit *m_pLineEdit_SpectroDbMax;

	QLabel *m_pLabel_CurrentAline;
	QSlider *m_pSlider_CurrentAline;

	QLabel *m_pLabel_Info;
};

#endif // SPECTROOCTDLG_H
