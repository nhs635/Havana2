
#include "NirfCrossTalkCompDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QResultTab.h>


#ifdef OCT_NIRF
#ifdef TWO_CHANNEL_NIRF
NirfCrossTalkCompDlg::NirfCrossTalkCompDlg(QWidget *parent) :
    QDialog(parent)
{
    // Set default size & frame
    setFixedSize(360, 330);
    setWindowFlags(Qt::Tool);
	setWindowTitle("2Ch NIRF Cross Talk Compensation");

    // Set main window objects
    m_pResultTab = (QResultTab*)parent;
    m_pMainWnd = m_pResultTab->getMainWnd();
	m_pConfig = m_pMainWnd->m_pConfiguration;
	

	m_pVBoxLayout = new QVBoxLayout;
	m_pVBoxLayout->setSpacing(5);

	
	// Set layout
	this->setLayout(m_pVBoxLayout);

	// Connect
}

NirfCrossTalkCompDlg::~NirfCrossTalkCompDlg()
{
}

void NirfCrossTalkCompDlg::keyPressEvent(QKeyEvent *e)
{
	if (e->key() != Qt::Key_Escape)
		QDialog::keyPressEvent(e);
}

#endif
#endif
