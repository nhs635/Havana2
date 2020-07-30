
#include "NirfEmissionProfileDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QStreamTab.h>
#include <Havana2/QResultTab.h>

#include <iostream>
#include <thread>


#ifdef OCT_NIRF
NirfEmissionProfileDlg::NirfEmissionProfileDlg(bool _isStreaming, QWidget *parent) :
    QDialog(parent), m_bIsStreaming(true)
{
    // Set default size & frame
    setFixedSize(600, 250);
    setWindowFlags(Qt::Tool);
    setWindowTitle("NIRF Emission Profile");

    // Set main window objects
    m_bIsStreaming = _isStreaming;
    if (m_bIsStreaming)
    {
        m_pStreamTab = (QStreamTab*)parent;
        m_pMainWnd = m_pStreamTab->getMainWnd();
    }
    else
    {
        m_pResultTab = (QResultTab*)parent;
        m_pMainWnd = m_pResultTab->getMainWnd();
    }
    m_pConfig = m_pMainWnd->m_pConfiguration;

    // Create scope object
#ifndef TWO_CHANNEL_NIRF
    if (m_bIsStreaming)
        m_pScope = new QScope({ 0, (double)m_pConfig->nAlines }, { m_pConfig->nirfRange.min, m_pConfig->nirfRange.max }, 2, 2, 1, 1, 0, 0, "", "", false, true);
    else
        m_pScope = new QScope({ 0, (double)m_pResultTab->m_nirfMap.size(0) }, { m_pConfig->nirfRange.min, m_pConfig->nirfRange.max });
#else
	if (m_bIsStreaming)
		m_pScope = new QScope2({ 0, (double)m_pConfig->nAlines }, { m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[0].max }, 2, 2, 1, 1, 0, 0, "", "", true);
	else
		m_pScope = new QScope2({ 0, (double)m_pResultTab->m_nirfMap1.size(0) }, { m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[0].max });
#endif


    // Set layout
    QGridLayout *pGridLayout = new QGridLayout;
    pGridLayout->setSpacing(0);

    pGridLayout->addWidget(m_pScope, 0, 0);
    setLayout(pGridLayout);
}

NirfEmissionProfileDlg::~NirfEmissionProfileDlg()
{
}

void NirfEmissionProfileDlg::keyPressEvent(QKeyEvent *e)
{
    if (e->key() != Qt::Key_Escape)
        QDialog::keyPressEvent(e);
}


void NirfEmissionProfileDlg::setTitle(const QString & str)
{
	setWindowTitle(str);
}

#endif
