
#include "NirfEmissionProfileDlg.h"

#include <Havana2/MainWindow.h>
#include <Havana2/QStreamTab.h>
#include <Havana2/QResultTab.h>
#include <Havana2/Dialog/NirfDistCompDlg.h>

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
		m_pScope = new QScope2({ 0, (double)NIRF_SCANS * 8 }, { min(m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[1].min), max(m_pConfig->nirfRange[0].max, m_pConfig->nirfRange[0].max) }, 2, 2, 1, 1, 0, 0, "", "", true);
	else
		m_pScope = new QScope2({ 0, (double)m_pResultTab->m_nirfMap1.size(0) }, { min(m_pConfig->nirfRange[0].min, m_pConfig->nirfRange[1].min), max(m_pConfig->nirfRange[0].max, m_pConfig->nirfRange[0].max) });
#endif
	if (!m_bIsStreaming)
	{
		m_pScope->getRender()->m_bSelectionAvailable = true;
		m_pScope->getRender()->DidMouseEvent += [&]() {
			if (!m_bIsStreaming)
			{
#ifndef TWO_CHANNEL_NIRF
				drawData(m_pScope->getRender()->m_pData);
#else
				drawData(m_pScope->getRender()->m_pData1, m_pScope->getRender()->m_pData2);
#endif
				m_pResultTab->invalidate();
			}
		};
	}

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

#ifndef TWO_CHANNEL_NIRF
void NirfEmissionProfileDlg::drawData(void* data)
#else
void NirfEmissionProfileDlg::drawData(void* data1, void* data2)
#endif
{
	if (m_bIsStreaming)
	{
#ifndef TWO_CHANNEL_NIRF
		m_pScope->drawData((double*)data);		

		Ipp64f mean, std;
		ippsMeanStdDev_64f((double*)data, m_pConfig->nAlines, &mean, &std);
		
		setWindowTitle(QString("NIRF Emission Profile (%1 + %2)").arg(mean, 3, 'f', 4).arg(std, 3, 'f', 4));
#else
		m_pScope->drawData((double*)data1, (double*)data2);

		Ipp64f mean[2], std[2];
		ippsMeanStdDev_64f((double*)data1, m_pConfig->nAlines, &mean[0], &std[0]);
		ippsMeanStdDev_64f((double*)data2, m_pConfig->nAlines, &mean[1], &std[1]);

		setWindowTitle(QString("NIRF Emission Profile (Ch1: %1 + %2 / Ch2: %3 + %4)").arg(mean[0], 3, 'f', 4).arg(std[0], 3, 'f', 4).arg(mean[1], 3, 'f', 4).arg(std[1], 3, 'f', 4));
#endif
	}
	else
	{
#ifndef TWO_CHANNEL_NIRF
		m_pScope->drawData((float*)data);

		np::FloatArray data0(m_pResultTab->m_nirfMap.size(0)), mask(m_pResultTab->m_nirfMap.size(0));
		memcpy(data0, data, sizeof(float) * data0.length());
		ippsConvert_8u32f(m_pScope->getRender()->m_pSelectedRegion, mask.raw_ptr(), mask.length());
				
		Ipp32f mean, std, maxi, mask_len;
		ippsSum_32f(mask.raw_ptr(), mask.length(), &mask_len, ippAlgHintFast);

		{
			np::Uint8Array valid_region_8u(m_pResultTab->m_nirfMap.size(0));
			np::FloatArray valid_region_32f(m_pResultTab->m_nirfMap.size(0));
			ippiCompareC_32f_C1R(data0.raw_ptr(), data0.size(0), 0.0f, valid_region_8u.raw_ptr(), valid_region_8u.size(0), { data0.size(0), 1 }, ippCmpEq);
			ippsConvert_8u32f(valid_region_8u, valid_region_32f, valid_region_32f.length());
			ippsDivC_32f_I(255.0f, valid_region_32f, valid_region_32f.length());
			ippsSubCRev_32f_I(1.0f, valid_region_32f, valid_region_32f.length());

			if (mask_len == 0.0f)
				memcpy(mask, valid_region_32f, sizeof(float) * mask.length());
			else
				ippsMul_32f_I(valid_region_32f, mask, mask.length());
		}

		ippsSum_32f(mask.raw_ptr(), mask.length(), &mask_len, ippAlgHintFast);
				
		if (mask_len > 0) ippsMul_32f_I(mask.raw_ptr(), data0.raw_ptr(), data0.length());
		ippsMeanStdDev_32f(data0.raw_ptr(), data0.length(), &mean, &std, ippAlgHintFast);
		ippsMax_32f(data0.raw_ptr(), data0.length(), &maxi);
		if (mask_len > 0)
		{
			mean = mean * data0.length() / mask_len;

			ippsSqr_32f_I(data0.raw_ptr(), data0.length());
			ippsSum_32f(data0.raw_ptr(), data0.length(), &std, ippAlgHintFast);
			std = sqrt(std / mask_len - mean * mean);
		}
		if (mask_len == 0)
			mean = 0, std = 0, maxi = 0;

		setWindowTitle(QString("NIRF Emission Profile (%1 + %2 / %3)").arg(mean, 3, 'f', 4).arg(std, 3, 'f', 4).arg(maxi, 3, 'f', 4));
#else
		m_pScope->drawData((float*)data1, (float*)data2);
		
		np::FloatArray data0_1(m_pResultTab->m_nirfMap1.size(0)), data0_2(m_pResultTab->m_nirfMap2.size(0)), mask(m_pResultTab->m_nirfMap2.size(0));
		memcpy(data0_1, data1, sizeof(float) * data0_1.length());
		memcpy(data0_2, data2, sizeof(float) * data0_2.length());
		ippsConvert_8u32f(m_pScope->getRender()->m_pSelectedRegion, mask.raw_ptr(), mask.length());

		Ipp32f mean[2], std[2], mask_len;
		ippsSum_32f(mask.raw_ptr(), mask.length(), &mask_len, ippAlgHintFast);

		if (mask_len > 0)
		{
			ippsMul_32f_I(mask.raw_ptr(), data0_1.raw_ptr(), data0_1.length());
			ippsMul_32f_I(mask.raw_ptr(), data0_2.raw_ptr(), data0_2.length());
		}
		ippsMeanStdDev_32f(data0_1.raw_ptr(), data0_1.length(), &mean[0], &std[0], ippAlgHintFast);
		ippsMeanStdDev_32f(data0_2.raw_ptr(), data0_2.length(), &mean[1], &std[1], ippAlgHintFast);
		if (mask_len > 0)
		{			
			mean[0] = mean[0] * data0_1.length() / mask_len;
			mean[1] = mean[1] * data0_2.length() / mask_len;

			ippsSqr_32f_I(data0_1.raw_ptr(), data0_1.length());
			ippsSum_32f(data0_1.raw_ptr(), data0_1.length(), &std[0], ippAlgHintFast);
			std[0] = sqrt(std[0] / mask_len - mean[0] * mean[0]);

			ippsSqr_32f_I(data0_2.raw_ptr(), data0_2.length());
			ippsSum_32f(data0_2.raw_ptr(), data0_2.length(), &std[1], ippAlgHintFast);
			std[1] = sqrt(std[1] / mask_len - mean[1] * mean[1]);
		}

		setWindowTitle(QString("NIRF Emission Profile (Ch1: %1 + %2 / Ch2: %3 + %4)").arg(mean[0], 3, 'f', 4).arg(std[0], 3, 'f', 4).arg(mean[1], 3, 'f', 4).arg(std[1], 3, 'f', 4));
#endif
	}
}
#endif
