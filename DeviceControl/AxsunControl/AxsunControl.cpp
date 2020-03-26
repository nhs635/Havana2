
#include "AxsunControl.h"

#ifdef AXSUN_OCT_LASER

AxsunControl::AxsunControl() :
	m_pAxsunOCTControl(nullptr),
    m_bIsConnected(DISCONNECTED)
{
}


AxsunControl::~AxsunControl()
{
	if (m_bIsConnected == CONNECTED)
	{
		unsigned long retvallong;
		//m_pAxsunOCTControl->CloseConnections();
		m_pAxsunOCTControl->StopNetworkControlInterface(&retvallong);
        CoUninitialize();
		printf("AXSUN: OCT Laser connection is successfully closed and uninitialized.\n");
	}
}


bool AxsunControl::initialize()
{
	HRESULT result;
	unsigned long retvallong;
	const char* pPreamble = "AXSUN: Failed to initialize Axsun OCT laser: ";

	printf("AXSUN: Initializing OCT laser devices...\n");

	// Co-Initialization
    result = CoInitialize(NULL);

	// Dynamic Object for Axsun OCT Control
	m_pAxsunOCTControl = IAxsunOCTControlPtr(__uuidof(struct AxsunOCTControl));
	
	// Start Ethernet Connection
	result = m_pAxsunOCTControl->StartNetworkControlInterface(&retvallong);
	if (result != S_OK)
	{
		dumpControlError(result, pPreamble);
		return false;
	}

	// Connect Device (Laser)
	unsigned long deviceNum = 0;
	BSTR systemTypeString = SysAllocString(L"");
	
	result = m_pAxsunOCTControl->ConnectToOCTDevice(0, &m_bIsConnected); // 0: Light Source[40]
	if (result != S_OK)
	{
		dumpControlError(result, pPreamble);
		return false;
	}

	if (m_bIsConnected == CONNECTED)
	{
		result = m_pAxsunOCTControl->GetSystemType(&deviceNum, &systemTypeString, &retvallong);
		if (result != S_OK)
		{
			dumpControlError(result, pPreamble);
			return false;
		}
						
		printf("AXSUN: %S (deviceNum: %d) is successfully connected.\n", systemTypeString, deviceNum);			
	}
	else
	{
		result = 80;
		dumpControlError(result, pPreamble);
		printf("AXSUN: Unable to connect to the devices.\n");
		return false;
	}
	
	SysFreeString(systemTypeString);
	
	return true;
}


bool AxsunControl::setLaserEmission(bool status)
{
	HRESULT result;
	unsigned long retvallong;
	const char* pPreamble = "AXSUN: Failed to set Laser Emission: ";
	
	result = m_pAxsunOCTControl->ConnectToOCTDevice(LASER_DEVICE, &m_bIsConnected);
	if (result != S_OK)
	{
		dumpControlError(result, pPreamble);
		return false;
	}

	if (m_bIsConnected == CONNECTED)
	{
		//std::this_thread::sleep_for(std::chrono::milliseconds(500));

		if (status)
			result = m_pAxsunOCTControl->StartScan(&retvallong);
		else
			result = m_pAxsunOCTControl->StopScan(&retvallong);
		if (result != S_OK)
		{
			dumpControlError(result, pPreamble);
			return false;
		}
	}
	else
	{
		result = 80;
		dumpControlError(result, pPreamble);
		printf("AXSUN: Unable to connect to the devices.\n");
		return false;
	}
		
	printf("AXSUN: Laser Emission is turned %s.\n", status ? "on" : "off");
	
	return true;
}


bool AxsunControl::setClockDelay(unsigned long delay)
{
	HRESULT result;
	unsigned long retvallong;
	const char* pPreamble = "AXSUN: Failed to set clock delay: ";

    result = m_pAxsunOCTControl->ConnectToOCTDevice(LASER_DEVICE, &m_bIsConnected);
	if (result != S_OK)
	{
		dumpControlError(result, pPreamble);
		return false;
	}

	if (m_bIsConnected == CONNECTED)
    {
        unsigned long delay0;
        result = m_pAxsunOCTControl->SetClockDelay(delay, &retvallong);
		if (result != S_OK)
		{
			dumpControlError(result, pPreamble);
			return false;
		}
        result = m_pAxsunOCTControl->GetClockDelay(&delay0, &retvallong);
				
        printf("AXSUN: Clock delay is set to %.3f nsec [%d].\n", CLOCK_GAIN * (double)delay + CLOCK_OFFSET, delay0);		
	}
	else
	{
		result = 80;
		dumpControlError(result, pPreamble);
		printf("AXSUN: Unable to connect to the devices.\n");
		return false;
	}

	return true;
}


bool AxsunControl::getDeviceState()
{
	HRESULT result;
	//unsigned long retvallong;
	const char* pPreamble = "AXSUN: Failed to get device state: ";

	result = m_pAxsunOCTControl->ConnectToOCTDevice(LASER_DEVICE, &m_bIsConnected);
	if (result != S_OK)
	{
		dumpControlError(result, pPreamble);
		return false;
	}

	if (m_bIsConnected == CONNECTED)
	{
		//unsigned long laser_time;
		//result = m_pAxsunOCTControl->GetLaserOnTime(&laser_time, &retvallong);
		//if (result != S_OK)
		//{
		//	dumpControlError(result, pPreamble);
		//	return false;
		//}

		//unsigned long rawVal;
		//float scaledVal;
		//BSTR temp; temp = _bstr_t("test");
		////result = m_pAxsunOCTControl->Get GetLowSpeedADOneChannel(5, &rawVal, &scaledVal, &temp);
		//if (result != S_OK)
		//{
		//	dumpControlError(result, pPreamble);
		//	return false;
		//}		
		//printf("[%d] %d %f %s\n", laser_time, rawVal, scaledVal, temp);

		//unsigned long delay0;
		//result = m_pAxsunOCTControl->SetClockDelay(delay, &retvallong);
		//if (result != S_OK)
		//{
		//	dumpControlError(result, pPreamble);
		//	return false;
		//}
		//result = m_pAxsunOCTControl->GetClockDelay(&delay0, &retvallong);

		//char msg[256];
		//sprintf(msg, "[Axsun Control] Clock delay is set to %.3f nsec [%d].", CLOCK_GAIN * (double)delay + CLOCK_OFFSET, delay0);
		//SendStatusMessage(msg, false);
	}
	else
	{
		result = 80;
		dumpControlError(result, pPreamble);
		printf("AXSUN: Unable to connect to the devices.\n");
		return false;
	}

	return true;
}


void AxsunControl::dumpControlError(long res, const char* pPreamble)
{
    char msg[256] = { 0, };
    memcpy(msg, pPreamble, strlen(pPreamble));

    char err[256] = { 0, };
    sprintf_s(err, 256, "Error code (%d)\n", res);
	
    strcat(msg, err);

	printf(msg);
}

#endif
