
#include "FaulhaberMotor.h"

#ifdef PULLBACK_DEVICE

FaulhaberMotor::FaulhaberMotor() :
	port_name(FAULHABER_PORT)
{
	m_pSerialComm = new QSerialComm;
}


FaulhaberMotor::~FaulhaberMotor()
{
	DisconnectDevice();
	if (m_pSerialComm) delete m_pSerialComm;
}


bool FaulhaberMotor::ConnectDevice()
{
	// Open a port
	if (!m_pSerialComm->m_bIsConnected)
	{
		if (m_pSerialComm->openSerialPort(port_name))
		{
			printf("FAULHABER: Success to connect to %s.\n", port_name);

			m_pSerialComm->DidReadBuffer += [&](char* buffer, qint64 len)
			{
				for (int i = 0; i < (int)len; i++)				
					printf("%c", buffer[i]);					
			};

			char* buff = (char*)"en\n";
			printf("FAULHABER: Send: %s", buff);
			m_pSerialComm->writeSerialPort(buff);
		}
		else
		{
			printf("FAULHABER: Fail to connect to %s.\n", port_name);
			return false;
		}
	}
	else
		printf("FAULHABER: Already connected.\n");

	return true;
}


void FaulhaberMotor::DisconnectDevice()
{
	if (m_pSerialComm->m_bIsConnected)
	{
		char* buff1 = (char*)"v0\n";
		printf("FAULHABER: Send: %s", buff1);
		m_pSerialComm->writeSerialPort(buff1);

		m_pSerialComm->waitUntilResponse();


		char* buff2 = (char*)"di\n";
		printf("FAULHABER: Send: %s", buff2);
		m_pSerialComm->writeSerialPort(buff2);	

		m_pSerialComm->waitUntilResponse();
		

		m_pSerialComm->closeSerialPort();
		printf("FAULHABER: Success to disconnect to %s.\n", port_name);
	}
}


void FaulhaberMotor::RotateMotor(int RPM)
{
	char buff[100];
	sprintf_s(buff, sizeof(buff), "v%d\n", (!FAULHABER_POSITIVE_ROTATION) ? -RPM : RPM);
	
	printf("FAULHABER: Send: %s", buff);
	m_pSerialComm->writeSerialPort(buff);
}


void FaulhaberMotor::StopMotor()
{
	char* buff = (char*)"v0\n";
	printf("FAULHABER: Send: %s", buff);
	m_pSerialComm->writeSerialPort(buff);
}

#endif
