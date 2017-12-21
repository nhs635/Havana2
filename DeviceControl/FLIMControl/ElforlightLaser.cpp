
#include "ElforlightLaser.h"
#include <Havana2/Configuration.h>

#ifdef OCT_FLIM

#include <iostream>
#include <thread>
#include <chrono>


ElforlightLaser::ElforlightLaser() :
	port_name(ELFORLIGHT_PORT)
{
	m_pSerialComm = new QSerialComm;
}


ElforlightLaser::~ElforlightLaser()
{
	DisconnectDevice();
	if (m_pSerialComm) delete m_pSerialComm;
}


bool ElforlightLaser::ConnectDevice()
{
	// Open a port
	if (!m_pSerialComm->m_bIsConnected)
	{
		if (m_pSerialComm->openSerialPort(port_name))
		{
			printf("ELFORLIGHT: Success to connect to %s.\n", port_name);

			m_pSerialComm->DidReadBuffer += [&](char* buffer, qint64 len)
			{
				for (int i = 0; i < (int)len; i++)
					printf("%c", buffer[i]);
			};
		}
		else
		{
			printf("ELFORLIGHT: Fail to connect to %s.\n", port_name);
			return false;
		}
	}
	else
		printf("ELFORLIGHT: Already connected.\n");

	return true;
}


void ElforlightLaser::DisconnectDevice()
{
	if (m_pSerialComm->m_bIsConnected)
	{
		m_pSerialComm->closeSerialPort();
		printf("ELFORLIGHT: Success to disconnect to %s.\n", port_name);
	}
}


void ElforlightLaser::IncreasePower()
{
	char buff[2] = "+";

	printf("ELFORLIGHT: Send: %s\n", buff);
	std::this_thread::sleep_for(std::chrono::milliseconds(250));
	m_pSerialComm->writeSerialPort(buff);
}


void ElforlightLaser::DecreasePower()
{
	char buff[2] = "-";

	printf("ELFORLIGHT: Send: %s\n", buff);
	std::this_thread::sleep_for(std::chrono::milliseconds(250));
	m_pSerialComm->writeSerialPort(buff);
}

#endif