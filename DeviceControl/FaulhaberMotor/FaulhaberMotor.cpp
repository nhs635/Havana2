
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
#ifndef FAULHABER_NEW_CONTROLLER
		if (m_pSerialComm->openSerialPort(port_name, QSerialPort::Baud9600))
#else
		if (m_pSerialComm->openSerialPort(port_name, QSerialPort::Baud115200))
#endif
		{
			printf("FAULHABER: Success to connect to %s.\n", port_name);

			m_pSerialComm->DidReadBuffer += [&](char* buffer, qint64 len)
			{
#ifdef FAULHABER_NEW_CONTROLLER
				printf("FAULHABER: Receive: ");
#endif
				for (int i = 0; i < (int)len; i++)
#ifndef FAULHABER_NEW_CONTROLLER		
					printf("%c", buffer[i]);
#else
					printf("%02X ", (unsigned char)buffer[i]);
				printf("\n");
#endif
			};
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
		StopMotor();			
		m_pSerialComm->closeSerialPort();
		printf("FAULHABER: Success to disconnect to %s.\n", port_name);
	}
}


void FaulhaberMotor::EnableMotor()
{
#ifndef FAULHABER_NEW_CONTROLLER
	char* buff = (char*)"en\n";
	printf("FAULHABER: Send: %s", buff);
	m_pSerialComm->writeSerialPort(buff);
	m_pSerialComm->waitUntilResponse();
#else
	controlword[7] = (char)0x06; 
	uint8_t crc_temp = 0xFF;
	for (int i = 1; i < controlword[1]; i++)
		crc_temp = CalcCRCByte(controlword[i], crc_temp);
	controlword[9] = (char)crc_temp;

	printf("FAULHABER: Send: ");
	for (int i = 0; i < (int)11; i++)
		printf("%02X ", (unsigned char)controlword[i]);
	printf("\n");
	m_pSerialComm->writeSerialPort(controlword, 11);
	m_pSerialComm->waitUntilResponse();

	controlword[7] = (char)0x0F;
	crc_temp = 0xFF;
	for (int i = 1; i < controlword[1]; i++)
		crc_temp = CalcCRCByte(controlword[i], crc_temp);
	controlword[9] = (char)crc_temp;

	printf("FAULHABER: Send: ");
	for (int i = 0; i < (int)11; i++)
		printf("%02X ", (unsigned char)controlword[i]);
	printf("\n");
	m_pSerialComm->writeSerialPort(controlword, 11);
	m_pSerialComm->waitUntilResponse();
#endif
}


void FaulhaberMotor::DisableMotor()
{
#ifndef FAULHABER_NEW_CONTROLLER
	char* buff = (char*)"di\n";
	printf("FAULHABER: Send: %s", buff);
	m_pSerialComm->writeSerialPort(buff);
	m_pSerialComm->waitUntilResponse();
#else
	controlword[7] = (char)0x0D;
	uint8_t crc_temp = 0xFF;
	for (int i = 1; i < controlword[1]; i++)
		crc_temp = CalcCRCByte(controlword[i], crc_temp);
	controlword[9] = (char)crc_temp;

	printf("FAULHABER: Send: ");
	for (int i = 0; i < (int)11; i++)
		printf("%02X ", (unsigned char)controlword[i]);
	printf("\n");
	m_pSerialComm->writeSerialPort(controlword, 11);
	m_pSerialComm->waitUntilResponse();
#endif
}


void FaulhaberMotor::RotateMotor(int RPM)
{
	EnableMotor();

#ifndef FAULHABER_NEW_CONTROLLER
	char buff[100];
	sprintf_s(buff, sizeof(buff), "v%d\n", (!FAULHABER_POSITIVE_ROTATION) ? -RPM : RPM);
	printf("FAULHABER: Send: %s", buff);
	m_pSerialComm->writeSerialPort(buff);
	m_pSerialComm->waitUntilResponse();
#else
	RPM = (!FAULHABER_POSITIVE_ROTATION) ? -RPM : RPM;
	target_velocity[7]  = (char)(0xFF & RPM);
	target_velocity[8]  = (char)(0xFF & (RPM >> 8));
	target_velocity[9]  = (char)(0xFF & (RPM >> 16));
	target_velocity[10] = (char)(0xFF & (RPM >> 24));	
	uint8_t crc_temp = 0xFF;
	for (int i = 1; i < target_velocity[1]; i++)
		crc_temp = CalcCRCByte(target_velocity[i], crc_temp);
	target_velocity[11] = (char)crc_temp;

	printf("FAULHABER: Send: ");
	for (int i = 0; i < (int)13; i++)
		printf("%02X ", (unsigned char)target_velocity[i]);
	printf("\n");
	m_pSerialComm->writeSerialPort(target_velocity, 13);
	m_pSerialComm->waitUntilResponse();
#endif
}


void FaulhaberMotor::StopMotor()
{
#ifndef FAULHABER_NEW_CONTROLLER
	char* buff = (char*)"v0\n";
	printf("FAULHABER: Send: %s", buff);
	m_pSerialComm->writeSerialPort(buff);
	m_pSerialComm->waitUntilResponse();
#else
	target_velocity[7]  = (char)0x00;
	target_velocity[8]  = (char)0x00;
	target_velocity[9]  = (char)0x00;
	target_velocity[10] = (char)0x00;
	uint8_t crc_temp = 0xFF;
	for (int i = 1; i < target_velocity[1]; i++)
		crc_temp = CalcCRCByte(target_velocity[i], crc_temp);
	target_velocity[11] = (char)crc_temp;

	printf("FAULHABER: Send: ");
	for (int i = 0; i < (int)13; i++)
		printf("%02X ", (unsigned char)target_velocity[i]);
	printf("\n");
	m_pSerialComm->writeSerialPort(target_velocity, 13);
	m_pSerialComm->waitUntilResponse();
#endif

	DisableMotor();
}

#ifdef FAULHABER_NEW_CONTROLLER
uint8_t FaulhaberMotor::CalcCRCByte(uint8_t u8Byte, uint8_t u8CRC)
{
	u8CRC = u8CRC ^ u8Byte;
	for (uint8_t i = 0; i < 8; i++)
	{
		if (u8CRC & 0x01)
			u8CRC = (u8CRC >> 1) ^ 0xD5;
		else
			u8CRC >>= 1;
	}

	return u8CRC;
}
#endif

#endif
