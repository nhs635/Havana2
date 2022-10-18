
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
#ifndef FAULHABER_NEW_CONTROLLER
				printf("FAULHABER: Receive: ");
				for (int i = 0; i < (int)len; i++)
					printf("%c", buffer[i]);
				printf("\n");
#else
				static unsigned char msg[256];
				static int n = 0;

				for (int i = 0; i < (int)len; i++)
					msg[n++] = buffer[i];

				if (buffer[len - 1] == 0x45)
				{
					printf("FAULHABER: Receive: ");
					for (int i = 0; i < n; i++)
						printf("%02X ", (unsigned char)msg[i]);
					printf("\n");

					if (n == 8)
					{
						// Pullback flag
						if ((msg[1] == 0x06) && (msg[3] == 0x05) && (msg[4] == 0x27) && (msg[5] == 0x14))
						{
							endTime = std::chrono::steady_clock::now();
							std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
							
							DidMovedAbsolute();
							printf("pullbacked: %d msec\n", (int)elapsed.count());
						}
						else if ((msg[1] == 0x06) && (msg[3] == 0x05) && (msg[4] == 0x27) && (msg[5] == 0x30))
						{							
							DisableMotor(2);	
							EnableMotor(2);
							
							std::this_thread::sleep_for(std::chrono::milliseconds(100));

							SetTargetPosition(2, -2048);
							printf("home sweet home\n");
						}
					}

					n = 0;
				}
#endif
			};			

			EnableMotor(2);
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
		m_pSerialComm->DidReadBuffer.clear();
		DisableMotor(1);
		DisableMotor(2);
		m_pSerialComm->closeSerialPort();
		printf("FAULHABER: Success to disconnect to %s.\n", port_name);
	}
}


void FaulhaberMotor::EnableMotor(char dev)
{
#ifndef FAULHABER_NEW_CONTROLLER
	char* buff = (char*)"en\n";
	printf("FAULHABER: Send: %s", buff);
	m_pSerialComm->writeSerialPort(buff);
	m_pSerialComm->waitUntilResponse();
#else
	// Send controlword command 1
	controlword[2] = dev;
	if (dev == 1)	
		controlword[7] = (char)0x06; 
	else if (dev == 2)
		controlword[7] = (char)0x0E;
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

	// Send controlword command 2
	controlword[2] = dev;
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


void FaulhaberMotor::DisableMotor(char dev)
{
#ifndef FAULHABER_NEW_CONTROLLER
	char* buff = (char*)"di\n";
	printf("FAULHABER: Send: %s", buff);
	m_pSerialComm->writeSerialPort(buff);
	m_pSerialComm->waitUntilResponse();
#else
	// Send controlword command
	controlword[2] = dev;
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


void FaulhaberMotor::RotateMotor(char dev, int RPM)
{
	// Enable motor first
	EnableMotor(dev);

#ifndef FAULHABER_NEW_CONTROLLER
	char buff[100];
	sprintf_s(buff, sizeof(buff), "v%d\n", (!FAULHABER_POSITIVE_ROTATION) ? -RPM : RPM);
	printf("FAULHABER: Send: %s", buff);
	m_pSerialComm->writeSerialPort(buff);
	m_pSerialComm->waitUntilResponse();
#else
	// Send target velocity command
	target_velocity[2] = dev;

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


void FaulhaberMotor::StopMotor(char dev)
{
#ifndef FAULHABER_NEW_CONTROLLER
	char* buff = (char*)"v0\n";
	printf("FAULHABER: Send: %s", buff);
	m_pSerialComm->writeSerialPort(buff);
	m_pSerialComm->waitUntilResponse();
#else
	//// Send target velocity command
	//target_velocity[2] = dev;

	//target_velocity[7]  = (char)0x00;
	//target_velocity[8]  = (char)0x00;
	//target_velocity[9]  = (char)0x00;
	//target_velocity[10] = (char)0x00;
	//uint8_t crc_temp = 0xFF;
	//for (int i = 1; i < target_velocity[1]; i++)
	//	crc_temp = CalcCRCByte(target_velocity[i], crc_temp);
	//target_velocity[11] = (char)crc_temp;

	//printf("FAULHABER: Send: ");
	//for (int i = 0; i < (int)13; i++)
	//	printf("%02X ", (unsigned char)target_velocity[i]);
	//printf("\n");
	//m_pSerialComm->writeSerialPort(target_velocity, 13);
	//m_pSerialComm->waitUntilResponse();
#endif

	// Disable motor
	DisableMotor(dev);

	if (dev == 2)
		EnableMotor(dev);
}


void FaulhaberMotor::SetTargetPosition(char dev, int position)
{
#ifdef FAULHABER_NEW_CONTROLLER
	// Enable motor first	
	//EnableMotor(dev);

	// Send target position command
	target_position[2] = dev;
	target_position[7] = (char)(0xFF & position);
	target_position[8] = (char)(0xFF & (position >> 8));
	target_position[9] = (char)(0xFF & (position >> 16));
	target_position[10] = (char)(0xFF & (position >> 24));
	uint8_t crc_temp = 0xFF;
	for (int i = 1; i < target_position[1]; i++)
		crc_temp = CalcCRCByte(target_position[i], crc_temp);
	target_position[11] = (char)crc_temp;

	printf("FAULHABER: Send: ");
	for (int i = 0; i < (int)13; i++)
		printf("%02X ", (unsigned char)target_position[i]);
	printf("\n");
	m_pSerialComm->writeSerialPort(target_position, 13);
	m_pSerialComm->waitUntilResponse();

	// Send control word command
	controlword[2] = dev;
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

	// Send control word command
	controlword[2] = dev;
	controlword[7] = (char)0x7F;
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

	startTime = std::chrono::steady_clock::now();
#endif
}


void FaulhaberMotor::SetProfileVelocity(char dev, int velocity)
{
#ifdef FAULHABER_NEW_CONTROLLER
	// Send profile velocity command
	profile_velocity[2] = dev;
	profile_velocity[7] = (char)(0xFF & velocity);
	profile_velocity[8] = (char)(0xFF & (velocity >> 8));
	profile_velocity[9] = (char)(0xFF & (velocity >> 16));
	profile_velocity[10] = (char)(0xFF & (velocity >> 24));
	uint8_t crc_temp = 0xFF;
	for (int i = 1; i < profile_velocity[1]; i++)
		crc_temp = CalcCRCByte(profile_velocity[i], crc_temp);
	profile_velocity[11] = (char)crc_temp;

	printf("FAULHABER: Send: ");
	for (int i = 0; i < (int)13; i++)
		printf("%02X ", (unsigned char)profile_velocity[i]);
	printf("\n");
	m_pSerialComm->writeSerialPort(profile_velocity, 13);
	m_pSerialComm->waitUntilResponse();
#endif
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
