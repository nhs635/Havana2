#ifndef _FAULHABER_MOTOR_H_
#define _FAULHABER_MOTOR_H_

#include <Havana2/Configuration.h>

#ifdef PULLBACK_DEVICE

#include "../QSerialComm.h"


class FaulhaberMotor 
{
public:
	FaulhaberMotor();
	~FaulhaberMotor();

public:
	bool ConnectDevice();
	void DisconnectDevice();
	void EnableMotor();
	void DisableMotor();
	void RotateMotor(int RPM);
	void StopMotor();

#ifdef FAULHABER_NEW_CONTROLLER
private:
	uint8_t CalcCRCByte(uint8_t u8Byte, uint8_t u8CRC);
#endif

private:
	QSerialComm* m_pSerialComm;
	const char* port_name;
#ifdef FAULHABER_NEW_CONTROLLER
	char controlword[11] = { (char)0x53, (char)0x09, (char)0x01, (char)0x02, (char)0x40, (char)0x60, (char)0x00, (char)0x00, (char)0x00, (char)0x00, (char)0x45 };
	char target_velocity[13] = { (char)0x53, (char)0x0B, (char)0x01, (char)0x02, (char)0xFF, (char)0x60, (char)0x00, (char)0x00, (char)0x00, (char)0x00, (char)0x00, (char)0x00, (char)0x45 };
#endif
};

#endif
#endif
