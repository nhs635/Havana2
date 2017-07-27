#ifndef _FAULHABER_MOTOR_H_
#define _FAULHABER_MOTOR_H_

#include "../QSerialComm.h"


class FaulhaberMotor 
{
public:
	FaulhaberMotor();
	~FaulhaberMotor();

public:
	bool ConnectDevice();
	void DisconnectDevice();
	void RotateMotor(int RPM);
	void StopMotor();

private:
	QSerialComm* m_pSerialComm;
	const char* port_name;
};

#endif