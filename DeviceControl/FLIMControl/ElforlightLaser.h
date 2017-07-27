#ifndef _ELFORLIGHT_LASER_H_
#define _ELFORLIGHT_LASER_H_

#include "../QSerialComm.h"


class ElforlightLaser
{
public:
	ElforlightLaser();
	~ElforlightLaser();

public:
	bool ConnectDevice();
	void DisconnectDevice();
	void IncreasePower();
	void DecreasePower();

private:
	QSerialComm* m_pSerialComm;
	const char* port_name;
};

#endif