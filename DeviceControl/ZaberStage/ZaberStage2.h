#ifndef _ZABER_STAGE_H_
#define _ZABER_STAGE_H_

#include <Havana2/Configuration.h>

#include <Common/callback.h>

#ifdef PULLBACK_DEVICE

#include <iostream>
#include <thread>


class QSerialComm;

class ZaberStage2
{
public:
	ZaberStage2();
    ~ZaberStage2();

public:
	bool ConnectDevice();
    void DisconnectDevice();    

public:
    inline bool getIsMoving() { return is_moving; }
    inline void setIsMoving(bool status) { is_moving = status; }

public:
    void Home(int _stageNumber);
    void Stop(int _stageNumber);
    void MoveAbsolute(int _stageNumber, double position);
    void MoveRelative(int _stageNumber, double position);
    void SetTargetSpeed(int _stageNumber, double speed);
    void GetPos(int _stageNumber);

public:
	callback<void> DidMovedAbsolute;
    
private:
	QSerialComm* m_pSerialComm;
	const char* port_name;
	    
    double microstep_size;
    double conversion_factor;

	int target_data;
    int cur_dev;
    int comm_num;
    bool is_moving;
};

#endif
#endif