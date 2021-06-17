
#include "ZaberStage2.h"

#ifdef PULLBACK_DEVICE

#include "../QSerialComm.h"

ZaberStage2::ZaberStage2() :
    m_pSerialComm(nullptr),
    port_name(ZABER_PORT),
    microstep_size(ZABER_MICRO_STEPSIZE),
    conversion_factor(ZABER_CONVERSION_FACTOR),    
	target_data(1),
    cur_dev(1),
    comm_num(1),
    is_moving(false)
{
	m_pSerialComm = new QSerialComm;
}


ZaberStage2::~ZaberStage2()
{
	DisconnectDevice();
	if (m_pSerialComm) delete m_pSerialComm;
}


bool ZaberStage2::ConnectDevice()
{
	// Open a port
	if (!m_pSerialComm->m_bIsConnected)
	{
		if (m_pSerialComm->openSerialPort(port_name, QSerialPort::Baud115200))
		{            
            printf("ZABER: Success to connect to %s.\n", port_name);
         
            // Define callback lambda function
			m_pSerialComm->DidReadBuffer += [&](char* buffer, qint64 len)
			{
                static char msg[256];
                static int j = 0;

                for (int i = 0; i < (int)len; i++)
                {
                    msg[j++] = buffer[i];

                    if (buffer[i] == '\n')
                    {
                        msg[j] = '\0';
                        printf(msg);

						if (is_moving)
						{
							int sp_pos = 0;
							for (int k = 0; k < j; k++)
								if (msg[k] == ' ')
									sp_pos = k;
							int cur_data = atoi(&msg[sp_pos + 1]);

							printf("%d %d\n", target_data, cur_data);

							if (target_data == cur_data)
							{
								if (target_data != 0)
									DidMovedAbsolute();
								is_moving = false;
							}
						}

                        j = 0;
                    }

                    if (j == 256) j = 0;
                }
            };
		}
		else
		{
            printf("ZABER: Success to disconnect to %s.\n", port_name);            
			return false;
		}
	}
	else
		printf("ZABER: Already connected.\n");

	return true;
}


void ZaberStage2::DisconnectDevice()
{
	if (m_pSerialComm->m_bIsConnected)
    {
		m_pSerialComm->closeSerialPort();
		        
        printf("ZABER: Success to disconnect to %s.\n", port_name);        
	}
}


void ZaberStage2::Home(int _stageNumber)
{
    cur_dev = _stageNumber;

	target_data = 0;
	
    char buff[100];
    sprintf_s(buff, sizeof(buff), "/%02d 0 %02d home\n", _stageNumber, comm_num);
    printf("ZABER: Send: %s", buff);
    
    comm_num++;
    if (comm_num == 96) comm_num = 1;

    m_pSerialComm->writeSerialPort(buff);
    is_moving = true;
}


void ZaberStage2::Stop(int _stageNumber)
{
    cur_dev = _stageNumber;
	
    char buff[100];
    sprintf_s(buff, sizeof(buff), "/%02d 0 %02d stop\n", _stageNumber, comm_num);
	printf("ZABER: Send: %s", buff);

    comm_num++;
    if (comm_num == 96) comm_num = 1;

    m_pSerialComm->writeSerialPort(buff);
}


void ZaberStage2::MoveAbsolute(int _stageNumber, double position)
{
    cur_dev = _stageNumber;

	target_data = (int)round(position * 1000.0 / microstep_size);
	
    char buff[100];
    sprintf_s(buff, sizeof(buff), "/%02d 0 %02d move abs %d\n", _stageNumber, comm_num, target_data);
	printf("ZABER: Send: %s", buff);

    comm_num++;
    if (comm_num == 96) comm_num = 1;

    m_pSerialComm->writeSerialPort(buff);
    is_moving = true;
}


void ZaberStage2::MoveRelative(int _stageNumber, double position)
{
    cur_dev = _stageNumber;

	target_data = (int)round(position * 1000.0 / microstep_size);
	
    char buff[100];
    sprintf_s(buff, sizeof(buff), "/%02d 0 %02d move rel %d\n", _stageNumber, comm_num, target_data);
	printf("ZABER: Send: %s", buff);

    comm_num++;
    if (comm_num == 96) comm_num = 1;

    m_pSerialComm->writeSerialPort(buff);
    is_moving = true;
}


void ZaberStage2::SetTargetSpeed(int _stageNumber, double speed)
{
    cur_dev = _stageNumber;

    int data = (int)round(speed * 1000.0 / microstep_size * conversion_factor);

    char buff[100];
    sprintf_s(buff, sizeof(buff), "/%02d 0 %02d set maxspeed %d\n", _stageNumber, comm_num, data);
	printf("ZABER: Send: %s", buff);

    comm_num++;
    if (comm_num == 96) comm_num = 1;

    m_pSerialComm->writeSerialPort(buff);
}


void ZaberStage2::GetPos(int _stageNumber)
{
    char buff[100];
    sprintf_s(buff, sizeof(buff), "/%02d 0 %02d get pos\n", _stageNumber, comm_num);
	printf("ZABER: Send: %s", buff);

    comm_num++;
    if (comm_num == 96) comm_num = 1;

    m_pSerialComm->writeSerialPort(buff);
}

#endif