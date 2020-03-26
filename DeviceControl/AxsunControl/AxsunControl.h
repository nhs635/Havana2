#ifndef AXSUN_CONTROL_H
#define AXSUN_CONTROL_H

#include <Havana2/Configuration.h>

#ifdef AXSUN_OCT_LASER

#include <Common/callback.h>

#include <iostream>
#include <thread>

#import "AxsunOCTControl.tlb" named_guids raw_interfaces_only
using namespace AxsunOCTControl;

// Devices
#define LASER_DEVICE         0

// Connection
#define DISCONNECTED         0
#define CONNECTED           -1

// Operation
#define MAX_SAMPLE_LENGTH    2048
#define CLOCK_OFFSET         28.285
#define CLOCK_GAIN           0.576112


class AxsunControl
{
// Methods
public: // Constructor & Destructor
    explicit AxsunControl();
    virtual ~AxsunControl();

private: // Not to call copy constrcutor and copy assignment operator
	AxsunControl(const AxsunControl&);
	AxsunControl& operator=(const AxsunControl&);

public:
    // For Control
    bool initialize();
    bool setLaserEmission(bool status);
    bool setClockDelay(unsigned long delay);
	bool getDeviceState();

private:
    void dumpControlError(long res, const char* pPreamble);

// Variables
private:
    // For Control
    IAxsunOCTControlPtr m_pAxsunOCTControl;
    VARIANT_BOOL m_bIsConnected;
};

#endif
#endif
