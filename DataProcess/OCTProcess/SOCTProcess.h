#ifndef _SOCT_PROCESS_H_
#define _SOCT_PROCESS_H_

#include <Havana2/Configuration.h>

#include <DataProcess/OCTProcess/OCTProcess.h>

#include <iostream>
#include <thread>
#include <complex>
#include <vector>

#include <QString>
#include <QFile>

#include <ipps.h>
#include <ippvm.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <Common/array.h>
#include <Common/callback.h>
using namespace np;

#define SURFACE_OFFSET	50


class SOCTProcess : public OCTProcess
{
// Methods
public: // Constructor & Destructor
	explicit SOCTProcess(int nScans, int nAlines, int nwin, int noverlap);
	~SOCTProcess();
    
private: // Not to call copy constrcutor and copy assignment operator
	SOCTProcess(const SOCTProcess&);
	SOCTProcess& operator=(const SOCTProcess&);
	
public:
	// Generate SOCT image
	void operator()(std::vector<np::FloatArray2> spectra, uint16_t* fringe);   
	void db_scaling(std::vector<np::FloatArray2> spectra);
	void spectrum_averaging(std::vector<np::FloatArray2>& spectra, np::Uint16Array& contour, 
							int in_plane, int out_of_plane, int roi_depth);

// Variables
private:
	// Size variables
	int nk;
	int nincrement;

	// SOCT image processing buffer
	FloatArray2 win_repmat;

public:
	// Callback
	callback<void> DidProcess;
};

#endif
