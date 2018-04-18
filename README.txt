/*** Required Software ***/

- Windows 7 64 bit OS

- Microsoft Visual Studio 2015 (vc14)
- VC_redistributable_MSVC_2015
- Intel C++ Compiler 17 (recommended)
- Qt 5.8
- QMake (Makefile by qmake (.pro))
- NI-DAQmx Run-Time Engine 9.3
- PX14400 Digitizer Setup

- Intel Performance Primitives Update 3
- Thread Building Blocks Update 3
- Math Kernel Library 2017 Update 6



/*** Untracked files on git ***/

.vs/
bin/
Debug/
GeneratedFiles/
include/
install/
lib/
Release/
x64/
bg.bin
calibration.dat
d1.bin
d2.bin
flim_mask.dat
Havana2.sln
Havana2.vcxproj*
Havana2.sdf
Havana2.VC*



/*** Update History ***/
- 170728 Havana2 v1.1.0 Final Version 
  (Lifetime colortable combobox was added. Saving current value for device control was added.)
  (.gitignore was added.)
- 170804 Havana2 v1.1.1 
  (ECG triggering writing mode + MATLAB access script were added.)
  (Faulhaber rotation direction & Zaber microstepsize were added to configuration.)
  (ECG triggering delay rate widget and delaying recording were added.)
- 170808 Havana2 v1.1.1 for Mac (code was modified.)
- 170915 Havana2 v1.2.0
  (Writing thread and its synchronization buffering were modified.)
  (Median filtering for circ images was added.)
- 171012 Havana2 v1.2.1
  (ECG view was modified. - peak position)
  (Software stability)
- 171024 Havana2 v1.2.1.1
  (Minor error correction) 
- 171107 Havana2 v1.2.2
  (Single frame processing version)
  (NIRF viewer was added.)
  (Median filtering for circ images was removed.)
  (Minor error correction)
- Havana2 v1.2.2.1
  (Matlab script for FLIM processing was modified.)
- Havana2 v1.2.2.2
  (GalvoScan was updated. - 2 axis scan / fast scan adjustment)
- Havana2 v1.2.2.3
  (Matlab script for FLIM processing was modified.)
- Havana2 v1.2.2.4
  (Configuration is categorized.)
- Havana2 v1.2.3
  (NIRF result view is modified.)
- Havana2 v1.2.3.1
  (Minor error modification & commit for microOCT version development)
- Havana2 v1.2.3.2
  (OCT intensity histogram)



To be updated:
- Synchronized NIRF emission aquisition using NIDAQ analog input
- Synchronized start for galvano mirror