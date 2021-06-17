#-------------------------------------------------
#
# Project created by QtCreator 2017-03-30T15:46:56
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets serialport

TARGET = Havana2
TEMPLATE = app

CONFIG += console
RC_FILE += Havana2.rc

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0



macx {
    INCLUDEPATH += /opt/intel/ipp/include \
                /opt/intel/tbb/include \
                /opt/intel/mkl/include

    IPPLIBS = -lippch -lippcore -lippi -lipps -lippvm
    MKLLIBS = -lmkl_core -lmkl_tbb_thread -lmkl_intel_lp64

    LIBS += -L/opt/intel/ipp/lib $$IPPLIBS
    debug {
        LIBS += -L/opt/intel/tbb/lib -ltbb_debug
    }
    release {
        LIBS += -L/opt/intel/tbb/lib -ltbb_debug \
                -L/opt/intel/tbb/lib -ltbb
    }
    LIBS += -L/opt/intel/mkl/lib $$MKLLIBS
}
win32 {
    INCLUDEPATH += $$PWD/include

    LIBS += $$PWD/lib/PX14_64.lib \
            $$PWD/lib/ATSApi.lib
    LIBS += $$PWD/lib/NIDAQmx.lib
    LIBS += $$PWD/lib/intel64_win/ippch.lib \
            $$PWD/lib/intel64_win/ippcc.lib \
            $$PWD/lib/intel64_win/ippcore.lib \
            $$PWD/lib/intel64_win/ippi.lib \
            $$PWD/lib/intel64_win/ipps.lib \
            $$PWD/lib/intel64_win/ippvm.lib
    debug {
        LIBS += $$PWD/lib/intel64_win/vc14/tbb_debug.lib \
                $$PWD/lib/opencv_world320d.lib
    }
    release {
        LIBS += $$PWD/lib/intel64_win/vc14/tbb.lib \
                $$PWD/lib/opencv_world320.lib
    }
    LIBS += $$PWD/lib/intel64_win/mkl_core.lib \
            $$PWD/lib/intel64_win/mkl_tbb_thread.lib \
            $$PWD/lib/intel64_win/mkl_intel_lp64.lib
    LIBS += $$PWD/lib/cuda.lib \
            $$PWD/lib/cudart.lib \
            $$PWD/lib/cufft.lib
}


SOURCES += Havana2/Havana2.cpp \
    Havana2/MainWindow.cpp \
    Havana2/QOperationTab.cpp \
    Havana2/QDeviceControlTab.cpp \
    Havana2/QStreamTab.cpp \
    Havana2/QResultTab.cpp \
    Havana2/Viewer/QScope.cpp \
    Havana2/Viewer/QScope2.cpp \
    Havana2/Viewer/QCalibScope.cpp \
    Havana2/Viewer/QEcgScope.cpp \
    Havana2/Viewer/QImageView.cpp \
    Havana2/Dialog/OctCalibDlg.cpp \
    Havana2/Dialog/OctIntensityHistDlg.cpp \
    Havana2/Dialog/LongitudinalViewDlg.cpp \
    Havana2/Dialog/FlimCalibDlg.cpp \
    Havana2/Dialog/DigitizerSetupDlg.cpp \
    Havana2/Dialog/SaveResultDlg.cpp \
    Havana2/Dialog/PulseReviewDlg.cpp \
    Havana2/Dialog/NirfEmissionProfileDlg.cpp \
    Havana2/Dialog/NirfDistCompDlg.cpp \
    Havana2/Dialog/NirfCrossTalkCompDlg.cpp

SOURCES += DataProcess/OCTProcess/OCTProcess.cpp \
    DataProcess/FLIMProcess/FLIMProcess.cpp \
    DataProcess/ThreadManager.cpp

SOURCES += DataAcquisition/SignatecDAQ/SignatecDAQ.cpp \
    DataAcquisition/AlazarDAQ/AlazarDAQ.cpp \
    DataAcquisition/DataAcquisition.cpp

SOURCES += MemoryBuffer/MemoryBuffer.cpp

SOURCES += DeviceControl/AxsunControl/AxsunControl.cpp \
    DeviceControl/FLIMControl/PmtGainControl.cpp \
    DeviceControl/FLIMControl/SyncFLIM.cpp \
    DeviceControl/FLIMControl/ElforlightLaser.cpp \
    DeviceControl/ECGMonitoring/EcgMonitoring.cpp \
    DeviceControl/ECGMonitoring/EcgMonitoringTrigger.cpp \
    DeviceControl/ECGMonitoring/Voltage800RPS.cpp \
    DeviceControl/NirfEmission/NirfEmission.cpp \
    DeviceControl/NirfEmission/NirfEmissionTrigger.cpp \
    DeviceControl/GalvoScan/GalvoScan.cpp \
    DeviceControl/ZaberStage/ZaberStage.cpp \
    DeviceControl/ZaberStage/zb_serial.cpp \
    DeviceControl/ZaberStage/ZaberStage2.cpp \
    DeviceControl/FaulhaberMotor/FaulhaberMotor.cpp


HEADERS += Havana2/Configuration.h \
    Havana2/MainWindow.h \
    Havana2/QOperationTab.h \
    Havana2/QDeviceControlTab.h \
    Havana2/QStreamTab.h \
    Havana2/QResultTab.h \
    Havana2/Viewer/QScope.h \
    Havana2/Viewer/QScope2.h \
    Havana2/Viewer/QCalibScope.h \
    Havana2/Viewer/QEcgScope.h \
    Havana2/Viewer/QImageView.h \
    Havana2/Dialog/OctCalibDlg.h \
    Havana2/Dialog/OctIntensityHistDlg.h \
    Havana2/Dialog/LongitudinalViewDlg.h \
    Havana2/Dialog/FlimCalibDlg.h \
    Havana2/Dialog/DigitizerSetupDlg.h \
    Havana2/Dialog/SaveResultDlg.h \
    Havana2/Dialog/PulseReviewDlg.h \
    Havana2/Dialog/NirfEmissionProfileDlg.h \
    Havana2/Dialog/NirfDistCompDlg.h \
    Havana2/Dialog/NirfCrossTalkCompDlg.h

HEADERS += DataProcess/OCTProcess/OCTProcess.h \
    DataProcess/FLIMProcess/FLIMProcess.h \
    DataProcess/ThreadManager.h \

HEADERS += DataAcquisition/SignatecDAQ/SignatecDAQ.h \
    DataAcquisition/AlazarDAQ/AlazarDAQ.h \
    DataAcquisition/DataAcquisition.h

HEADERS += MemoryBuffer/MemoryBuffer.h

HEADERS += DeviceControl/AxsunControl/AxsunControl.h \
    DeviceControl/FLIMControl/PmtGainControl.h \
    DeviceControl/FLIMControl/SyncFLIM.h \
    DeviceControl/FLIMControl/ElforlightLaser.h \
    DeviceControl/ECGMonitoring/EcgMonitoring.h \
    DeviceControl/ECGMonitoring/EcgMonitoringTrigger.h \
    DeviceControl/ECGMonitoring/Voltage800RPS.h \
    DeviceControl/NirfEmission/NirfEmission.h \
    DeviceControl/NirfEmission/NirfEmissionTrigger.h \
    DeviceControl/GalvoScan/GalvoScan.h \
    DeviceControl/ZaberStage/ZaberStage.h \
    DeviceControl/ZaberStage/zb_serial.h \
    DeviceControl/ZaberStage/ZaberStage2.h \
    DeviceControl/FaulhaberMotor/FaulhaberMotor.h \
    DeviceControl/QSerialComm.h


OTHER_FILES += CUDA/CudaOCTProcess.cuh \
    CUDA/CudaOCTProcess.cu \
    CUDA/CudaCircularize.cuh \
    CUDA/CudaCircularize.cu


FORMS    += Havana2/MainWindow.ui
