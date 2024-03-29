﻿#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#define VERSION						"1.2.9.0"

#define POWER_2(x)					(1 << x)
#define NEAR_2_POWER(x)				(int)(1 << (int)ceil(log2(x)))

///////////////////// Library enabling //////////////////////
#define PX14_ENABLE                 false						// Oh short cavity 120 kHz
#define ALAZAR_ENABLE               true							// Axsun 200 kHz

#define NI_ENABLE					true

#if PX14_ENABLE && ALAZAR_ENABLE
#error("PX14_ENABLE and ALAZAR_ENABLE cannot be defined at the same time.");
#endif

/////////////////////// System setup ////////////////////////
//#define OCT_FLIM // deprecated
#define STANDALONE_OCT

#ifdef STANDALONE_OCT
///#define DUAL_CHANNEL // in the Streaming tab.. but it is not supported yet...
//#define OCT_NIRF // NIRF data can be loaded in the Result tab.
#endif

#if defined(STANDALONE_OCT) && defined(OCT_FLIM)
#error("STANDALONE_OCT and OCT_FLIM cannot be defined at the same time.");
#endif

#define AXSUN_OCT_LASER  // axsun only
#ifdef AXSUN_OCT_LASER
//#define AXSUN_VDL_K_CLOCK_DELAY
#endif

#ifndef OCT_NIRF
#if NI_ENABLE
//#define ECG_TRIGGERING
#endif
#else
//#define PROGRAMMATIC_GAIN_CONTROL
#if ALAZAR_ENABLE
#define ALAZAR_NIRF_ACQUISITION
#endif
//#define TWO_CHANNEL_NIRF
#endif
#define GALVANO_MIRROR
//#define PULLBACK_DEVICE
#ifdef PULLBACK_DEVICE
//#define DOTTER_STAGE
#endif


////////////////////// Digitizer setup //////////////////////
#if PX14_ENABLE
#define ADC_RATE					340 // MS/sec

#define DIGITIZER_VOLTAGE			0.220
#define DIGITIZER_VOLTAGE_RATIO		1.122018
#elif ALAZAR_ENABLE
#define ADC_RATE                    1000 // MS/sec
#define USE_EXTERNAL_K_CLOCK		false
#endif

/////////////////////// Device setup ////////////////////////
#ifdef ECG_TRIGGERING
#define NI_ECG_TRIGGER_CHANNEL		"Dev1/ctr0"
#define NI_ECG_TRIGGER_SOURCE		"/Dev1/PFI15"
#define NI_ECG_CHANNEL				"Dev1/ai4"
#define ECG_SAMPLING_RATE			1000.0 // Hz
#define N_VIS_SAMPS_ECG				2000 // N_VIS_SAMPS_ECG / ECG_SAMPLING_RATE = time span
#define ECG_VOLTAGE					0.10 // peak-to-peak voltage for analog input
#define ECG_THRES_VALUE				0.12 // volt
#define ECG_THRES_TIME				500 // millisecond
#define ECG_VIEW_RENEWAL_COUNT		50
#define NI_800RPS_CHANNEL			"Dev1/ao1"
#endif

#ifdef OCT_FLIM
#define NI_PMT_GAIN_CHANNEL		    "Dev1/ao2"
#define NI_FLIM_SYNC_CHANNEL		"Dev1/ctr0"
#define NI_FLIM_SYNC_SOURCE			"/Dev1/PFI13"

#define ELFORLIGHT_PORT				"COM1"
#endif

#ifdef OCT_NIRF
#define ALINE_RATE					200500 // approximated value (should be larger than the actual value)
#ifndef ALAZAR_NIRF_ACQUISITION
// It may be deprecated in the next release... //
///#define NI_NIRF_TRIGGER_SOURCE		"/Dev2/PFI0" // "/Dev3/PFI0"
///#ifndef TWO_CHANNEL_NIRF
///#define NI_NIRF_EMISSION_CHANNEL	"Dev2/ai0" // "Dev3/ai7"
///#else
///#define NI_NIRF_EMISSION_CHANNEL	"Dev1/ai0, Dev1/ai7" // "Dev3/ai0"
///#endif
///#define NI_NIRF_ALINES_COUNTER		"Dev1/ctr0" // 12  "Dev3/ctr0" // ctr0,1,2,3 => PFI12,13,14,15
///#define NI_NIRF_ALINES_SOURCE		"/Dev1/PFI10" // "/Dev3/PFI1"
#else
#define NIRF_SCANS					240

#ifdef TWO_CHANNEL_NIRF

#define MODULATION_FREQ				4

#define NI_NIRF_MODUL_CHANNEL		"Dev1/ao2:3"
#define NI_NIRF_MODUL_SOURCE		"/Dev1/PFI12"
#endif
#endif

#ifdef PROGRAMMATIC_GAIN_CONTROL
#ifndef TWO_CHANNEL_NIRF
#define NI_PMT_GAIN_CHANNEL		    "Dev1/ao0" //ch1 -> ao0 / ch2 -> ao1
#else
#define NI_PMT_GAIN_CHANNEL		    "Dev1/ao0:1"
#endif
#endif

#endif

#ifdef GALVANO_MIRROR
#define NI_GALVO_CHANNEL			"Dev1/ao0:1"
#define NI_GAVLO_SOURCE				"/Dev1/PFI8"			 // Axsun 8  Short-cavity 12
#endif

#ifdef PULLBACK_DEVICE
//#define ZABER_NEW_STAGE				
#define ZABER_PORT					"COM6"
#define ZABER_MAX_MICRO_RESOLUTION  64
#define ZABER_MICRO_RESOLUTION		32
#define ZABER_CONVERSION_FACTOR		1.0 / 9.375 // BENCHTOP_MODE ? 1.0 / 9.375 : 1.6384;
#define ZABER_MICRO_STEPSIZE		0.49609375 // micro-meter ///0.09921875
#define ZABER_HOME_OFFSET			0.0

#define FAULHABER_NEW_CONTROLLER
#define FAULHABER_PORT				"COM4"
#define FAULHABER_POSITIVE_ROTATION false
#endif

//////////////////////// Processing /////////////////////////
#define CUDA_ENABLED				// Only valid in visual studio environment

#ifdef CUDA_ENABLED
#define N_CUDA_THREADS				32
#define N_CUDA_STREAMS				4
#define N_CUDA_PARTITIONS			4
#endif

//#define FREQ_SHIFTING				// short cavity only
//#define K_CLOCKING

//#define OCT_VERTICAL_MIRRORING

///#define DATA_HALVING				 // to be updated... 

#define PROCESSING_BUFFER_SIZE		20

#ifdef _DEBUG
#define WRITING_BUFFER_SIZE			200
#else
#define WRITING_BUFFER_SIZE	        800
#endif

#define	RECORDING_SKIP_FRAMES		1

/////////////////////// FLIM system /////////////////////////
#ifdef OCT_FLIM
#define FLIM_CH_START_5				30
#define GAUSSIAN_FILTER_WIDTH		200
#define GAUSSIAN_FILTER_STD			48
#define FLIM_SPLINE_FACTOR			20
#define INTENSITY_THRES				0.001f
#endif

/////////////////////// Visualization ///////////////////////
#ifdef OCT_FLIM
#define N_VIS_SAMPS_FLIM			200
#endif

#define PIXEL_SIZE					4.4 // (100.0 / 43.0) // um/px

#ifdef OCT_FLIM
#define INTENSITY_COLORTABLE		6 // fire
#endif

#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
#define NIRF_COLORTABLE1			5 // hot
#else
#define NIRF_COLORTABLE1			5 // hot
#define NIRF_COLORTABLE2			18 // cyan
#define CH_DIVIDING_LINE
#endif
#endif

#define RENEWAL_COUNT				11






template <typename T>
struct Range
{
    T min = 0;
    T max = 0;
};

#if PX14_ENABLE
enum voltage_range
{
    v0_220 = 1, v0_247, v0_277, v0_311, v0_349,
    v0_391, v0_439, v0_493, v0_553, v0_620,
    v0_696, v0_781, v0_876, v0_983, v1_103,
    v1_237, v1_388, v1_557, v1_748, v1_961,
    v2_200, v2_468, v2_770, v3_108, v3_487
};
#elif ALAZAR_ENABLE
enum voltage_range
{
    v0_002 = 1, v0_005, v0_01, v0_02, v0_04,
    v0_05, v0_1, v0_2, v0_5, v1, v2, v5, v10, v20
};
#endif


#include <QString>
#include <QSettings>
#include <QDateTime>

class Configuration
{
public:
    explicit Configuration() : nChannels(0), systemType(""), erasmus(false), octDbGamma(1.0f)
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
    , nirf(false)
#ifdef TWO_CHANNEL_NIRF
    , _2ch_nirf(false)
#endif
#endif
    , oldUhs(false)
#endif
    {}
    ~Configuration() {}

public:
    void getConfigFile(const QString& inipath) // const char* inipath)
    {
        QSettings settings(inipath, QSettings::IniFormat);
        settings.beginGroup("configuration");

        // Digitizer setup
#if PX14_ENABLE
        bootTimeBufferIndex = settings.value("bootTimeBufferIndex").toInt();
#endif
        ch1VoltageRange = settings.value("ch1VoltageRange").toInt();
        ch2VoltageRange = settings.value("ch2VoltageRange").toInt();
#if PX14_ENABLE | defined(OCT_FLIM)
        preTrigSamps = settings.value("preTrigSamps").toInt();
#endif
#if ALAZAR_ENABLE
        triggerDelay = settings.value("triggerDelay").toInt();
#endif

        nChannels = settings.value("nChannels").toInt(); if (nChannels == 0) nChannels = 2;
        nScans = settings.value("nScans").toInt();
        fnScans = nScans * 4;
        nScansFFT = NEAR_2_POWER((float)nScans);
        n2ScansFFT = nScansFFT / 2;
        nAlines = settings.value("nAlines").toInt();
        n4Alines = nAlines / 4;
        nAlines4 = ((nAlines + 3) >> 2) << 2;
        n4Alines4 = ((n4Alines + 3) >> 2) << 2;
        nFrameSize = nChannels * nScans * nAlines;

        // OCT processing
        octDiscomVal = settings.value("octDiscomVal").toInt();
		
		// SOCT processing
		spectroDbRange.max = settings.value("spectroDbRangeMax").toFloat();
		spectroDbRange.min = settings.value("spectroDbRangeMin").toFloat();
		spectroWindow = settings.value("spectroWindow").toInt();
		spectroOverlap = settings.value("spectroOverlap").toInt();
		spectroInPlaneAvgSize = settings.value("spectroInPlaneAvgSize").toInt();
		spectroOutOfPlaneAvgSize = settings.value("spectroOutOfPlaneAvgSize").toInt();
		spectroRoiDepth = settings.value("spectroRoiDepth").toInt();

#ifdef OCT_FLIM
        // FLIM processing
        flimCh = settings.value("flimCh").toInt();
        flimBg = settings.value("flimBg").toFloat();
        flimWidthFactor = settings.value("flimWidthFactor").toFloat();

        for (int i = 0; i < 4; i++)
        {
            flimChStartInd[i] = settings.value(QString("flimChStartInd_%1").arg(i)).toInt();
            if (i != 0)
                flimDelayOffset[i - 1] = settings.value(QString("flimDelayOffset_%1").arg(i)).toFloat();
        }

        // FLIM classification
        clfAnnXNode = settings.value("clfAnnXNode").toInt();
        clfAnnHNode = settings.value("clfAnnHNode").toInt();
        clfAnnYNode = settings.value("clfAnnYNode").toInt();
#endif
        // Visualization
        circCenter = settings.value("circCenter").toInt();
        ballRadius = settings.value("ballRadius").toInt();
        circRadius = settings.value("circRadius").toInt();
        sheathRadius = settings.value("sheathRadius").toInt();
#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
        ringThickness = settings.value("ringThickness").toInt();
#endif
        octColorTable = settings.value("octColorTable").toInt();
        octDbRange.max = settings.value("octDbRangeMax").toInt();
        octDbRange.min = settings.value("octDbRangeMin").toInt();
        octDbGamma = settings.value("octDbGamma").toFloat();
#ifdef OCT_FLIM
        flimLifetimeColorTable = settings.value("flimLifetimeColorTable").toInt();
        flimIntensityRange.max = settings.value("flimIntensityRangeMax").toFloat();
        flimIntensityRange.min = settings.value("flimIntensityRangeMin").toFloat();
        flimLifetimeRange.max = settings.value("flimLifetimeRangeMax").toFloat();
        flimLifetimeRange.min = settings.value("flimLifetimeRangeMin").toFloat();

        for (int i = 0; i < 3; i++)
        {
            float temp = settings.value(QString("flimIntensityComp_%1").arg(i + 1)).toFloat();
            flimIntensityComp[i] = (temp == 0.0f) ? 1.0f : temp;
        }
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
        nirfRange.max = settings.value("nirfRangeMax").toFloat();
        nirfRange.min = settings.value("nirfRangeMin").toFloat();
#else
        nirfRange[0].max = settings.value("nirfRange1Max").toFloat();
        nirfRange[0].min = settings.value("nirfRange1Min").toFloat();
        nirfRange[1].max = settings.value("nirfRange2Max").toFloat();
        nirfRange[1].min = settings.value("nirfRange2Min").toFloat();
#endif
#endif
        // 2 Ch NIRF cross-talk compensation
#ifdef OCT_NIRF
#ifdef TWO_CHANNEL_NIRF
        nirfCrossTalkRatio = settings.value("nirfCrossTalkRatio").toFloat();

		// 2 Ch integration window
		nirfIntegWindow[0].max = settings.value("nirfIntegWindow1Max").toInt();
		nirfIntegWindow[0].min = settings.value("nirfIntegWindow1Min").toInt();
		nirfIntegWindow[1].max = settings.value("nirfIntegWindow2Max").toInt();
		nirfIntegWindow[1].min = settings.value("nirfIntegWindow2Min").toInt();
#endif
#endif
        // NIRF distance compensation
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
        nirfCompCoeffs_a = settings.value("nirfCompCoeffs_a").toFloat();
        nirfCompCoeffs_b = settings.value("nirfCompCoeffs_b").toFloat();
        nirfCompCoeffs_c = settings.value("nirfCompCoeffs_c").toFloat();
        nirfCompCoeffs_d = settings.value("nirfCompCoeffs_d").toFloat();
#else
		nirfCompCoeffs_a[0] = settings.value("nirfCompCoeffs_a1").toFloat();
		nirfCompCoeffs_b[0] = settings.value("nirfCompCoeffs_b1").toFloat();
		nirfCompCoeffs_c[0] = settings.value("nirfCompCoeffs_c1").toFloat();
		nirfCompCoeffs_d[0] = settings.value("nirfCompCoeffs_d1").toFloat();

		nirfCompCoeffs_a[1] = settings.value("nirfCompCoeffs_a2").toFloat();
		nirfCompCoeffs_b[1] = settings.value("nirfCompCoeffs_b2").toFloat();
		nirfCompCoeffs_c[1] = settings.value("nirfCompCoeffs_c2").toFloat();
		nirfCompCoeffs_d[1] = settings.value("nirfCompCoeffs_d2").toFloat();
#endif

        nirfFactorThres = settings.value("nirfFactorThres").toFloat();
        nirfFactorPropConst = settings.value("nirfFactorPropConst").toFloat();
        nirfDistPropConst = settings.value("nirfDistPropConst").toFloat();

        nirfLumContourOffset = settings.value("nirfLumContourOffset").toInt();
        nirfOuterSheathPos = settings.value("nirfOuterSheathPos").toInt();
#endif
        // Device control
#ifdef AXSUN_OCT_LASER
		axsunVDLLength = settings.value("axsunVDLLength").toFloat();
		axsunkClockDelay = settings.value("axsunkClockDelay").toInt();
#endif
#ifdef ECG_TRIGGERING
        ecgDelayRate = settings.value("ecgDelayRate").toFloat();
#endif
#if defined(OCT_FLIM) || defined(PROGRAMMATIC_GAIN_CONTROL)
#ifndef TWO_CHANNEL_NIRF
        pmtGainVoltage = settings.value("pmtGainVoltage").toFloat();
#else
        pmtGainVoltage[0] = settings.value("pmtGainVoltage1").toFloat();
        pmtGainVoltage[1] = settings.value("pmtGainVoltage2").toFloat();
#endif
#endif
#ifdef GALVANO_MIRROR
        galvoFastScanVoltage = settings.value("galvoFastScanVoltage").toFloat();
        galvoFastScanVoltageOffset = settings.value("galvoFastScanVoltageOffset").toFloat();
        galvoSlowScanVoltage = settings.value("galvoSlowScanVoltage").toFloat();
        galvoSlowScanVoltageOffset = settings.value("galvoSlowScanVoltageOffset").toFloat();
        galvoSlowScanIncrement = settings.value("galvoSlowScanIncrement").toFloat();
        galvoHorizontalShift = 0;
#endif
#ifdef PULLBACK_DEVICE
        zaberPullbackSpeed = settings.value("zaberPullbackSpeed").toInt();
        zaberPullbackLength = settings.value("zaberPullbackLength").toInt();
        faulhaberRpm = settings.value("faulhaberRpm").toInt();
#endif
        // System type
        systemType = settings.value("system").toString(); if (systemType == "") systemType = "OCT-FLIM";

#ifdef OCT_FLIM
        // Havana1 (MFC ver.) support
        if (settings.contains("DiscomValue"))
            octDiscomVal = settings.value("DiscomValue").toInt();
        if (settings.contains("FLIMwidth_factor"))
            flimWidthFactor = settings.value("FLIMwidth_factor").toFloat();
        for (int i = 0; i < 4; i++)
        {
            if (settings.contains(QString("ChannelStart_%1").arg(i)))
#if PX14_ENABLE
                flimChStartInd[i] = (int)round(settings.value(QString("ChannelStart_%1").arg(i)).toFloat() / (1000.0f / (float)ADC_RATE));
#else
                flimChStartInd[i] = (int)round(settings.value(QString("ChannelStart_%1").arg(i)).toFloat() / (1000.0f / 340.0f));
#endif

            if (i != 0)
                if (settings.contains(QString("DelayTimeOffset_%1").arg(i)))
                    flimDelayOffset[i - 1] = settings.value(QString("DelayTimeOffset_%1").arg(i)).toFloat();
        }
#endif
        // Erasmus support
        if (settings.contains("processType"))
            erasmus = true;

        // Old UHS data support
#ifdef STANDALONE_OCT
        if (settings.contains("OldUHS"))
            oldUhs = true;
#endif

        settings.endGroup();
    }

    void setConfigFile(const QString& inipath)
    {
        QSettings settings(inipath, QSettings::IniFormat);
        settings.beginGroup("configuration");

        // Digitizer setup
#if PX14_ENABLE
        settings.setValue("bootTimeBufferIndex", bootTimeBufferIndex);
#endif
        settings.setValue("ch1VoltageRange", ch1VoltageRange);
        settings.setValue("ch2VoltageRange", ch2VoltageRange);
#if PX14_ENABLE | defined(OCT_FLIM)
        settings.setValue("preTrigSamps", preTrigSamps);
#endif
#if ALAZAR_ENABLE
        settings.setValue("triggerDelay", triggerDelay);
#endif

        settings.setValue("nChannels", nChannels);
        settings.setValue("nScans", nScans);
        settings.setValue("nAlines", nAlines);

        // OCT processing
        settings.setValue("octDiscomVal", octDiscomVal);

		// SOCT processing
		settings.setValue("spectroDbRangeMax", spectroDbRange.max);
		settings.setValue("spectroDbRangeMin", spectroDbRange.min);
		settings.setValue("spectroWindow", spectroWindow);
		settings.setValue("spectroOverlap", spectroOverlap);
		settings.setValue("spectroInPlaneAvgSize", spectroInPlaneAvgSize);
		settings.setValue("spectroOutOfPlaneAvgSize", spectroOutOfPlaneAvgSize);
		settings.setValue("spectroRoiDepth", spectroRoiDepth);
		
#ifdef OCT_FLIM
        // FLIM processing
        settings.setValue("flimCh", flimCh);
        settings.setValue("flimBg", QString::number(flimBg, 'f', 2));
        settings.setValue("flimWidthFactor", QString::number(flimWidthFactor, 'f', 2));

        for (int i = 0; i < 4; i++)
        {
            settings.setValue(QString("flimChStartInd_%1").arg(i), flimChStartInd[i]);
            if (i != 0)
                settings.setValue(QString("flimDelayOffset_%1").arg(i), QString::number(flimDelayOffset[i - 1], 'f', 3));
        }
#endif
        // Visualization
        settings.setValue("circCenter", circCenter);
        settings.setValue("ballRadius", ballRadius);
        settings.setValue("circRadius", circRadius);
        settings.setValue("sheathRadius", sheathRadius);
#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
        settings.setValue("ringThickness", ringThickness);
#endif
        settings.setValue("octColorTable", octColorTable);
        settings.setValue("octDbRangeMax", octDbRange.max);
        settings.setValue("octDbRangeMin", octDbRange.min);
        settings.setValue("octDbGamma", QString::number(octDbGamma, 'f', 2));
#ifdef OCT_FLIM
        settings.setValue("flimLifetimeColorTable", flimLifetimeColorTable);
        settings.setValue("flimIntensityRangeMax", QString::number(flimIntensityRange.max, 'f', 1));
        settings.setValue("flimIntensityRangeMin", QString::number(flimIntensityRange.min, 'f', 1));
        settings.setValue("flimLifetimeRangeMax", QString::number(flimLifetimeRange.max, 'f', 1));
        settings.setValue("flimLifetimeRangeMin", QString::number(flimLifetimeRange.min, 'f', 1));
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
        settings.setValue("nirfRangeMax", QString::number(nirfRange.max, 'f', 2));
        settings.setValue("nirfRangeMin", QString::number(nirfRange.min, 'f', 2));
#else
        settings.setValue("nirfRange1Max", QString::number(nirfRange[0].max, 'f', 2));
        settings.setValue("nirfRange1Min", QString::number(nirfRange[0].min, 'f', 2));
        settings.setValue("nirfRange2Max", QString::number(nirfRange[1].max, 'f', 2));
        settings.setValue("nirfRange2Min", QString::number(nirfRange[1].min, 'f', 2));
#endif
#endif
        // 2 Ch NIRF cross-talk compensation
#ifdef OCT_NIRF
#ifdef TWO_CHANNEL_NIRF
        settings.setValue("nirfCrossTalkRatio", QString::number(nirfCrossTalkRatio, 'f', 4));

		// 2 Ch integration window
		settings.setValue("nirfIntegWindow1Max", nirfIntegWindow[0].max);
		settings.setValue("nirfIntegWindow1Min", nirfIntegWindow[0].min);
		settings.setValue("nirfIntegWindow2Max", nirfIntegWindow[1].max);
		settings.setValue("nirfIntegWindow2Min", nirfIntegWindow[1].min);		
#endif
#endif
        // NIRF distance compensation
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
        settings.setValue("nirfCompCoeffs_a", QString::number(nirfCompCoeffs_a, 'f', 10));
        settings.setValue("nirfCompCoeffs_b", QString::number(nirfCompCoeffs_b, 'f', 10));
        settings.setValue("nirfCompCoeffs_c", QString::number(nirfCompCoeffs_c, 'f', 10));
        settings.setValue("nirfCompCoeffs_d", QString::number(nirfCompCoeffs_d, 'f', 10));
#else
		settings.setValue("nirfCompCoeffs_a1", QString::number(nirfCompCoeffs_a[0], 'f', 10));
		settings.setValue("nirfCompCoeffs_b1", QString::number(nirfCompCoeffs_b[0], 'f', 10));
		settings.setValue("nirfCompCoeffs_c1", QString::number(nirfCompCoeffs_c[0], 'f', 10));
		settings.setValue("nirfCompCoeffs_d1", QString::number(nirfCompCoeffs_d[0], 'f', 10));

		settings.setValue("nirfCompCoeffs_a2", QString::number(nirfCompCoeffs_a[1], 'f', 10));
		settings.setValue("nirfCompCoeffs_b2", QString::number(nirfCompCoeffs_b[1], 'f', 10));
		settings.setValue("nirfCompCoeffs_c2", QString::number(nirfCompCoeffs_c[1], 'f', 10));
		settings.setValue("nirfCompCoeffs_d2", QString::number(nirfCompCoeffs_d[1], 'f', 10));
#endif

        settings.setValue("nirfFactorThres", QString::number(nirfFactorThres, 'f', 1));
        settings.setValue("nirfFactorPropConst", QString::number(nirfFactorPropConst, 'f', 3));
        settings.setValue("nirfDistPropConst", QString::number(nirfDistPropConst, 'f', 3));

        settings.setValue("nirfLumContourOffset", nirfLumContourOffset);
        settings.setValue("nirfOuterSheathPos", nirfOuterSheathPos);
#endif

        // Device control
#ifdef AXSUN_OCT_LASER
		settings.setValue("axsunVDLLength", QString::number(axsunVDLLength, 'f', 2));
		settings.setValue("axsunkClockDelay", axsunkClockDelay);
#endif
#ifdef ECG_TRIGGERING
        settings.setValue("ecgDelayRate", QString::number(ecgDelayRate, 'f', 2));
        settings.setValue("ecgHeartRate", QString::number(ecgHeartRate, 'f', 2));
#endif
#if defined(OCT_FLIM) || defined(PROGRAMMATIC_GAIN_CONTROL)
#ifndef TWO_CHANNEL_NIRF
        settings.setValue("pmtGainVoltage", QString::number(pmtGainVoltage, 'f', 3));
#else
        settings.setValue("pmtGainVoltage1", QString::number(pmtGainVoltage[0], 'f', 3));
        settings.setValue("pmtGainVoltage2", QString::number(pmtGainVoltage[1], 'f', 3));
#endif
#endif
#ifdef GALVANO_MIRROR
        settings.setValue("galvoFastScanVoltage", QString::number(galvoFastScanVoltage, 'f', 1));
        settings.setValue("galvoFastScanVoltageOffset", QString::number(galvoFastScanVoltageOffset, 'f', 1));
        settings.setValue("galvoSlowScanVoltage", QString::number(galvoSlowScanVoltage, 'f', 1));
        settings.setValue("galvoSlowScanVoltageOffset", QString::number(galvoSlowScanVoltageOffset, 'f', 1));
        settings.setValue("galvoSlowScanIncrement", QString::number(galvoSlowScanIncrement, 'f', 3));
        settings.setValue("galvoHorizontalShift", galvoHorizontalShift);
#endif
#ifdef PULLBACK_DEVICE
        settings.setValue("zaberPullbackSpeed", zaberPullbackSpeed);
        settings.setValue("zaberPullbackLength", zaberPullbackLength);
        settings.setValue("faulhaberRpm", faulhaberRpm);
#endif
        // System type
#ifdef OCT_FLIM
        settings.setValue("system", "OCT-FLIM");
#elif defined (STANDALONE_OCT)
#ifndef OCT_NIRF
        settings.setValue("system", "Standalone OCT");
#else
#ifndef TWO_CHANNEL_NIRF
        settings.setValue("system", "OCT-NIRF");
#else
        settings.setValue("system", "2Ch OCT-NIRF");
#endif
#endif
#endif
        // Current Time
        QDate date = QDate::currentDate();
        QTime time = QTime::currentTime();
        settings.setValue("time", QString("%1-%2-%3 %4-%5-%6")
            .arg(date.year()).arg(date.month(), 2, 10, (QChar)'0').arg(date.day(), 2, 10, (QChar)'0')
            .arg(time.hour(), 2, 10, (QChar)'0').arg(time.minute(), 2, 10, (QChar)'0').arg(time.second(), 2, 10, (QChar)'0'));


        settings.endGroup();
    }

public:
    // Digitizer setup
#if PX14_ENABLE
    int bootTimeBufferIndex;
#endif
    int ch1VoltageRange, ch2VoltageRange;
#if ALAZAR_ENABLE
    double voltRange[14] = { 0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0 };
#endif
#if PX14_ENABLE | defined(OCT_FLIM)
    int preTrigSamps;
#endif
#if ALAZAR_ENABLE
    int triggerDelay;
#endif

    int nFrames;
    int nChannels, nScans, nAlines;
    int fnScans;
    int nScansFFT, n2ScansFFT;
    int n4Alines;
    int nAlines4;
    int n4Alines4;
    int nFrameSize;

    // OCT processing
    int octDiscomVal;

	// SOCT processing
	Range<float> spectroDbRange;
	int spectroWindow;
	int spectroOverlap;
	int spectroInPlaneAvgSize;
	int spectroOutOfPlaneAvgSize;
	int spectroRoiDepth;

    // FLIM processing
#ifdef OCT_FLIM
    int flimCh;
    float flimBg;
    float flimWidthFactor;
    int flimChStartInd[4];
    float flimDelayOffset[3];
#endif

    // FLIM classification
#ifdef OCT_FLIM
    int clfAnnXNode;
    int clfAnnHNode;
    int clfAnnYNode;
#endif

    // Visualization
    int circCenter;
    int ballRadius;
    int circRadius;
    int sheathRadius;
#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
    int ringThickness;
#endif
    int octColorTable;
    Range<int> octDbRange;
    float octDbGamma;
#ifdef OCT_FLIM
    int flimLifetimeColorTable;
    Range<float> flimIntensityRange;
    Range<float> flimLifetimeRange;
    float flimIntensityComp[3];
#endif
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
    Range<float> nirfRange;
#else
    Range<float> nirfRange[2];
#endif
#endif

#ifdef OCT_NIRF
#ifdef TWO_CHANNEL_NIRF
    // 2 Ch NIRF cross-talk compensation
    float nirfCrossTalkRatio;

	// 2 Ch integration window
	Range<int> nirfIntegWindow[2];
#endif
#endif

#ifdef OCT_NIRF
    // NIRF ditance compensation
#ifndef TWO_CHANNEL_NIRF
    float nirfCompCoeffs_a;
	float nirfCompCoeffs_b;
	float nirfCompCoeffs_c;
	float nirfCompCoeffs_d;
#else
	float nirfCompCoeffs_a[2];
	float nirfCompCoeffs_b[2];
	float nirfCompCoeffs_c[2];
	float nirfCompCoeffs_d[2];
#endif
    float nirfFactorThres;
    float nirfFactorPropConst;
    float nirfDistPropConst;
    int nirfLumContourOffset;
    int nirfOuterSheathPos;
#endif

    // System type
    QString systemType;
    bool erasmus;
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
    bool nirf;
#ifdef TWO_CHANNEL_NIRF
    bool _2ch_nirf;
#endif
#endif
    bool oldUhs;
#endif

    // Device control
#ifdef AXSUN_OCT_LASER
	float axsunVDLLength;
	int axsunkClockDelay;
#endif
#ifdef ECG_TRIGGERING
    float ecgDelayRate;
    float ecgHeartRate;
#endif
#if defined(OCT_FLIM) || defined(PROGRAMMATIC_GAIN_CONTROL)
#ifndef TWO_CHANNEL_NIRF
    float pmtGainVoltage;
#else
    float pmtGainVoltage[2];
#endif
#endif
#ifdef GALVANO_MIRROR
    float galvoFastScanVoltage;
    float galvoFastScanVoltageOffset;
    float galvoSlowScanVoltage;
    float galvoSlowScanVoltageOffset;
    float galvoSlowScanIncrement;
    int galvoHorizontalShift;
#endif
#ifdef PULLBACK_DEVICE
    int zaberPullbackSpeed;
    int zaberPullbackLength;
    int faulhaberRpm;
#endif
};


#endif // CONFIGURATION_H
