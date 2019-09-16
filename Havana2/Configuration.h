#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#define VERSION						"1.2.6.1"

#define POWER_2(x)					(1 << x)
#define NEAR_2_POWER(x)				(int)(1 << (int)ceil(log2(x)))

///////////////////// Library enabling //////////////////////
#define PX14_ENABLE                 false
#define NI_ENABLE					false

/////////////////////// System setup ////////////////////////
#define OCT_FLIM
//#define STANDALONE_OCT

#ifdef STANDALONE_OCT
///#define DUAL_CHANNEL // in the Streaming tab.. but it is not supported yet...
#define OCT_NIRF // NIRF data can be loaded in the Result tab.
#endif

#if defined(STANDALONE_OCT) && defined(OCT_FLIM)
#error("STANDALONE_OCT and OCT_FLIM cannot be defined at the same time.");
#endif

#ifndef OCT_NIRF
#if NI_ENABLE
//#define ECG_TRIGGERING
#endif
#else
#define PROGRAMMATIC_GAIN_CONTROL
//#define TWO_CHANNEL_NIRF
//#define NI_NIRF_SYNC
#endif
#define GALVANO_MIRROR
//#define PULLBACK_DEVICE


////////////////////// Digitizer setup //////////////////////
#if PX14_ENABLE
#define ADC_RATE					340 // MHz

#define DIGITIZER_VOLTAGE			0.220
#define DIGITIZER_VOLTAGE_RATIO		1.122018
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
#define ALINE_RATE					51200

#define NI_NIRF_TRIGGER_SOURCE		"/Dev1/PFI8" // "/Dev3/PFI0"
#ifndef TWO_CHANNEL_NIRF
#define NI_NIRF_EMISSION_CHANNEL	"Dev1/ai0" // "Dev3/ai7"
#else
#define NI_NIRF_EMISSION_CHANNEL	"Dev1/ai0, Dev1/ai7" // "Dev3/ai0"
#endif
#define NI_NIRF_ALINES_COUNTER		"Dev1/ctr0" // 12  "Dev3/ctr0" // ctr0,1,2,3 => PFI12,13,14,15
#ifdef NI_NIRF_SYNC
#define NI_NIRF_CTR_EQV_PORT		"/Dev1/port2/line4"
#endif
#define NI_NIRF_ALINES_SOURCE		"/Dev1/PFI8" // "/Dev3/PFI1"

#ifdef PROGRAMMATIC_GAIN_CONTROL
#ifndef TWO_CHANNEL_NIRF
#define NI_PMT_GAIN_CHANNEL		    "Dev1/ao0" //ch1 -> ao0 / ch2 -> ao1
#else
#define NI_PMT_GAIN_CHANNEL		    "Dev1/ao0:1"
#endif
#endif

#ifdef NI_NIRF_SYNC
#define NI_NIRF_SYNC_PORT			"/Dev1/port0/line0:1" // power & reset
#endif
#endif

#ifdef GALVANO_MIRROR
#define NI_GALVO_CHANNEL			"Dev1/ao0:1"
#define NI_GAVLO_SOURCE				"/Dev1/PFI13"
#endif

#ifdef PULLBACK_DEVICE
#define ZABER_PORT					"COM6"
#define ZABER_MAX_MICRO_RESOLUTION  64
#define ZABER_MICRO_RESOLUTION		32 
#define ZABER_CONVERSION_FACTOR		1.6384 //1.0 / 9.375 // BENCHTOP_MODE ? 1.0 / 9.375 : 1.6384;
#define ZABER_MICRO_STEPSIZE		0.09921875 // 0.49609375 // micro-meter ///
#define ZABER_HOME_OFFSET			0.0

#define FAULHABER_NEW_CONTROLLER
#define FAULHABER_PORT				"COM8"
#define FAULHABER_POSITIVE_ROTATION false
#endif

//////////////////////// Processing /////////////////////////
#define DATA_HALVING				false // to be updated...

#define PROCESSING_BUFFER_SIZE		20

#ifdef _DEBUG
#define WRITING_BUFFER_SIZE			200
#else
#define WRITING_BUFFER_SIZE	        1000
#endif

//////////////////////// OCT system /////////////////////////
#define DISCOM_VAL					0 

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

#define PIXEL_SIZE					1 // (100.0 / 43.0) // um/px

#ifdef OCT_FLIM
#define INTENSITY_COLORTABLE		6 // fire
#endif

#ifdef OCT_NIRF
#define NIRF_COLORTABLE1			5 // hot
#ifdef TWO_CHANNEL_NIRF
#define NIRF_COLORTABLE2			18 // cyan

//#define CH_DIVIDING_LINE
#endif

#endif

#define RENEWAL_COUNT				5






template <typename T>
struct Range
{
	T min = 0;
	T max = 0;
};

enum voltage_range
{
	v0_220 = 1, v0_247, v0_277, v0_311, v0_349,
	v0_391, v0_439, v0_493, v0_553, v0_620,
	v0_696, v0_781, v0_876, v0_983, v1_103,
	v1_237, v1_388, v1_557, v1_748, v1_961,
	v2_200, v2_468, v2_770, v3_108, v3_487
};


#include <QString>
#include <QSettings>
#include <QDateTime>

class Configuration
{
public:
	explicit Configuration() : nChannels(0), systemType(""), erasmus(false)		
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
		bootTimeBufferIndex = settings.value("bootTimeBufferIndex").toInt();
		ch1VoltageRange = settings.value("ch1VoltageRange").toInt();
		ch2VoltageRange = settings.value("ch2VoltageRange").toInt();
		preTrigSamps = settings.value("preTrigSamps").toInt();

		nChannels = settings.value("nChannels").toInt(); if (nChannels == 0) nChannels = 2;
		nScans = settings.value("nScans").toInt();
		fnScans = nScans * 4;
		nScansFFT = NEAR_2_POWER((double)nScans);
		n2ScansFFT = nScansFFT / 2;
		nAlines = settings.value("nAlines").toInt();
		n4Alines = nAlines / 4;
		nAlines4 = ((nAlines + 3) >> 2) << 2;
		nFrameSize = nChannels * nScans * nAlines;

		// OCT processing
		octDiscomVal = settings.value("octDiscomVal").toInt();

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
#ifdef OCT_FLIM
		flimLifetimeColorTable = settings.value("flimLifetimeColorTable").toInt();
		flimIntensityRange.max = settings.value("flimIntensityRangeMax").toFloat();
		flimIntensityRange.min = settings.value("flimIntensityRangeMin").toFloat();
		flimLifetimeRange.max = settings.value("flimLifetimeRangeMax").toFloat();
		flimLifetimeRange.min = settings.value("flimLifetimeRangeMin").toFloat();
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
#endif
#endif
		// NIRF distance compensation
#ifdef OCT_NIRF
		nirfCompCoeffs[0] = settings.value("nirfCompCoeffs_a").toFloat();
		nirfCompCoeffs[1] = settings.value("nirfCompCoeffs_b").toFloat();
		nirfCompCoeffs[2] = settings.value("nirfCompCoeffs_c").toFloat();
		nirfCompCoeffs[3] = settings.value("nirfCompCoeffs_d").toFloat();

		nirfFactorThres = settings.value("nirfFactorThres").toFloat();
		nirfFactorPropConst = settings.value("nirfFactorPropConst").toFloat();
		nirfDistPropConst = settings.value("nirfDistPropConst").toFloat();

		nirfLumContourOffset = settings.value("nirfLumContourOffset").toInt();
		nirfOuterSheathPos = settings.value("nirfOuterSheathPos").toInt();
#endif
		// Device control
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
				flimChStartInd[i] = (int)(settings.value(QString("ChannelStart_%1").arg(i)).toFloat() / (1000.0f / (float)ADC_RATE));
#else
				flimChStartInd[i] = (int)(settings.value(QString("ChannelStart_%1").arg(i)).toFloat() / (1000.0f / 340.0f));
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
		settings.setValue("bootTimeBufferIndex", bootTimeBufferIndex);
		settings.setValue("ch1VoltageRange", ch1VoltageRange);
		settings.setValue("ch2VoltageRange", ch2VoltageRange);
		settings.setValue("preTrigSamps", preTrigSamps);

		settings.setValue("nChannels", nChannels);
		settings.setValue("nScans", nScans);
		settings.setValue("nAlines", nAlines);

		// OCT processing
		settings.setValue("octDiscomVal", octDiscomVal);

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
#ifdef OCT_FLIM
		settings.setValue("flimLifetimeColorTable", flimLifetimeColorTable);
		settings.setValue("flimIntensityRangeMax", QString::number(flimIntensityRange.max, 'f', 1)); 
		settings.setValue("flimIntensityRangeMin", QString::number(flimIntensityRange.min, 'f', 1)); 
		settings.setValue("flimLifetimeRangeMax", QString::number(flimLifetimeRange.max, 'f', 1)); 
		settings.setValue("flimLifetimeRangeMin", QString::number(flimLifetimeRange.min, 'f', 1)); 
#endif	
#ifdef OCT_NIRF
#ifndef TWO_CHANNEL_NIRF
		settings.setValue("nirfRangeMax", QString::number(nirfRange.max, 'f', 21));
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
#endif
#endif
		// NIRF distance compensation
#ifdef OCT_NIRF
		settings.setValue("nirfCompCoeffs_a", QString::number(nirfCompCoeffs[0], 'f', 10));
		settings.setValue("nirfCompCoeffs_b", QString::number(nirfCompCoeffs[1], 'f', 10));
		settings.setValue("nirfCompCoeffs_c", QString::number(nirfCompCoeffs[2], 'f', 10));
		settings.setValue("nirfCompCoeffs_d", QString::number(nirfCompCoeffs[3], 'f', 10));
		
		settings.setValue("nirfFactorThres", QString::number(nirfFactorThres, 'f', 1));
		settings.setValue("nirfFactorPropConst", QString::number(nirfFactorPropConst, 'f', 3));
		settings.setValue("nirfDistPropConst", QString::number(nirfDistPropConst, 'f', 3));

		settings.setValue("nirfLumContourOffset", nirfLumContourOffset);
		settings.setValue("nirfOuterSheathPos", nirfOuterSheathPos);
#endif

		// Device control
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
	int bootTimeBufferIndex;
	int ch1VoltageRange, ch2VoltageRange;
	int preTrigSamps;

	int nFrames;
	int nChannels, nScans, nAlines;
	int fnScans;
	int nScansFFT, n2ScansFFT;
	int n4Alines;
	int nAlines4;
	int nFrameSize;
	
	// OCT processing
	int octDiscomVal;

	// FLIM processing
#ifdef OCT_FLIM
	int flimCh;
	float flimBg;
	float flimWidthFactor;
	int flimChStartInd[4];
	float flimDelayOffset[3];
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
#ifdef OCT_FLIM
	int flimLifetimeColorTable;
	Range<float> flimIntensityRange;
	Range<float> flimLifetimeRange;
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
#endif
#endif

#ifdef OCT_NIRF
	// NIRF ditance compensation
	float nirfCompCoeffs[4];
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
