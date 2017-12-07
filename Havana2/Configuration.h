#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#define VERSION						"1.2.2.3"

#define POWER_2(x)					(1 << x)
#define NEAR_2_POWER(x)				(int)(1 << (int)ceil(log2(x)))

/////////////////////// System setup ////////////////////////
#define OCT_FLIM
//#define STANDALONE_OCT

#ifdef STANDALONE_OCT
//#define DUAL_CHANNEL // not supported dual channel
#define OCT_NIRF // NIRF data can be loaded in Result pannel.
#endif

#if defined(STANDALONE_OCT) && defined(OCT_FLIM)
#error("STANDALONE_OCT and OCT_FLIM cannot be defined at the same time.");
#endif

#ifndef OCT_NIRF
//#define ECG_TRIGGERING
#endif
#define GALVANO_MIRROR
#define PULLBACK_DEVICE

///////////////////// Library enabling //////////////////////
#define PX14_ENABLE                 true
#define NI_ENABLE					true

////////////////////// Digitizer setup //////////////////////
#define ADC_RATE					340 // MHz

#define DIGITIZER_VOLTAGE			0.220
#define DIGITIZER_VOLTAGE_RATIO		1.122018

/////////////////////// Device setup ////////////////////////
#define NI_ECG_TRIGGER_CHANNEL		"Dev1/ctr1"
#define NI_ECG_TRIG_SOURCE			"/Dev1/PFI15"
#define NI_ECG_CHANNEL				"Dev1/ai4"
#define ECG_SAMPLING_RATE			1000 // Hz
#define N_VIS_SAMPS_ECG				2000 // N_VIS_SAMPS_ECG / ECG_SAMPLING_RATE = time span
#define ECG_VOLTAGE					1 // peak-to-peak voltage for analog input 
#define ECG_THRES_VALUE				0.25 // volt
#define ECG_THRES_TIME				500 // millisecond
#define ECG_VIEW_RENEWAL_COUNT		20
#define NI_800RPS_CHANNEL			"Dev1/ao1"

#define NI_GALVO_CHANNEL			"Dev1/ao0:1"
#define NI_GAVLO_SOURCE				"/Dev1/PFI13"

#define NI_PMT_GAIN_CHANNEL		    "Dev1/ao2"
#define NI_FLIM_SYNC_CHANNEL		"Dev1/ctr0"
#define NI_FLIM_SYNC_SOURCE			"/Dev1/PFI13"

#define ELFORLIGHT_PORT				"COM1"

#define ZABER_PORT					"COM9"
#define ZABER_MAX_MICRO_RESOLUTION  64 // BENCHTOP_MODE ? 128 : 64;
#define ZABER_MICRO_RESOLUTION		32
#define ZABER_CONVERSION_FACTOR		1.6384 //1.0 / 9.375 //1.0 / 9.375 // BENCHTOP_MODE ? 1.0 / 9.375 : 1.6384;
#define ZABER_MICRO_STEPSIZE		0.09921875 // 0.49609375 // micro-meter ///

#define FAULHABER_PORT				"COM2"
#define FAULHABER_POSITIVE_ROTATION false

//////////////////////// Processing /////////////////////////
#define DATA_HALVING				false // to be updated...

#define PROCESSING_BUFFER_SIZE		20

#ifdef _DEBUG
#define WRITING_BUFFER_SIZE			50
#else
#define WRITING_BUFFER_SIZE	        500
#endif

//////////////////////// OCT system /////////////////////////
#define DISCOM_VAL					0

/////////////////////// FLIM system /////////////////////////
#define FLIM_CH_START_5				30
#define GAUSSIAN_FILTER_WIDTH		200
#define GAUSSIAN_FILTER_STD			48
#define FLIM_SPLINE_FACTOR			20
#define INTENSITY_THRES				0.001f

/////////////////////// Visualization ///////////////////////
#ifdef OCT_FLIM
#define N_VIS_SAMPS_FLIM			200
#endif
#if defined OCT_FLIM || (defined(STANDALONE_OCT) && defined(OCT_NIRF))
#define RING_THICKNESS				60 
#endif

#define CIRC_RADIUS					1300
#define PROJECTION_OFFSET			100

#define INTENSITY_COLORTABLE		6 // fire

#define RENEWAL_COUNT				20






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
#endif
	, oldUhs(false)
#endif
	{}
	~Configuration() {}

public:
	void getConfigFile(QString inipath) // const char* inipath)
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
		nirfRange.max = settings.value("nirfRangeMax").toFloat();
		nirfRange.min = settings.value("nirfRangeMin").toFloat();
#endif
		// Device control
#ifdef ECG_TRIGGERING
		ecgDelayRate = settings.value("ecgDelayRate").toFloat();
#endif
#ifdef OCT_FLIM
		pmtGainVoltage = settings.value("pmtGainVoltage").toFloat();
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
				flimChStartInd[i] = (int)(settings.value(QString("ChannelStart_%1").arg(i)).toFloat() / (1000.0f / (float)ADC_RATE));
			
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

	void setConfigFile(QString inipath)
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
		settings.setValue("nirfRangeMax", QString::number(nirfRange.max, 'f', 1));
		settings.setValue("nirfRangeMin", QString::number(nirfRange.min, 'f', 1));
#endif

		// Device control
#ifdef ECG_TRIGGERING
		settings.setValue("ecgDelayRate", QString::number(ecgDelayRate, 'f', 2));
#endif
#ifdef OCT_FLIM
		settings.setValue("pmtGainVoltage", QString::number(pmtGainVoltage, 'f', 2));
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
		settings.setValue("system", "Standalone OCT");		
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
	int octColorTable;
	Range<int> octDbRange;
#ifdef OCT_FLIM
	int flimLifetimeColorTable;
	Range<float> flimIntensityRange;
	Range<float> flimLifetimeRange;
#endif
#ifdef OCT_NIRF
	Range<float> nirfRange;
#endif

	// System type
	QString systemType;
	bool erasmus;
#ifdef STANDALONE_OCT
#ifdef OCT_NIRF
	bool nirf;
#endif
	bool oldUhs;
#endif

	// Device control
#ifdef ECG_TRIGGERING
	float ecgDelayRate;
#endif
#ifdef OCT_FLIM
	float pmtGainVoltage;
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
