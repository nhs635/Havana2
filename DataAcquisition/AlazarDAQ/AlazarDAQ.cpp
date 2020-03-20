
#include "AlazarDAQ.h"

#if ALAZAR_ENABLE

#include <AlazarError.h>
#include <AlazarApi.h>
#include <AlazarCmd.h>

using namespace std;

AlazarDAQ::AlazarDAQ() :
    nChannels(2),
    nScans(1600),
    nAlines(1024),
    VoltRange1(INPUT_RANGE_PM_500_MV),
    VoltRange2(INPUT_RANGE_PM_500_MV),
    TriggerDelay(0),
    UseExternalClock(false),
    UseAutoTrigger(false),
    _dirty(true), 
	_running(false), 
	boardHandle(nullptr)
{
}

AlazarDAQ::~AlazarDAQ()
{
    if (_thread.joinable())
    {
        _running = false;
        _thread.join();
    }

    // Abort the acquisition
    AlazarAbortAsyncRead(boardHandle);

    // Free all memory allocated
    for (U32 bufferIndex = 0; bufferIndex < BUFFER_COUNT; bufferIndex++)
    {
        if (BufferArray[bufferIndex] != NULL)
        {
#ifdef _WIN32
            VirtualFree(BufferArray[bufferIndex], 0, MEM_RELEASE);
#else
            free(BufferArray[bufferIndex]);
#endif
        }
    }
}


bool AlazarDAQ::initialize()
{    
    RETURN_CODE retCode = ApiSuccess;

	// Select a board
	U32 systemId = 1;
	U32 boardId = 1;

	// Get a handle to the board
	boardHandle = AlazarGetBoardBySystemID(systemId, boardId);
    if (boardHandle == nullptr)
	{
        dumpError(retCode, "Error: Unable to open board system Id 1 board Id 1");
		return false;
	}

	// Specify the sample rate (see sample rate id below)
    // double samplesPerSec = 500.e6;
    const U32 SamplingRate = SAMPLE_RATE_800MSPS;  // SAMPLE_RATE_500MSPS; // only for internal clock

	// Select clock parameters as required to generate this sample rate.
	//
	// For example: if samplesPerSec is 100.e6 (100 MS/s), then:
	// - select clock source INTERNAL_CLOCK and sample rate SAMPLE_RATE_100MSPS
	// - select clock source FAST_EXTERNAL_CLOCK, sample rate SAMPLE_RATE_USER_DEF,
	//   and connect a 100 MHz signalto the EXT CLK BNC connector.
	retCode = 
		AlazarSetCaptureClock(
			boardHandle,			// HANDLE -- board handle
            UseExternalClock ? EXTERNAL_CLOCK_AC : INTERNAL_CLOCK,		// U32 -- clock source id
            UseExternalClock ? SAMPLE_RATE_USER_DEF : SamplingRate,	// U32 -- sample rate id
			CLOCK_EDGE_RISING,		// U32 -- clock edge id
			0						// U32 -- clock decimation 
			);
	if (retCode != ApiSuccess)
	{
        dumpError(retCode, "Error: AlazarSetCaptureClock failed: ");
		return false;
	}

    // External Clock Settings
    retCode =
        AlazarSetExternalClockLevel(
        boardHandle,			// HANDLE -- board handle
        30.f                    // Voltage level in percentage
        );
    if (retCode != ApiSuccess)
    {
        dumpError(retCode, "Error: AlazarSetExternalClockLevel failed: ");
        return false;
    }


	// Select CHA input parameters as required
	retCode = 
		AlazarInputControl(
			boardHandle,			// HANDLE -- board handle
			CHANNEL_A,			    // U8 -- input channel 
            DC_COUPLING,			// U32 -- input coupling id
            VoltRange1,     		// U32 -- input range id
			IMPEDANCE_50_OHM		// U32 -- input impedance id
			);
	if (retCode != ApiSuccess)
	{
        dumpError(retCode, "Error: AlazarInputControl failed: ");
		return false;
	}

	// Select CHB input parameters as required
	retCode = 
		AlazarInputControl(
			boardHandle,			// HANDLE -- board handle
			CHANNEL_B,				// U8 -- channel identifier
            DC_COUPLING,			// U32 -- input coupling id
            VoltRange2,         	// U32 -- input range id
            IMPEDANCE_50_OHM		// U32 -- input impedance id
			);
	if (retCode != ApiSuccess)
	{
        dumpError(retCode, "Error: AlazarInputControl failed: ");
		return false;
	}

	// Select trigger inputs and levels as required
	retCode = 
		AlazarSetTriggerOperation(
			boardHandle,			// HANDLE -- board handle
			TRIG_ENGINE_OP_J,		// U32 -- trigger operation 
			TRIG_ENGINE_J,			// U32 -- trigger engine id
			TRIG_EXTERNAL,			// U32 -- trigger source id
			TRIGGER_SLOPE_POSITIVE,	// U32 -- trigger slope id
			128 + (int)(128 * 0.5),// Utrigger32 -- trigger level from 0 (-range) to 255 (+range)
			TRIG_ENGINE_K,			// U32 -- trigger engine id
			TRIG_DISABLE,			// U32 -- trigger source id for engine K
			TRIGGER_SLOPE_POSITIVE,	// U32 -- trigger slope id
			128						// U32 -- trigger level from 0 (-range) to 255 (+range)
			);
	if (retCode != ApiSuccess)
	{
        dumpError(retCode, "Error: AlazarSetTriggerOperation failed: ");
		return false;
	}

	// Select external trigger parameters as required
	retCode =
		AlazarSetExternalTrigger( 
			boardHandle,			// HANDLE -- board handle
			DC_COUPLING,			// U32 -- external trigger coupling id
            ETR_2V5					// U32 -- external trigger range id
			);

    // Set trigger delay as required.
    retCode = AlazarSetTriggerDelay(boardHandle, TriggerDelay);
    if (retCode != ApiSuccess)
    {
        dumpError(retCode, "Error: AlazarSetTriggerDelay failed: ");
        return FALSE;
    }

    // Set trigger timeout as required.
    // NOTE:
    // The board will wait for a for this amount of time for a trigger event.
    // If a trigger event does not arrive, then the board will automatically
    // trigger. Set the trigger timeout value to 0 to force the board to wait
    // forever for a trigger event.
    //
    // IMPORTANT:
    // The trigger timeout value should be set to zero after appropriate
    // trigger parameters have been determined, otherwise the
    // board may trigger if the timeout interval expires before a
    // hardware trigger event arrives.

    //double triggerTimeout_sec = 0.;
    //U32 triggerTimeout_clocks = (U32)(triggerTimeout_sec / 10.e-6 + 0.5);
    U32 triggerTimeout_clocks = UseAutoTrigger ? 1U : 0U; // ns

    retCode =
        AlazarSetTriggerTimeOut(
            boardHandle,			// HANDLE -- board handle
            triggerTimeout_clocks	// U32 -- timeout_sec / 10.e-6 (0 means wait forever)
            );
    if (retCode != ApiSuccess)
    {
        dumpError(retCode, "Error: AlazarSetTriggerTimeOut failed :");
        return FALSE;
    }

	// Configure AUX I/O connector as required
	//retCode = 
	//	AlazarConfigureAuxIO(
	//		boardHandle,			// HANDLE -- board handle
	//		AUX_OUT_TRIGGER,		// U32 -- mode
	//		0						// U32 -- parameter
	//		);	
	//if (retCode != ApiSuccess)
	//{
	//	printf("Error: AlazarConfigureAuxIO failed -- %s\n", AlazarErrorToText(retCode));
	//	return FALSE;
	//}

	return true;
}

bool AlazarDAQ::set_init()
{
    if (_dirty)
    {
        if (!initialize())
            return false;

        _dirty = false;
    }

    return true;
}

bool AlazarDAQ::startAcquisition()
{
    if (_thread.joinable())
    {
        dumpErrorSystem(::GetLastError(), "ERROR: Acquisition is already running: ");
        return false;
    }

    _thread = std::thread(&AlazarDAQ::run, this); // thread executing
    if (::SetThreadPriority(_thread.native_handle(), THREAD_PRIORITY_TIME_CRITICAL) == 0)
    {
        dumpErrorSystem(::GetLastError(), "ERROR: Failed to set acquisition thread priority: ");
        return false;
    }

    printf("Data acquisition thread is started.\n");

    return true;
}

void AlazarDAQ::stopAcquisition()
{
    if (_thread.joinable())
    {
        DidStopData();
        _thread.join();
    }

    printf("Data acquisition thread is finished normally.\n");
}


// Acquisition Thread
void AlazarDAQ::run()
{
    RETURN_CODE retCode = ApiSuccess;

    // Select the number of pre-trigger samples per record
    U32 preTriggerSamples = TriggerDelay;

    // Select the number of post-trigger samples per record
    U32 postTriggerSamples = nScans - TriggerDelay;

    // Specify the number of records per DMA buffer
    U32 recordsPerBuffer = nAlines;

    // MEMO: we always acquire two channel and if nChannels == 1, interlace and send only 1 channel

    // Calculate the number of enabled channels from the channel mask
    int channelCount = nChannels;

    // Select which channels to capture (A, B, or both)
    U32 channelMask = CHANNEL_A | CHANNEL_B;

    // Get the sample size in bits, and the on-board memory size in samples per channel
    U8 bitsPerSample;
    U32 maxSamplesPerChannel;
    retCode = AlazarGetChannelInfo(boardHandle, &maxSamplesPerChannel, &bitsPerSample);
    if (retCode != ApiSuccess)
    {
        dumpError(retCode, "Error: AlazarGetChannelInfo failed: ");
        return;
    }

    // Calculate the size of each DMA buffer in bytes
    U32 bytesPerSample = (bitsPerSample + 7) / 8;
    U32 samplesPerRecord = preTriggerSamples + postTriggerSamples; // nScans
    U32 bytesPerRecord = bytesPerSample * samplesPerRecord; // sizeof(UINT16) * nScans
    U32 bytesPerBuffer = bytesPerRecord * recordsPerBuffer * channelCount; // 2 * sizeof(UINT16) * nScans

    // Allocate memory for DMA buffers
    BOOL success = TRUE;

    U32 bufferIndex;
    for (bufferIndex = 0; (bufferIndex < BUFFER_COUNT) && success; bufferIndex++)
    {
#ifdef _WIN32	// Allocate page aligned memory
        BufferArray[bufferIndex] = (U16*) VirtualAlloc(NULL, bytesPerBuffer, MEM_COMMIT, PAGE_READWRITE);
#else
        BufferArray[bufferIndex] = (U16*) malloc(bytesPerBuffer);
#endif
        if (BufferArray[bufferIndex] == NULL)
        {
            dumpError(retCode, "Error: Alloc failed");
            success = FALSE;
        }
    }

    // Configure the record size
    if (success)
    {
        retCode =
            AlazarSetRecordSize (
                boardHandle,			// HANDLE -- board handle
                preTriggerSamples,		// U32 -- pre-trigger samples
                postTriggerSamples		// U32 -- post-trigger samples
                );
        if (retCode != ApiSuccess)
        {
            dumpError(retCode, "Error: AlazarSetRecordSize failed: ");
            success = FALSE;
        }
    }

    // Configure the board to make a traditional AutoDMA acquisition
    if (success)
    {
        U32 admaFlags;

        if (UseAutoTrigger)
        {
            // Continuous mode (for debugging)
            admaFlags = ADMA_EXTERNAL_STARTCAPTURE | // Start acquisition when we call AlazarStartCapture
                ADMA_CONTINUOUS_MODE |		 // Acquire a continuous stream of sample data without trigger
                ADMA_FIFO_ONLY_STREAMING;	 // The ATS9360-FIFO does not have on-board memory
        }
        else
        {
            // Acquire records per each trigger
            admaFlags = ADMA_EXTERNAL_STARTCAPTURE |	// Start acquisition when AlazarStartCapture is called
                ADMA_FIFO_ONLY_STREAMING |		// The ATS9360-FIFO does not have on-board memory
                ADMA_TRADITIONAL_MODE;			// Acquire multiple records optionally with pretrigger
        }

        // samples and record headers
        retCode =
            AlazarBeforeAsyncRead(
                boardHandle,			// HANDLE -- board handle
                channelMask,			// U32 -- enabled channel mask
                -(long)preTriggerSamples,	// long -- offset from trigger in samples
                samplesPerRecord,		// U32 -- samples per record
                recordsPerBuffer,		// U32 -- records per buffer
                0x7fffffff,					// U32 -- records per acquisition (infinitly)
                admaFlags				// U32 -- AutoDMA flags
                );
        if (retCode != ApiSuccess)
        {
            dumpError(retCode, "Error: AlazarBeforeAsyncRead failed: ");
            success = FALSE;
        }
    }

    // Add the buffers to a list of buffers available to be filled by the board
    for (bufferIndex = 0; (bufferIndex < BUFFER_COUNT) && success; bufferIndex++)
    {
        U16* pBuffer = BufferArray[bufferIndex];
        retCode = AlazarPostAsyncBuffer(boardHandle, pBuffer, bytesPerBuffer);
        if (retCode != ApiSuccess)
        {
            dumpError(retCode, "Error: AlazarPostAsyncBuffer failed: ");
            success = FALSE;
        }
    }

    // Arm the board system to wait for a trigger event to begin the acquisition
    if (success)
    {
        retCode = AlazarStartCapture(boardHandle);
        if (retCode != ApiSuccess)
        {
            dumpError(retCode, "Error: AlazarStartCapture failed: ");
            success = FALSE;
        }
    }
	
    // Wait for each buffer to be filled, process the buffer, and re-post it to the board.
    if (success)
    {

        U32 buffersCompleted = 0, buffersCompletedUpdate = 0;
        UINT64 bytesTransferred = 0, bytesTransferredPerUpdate = 0;
        ULONG dwTickStart = 0, dwTickLastUpdate;

        _running = true;
        while (_running)
        {
            // Set a buffer timeout that is longer than the time
            // required to capture all the records in one buffer.
            DWORD timeout_ms = 5000;

            // Wait for the buffer at the head of the list of available buffers
            // to be filled by the board.
            bufferIndex = buffersCompleted % BUFFER_COUNT;
            U16* pBuffer = BufferArray[bufferIndex];
			
            retCode = AlazarWaitAsyncBufferComplete(boardHandle, pBuffer, timeout_ms);
            if (retCode != ApiSuccess)
            {
                dumpError(retCode, "Error: AlazarWaitAsyncBufferComplete failed: ");
                success = FALSE;
//                return;
            }
			
            if (success)
            {
                // The buffer is full and has been removed from the list
                // of buffers available for the board        
                buffersCompletedUpdate++;
                bytesTransferred += bytesPerBuffer;
                bytesTransferredPerUpdate += bytesPerBuffer;

                // Process sample data in this buffer.

                // NOTE:
                //
                // While you are processing this buffer, the board is already
                // filling the next available buffer(s).
                //
                // You MUST finish processing this buffer and post it back to the
                // board before the board fills all of its available DMA buffers
                // and on-board memory.
                //
                // Records are arranged in the buffer as follows:
                // R0[AB], R1[AB], R2[AB] ... Rn[AB]
                //
                // Samples values are arranged contiguously in each record.
                // A 12-bit sample code is stored in the most significant
                // bits of each 16-bit sample value.
                //
                // Sample codes are unsigned by default. As a result:
                // - a sample code of 0x000 represents a negative full scale input signal.
                // - a sample code of 0x800 represents a ~0V signal.
                // - a sample code of 0xFFF represents a positive full scale input signal.

                if (nChannels == 2)
                {
                    // Callback
                    np::Array<uint16_t, 2> frame(pBuffer, channelCount * nScans, nAlines);

                    // MEMO: -1 to make buffersCompleted start with 0 (same as signatec system)                    
                    DidAcquireData(buffersCompleted++, frame);
                }
//                else if (nChannels == 1)
//                {
//                    // deinterlace data and send only 1 channel

//                    // lazy initialize buffer
//                    if (buffer1ch.size(0) != nScans || buffer1ch.size(1) != nAlines)
//                    {
//                        buffer1ch = np::Array<uint16_t, 2>(nScans, nAlines);
//                        buffer2ch = np::Array<uint16_t, 2>(nScans, nAlines);
//                    }

//                    // deinterlace fringe
//                    Ipp16u *deinterlaced_fringe[2] = { buffer1ch, buffer2ch};
//                    ippsDeinterleave_16s((Ipp16s *)pBuffer, 2, nAlines * nScans, (Ipp16s **)deinterlaced_fringe);

//                    // MEMO: -1 to make buffersCompleted start with 0 (same as signatec system)
//                    DidAcquireData(buffersCompleted - 1, buffer1ch);
//                }
            }
				
            // Add the buffer to the end of the list of available buffers.
            if (success)
            {
                retCode = AlazarPostAsyncBuffer(boardHandle, pBuffer, bytesPerBuffer);
                if (retCode != ApiSuccess)
                {
                    dumpError(retCode, "Error: AlazarPostAsyncBuffer failed: ");
                    success = FALSE;
                }
            }

            // If the acquisition failed, exit the acquisition loop
            if (!success)
                break;

            // Acquisition Status
            if (!dwTickStart)
                dwTickStart = dwTickLastUpdate = GetTickCount();

            // Periodically update progress
            ULONG dwTickNow = GetTickCount();
            if (dwTickNow - dwTickLastUpdate > 5000)
            {
                double dRate, dRateUpdate;

                ULONG dwElapsed = dwTickNow - dwTickStart;
                ULONG dwElapsedUpdate = dwTickNow - dwTickLastUpdate;

                dwTickLastUpdate = dwTickNow;

                if (dwElapsed)
                {
                    dRate = (bytesTransferred / 1000000.0) / (dwElapsed / 1000.0);
                    dRateUpdate = (bytesTransferredPerUpdate / 1000000.0) / (dwElapsedUpdate / 1000.0);

                    unsigned h = 0, m = 0, s = 0;
                    if (dwElapsed >= 1000)
                    {
                        if ((s = dwElapsed / 1000) >= 60)	// Seconds
                        {
                            if ((m = s / 60) >= 60)			// Minutes
                            {
                                if (h = m / 60)				// Hours
                                    m %= 60;
                            }
                            s %= 60;
                        }
                    }

                    printf("[Elapsed Time] %u:%02u:%02u [DAQ Rate] %3.2f MB/s [Frame Rate] %.2f fps \n", h, m, s,
                           dRateUpdate, (double)buffersCompletedUpdate / (double)(dwElapsedUpdate) * 1000.0);
                }

                // reset
                buffersCompletedUpdate = 0;
                bytesTransferredPerUpdate = 0;
            }
        }
    }

    // Abort the acquisition
    retCode = AlazarAbortAsyncRead(boardHandle);
    if (retCode != ApiSuccess)
    {
        dumpError(retCode, "Error: AlazarAbortAsyncRead failed: ");
        success = FALSE;
    }

    // Free all memory allocated
    for (bufferIndex = 0; bufferIndex < BUFFER_COUNT; bufferIndex++)
    {
        if (BufferArray[bufferIndex] != NULL)
        {
#ifdef _WIN32
            VirtualFree(BufferArray[bufferIndex], 0, MEM_RELEASE);
#else
            free(BufferArray[bufferIndex]);
#endif
        }
    }
}

// Dump a Alazar library error
void AlazarDAQ::dumpError(RETURN_CODE retCode, const char* pPreamble)
{
    char *pErr = nullptr;
    int my_res;
    char msg[MAX_MSG_LENGTH];
    memcpy(msg, pPreamble, strlen(pPreamble));

    strcat(msg, AlazarErrorToText(retCode));

    printf("%s\n", msg);
//    SendStatusMessage(msg);

    // Abort the acquisition
    AlazarAbortAsyncRead(boardHandle);

    // Free all memory allocated
    for (U32 bufferIndex = 0; bufferIndex < BUFFER_COUNT; bufferIndex++)
    {
        if (BufferArray[bufferIndex] != NULL)
        {
#ifdef _WIN32
            VirtualFree(BufferArray[bufferIndex], 0, MEM_RELEASE);
#else
            free(BufferArray[bufferIndex]);
#endif
        }
    }
}


void AlazarDAQ::dumpErrorSystem(int res, const char* pPreamble)
{
    char* pErr = nullptr;
    char msg[MAX_MSG_LENGTH];
    memcpy(msg, pPreamble, strlen(pPreamble));

    sprintf(pErr, "Error code (%d)", res);
    strcat(msg, pErr);

    printf("%s\n", msg);
//    SendStatusMessage(msg);

    // Abort the acquisition
    AlazarAbortAsyncRead(boardHandle);

    // Free all memory allocated
    for (U32 bufferIndex = 0; bufferIndex < BUFFER_COUNT; bufferIndex++)
    {
        if (BufferArray[bufferIndex] != NULL)
        {
#ifdef _WIN32
            VirtualFree(BufferArray[bufferIndex], 0, MEM_RELEASE);
#else
            free(BufferArray[bufferIndex]);
#endif
        }
    }
}

#endif
