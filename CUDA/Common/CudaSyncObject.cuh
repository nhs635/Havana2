#ifndef CUDA_SYNCOBJECT_CUH
#define CUDA_SYNCOBJECT_CUH

#include <iostream>
#include <queue>
#include <mutex>

#include <cuda/cuda_runtime.h>
#include <cuda/cufft.h>

#include <Common/Queue.h>

#include "CudaErrorCheck.cuh"

template <typename T>
class CudaSyncObject
{
public:
	explicit CudaSyncObject() {}
    ~CudaSyncObject() {	deallocate_queue_buffer(); }

public:
    void allocate_queue_buffer(int width, int height, int n)
    {
		n_buffer = n;
        for (int i = 0; i < n_buffer; i++)
        {			
			T* buffer; CUDA_CHECK_ERROR(cudaHostAlloc((void**)&buffer, sizeof(T) * width * height, cudaHostAllocDefault));
            memset(buffer, 0, width * height * sizeof(T));
            queue_buffer.push(buffer);
        }
    }

	void deallocate_queue_buffer()
	{
		for (int i = 0; i < n_buffer; i++)
		{
			if (!queue_buffer.empty())
			{
				T* buffer = queue_buffer.front();
				queue_buffer.pop();
				cudaFreeHost(buffer);
			}
		}
	}

public:
    std::queue<T*> queue_buffer; // Buffers for threading operations
    std::mutex mtx; // Mutex for buffering operation
    Queue<T*> Queue_sync; // Synchronization objects for threading operations

private:
	int n_buffer;
};

#endif // CUDA_SYNCOBJECT_H
