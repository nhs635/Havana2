#ifndef SYNCOBJECT_H
#define SYNCOBJECT_H

#include <iostream>
#include <queue>
#include <mutex>

#include <Common/Queue.h>

template <typename T>
class SyncObject
{
public:
    explicit SyncObject() {}
    ~SyncObject() {	deallocate_queue_buffer(); }

public:
    void allocate_queue_buffer(int width, int height, int n)
    {
		n_buffer = n;
        for (int i = 0; i < n_buffer; i++)
        {
            T* buffer = new T[width * height];
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
				delete[] buffer;
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

#endif // SYNCOBJECT_H
