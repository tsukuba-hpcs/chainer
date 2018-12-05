#include "chainerx/cuda/memory_pool.h"

#include <algorithm>
#include <iterator>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace cuda {

MallocStatus DeviceMemoryAllocator::Malloc(void** ptr, size_t bytesize) {
    cudaError_t status = cudaMallocManaged(ptr, bytesize, cudaMemAttachGlobal);
    switch (status) {
        case cudaSuccess:
            return MallocStatus::kSuccess;
        case cudaErrorMemoryAllocation:
            return MallocStatus::kErrorMemoryAllocation;
        default:
            Throw(status);
    }
    CHAINERX_NEVER_REACH();
}

MallocStatus PinnedMemoryAllocator::Malloc(void** ptr, size_t bytesize) {
    cudaError_t status = cudaHostAlloc(ptr, bytesize, cudaHostAllocWriteCombined);
    switch (status) {
        case cudaSuccess:
            return MallocStatus::kSuccess;
        case cudaErrorMemoryAllocation:
            return MallocStatus::kErrorMemoryAllocation;
        default:
            Throw(status);
    }
    CHAINERX_NEVER_REACH();
}

std::unique_ptr<Chunk> Chunk::Split(size_t bytesize) {
    CHAINERX_ASSERT(bytesize_ >= bytesize);
    if (bytesize_ == bytesize) {
        return nullptr;
    }

    std::unique_ptr<Chunk> remaining = std::make_unique<Chunk>(ptr(), bytesize, bytesize_ - bytesize);
    bytesize_ = bytesize;

    if (next_ != nullptr) {
        remaining->SetNext(next_);
        remaining->next()->SetPrev(remaining.get());
    }
    next_ = remaining.get();
    remaining->SetPrev(this);

    return remaining;
}

void Chunk::MergeWithNext() {
    CHAINERX_ASSERT(next_ != nullptr);
    bytesize_ += next_->bytesize();
    if (next_->next() != nullptr) {
        next_->next()->SetPrev(this);
    }
    next_ = next_->next();
}

// Pushes a chunk into an appropriate free list
//
// Not thread-safe
void MemoryPool::PushIntoFreeList(std::unique_ptr<Chunk> chunk) {
    FreeList& free_list = free_bins_[chunk->bytesize()];
    free_list.emplace_back(std::move(chunk));
}

void MemoryPool::CompactFreeBins(std::map<size_t, FreeList>::iterator it_start, std::map<size_t, FreeList>::iterator it_end) {
    auto it_start_rev = std::make_reverse_iterator(it_start);
    it_start = std::find_if(it_start_rev, free_bins_.rend(), [](const auto& p) { return !p.second.empty(); }).base();
    it_end = std::find_if(it_end, free_bins_.end(), [](const auto& p) { return !p.second.empty(); });
    free_bins_.erase(it_start, it_end);
}

// Finds best-fit, or a smallest larger allocation if available
//
// Not thread-safe
std::unique_ptr<Chunk> MemoryPool::PopFromFreeList(size_t allocation_size) {
    auto it_start = free_bins_.lower_bound(allocation_size);
    size_t distance{0};
    for (auto it = it_start; it != free_bins_.end(); ++it, ++distance) {
        FreeList& free_list = it->second;
        if (free_list.empty()) {
            continue;
        }
        std::unique_ptr<Chunk> chunk = std::move(free_list.back());
        CHAINERX_ASSERT(chunk != nullptr);
        free_list.pop_back();
        if (distance > kCompactionThreashold) {
            CompactFreeBins(it_start, it);
        }
        return chunk;
    }

    return nullptr;
}

// Removes a chunk from an appropriate free list, and returns the removed chunk
//
// Not thread-safe
std::unique_ptr<Chunk> MemoryPool::RemoveChunkFromFreeList(Chunk* chunk) {
    CHAINERX_ASSERT(chunk != nullptr);

    // Find an appropriate free list
    auto free_bins_it = free_bins_.find(chunk->bytesize());
    if (free_bins_it == free_bins_.end()) {
        return nullptr;
    }
    FreeList& free_list = free_bins_it->second;

    // Remove the given chunk from the found free list
    auto it = std::find_if(free_list.begin(), free_list.end(), [chunk](const auto& ptr) { return ptr.get() == chunk; });
    if (it == free_list.end()) return nullptr;
    std::unique_ptr<Chunk> chunk_uniq_ptr = std::move(*it);
    CHAINERX_ASSERT(chunk_uniq_ptr != nullptr);
    free_list.erase(it);
    return chunk_uniq_ptr;
}

MemoryPool::~MemoryPool() {
    // NOTE: CudaSetDeviceScope is not available at dtor because it may throw
    int orig_device_index{0};
    cudaGetDevice(&orig_device_index);
    cudaSetDevice(device_index_);

    for (std::pair<const size_t, FreeList>& item : free_bins_) {
        FreeList& free_list = item.second;
        for (const std::unique_ptr<Chunk>& chunk : free_list) {
            if (chunk->prev() == nullptr) {
                allocator_->Free(chunk->ptr());
            }
        }
    }
    // Ideally, in_use_ should be empty, but it could happen that shared ptrs to memories allocated
    // by this memory pool are released after this memory pool is destructed.
    // Our approach is that we anyway free CUDA memories held by this memory pool here in such case.
    // Operators of arrays holding such memories will be broken, but are not supported.
    for (const std::pair<void* const, std::unique_ptr<Chunk>>& item : in_use_) {
        const std::unique_ptr<Chunk>& chunk = item.second;
        if (chunk->prev() == nullptr) {
            allocator_->Free(chunk->ptr());
        }
    }

    cudaSetDevice(orig_device_index);
}

void MemoryPool::FreeUnusedBlocks() {
    CudaSetDeviceScope scope{device_index_};

    // Frees unused memory blocks
    for (std::pair<const size_t, FreeList>& pair : free_bins_) {
        FreeList& free_list = pair.second;
        for (std::unique_ptr<Chunk>& chunk : free_list) {
            if (chunk->next() == nullptr && chunk->prev() == nullptr) {
                allocator_->Free(chunk->ptr());
                chunk.reset();
            }
        }
        free_list.erase(std::remove(free_list.begin(), free_list.end(), nullptr), free_list.end());
    }

    // Erase empty free lists from free bins.
    for (auto free_bins_it = free_bins_.begin(); free_bins_it != free_bins_.end();) {
        if (free_bins_it->second.empty()) {
            free_bins_it = free_bins_.erase(free_bins_it);
        } else {
            free_bins_it++;
        }
    }
}

void* MemoryPool::Malloc(size_t bytesize) {
    if (bytesize == 0) {
        return nullptr;
    }

    // TODO(niboshi): Currently the deleter of allocated memory assumes that
    // the memory is stored in the memory pool (in `in_use_`), but this may not hold if some exception is thrown before it is stored.
    // `std::lock_guard` and `in_use_.emplace` are the sources of possible exceptions.

    size_t allocation_size = GetAllocationSize(bytesize);
    std::unique_ptr<Chunk> chunk{nullptr};

    {
        std::lock_guard<std::mutex> lock{free_bins_mutex_};
        chunk = PopFromFreeList(allocation_size);
    }

    if (chunk != nullptr) {
        std::unique_ptr<Chunk> remaining = chunk->Split(allocation_size);
        if (remaining != nullptr) {
            std::lock_guard<std::mutex> lock{free_bins_mutex_};
            PushIntoFreeList(std::move(remaining));
        }
    } else {
        void* ptr{nullptr};
        CudaSetDeviceScope scope{device_index_};
        MallocStatus status = allocator_->Malloc(&ptr, allocation_size);
        if (status == MallocStatus::kErrorMemoryAllocation) {
            FreeUnusedBlocks();
            status = allocator_->Malloc(&ptr, allocation_size);
            if (status == MallocStatus::kErrorMemoryAllocation) {
                // TODO(sonots): Include total pooled bytes in the error message
                throw OutOfMemoryError{bytesize};
            }
        }
        chunk = std::make_unique<Chunk>(ptr, 0, allocation_size);
    }

    CHAINERX_ASSERT(chunk != nullptr);
    void* ptr = chunk->ptr();
    {
        std::lock_guard<std::mutex> lock{in_use_mutex_};
        in_use_.emplace(ptr, std::move(chunk));
    }
    return ptr;
}

void MemoryPool::Free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    std::unique_ptr<Chunk> chunk{nullptr};

    {
        std::lock_guard<std::mutex> lock{in_use_mutex_};
        auto it = in_use_.find(ptr);
        if (it == in_use_.end()) {
            throw ChainerxError{"Cannot free out-of-pool memory"};
        }
        chunk = std::move(it->second);
        in_use_.erase(it);
    }

    CHAINERX_ASSERT(chunk != nullptr);
    {
        std::lock_guard<std::mutex> lock{free_bins_mutex_};

        // If the next chunk is free, merges it with them.
        if (chunk->next() != nullptr) {
            std::unique_ptr<Chunk> chunk_next = RemoveChunkFromFreeList(chunk->next());
            if (chunk_next != nullptr) {
                chunk->MergeWithNext();
            }
        }

        // If the previous chunk is free, merges it with them.
        if (chunk->prev() != nullptr) {
            std::unique_ptr<Chunk> chunk_prev = RemoveChunkFromFreeList(chunk->prev());
            if (chunk_prev != nullptr) {
                chunk_prev->MergeWithNext();
                chunk = std::move(chunk_prev);
            }
        }

        PushIntoFreeList(std::move(chunk));
    }
}

void MemoryPool::FreeNoExcept(void* ptr) noexcept {
    try {
        Free(ptr);
    } catch (...) {
        CHAINERX_NEVER_REACH();
    }
}

}  // namespace cuda
}  // namespace chainerx
