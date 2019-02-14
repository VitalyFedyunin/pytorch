#pragma once
#include <c10/core/Allocator.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace torch {

constexpr uint CUDA_IPC_REF_COUNTER_FILE_SIZE = 10000;
constexpr uint CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;

void CudaIPCCollect();

at::DataPtr GetNewRefCountedSentData(void* data, at::Device device);

bool CudaIPCHaveRefCounter();
void CudaIPCCreateRefCounter(
    std::string handle,
    uint64_t size,
    at::DataPtr data_ptr);
void ReturnRefCounter(std::string handle, uint64_t offset);

struct CudaIPCSentData final {
  std::string handle_;
  int64_t offset_;
  int64_t* counter_ptr_; // Reference counter shared memory block
  at::DataPtr original_ptr_; // Original mem allocation
  cudaEvent_t event_; // Sync cuEventDestroy

  CudaIPCSentData(std::string handle, int64_t offset, int64_t* counter_ptr, at::Device device)
      : handle_(handle),
        offset_(offset),
        counter_ptr_(counter_ptr),
        original_ptr_() {
    // TODO: More efficient would be to create event inside of main thread (at
    // the moment of the queue.put)
    C10_CUDA_CHECK(cudaEventCreateWithFlags(
        &event_,
        cudaEventDisableTiming | cudaEventInterprocess |
            cudaEventBlockingSync));
    C10_CUDA_CHECK(
        cudaEventRecord(event_, c10::cuda::getCurrentCUDAStream(device.index())));
  }
  ~CudaIPCSentData();

  int64_t counter_value();
  std::string handle() {
    return handle_;
  }
  int64_t offset() {
    return offset_;
  }
  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }
};

// All to be deleted data blocks with non zero reference counter goes there
struct CudaIPCSentDataLimbo final {
  ~CudaIPCSentDataLimbo();

  void collect();
  void add(std::unique_ptr<CudaIPCSentData> shared_block);
  uint64_t size() {
    return shared_blocks_.size();
  };

 private:
  // TODO: Can be changed to FIFO in order to avoid full traverse on every
  // collect()
  std::vector<std::unique_ptr<CudaIPCSentData>> shared_blocks_;
};


struct CudaIPCRefCountersFile final {
  ~CudaIPCRefCountersFile();
  CudaIPCRefCountersFile(
      std::string handle,
      uint64_t size,
      at::DataPtr data_ptr)
      : next_offset_(0),
        size_(size),
        used_slots_(0),
        handle_(handle),
        refcounted_shared_mem_(std::move(data_ptr)) {}

  int64_t* counter_ptr() {
    return static_cast<int64_t*>(refcounted_shared_mem_.get()) + next_offset_;
  }

  void set_counter(uint64_t value) {
    *counter_ptr() = value;
  }

  bool have_offsets() {
    return next_offset_ < size_;
  }

  bool offsets_in_use() {
    return used_slots_;
  }

  int64_t get_offset() {
    return next_offset_;
  }

  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }

  void return_offset(uint64_t offset /* unused */) {
    used_slots_--;
  }

  std::string handle() {
    return handle_;
  }

 private:
  uint64_t next_offset_;
  uint64_t size_;
  uint64_t used_slots_;
  std::string handle_;
  at::DataPtr refcounted_shared_mem_;
};

struct CudaIPCReceivedData final {
  explicit CudaIPCReceivedData(std::shared_ptr<void> shared_ptr);
  std::shared_ptr<void> shared_ptr_;
};


} // namespace torch
