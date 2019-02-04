#include <c10/core/StorageImpl.h>

namespace c10 {

struct MPSharedDataBlock_ {
  DataPtr data_ptr_;
  std::shared_ptr<MPRefCounter> ref_counter_;
  MPSharedDataBlock_(DataPtr data_ptr, std::shared_ptr<MPRefCounter> ref_counter);
  ~MPSharedDataBlock_();
};

struct MPSharedStorageLimbo_ {
  std::vector<std::shared_ptr<MPSharedDataBlock_>> shared_blocks_;
  void collect();
  void collect(bool require_everything_released);
  void final_collect();
  ~MPSharedStorageLimbo_();
  // void add(MPSharedDataBlock shared_block);
};


MPSharedDataBlock_::~MPSharedDataBlock_() {
  data_ptr_.clear();
  // delete ref_counter_;
};

MPSharedDataBlock_::MPSharedDataBlock_(
    DataPtr data_ptr,
    std::shared_ptr<MPRefCounter>  ref_counter)
    : data_ptr_(std::move(data_ptr)) {
  ref_counter_ = std::move(ref_counter);
};

void MPSharedStorageLimbo_::final_collect() {
  collect(true);
}

MPSharedStorageLimbo_::~MPSharedStorageLimbo_() {
  // std::cout << "Limbo final collect\n";
  final_collect();
  // std::cout << "Limbo final collect completed\n";
}

void MPSharedStorageLimbo_::collect() {
  collect(false);
}
void MPSharedStorageLimbo_::collect(bool require_everything_released) {
  std::vector<std::shared_ptr<MPSharedDataBlock_>> referenced_blocks;
  int64_t col = 0;
  for (auto const& sd : shared_blocks_) {
    if (sd->ref_counter_->get_count() > 0) {
      // std::cout << "Remaining ref_counter " << sd->ref_counter_->get_handle()
                // << "\n";
      referenced_blocks.push_back(sd);
    } else {
      col += 1;

    }
  }
  if (require_everything_released && referenced_blocks.size() > 0) {
    AT_ERROR("Process terminated before sent tensors were released");
  }
  if (col > 0 ){
   // std::cout << "Collected ref_counter (s) " << col             << "\n";
           }
  // std::cout << "Limbo collect completed\n";
  shared_blocks_ = referenced_blocks;
  // std::cout << "Moving vector completed\n";
}

MPSharedStorageLimbo_ limbo;

StorageImpl::StorageImpl(
    caffe2::TypeMeta data_type,
    int64_t numel,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizable)
    : data_type_(data_type),
      data_ptr_(std::move(data_ptr)),
      numel_(numel),
      resizable_(resizable),
      cuda_ipc_sent_(false),
      cuda_ipc_received_(false),
      allocator_(allocator),
      ref_counter_(nullptr) {
  if (numel > 0) {
    if (data_type_.id() == caffe2::TypeIdentifier::uninitialized()) {
      AT_ERROR(
          "Constructing a storage with meta of unknown type and non-zero numel");
    }
  }
  // std::cout << "(v3) Constructing storage " << numel << " ptr " << data_ptr
  // << "\n";
}

StorageImpl::~StorageImpl() {
  if (cuda_ipc_sent_ || cuda_ipc_received_) {
    // std::cout << "DES Storage distructor called sent: " << cuda_ipc_sent_
    //           << " received: " << cuda_ipc_received_ << "\n";
  }
  if (have_refcounter()) {
    // std::cout << "DEST called shared resource and ref_counter "
    //           << get_refcounter()->get_handle() << " now at "
    //           << get_refcounter()->get_count() << "\n";
  }

  release_resources();
  limbo.collect();
}

void StorageImpl::release_resources() {
  if (data_ptr_.get() != nullptr) {
    if (cuda_ipc_sent_ || cuda_ipc_received_) {
      if (cuda_ipc_received_) {
        if (have_refcounter()) {
          get_refcounter()->dec_counter();
          // std::cout << "DEST Free shared resource and ref_counter "
          //           << get_refcounter()->get_handle() << " now at "
          //           << get_refcounter()->get_count() << "\n";
          ref_counter_ = nullptr;
          data_ptr_.clear();
        }
      } else {
        if (cuda_ipc_sent_ && have_refcounter()) {
          if (get_refcounter()->get_count() > 0) {
            // std::cout << "DEST Keeping shared resource and ref_counter "
            //           << get_refcounter()->get_handle() << " ( now at "
            //           << get_refcounter()->get_count() << " )\n";
            std::shared_ptr<c10::MPSharedDataBlock_> shared_block(
                new MPSharedDataBlock_(std::move(data_ptr_), get_refcounter()));
            ref_counter_ = nullptr;
            limbo.shared_blocks_.push_back(shared_block);
            data_ptr_.clear();
          } else {
            // TODO: Delete counter file
            ref_counter_ = nullptr;
            data_ptr_.clear();
          }
        }
      }
    } else {
      data_ptr_.clear();
    }
  }
}

void StorageImpl::reset() {
  release_resources();
  numel_ = 0;
}

std::shared_ptr<MPRefCounter> StorageImpl::get_refcounter() {
  // TODO: lock here
  if (ref_counter_ == nullptr) {
    AT_ERROR("Undefined refcounter");
    // ref_counter_->counter = new int64_t(0);
  }
  return ref_counter_;
};

bool StorageImpl::have_refcounter() {
  return ref_counter_ != nullptr;
};

// void StorageImpl::set_refcounter(MPRefCounter* ref_counter) {
//   ref_counter_ = ref_counter;
// };

void StorageImpl::set_refcounter(std::string handle, DataPtr data_ptr) {
  ref_counter_ = std::shared_ptr<MPRefCounter>(new MPRefCounter(handle, std::move(data_ptr)));
};

void MPRefCounter::inc_counter() {
  *(int64_t*)(data_ptr_.get()) += 1;
  // TODO: Lock
  // *counter+=1;
};

void MPRefCounter::dec_counter() {
  *(int64_t*)(data_ptr_.get()) -= 1;
  // TODO: Lock
  // *counter-=1;
};

int64_t MPRefCounter::get_count() {
  return *(int64_t*)(data_ptr_.get());
  // TODO: Lock
  // *counter-=1;
};

std::string MPRefCounter::get_handle() {
  return handle_;
};

//
// int64_t MPRefCounter::get_count() {
//   return *counter;
// }

void StorageImpl::inc_refcounter() {
  // std::cout << "inc_refcounter\n";
  get_refcounter()->inc_counter();
};

std::string StorageImpl::get_refcounter_handle() {
  // std::cout << "get_refcounter_handle\n";
  auto ref_counter = get_refcounter().get();
  return ref_counter->get_handle();
};

int64_t StorageImpl::get_refcounter_value() {
  // std::cout << "get_refcounter_value\n";
  return get_refcounter()->get_count();
}

MPRefCounter::~MPRefCounter() {
  // std::cout << "Destroying ref_counter with handle " << handle_ << "\n";
}

MPRefCounter::MPRefCounter(std::string handle, DataPtr data_ptr)
    : data_ptr_(std::move(data_ptr)) {
  // TODO: Lock
  // counter = new int64_t(0);
  handle_ = handle;
  // std::cout << "Created ref_counter with handle " << handle << "\n";
};

} // namespace c10
