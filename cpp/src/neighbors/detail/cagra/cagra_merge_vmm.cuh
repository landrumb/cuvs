/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/process_is_exiting.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra::detail {

class cuda_vmm_error : public std::runtime_error {
 public:
  cuda_vmm_error(CUresult result, std::string message)
    : std::runtime_error(std::move(message)), result_(result)
  {
  }

  [[nodiscard]] auto result() const noexcept -> CUresult { return result_; }

 private:
  CUresult result_;
};

inline auto cuda_driver_error_message(CUresult result, char const* operation) -> std::string
{
  char const* name        = nullptr;
  char const* description = nullptr;
  static_cast<void>(cuGetErrorName(result, &name));
  static_cast<void>(cuGetErrorString(result, &description));

  std::string message{operation};
  message += " failed";
  if (name != nullptr) {
    message += ": ";
    message += name;
  }
  if (description != nullptr) {
    message += " (";
    message += description;
    message += ")";
  }
  return message;
}

inline void check_cuda_driver(CUresult result, char const* operation)
{
  if (result != CUDA_SUCCESS) {
    throw cuda_vmm_error{result, cuda_driver_error_message(result, operation)};
  }
}

inline void log_cuda_driver_cleanup_error(CUresult result, char const* operation) noexcept
{
  if (result == CUDA_SUCCESS) { return; }
  char const* name        = nullptr;
  char const* description = nullptr;
  static_cast<void>(cuGetErrorName(result, &name));
  static_cast<void>(cuGetErrorString(result, &description));
  RAFT_LOG_WARN("cagra::merge: %s failed: %s (%s)",
                operation,
                name == nullptr ? "unknown CUDA driver error" : name,
                description == nullptr ? "no description" : description);
}

/**
 * VMM physical allocations bypass RMM. Only the direct cudaMalloc/cudaFree resource reliably
 * returns each consumed input allocation to CUDA so that the next VMM chunk can reuse its physical
 * capacity. Caching resources retain physical pages in their own pools and therefore use the
 * legacy direct device-copy fallback instead.
 */
inline auto current_device_resource_releases_to_cuda() -> bool
{
  rmm::mr::cuda_memory_resource direct_cuda_resource;
  auto direct_ref = rmm::device_async_resource_ref{direct_cuda_resource};
  return rmm::mr::get_current_device_resource_ref() == direct_ref;
}

inline auto cuda_vmm_supported() noexcept -> bool
{
  int ordinal = 0;
  if (cudaGetDevice(&ordinal) != cudaSuccess) { return false; }

  CUdevice device{};
  if (cuDeviceGet(&device, ordinal) != CUDA_SUCCESS) { return false; }

  int supported = 0;
  if (cuDeviceGetAttribute(&supported,
                           CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                           device) != CUDA_SUCCESS) {
    return false;
  }
  return supported != 0;
}

template <typename T>
auto checked_dataset_bytes(int64_t n_rows, int64_t dim) -> std::size_t
{
  RAFT_EXPECTS(n_rows >= 0 && dim >= 0, "Dataset extents must be nonnegative");
  auto const rows = static_cast<std::size_t>(n_rows);
  auto const cols = static_cast<std::size_t>(dim);
  RAFT_EXPECTS(cols == 0 || rows <= std::numeric_limits<std::size_t>::max() / cols,
               "Dataset element count overflows size_t");
  auto const elements = rows * cols;
  RAFT_EXPECTS(elements <= std::numeric_limits<std::size_t>::max() / sizeof(T),
               "Dataset byte size overflows size_t");
  return elements * sizeof(T);
}

template <typename T>
class contiguous_dataset_storage {
 public:
  virtual ~contiguous_dataset_storage() noexcept = default;

  [[nodiscard]] virtual auto data_handle() noexcept -> T*                   = 0;
  [[nodiscard]] virtual auto data_handle() const noexcept -> T const*       = 0;
  [[nodiscard]] virtual auto n_rows() const noexcept -> int64_t             = 0;
  [[nodiscard]] virtual auto dim() const noexcept -> int64_t                = 0;
  [[nodiscard]] virtual auto physical_bytes() const noexcept -> std::size_t = 0;
  [[nodiscard]] virtual auto is_vmm() const noexcept -> bool                = 0;

  [[nodiscard]] auto extent(std::size_t rank) const noexcept -> int64_t
  {
    return rank == 0 ? n_rows() : dim();
  }

  [[nodiscard]] auto view() noexcept
  {
    return raft::make_device_matrix_view<T, int64_t>(data_handle(), n_rows(), dim());
  }

  [[nodiscard]] auto view() const noexcept
  {
    return raft::make_device_matrix_view<T const, int64_t>(data_handle(), n_rows(), dim());
  }
};

template <typename T>
class device_matrix_dataset_storage final : public contiguous_dataset_storage<T> {
 public:
  device_matrix_dataset_storage(raft::resources const& handle, int64_t n_rows, int64_t dim)
    : matrix_(raft::make_device_matrix<T, int64_t>(handle, n_rows, dim))
  {
  }

  [[nodiscard]] auto data_handle() noexcept -> T* final { return matrix_.data_handle(); }
  [[nodiscard]] auto data_handle() const noexcept -> T const* final
  {
    return matrix_.data_handle();
  }
  [[nodiscard]] auto n_rows() const noexcept -> int64_t final { return matrix_.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> int64_t final { return matrix_.extent(1); }
  [[nodiscard]] auto physical_bytes() const noexcept -> std::size_t final
  {
    return matrix_.size() * sizeof(T);
  }
  [[nodiscard]] auto is_vmm() const noexcept -> bool final { return false; }

 private:
  raft::device_matrix<T, int64_t> matrix_;
};

/**
 * Owns a contiguous CUDA virtual address reservation backed by independently mapped physical
 * allocations. Mapping one input-sized range at a time lets Fastener D2D-copy and release each
 * source before committing the next range, bounding temporary physical memory by one subgraph.
 */
template <typename T>
class vmm_dataset_storage final : public contiguous_dataset_storage<T> {
 public:
  vmm_dataset_storage(int64_t n_rows, int64_t dim, std::size_t max_mappings)
    : n_rows_(n_rows), dim_(dim), logical_bytes_(checked_dataset_bytes<T>(n_rows, dim))
  {
    RAFT_EXPECTS(logical_bytes_ > 0, "CUDA VMM storage requires a nonempty dataset");
    RAFT_CUDA_TRY(cudaGetDevice(&device_ordinal_));

    CUdevice device{};
    check_cuda_driver(cuDeviceGet(&device, device_ordinal_), "cuDeviceGet");

    int supported = 0;
    check_cuda_driver(
      cuDeviceGetAttribute(
        &supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device),
      "cuDeviceGetAttribute(VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)");
    if (supported == 0) {
      throw cuda_vmm_error{CUDA_ERROR_NOT_SUPPORTED,
                           "CUDA virtual memory management is not supported on this device"};
    }

    allocation_properties_.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocation_properties_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocation_properties_.location.id   = device_ordinal_;
    check_cuda_driver(cuMemGetAllocationGranularity(
                        &granularity_, &allocation_properties_, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                      "cuMemGetAllocationGranularity");
    RAFT_EXPECTS(granularity_ > 0, "CUDA VMM returned zero allocation granularity");

    RAFT_EXPECTS(logical_bytes_ <= std::numeric_limits<std::size_t>::max() - (granularity_ - 1),
                 "Rounded CUDA VMM allocation size overflows size_t");
    reserved_bytes_ = align_up(logical_bytes_);
    mappings_.reserve(max_mappings);
    check_cuda_driver(cuMemAddressReserve(&address_, reserved_bytes_, granularity_, 0, 0),
                      "cuMemAddressReserve");
  }

  vmm_dataset_storage(vmm_dataset_storage const&)                    = delete;
  auto operator=(vmm_dataset_storage const&) -> vmm_dataset_storage& = delete;

  ~vmm_dataset_storage() noexcept final { release(); }

  void map_until(std::size_t logical_end)
  {
    RAFT_EXPECTS(logical_end <= logical_bytes_, "CUDA VMM mapping exceeds dataset size");
    RAFT_EXPECTS(logical_end >= previous_logical_end_,
                 "CUDA VMM mappings must be populated in logical order");
    auto const target_mapped = align_up(logical_end);
    if (target_mapped <= mapped_bytes_) {
      previous_logical_end_ = logical_end;
      return;
    }

    auto const map_bytes   = target_mapped - mapped_bytes_;
    auto const map_address = address_ + mapped_bytes_;
    CUmemGenericAllocationHandle handle{};
    bool handle_created = false;
    bool mapped         = false;
    try {
      check_cuda_driver(cuMemCreate(&handle, map_bytes, &allocation_properties_, 0), "cuMemCreate");
      handle_created = true;
      check_cuda_driver(cuMemMap(map_address, map_bytes, 0, handle, 0), "cuMemMap");
      mapped = true;

      CUmemAccessDesc access{};
      access.location = allocation_properties_.location;
      access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      check_cuda_driver(cuMemSetAccess(map_address, map_bytes, &access, 1), "cuMemSetAccess");

      mappings_.push_back(mapping{map_address, map_bytes, handle});
      mapped_bytes_      = target_mapped;
      max_mapping_bytes_ = std::max(max_mapping_bytes_, map_bytes);
      max_incremental_scratch_bytes_ =
        std::max(max_incremental_scratch_bytes_, target_mapped - previous_logical_end_);
      previous_logical_end_ = logical_end;
    } catch (...) {
      if (mapped) {
        log_cuda_driver_cleanup_error(cuMemUnmap(map_address, map_bytes), "cuMemUnmap");
      }
      if (handle_created) { log_cuda_driver_cleanup_error(cuMemRelease(handle), "cuMemRelease"); }
      throw;
    }
  }

  [[nodiscard]] auto data_handle() noexcept -> T* final { return reinterpret_cast<T*>(address_); }
  [[nodiscard]] auto data_handle() const noexcept -> T const* final
  {
    return reinterpret_cast<T const*>(address_);
  }
  [[nodiscard]] auto n_rows() const noexcept -> int64_t final { return n_rows_; }
  [[nodiscard]] auto dim() const noexcept -> int64_t final { return dim_; }
  [[nodiscard]] auto physical_bytes() const noexcept -> std::size_t final { return mapped_bytes_; }
  [[nodiscard]] auto is_vmm() const noexcept -> bool final { return true; }
  [[nodiscard]] auto granularity() const noexcept -> std::size_t { return granularity_; }
  [[nodiscard]] auto max_mapping_bytes() const noexcept -> std::size_t
  {
    return max_mapping_bytes_;
  }
  [[nodiscard]] auto max_incremental_scratch_bytes() const noexcept -> std::size_t
  {
    return max_incremental_scratch_bytes_;
  }
  [[nodiscard]] auto fully_mapped() const noexcept -> bool
  {
    return mapped_bytes_ == reserved_bytes_;
  }

 private:
  struct mapping {
    CUdeviceptr address;
    std::size_t bytes;
    CUmemGenericAllocationHandle handle;
  };

  [[nodiscard]] auto align_up(std::size_t bytes) const noexcept -> std::size_t
  {
    return ((bytes + granularity_ - 1) / granularity_) * granularity_;
  }

  void release() noexcept
  {
    if (address_ == 0 || rmm::process_is_exiting()) { return; }

    int previous_device          = device_ordinal_;
    auto const get_device_status = cudaGetDevice(&previous_device);
    if (get_device_status != cudaSuccess) {
      RAFT_LOG_WARN("cagra::merge: cudaGetDevice failed during VMM cleanup: %s",
                    cudaGetErrorString(get_device_status));
      return;
    }

    bool const switch_device = previous_device != device_ordinal_;
    if (switch_device) {
      auto const set_device_status = cudaSetDevice(device_ordinal_);
      if (set_device_status != cudaSuccess) {
        RAFT_LOG_WARN("cagra::merge: cudaSetDevice failed during VMM cleanup: %s",
                      cudaGetErrorString(set_device_status));
        return;
      }
    }

    for (auto it = mappings_.rbegin(); it != mappings_.rend(); ++it) {
      log_cuda_driver_cleanup_error(cuMemUnmap(it->address, it->bytes), "cuMemUnmap");
      log_cuda_driver_cleanup_error(cuMemRelease(it->handle), "cuMemRelease");
    }
    mappings_.clear();
    mapped_bytes_ = 0;
    log_cuda_driver_cleanup_error(cuMemAddressFree(address_, reserved_bytes_), "cuMemAddressFree");
    address_        = 0;
    reserved_bytes_ = 0;

    if (switch_device) {
      auto const restore_status = cudaSetDevice(previous_device);
      if (restore_status != cudaSuccess) {
        RAFT_LOG_WARN("cagra::merge: failed to restore CUDA device after VMM cleanup: %s",
                      cudaGetErrorString(restore_status));
      }
    }
  }

  int64_t n_rows_;
  int64_t dim_;
  std::size_t logical_bytes_;
  int device_ordinal_ = 0;
  CUmemAllocationProp allocation_properties_{};
  std::size_t granularity_                   = 0;
  CUdeviceptr address_                       = 0;
  std::size_t reserved_bytes_                = 0;
  std::size_t mapped_bytes_                  = 0;
  std::size_t max_mapping_bytes_             = 0;
  std::size_t previous_logical_end_          = 0;
  std::size_t max_incremental_scratch_bytes_ = 0;
  std::vector<mapping> mappings_;
};

}  // namespace cuvs::neighbors::cagra::detail
