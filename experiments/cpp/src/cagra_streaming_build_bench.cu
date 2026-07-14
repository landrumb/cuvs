/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/cagra.hpp>
#include <neighbors/detail/cagra/cagra_merge.cuh>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda/std/array>
#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
using time_point = clock_type::time_point;
using index_type = cuvs::neighbors::cagra::index<float, uint32_t>;
using device_matrix_type = raft::device_matrix<float, int64_t>;

constexpr std::uint64_t fbin_header_bytes = 2 * sizeof(std::uint32_t);
constexpr std::uint64_t minimum_free_headroom = 2ull << 30;
constexpr char temp_marker_name[] = ".cuvs_streaming_build_bench";

struct options {
  std::filesystem::path dataset;
  std::filesystem::path temp_dir;
  std::filesystem::path summary_csv;
  std::filesystem::path parts_csv;
  std::string mode;
  std::string build_path = "prefetch-device";
  std::uint32_t parts = 1;
  std::uint32_t run = 1;
  std::uint32_t row_limit = 0;
  double network_seconds = 36.067;
  std::size_t io_chunk_bytes = 64ull << 20;
};

struct source_info {
  std::uint32_t rows = 0;
  std::uint32_t dim = 0;
  std::uint64_t payload_bytes = 0;
};

struct part_spec {
  std::uint32_t id = 0;
  std::uint32_t row_offset = 0;
  std::uint32_t rows = 0;
  std::uint32_t dim = 0;
  std::uint64_t source_offset = 0;
  std::uint64_t payload_bytes = 0;
  std::filesystem::path path;
};

struct interval {
  double start_ms = 0.0;
  double end_ms = 0.0;
};

struct part_metrics {
  interval download;
  interval load;
  interval build;
  double disk_read_ms = 0.0;
  double h2d_ms = 0.0;
  double build_wait_ms = 0.0;
};

struct pipeline_state {
  explicit pipeline_state(std::size_t n)
      : downloaded(n, false), loaded(n, false) {}

  std::mutex mutex;
  std::condition_variable cv;
  std::vector<bool> downloaded;
  std::vector<bool> loaded;
  std::exception_ptr error;
  std::atomic<bool> cancel{false};
};

std::string usage() {
  return R"(Usage: CAGRA_STREAMING_BUILD_BENCH
  --dataset <openai_5m/base.5M.fbin>
  --temp-dir </raid/blandrum/...>
  --summary-csv <summary.csv>
  --parts-csv <parts.csv>
  --mode <local|network>
  --build-path <naive-host|prefetch-device|prefetch-single-build>
  --parts <1|2|4|8>
  --run <positive integer>
  [--row-limit <rows>]
  [--network-seconds <seconds> (default 36.067)]
  [--io-chunk-mib <MiB> (default 64)]

The local mode materializes split files before the timed region, flushes and evicts them from the
page cache, then measures allocation + pipelined disk/H2D/build + Fastener merge. The network mode
paces materialization across --network-seconds and includes it in the timed pipeline.
)";
}

std::uint64_t parse_u64(std::string const &value, std::string const &flag) {
  std::size_t parsed_chars = 0;
  auto parsed = std::stoull(value, &parsed_chars);
  if (parsed_chars != value.size()) {
    throw std::runtime_error("Invalid " + flag);
  }
  return parsed;
}

double parse_double(std::string const &value, std::string const &flag) {
  std::size_t parsed_chars = 0;
  auto parsed = std::stod(value, &parsed_chars);
  if (parsed_chars != value.size() || !std::isfinite(parsed)) {
    throw std::runtime_error("Invalid " + flag);
  }
  return parsed;
}

options parse_args(int argc, char **argv) {
  options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto value = [&]() {
      if (++i >= argc) {
        throw std::runtime_error("Missing value for " + arg);
      }
      return std::string{argv[i]};
    };
    if (arg == "--dataset") {
      opts.dataset = value();
    } else if (arg == "--temp-dir") {
      opts.temp_dir = value();
    } else if (arg == "--summary-csv") {
      opts.summary_csv = value();
    } else if (arg == "--parts-csv") {
      opts.parts_csv = value();
    } else if (arg == "--mode") {
      opts.mode = value();
    } else if (arg == "--build-path") {
      opts.build_path = value();
    } else if (arg == "--parts") {
      opts.parts = static_cast<std::uint32_t>(parse_u64(value(), arg));
    } else if (arg == "--run") {
      opts.run = static_cast<std::uint32_t>(parse_u64(value(), arg));
    } else if (arg == "--row-limit") {
      opts.row_limit = static_cast<std::uint32_t>(parse_u64(value(), arg));
    } else if (arg == "--network-seconds") {
      opts.network_seconds = parse_double(value(), arg);
    } else if (arg == "--io-chunk-mib") {
      opts.io_chunk_bytes = static_cast<std::size_t>(parse_u64(value(), arg))
                            << 20;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << usage();
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (opts.dataset.empty() || opts.temp_dir.empty() ||
      opts.summary_csv.empty() || opts.parts_csv.empty() || opts.mode.empty()) {
    throw std::runtime_error("All required arguments must be provided\n" +
                             usage());
  }
  if (opts.mode != "local" && opts.mode != "network") {
    throw std::runtime_error("--mode must be local or network");
  }
  if (opts.build_path != "naive-host" && opts.build_path != "prefetch-device" &&
      opts.build_path != "prefetch-single-build") {
    throw std::runtime_error(
        "--build-path must be naive-host, prefetch-device, or "
        "prefetch-single-build");
  }
  if (opts.build_path == "naive-host" && opts.parts != 1) {
    throw std::runtime_error("--build-path naive-host requires --parts 1");
  }
  if (opts.build_path == "prefetch-single-build" &&
      (opts.mode != "network" || opts.parts != 8)) {
    throw std::runtime_error(
        "--build-path prefetch-single-build requires --mode network --parts 8");
  }
  if (opts.parts != 1 && opts.parts != 2 && opts.parts != 4 &&
      opts.parts != 8) {
    throw std::runtime_error("--parts must be 1, 2, 4, or 8");
  }
  if (opts.run == 0) {
    throw std::runtime_error("--run must be positive");
  }
  if (opts.network_seconds <= 0.0) {
    throw std::runtime_error("--network-seconds must be positive");
  }
  if (opts.io_chunk_bytes == 0 || opts.io_chunk_bytes > (1ull << 30)) {
    throw std::runtime_error("--io-chunk-mib must be between 1 and 1024");
  }

  auto normalized = opts.temp_dir.lexically_normal().string();
  if (!normalized.starts_with("/raid/blandrum/")) {
    throw std::runtime_error("--temp-dir must be below /raid/blandrum/");
  }
  return opts;
}

double elapsed_ms(time_point start, time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double relative_ms(time_point origin, time_point point) {
  return elapsed_ms(origin, point);
}

[[noreturn]] void throw_system_error(std::string const &operation,
                                     std::filesystem::path const &path) {
  int error = errno;
  throw std::runtime_error(operation + " " + path.string() + ": " +
                           std::strerror(error));
}

class file_descriptor {
public:
  file_descriptor(std::filesystem::path path, int flags, mode_t mode = 0)
      : path_(std::move(path)) {
    fd_ = mode == 0 ? ::open(path_.c_str(), flags)
                    : ::open(path_.c_str(), flags, mode);
    if (fd_ < 0) {
      throw_system_error("Could not open", path_);
    }
  }

  file_descriptor(file_descriptor const &) = delete;
  auto operator=(file_descriptor const &) -> file_descriptor & = delete;

  ~file_descriptor() {
    if (fd_ >= 0) {
      static_cast<void>(::close(fd_));
    }
  }

  [[nodiscard]] auto get() const noexcept -> int { return fd_; }
  [[nodiscard]] auto path() const noexcept -> std::filesystem::path const & {
    return path_;
  }

private:
  std::filesystem::path path_;
  int fd_ = -1;
};

void pread_exact(file_descriptor const &file, void *destination,
                 std::size_t bytes, std::uint64_t offset) {
  auto *out = static_cast<std::byte *>(destination);
  std::size_t done = 0;
  while (done < bytes) {
    auto result = ::pread(file.get(), out + done, bytes - done,
                          static_cast<off_t>(offset + done));
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result < 0) {
      throw_system_error("Could not read", file.path());
    }
    if (result == 0) {
      throw std::runtime_error("Unexpected EOF in " + file.path().string());
    }
    done += static_cast<std::size_t>(result);
  }
}

void write_exact(file_descriptor const &file, void const *source,
                 std::size_t bytes) {
  auto const *input = static_cast<std::byte const *>(source);
  std::size_t done = 0;
  while (done < bytes) {
    auto result = ::write(file.get(), input + done, bytes - done);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result < 0) {
      throw_system_error("Could not write", file.path());
    }
    done += static_cast<std::size_t>(result);
  }
}

void sync_file(file_descriptor const &file) {
  while (::fdatasync(file.get()) != 0) {
    if (errno == EINTR) {
      continue;
    }
    throw_system_error("Could not sync", file.path());
  }
}

void evict_file(file_descriptor const &file) {
  int result = ::posix_fadvise(file.get(), 0, 0, POSIX_FADV_DONTNEED);
  if (result != 0) {
    throw std::runtime_error("posix_fadvise failed for " +
                             file.path().string() + ": " +
                             std::strerror(result));
  }
}

source_info inspect_source(std::filesystem::path const &path) {
  file_descriptor file(path, O_RDONLY | O_CLOEXEC);
  std::array<std::uint32_t, 2> header{};
  pread_exact(file, header.data(), fbin_header_bytes, 0);
  source_info info;
  info.rows = header[0];
  info.dim = header[1];
  info.payload_bytes =
      static_cast<std::uint64_t>(info.rows) * info.dim * sizeof(float);
  auto expected_size = fbin_header_bytes + info.payload_bytes;
  if (std::filesystem::file_size(path) != expected_size) {
    throw std::runtime_error(
        "Dataset is not a float32 fbin with the declared shape");
  }
  return info;
}

std::vector<part_spec> make_parts(source_info const &source, std::uint32_t rows,
                                  std::uint32_t count,
                                  std::filesystem::path const &temp_dir) {
  std::vector<part_spec> result;
  result.reserve(count);
  std::uint32_t row_offset = 0;
  auto base_rows = rows / count;
  auto remainder = rows % count;
  for (std::uint32_t id = 0; id < count; ++id) {
    auto part_rows = base_rows + (id < remainder ? 1 : 0);
    auto payload =
        static_cast<std::uint64_t>(part_rows) * source.dim * sizeof(float);
    auto source_start =
        fbin_header_bytes +
        static_cast<std::uint64_t>(row_offset) * source.dim * sizeof(float);
    result.push_back(
        part_spec{id, row_offset, part_rows, source.dim, source_start, payload,
                  temp_dir / ("part_" + std::to_string(id) + ".fbin")});
    row_offset += part_rows;
  }
  return result;
}

class temp_dir_guard {
public:
  explicit temp_dir_guard(std::filesystem::path dir) : dir_(std::move(dir)) {
    auto marker = dir_ / temp_marker_name;
    if (std::filesystem::exists(dir_)) {
      if (!std::filesystem::exists(marker)) {
        throw std::runtime_error("Refusing to remove unmarked temp directory " +
                                 dir_.string());
      }
      std::filesystem::remove_all(dir_);
    }
    std::filesystem::create_directories(dir_);
    std::ofstream out(marker);
    if (!out) {
      throw std::runtime_error("Could not create temp directory marker");
    }
    out << "Temporary OpenAI-5M streaming benchmark files. Safe to remove.\n";
  }

  temp_dir_guard(temp_dir_guard const &) = delete;
  auto operator=(temp_dir_guard const &) -> temp_dir_guard & = delete;

  ~temp_dir_guard() {
    std::error_code error;
    std::filesystem::remove_all(dir_, error);
    if (error) {
      std::cerr << "warning: failed to clean " << dir_ << ": "
                << error.message() << '\n';
    }
  }

  [[nodiscard]] auto path() const noexcept -> std::filesystem::path const & {
    return dir_;
  }

private:
  std::filesystem::path dir_;
};

void set_error(pipeline_state &state, std::exception_ptr error) {
  {
    std::lock_guard lock(state.mutex);
    if (!state.error) {
      state.error = std::move(error);
    }
  }
  state.cv.notify_all();
}

void notify_downloaded(pipeline_state &state, std::size_t part) {
  {
    std::lock_guard lock(state.mutex);
    state.downloaded[part] = true;
  }
  state.cv.notify_all();
}

void copy_parts(std::filesystem::path const &source_path,
                std::vector<part_spec> const &parts, std::size_t chunk_bytes,
                double pace_seconds, time_point origin,
                std::vector<part_metrics> *metrics, pipeline_state *state,
                bool evict_outputs) {
  file_descriptor source(source_path, O_RDONLY | O_CLOEXEC);
  std::vector<std::byte> buffer(chunk_bytes);
  std::uint64_t total_bytes = 0;
  for (auto const &part : parts) {
    total_bytes += part.payload_bytes;
  }

  auto pace_start = clock_type::now();
  std::uint64_t cumulative = 0;
  for (auto const &part : parts) {
    if (state != nullptr && state->cancel.load()) {
      return;
    }
    if (metrics != nullptr) {
      (*metrics)[part.id].download.start_ms =
          relative_ms(origin, clock_type::now());
    }

    file_descriptor output(part.path, O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC,
                           0644);
    std::array<std::uint32_t, 2> header{part.rows, part.dim};
    write_exact(output, header.data(), fbin_header_bytes);

    std::uint64_t copied = 0;
    while (copied < part.payload_bytes) {
      if (state != nullptr && state->cancel.load()) {
        return;
      }
      auto bytes = static_cast<std::size_t>(
          std::min<std::uint64_t>(chunk_bytes, part.payload_bytes - copied));
      pread_exact(source, buffer.data(), bytes, part.source_offset + copied);
      write_exact(output, buffer.data(), bytes);
      copied += bytes;
      cumulative += bytes;
      if (pace_seconds > 0.0) {
        auto target_seconds = pace_seconds * static_cast<double>(cumulative) /
                              static_cast<double>(total_bytes);
        auto target =
            pace_start + std::chrono::duration_cast<clock_type::duration>(
                             std::chrono::duration<double>(target_seconds));
        std::this_thread::sleep_until(target);
      }
    }
    sync_file(output);
    if (evict_outputs) {
      evict_file(output);
    }
    static_cast<void>(::posix_fadvise(
        source.get(), static_cast<off_t>(part.source_offset),
        static_cast<off_t>(part.payload_bytes), POSIX_FADV_DONTNEED));

    if (metrics != nullptr) {
      (*metrics)[part.id].download.end_ms =
          relative_ms(origin, clock_type::now());
    }
    if (state != nullptr) {
      notify_downloaded(*state, part.id);
    }
  }
}

std::vector<std::byte> preload_payload(std::filesystem::path const &source_path,
                                       std::uint64_t payload_bytes,
                                       std::size_t chunk_bytes) {
  file_descriptor source(source_path, O_RDONLY | O_CLOEXEC);
  std::vector<std::byte> payload(static_cast<std::size_t>(payload_bytes));
  std::uint64_t copied = 0;
  while (copied < payload_bytes) {
    auto bytes = static_cast<std::size_t>(
        std::min<std::uint64_t>(chunk_bytes, payload_bytes - copied));
    pread_exact(source, payload.data() + copied, bytes,
                fbin_header_bytes + copied);
    copied += bytes;
  }
  static_cast<void>(::posix_fadvise(source.get(), 0, 0, POSIX_FADV_DONTNEED));
  return payload;
}

void write_memory_parts(std::vector<std::byte> const &payload,
                        std::vector<part_spec> const &parts,
                        std::size_t chunk_bytes, double pace_seconds,
                        time_point origin, std::vector<part_metrics> &metrics,
                        pipeline_state &state) {
  std::uint64_t total_bytes = 0;
  for (auto const &part : parts) {
    total_bytes += part.payload_bytes;
  }
  if (payload.size() != total_bytes) {
    throw std::runtime_error(
        "Preloaded payload size does not match requested rows");
  }

  auto pace_start = clock_type::now();
  std::uint64_t cumulative = 0;
  for (auto const &part : parts) {
    if (state.cancel.load()) {
      return;
    }
    metrics[part.id].download.start_ms = relative_ms(origin, clock_type::now());
    file_descriptor output(part.path, O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC,
                           0644);
    std::array<std::uint32_t, 2> header{part.rows, part.dim};
    write_exact(output, header.data(), fbin_header_bytes);

    auto memory_offset =
        static_cast<std::uint64_t>(part.row_offset) * part.dim * sizeof(float);
    std::uint64_t copied = 0;
    while (copied < part.payload_bytes) {
      if (state.cancel.load()) {
        return;
      }
      auto bytes = static_cast<std::size_t>(
          std::min<std::uint64_t>(chunk_bytes, part.payload_bytes - copied));
      write_exact(output, payload.data() + memory_offset + copied, bytes);
      copied += bytes;
      cumulative += bytes;

      auto target_seconds = pace_seconds * static_cast<double>(cumulative) /
                            static_cast<double>(total_bytes);
      auto target =
          pace_start + std::chrono::duration_cast<clock_type::duration>(
                           std::chrono::duration<double>(target_seconds));
      std::this_thread::sleep_until(target);
    }
    metrics[part.id].download.end_ms = relative_ms(origin, clock_type::now());
    notify_downloaded(state, part.id);
  }
}

class pinned_copy_engine {
public:
  explicit pinned_copy_engine(std::size_t bytes) : bytes_(bytes) {
    RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    for (std::size_t i = 0; i < buffers_.size(); ++i) {
      RAFT_CUDA_TRY(cudaHostAlloc(&buffers_[i], bytes_, cudaHostAllocPortable));
      RAFT_CUDA_TRY(cudaEventCreate(&start_events_[i]));
      RAFT_CUDA_TRY(cudaEventCreate(&end_events_[i]));
    }
  }

  pinned_copy_engine(pinned_copy_engine const &) = delete;
  auto operator=(pinned_copy_engine const &) -> pinned_copy_engine & = delete;

  ~pinned_copy_engine() {
    for (std::size_t i = 0; i < buffers_.size(); ++i) {
      if (start_events_[i] != nullptr) {
        static_cast<void>(cudaEventDestroy(start_events_[i]));
      }
      if (end_events_[i] != nullptr) {
        static_cast<void>(cudaEventDestroy(end_events_[i]));
      }
      if (buffers_[i] != nullptr) {
        static_cast<void>(cudaFreeHost(buffers_[i]));
      }
    }
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  void load(part_spec const &part, float *destination, time_point origin,
            part_metrics &metrics, bool evict_after_read) {
    file_descriptor input(part.path, O_RDONLY | O_CLOEXEC);
    std::array<std::uint32_t, 2> header{};
    pread_exact(input, header.data(), fbin_header_bytes, 0);
    if (header[0] != part.rows || header[1] != part.dim) {
      throw std::runtime_error("Part header mismatch in " + part.path.string());
    }

    metrics.load.start_ms = relative_ms(origin, clock_type::now());
    std::array<bool, 2> in_flight{false, false};
    auto finish = [&](std::size_t slot) {
      if (!in_flight[slot]) {
        return;
      }
      RAFT_CUDA_TRY(cudaEventSynchronize(end_events_[slot]));
      float milliseconds = 0.0f;
      RAFT_CUDA_TRY(cudaEventElapsedTime(&milliseconds, start_events_[slot],
                                         end_events_[slot]));
      metrics.h2d_ms += milliseconds;
      in_flight[slot] = false;
    };

    std::uint64_t copied = 0;
    std::size_t chunk = 0;
    while (copied < part.payload_bytes) {
      auto slot = chunk % buffers_.size();
      finish(slot);
      auto bytes = static_cast<std::size_t>(
          std::min<std::uint64_t>(bytes_, part.payload_bytes - copied));
      auto read_start = clock_type::now();
      pread_exact(input, buffers_[slot], bytes, fbin_header_bytes + copied);
      metrics.disk_read_ms += elapsed_ms(read_start, clock_type::now());

      RAFT_CUDA_TRY(cudaEventRecord(start_events_[slot], stream_));
      RAFT_CUDA_TRY(cudaMemcpyAsync(destination + copied / sizeof(float),
                                    buffers_[slot], bytes,
                                    cudaMemcpyHostToDevice, stream_));
      RAFT_CUDA_TRY(cudaEventRecord(end_events_[slot], stream_));
      in_flight[slot] = true;
      copied += bytes;
      ++chunk;
    }
    for (std::size_t slot = 0; slot < buffers_.size(); ++slot) {
      finish(slot);
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream_));
    if (evict_after_read) {
      evict_file(input);
    }
    metrics.load.end_ms = relative_ms(origin, clock_type::now());
  }

private:
  std::size_t bytes_;
  cudaStream_t stream_ = nullptr;
  std::array<void *, 2> buffers_{};
  std::array<cudaEvent_t, 2> start_events_{};
  std::array<cudaEvent_t, 2> end_events_{};
};

void load_pipeline(std::vector<part_spec> const &parts,
                   std::vector<device_matrix_type> &device_parts,
                   pinned_copy_engine &copy_engine, time_point origin,
                   std::vector<part_metrics> &metrics, pipeline_state &state,
                   bool evict_after_read) {
  try {
    for (auto const &part : parts) {
      {
        std::unique_lock lock(state.mutex);
        state.cv.wait(lock, [&] {
          return state.downloaded[part.id] || state.error ||
                 state.cancel.load();
        });
        if (state.error || state.cancel.load()) {
          return;
        }
      }
      copy_engine.load(part, device_parts[part.id].data_handle(), origin,
                       metrics[part.id], evict_after_read);
      {
        std::lock_guard lock(state.mutex);
        state.loaded[part.id] = true;
      }
      state.cv.notify_all();
    }
  } catch (...) {
    set_error(state, std::current_exception());
  }
}

void rethrow_pipeline_error(pipeline_state &state) {
  std::exception_ptr error;
  {
    std::lock_guard lock(state.mutex);
    error = state.error;
  }
  if (error) {
    std::rethrow_exception(error);
  }
}

double intersection_ms(interval lhs, interval rhs) {
  return std::max(0.0, std::min(lhs.end_ms, rhs.end_ms) -
                           std::max(lhs.start_ms, rhs.start_ms));
}

template <typename Matrix>
void attach_owned_dataset(raft::resources const &resources, index_type &index,
                          Matrix &&matrix) {
  using matrix_type = std::remove_reference_t<Matrix>;
  using layout_type = typename matrix_type::layout_type;
  using container_type = typename matrix_type::container_policy_type;
  using owning_dataset_type =
      cuvs::neighbors::owning_dataset<float, int64_t, layout_type,
                                      container_type>;
  auto dim = matrix.extent(1);
  auto layout = raft::make_strided_layout(matrix.view().extents(),
                                          cuda::std::array<int64_t, 2>{dim, 1});
  index.update_dataset(resources,
                       owning_dataset_type{std::move(matrix), layout});
}

void append_summary(options const &opts, source_info const &source,
                    std::uint32_t rows, double prepare_ms, interval allocation,
                    std::vector<part_metrics> const &metrics, interval merge,
                    double total_ms, bool valid) {
  double download_end = 0.0;
  double download_start = std::numeric_limits<double>::max();
  double disk_read_ms = 0.0;
  double h2d_ms = 0.0;
  double load_wall_sum = 0.0;
  double build_sum = 0.0;
  double wait_sum = 0.0;
  for (auto const &metric : metrics) {
    download_end = std::max(download_end, metric.download.end_ms);
    if (metric.download.end_ms > 0.0) {
      download_start = std::min(download_start, metric.download.start_ms);
    }
    disk_read_ms += metric.disk_read_ms;
    h2d_ms += metric.h2d_ms;
    load_wall_sum += metric.load.end_ms - metric.load.start_ms;
    build_sum += metric.build.end_ms - metric.build.start_ms;
    wait_sum += metric.build_wait_ms;
  }
  if (download_start == std::numeric_limits<double>::max()) {
    download_start = 0.0;
  }
  auto download_wall = std::max(0.0, download_end - download_start);
  auto allocation_ms = allocation.end_ms - allocation.start_ms;
  auto merge_ms = merge.end_ms - merge.start_ms;
  auto serial_work =
      allocation_ms + download_wall + load_wall_sum + build_sum + merge_ms;
  auto overlap_saved = serial_work - total_ms;

  double load_build_overlap = 0.0;
  double download_build_overlap = 0.0;
  double download_load_overlap = 0.0;
  interval download_interval{download_start, download_end};
  for (auto const &lhs : metrics) {
    for (auto const &rhs : metrics) {
      load_build_overlap += intersection_ms(lhs.load, rhs.build);
    }
    download_build_overlap += intersection_ms(download_interval, lhs.build);
    download_load_overlap += intersection_ms(download_interval, lhs.load);
  }
  auto allocation_download_overlap =
      intersection_ms(allocation, download_interval);
  auto bytes = static_cast<std::uint64_t>(rows) * source.dim * sizeof(float);
  auto achieved_mib_s =
      download_wall > 0.0
          ? static_cast<double>(bytes) / (1ull << 20) / (download_wall / 1000.0)
          : 0.0;

  bool exists = std::filesystem::exists(opts.summary_csv);
  std::ofstream out(opts.summary_csv, std::ios::app);
  if (!out) {
    throw std::runtime_error("Could not write " + opts.summary_csv.string());
  }
  if (!exists) {
    out << "dataset,mode,build_path,parts,run,rows,dim,data_bytes,network_"
           "target_s,"
           "prepare_ms_excluded,"
           "device_alloc_ms,download_wall_ms,download_achieved_mib_s,disk_read_"
           "active_ms,"
           "h2d_active_ms,load_wall_sum_ms,build_sum_ms,build_wait_ms,merge_ms,"
           "total_ms,"
           "serial_work_ms,overlap_saved_ms,load_build_overlap_ms,download_"
           "build_overlap_ms,"
           "download_load_overlap_ms,allocation_download_overlap_ms,graph_"
           "degree,"
           "intermediate_graph_degree,valid\n";
  }
  cuvs::neighbors::cagra::index_params defaults;
  out << "openai_5m," << opts.mode << ',' << opts.build_path << ','
      << opts.parts << ',' << opts.run << ',' << rows << ',' << source.dim
      << ',' << bytes << ',' << std::fixed << std::setprecision(6)
      << (opts.mode == "network" ? opts.network_seconds : 0.0) << ','
      << prepare_ms << ',' << allocation_ms << ',' << download_wall << ','
      << achieved_mib_s << ',' << disk_read_ms << ',' << h2d_ms << ','
      << load_wall_sum << ',' << build_sum << ',' << wait_sum << ',' << merge_ms
      << ',' << total_ms << ',' << serial_work << ',' << overlap_saved << ','
      << load_build_overlap << ',' << download_build_overlap << ','
      << download_load_overlap << ',' << allocation_download_overlap << ','
      << defaults.graph_degree << ',' << defaults.intermediate_graph_degree
      << ',' << (valid ? 1 : 0) << '\n';
}

void append_parts(options const &opts, std::vector<part_spec> const &parts,
                  std::vector<part_metrics> const &metrics) {
  bool exists = std::filesystem::exists(opts.parts_csv);
  std::ofstream out(opts.parts_csv, std::ios::app);
  if (!out) {
    throw std::runtime_error("Could not write " + opts.parts_csv.string());
  }
  if (!exists) {
    out << "dataset,mode,build_path,parts,run,part,row_offset,rows,payload_"
           "bytes,download_"
           "start_ms,"
           "download_end_ms,load_start_ms,load_end_ms,load_wall_ms,disk_read_"
           "active_ms,"
           "h2d_active_ms,build_wait_ms,build_start_ms,build_end_ms,build_ms\n";
  }
  for (auto const &part : parts) {
    auto const &metric = metrics[part.id];
    out << "openai_5m," << opts.mode << ',' << opts.build_path << ','
        << opts.parts << ',' << opts.run << ',' << part.id << ','
        << part.row_offset << ',' << part.rows << ',' << part.payload_bytes
        << ',' << std::fixed << std::setprecision(6) << metric.download.start_ms
        << ',' << metric.download.end_ms << ',' << metric.load.start_ms << ','
        << metric.load.end_ms << ','
        << (metric.load.end_ms - metric.load.start_ms) << ','
        << metric.disk_read_ms << ',' << metric.h2d_ms << ','
        << metric.build_wait_ms << ',' << metric.build.start_ms << ','
        << metric.build.end_ms << ','
        << (metric.build.end_ms - metric.build.start_ms) << '\n';
  }
}

int run_naive_host(options const &opts, source_info const &source,
                   std::uint32_t rows, std::vector<part_spec> const &parts,
                   std::vector<part_metrics> &metrics, double prepare_ms,
                   std::vector<std::byte> const &network_payload) {
  auto const &part = parts.front();
  raft::resources resources;
  pipeline_state state(1);
  if (opts.mode == "local") {
    state.downloaded[0] = true;
  }

  auto total_start = clock_type::now();
  std::thread downloader;
  if (opts.mode == "network") {
    downloader = std::thread([&] {
      try {
        write_memory_parts(network_payload, parts, opts.io_chunk_bytes,
                           opts.network_seconds, total_start, metrics, state);
      } catch (...) {
        set_error(state, std::current_exception());
      }
    });
  }

  try {
    auto wait_start = clock_type::now();
    {
      std::unique_lock lock(state.mutex);
      state.cv.wait(lock, [&] {
        return state.downloaded[0] || state.error || state.cancel.load();
      });
    }
    metrics[0].build_wait_ms = elapsed_ms(wait_start, clock_type::now());
    rethrow_pipeline_error(state);
  } catch (...) {
    state.cancel.store(true);
    state.cv.notify_all();
    if (downloader.joinable()) {
      downloader.join();
    }
    throw;
  }
  if (downloader.joinable()) {
    downloader.join();
  }
  rethrow_pipeline_error(state);

  metrics[0].load.start_ms = relative_ms(total_start, clock_type::now());
  file_descriptor input(part.path, O_RDONLY | O_CLOEXEC);
  std::array<std::uint32_t, 2> header{};
  pread_exact(input, header.data(), fbin_header_bytes, 0);
  if (header[0] != rows || header[1] != source.dim) {
    throw std::runtime_error("Naive input header mismatch");
  }
  std::vector<float> host_data(
      static_cast<std::size_t>(part.payload_bytes / sizeof(float)));
  std::uint64_t copied = 0;
  while (copied < part.payload_bytes) {
    auto bytes = static_cast<std::size_t>(std::min<std::uint64_t>(
        opts.io_chunk_bytes, part.payload_bytes - copied));
    auto read_start = clock_type::now();
    pread_exact(input, reinterpret_cast<std::byte *>(host_data.data()) + copied,
                bytes, fbin_header_bytes + copied);
    metrics[0].disk_read_ms += elapsed_ms(read_start, clock_type::now());
    copied += bytes;
  }
  if (opts.mode == "local") {
    evict_file(input);
  }
  metrics[0].load.end_ms = relative_ms(total_start, clock_type::now());

  metrics[0].build.start_ms = relative_ms(total_start, clock_type::now());
  cuvs::neighbors::cagra::index_params build_params;
  auto view = raft::make_host_matrix_view<const float, int64_t>(
      host_data.data(), rows, source.dim);
  auto final_index =
      cuvs::neighbors::cagra::build(resources, build_params, view);
  raft::resource::sync_stream(resources);
  metrics[0].build.end_ms = relative_ms(total_start, clock_type::now());

  bool valid = final_index.size() == rows && final_index.dim() == source.dim &&
               final_index.graph_degree() == 64 &&
               final_index.graph().extent(0) == rows &&
               final_index.dataset().extent(0) == rows &&
               final_index.dataset().extent(1) == source.dim &&
               final_index.data().is_owning();
  if (!valid) {
    throw std::runtime_error("Naive CAGRA index failed validity checks");
  }
  auto total_ms = elapsed_ms(total_start, clock_type::now());
  interval allocation;
  interval merge;
  append_summary(opts, source, rows, prepare_ms, allocation, metrics, merge,
                 total_ms, valid);
  append_parts(opts, parts, metrics);
  std::cout << std::fixed << std::setprecision(3) << "RESULT mode=" << opts.mode
            << " build_path=" << opts.build_path << " parts=1 run=" << opts.run
            << " rows=" << rows << " total_ms=" << total_ms
            << " merge_ms=0 valid=" << valid << '\n';
  return 0;
}

int run_prefetch_single_build(options const &opts, source_info const &source,
                              std::uint32_t rows,
                              std::vector<part_spec> const &parts,
                              std::vector<part_metrics> &metrics,
                              double prepare_ms,
                              std::vector<std::byte> const &network_payload) {
  raft::resources resources;
  pinned_copy_engine copy_engine(opts.io_chunk_bytes);
  pipeline_state state(opts.parts);

  auto total_start = clock_type::now();
  std::thread downloader([&] {
    try {
      write_memory_parts(network_payload, parts, opts.io_chunk_bytes,
                         opts.network_seconds, total_start, metrics, state);
    } catch (...) {
      set_error(state, std::current_exception());
    }
  });

  interval allocation;
  allocation.start_ms = relative_ms(total_start, clock_type::now());
  auto device_dataset =
      raft::make_device_matrix<float, int64_t>(resources, rows, source.dim);
  allocation.end_ms = relative_ms(total_start, clock_type::now());

  std::thread loader([&] {
    try {
      for (auto const &part : parts) {
        {
          std::unique_lock lock(state.mutex);
          state.cv.wait(lock, [&] {
            return state.downloaded[part.id] || state.error ||
                   state.cancel.load();
          });
          if (state.error || state.cancel.load()) {
            return;
          }
        }
        auto offset = static_cast<std::uint64_t>(part.row_offset) * part.dim;
        copy_engine.load(part, device_dataset.data_handle() + offset,
                         total_start, metrics[part.id], false);
        {
          std::lock_guard lock(state.mutex);
          state.loaded[part.id] = true;
        }
        state.cv.notify_all();
      }
    } catch (...) {
      set_error(state, std::current_exception());
    }
  });

  auto wait_start = clock_type::now();
  try {
    loader.join();
    downloader.join();
    rethrow_pipeline_error(state);
  } catch (...) {
    state.cancel.store(true);
    state.cv.notify_all();
    if (loader.joinable()) {
      loader.join();
    }
    if (downloader.joinable()) {
      downloader.join();
    }
    throw;
  }

  auto &build_metric = metrics.back();
  build_metric.build_wait_ms = elapsed_ms(wait_start, clock_type::now());
  build_metric.build.start_ms = relative_ms(total_start, clock_type::now());
  cuvs::neighbors::cagra::index_params build_params;
  build_params.attach_dataset_on_build = false;
  auto view = raft::make_device_matrix_view<const float, int64_t>(
      device_dataset.data_handle(), rows, source.dim);
  auto final_index =
      cuvs::neighbors::cagra::build(resources, build_params, view);
  attach_owned_dataset(resources, final_index, std::move(device_dataset));
  raft::resource::sync_stream(resources);
  build_metric.build.end_ms = relative_ms(total_start, clock_type::now());

  bool valid = final_index.size() == rows && final_index.dim() == source.dim &&
               final_index.graph_degree() == 64 &&
               final_index.graph().extent(0) == rows &&
               final_index.dataset().extent(0) == rows &&
               final_index.dataset().extent(1) == source.dim &&
               final_index.data().is_owning();
  if (!valid) {
    throw std::runtime_error(
        "Prefetch single-build CAGRA index failed validity checks");
  }

  auto total_ms = elapsed_ms(total_start, clock_type::now());
  interval merge;
  append_summary(opts, source, rows, prepare_ms, allocation, metrics, merge,
                 total_ms, valid);
  append_parts(opts, parts, metrics);
  std::cout << std::fixed << std::setprecision(3) << "RESULT mode=" << opts.mode
            << " build_path=" << opts.build_path << " parts=" << opts.parts
            << " run=" << opts.run << " rows=" << rows
            << " total_ms=" << total_ms << " merge_ms=0 valid=" << valid
            << '\n';
  return 0;
}

int run(options const &opts) {
  auto source = inspect_source(opts.dataset);
  auto rows = opts.row_limit == 0 ? source.rows : opts.row_limit;
  if (rows == 0 || rows > source.rows) {
    throw std::runtime_error("Invalid --row-limit");
  }
  if (opts.parts > rows) {
    throw std::runtime_error("More parts than rows");
  }

  temp_dir_guard temp_dir(opts.temp_dir);
  auto parts = make_parts(source, rows, opts.parts, temp_dir.path());
  auto required_bytes =
      static_cast<std::uint64_t>(rows) * source.dim * sizeof(float) +
      minimum_free_headroom;
  auto space = std::filesystem::space(temp_dir.path().parent_path());
  if (space.available < required_bytes) {
    throw std::runtime_error(
        "Insufficient free space for one temporary split set");
  }

  std::vector<part_metrics> metrics(opts.parts);
  std::vector<std::byte> network_payload;
  double prepare_ms = 0.0;
  if (opts.mode == "local") {
    auto prepare_start = clock_type::now();
    copy_parts(opts.dataset, parts, opts.io_chunk_bytes, 0.0, prepare_start,
               nullptr, nullptr, true);
    prepare_ms = elapsed_ms(prepare_start, clock_type::now());
  } else {
    auto prepare_start = clock_type::now();
    auto payload_bytes =
        static_cast<std::uint64_t>(rows) * source.dim * sizeof(float);
    network_payload =
        preload_payload(opts.dataset, payload_bytes, opts.io_chunk_bytes);
    prepare_ms = elapsed_ms(prepare_start, clock_type::now());
  }

  if (opts.build_path == "naive-host") {
    return run_naive_host(opts, source, rows, parts, metrics, prepare_ms,
                          network_payload);
  }
  if (opts.build_path == "prefetch-single-build") {
    return run_prefetch_single_build(opts, source, rows, parts, metrics,
                                     prepare_ms, network_payload);
  }

  if (opts.parts > 1) {
    if (!cuvs::neighbors::cagra::detail::cuda_vmm_supported()) {
      throw std::runtime_error("Fastener VMM is not supported on this device");
    }
    if (!cuvs::neighbors::cagra::detail::
            current_device_resource_releases_to_cuda()) {
      throw std::runtime_error(
          "Fastener VMM requires RMM's direct cuda_memory_resource");
    }
  }

  raft::resources resources;
  pinned_copy_engine copy_engine(opts.io_chunk_bytes);
  pipeline_state state(opts.parts);
  if (opts.mode == "local") {
    std::fill(state.downloaded.begin(), state.downloaded.end(), true);
  }

  auto total_start = clock_type::now();
  std::thread downloader;
  if (opts.mode == "network") {
    downloader = std::thread([&] {
      try {
        write_memory_parts(network_payload, parts, opts.io_chunk_bytes,
                           opts.network_seconds, total_start, metrics, state);
      } catch (...) {
        set_error(state, std::current_exception());
      }
    });
  }

  interval allocation;
  allocation.start_ms = relative_ms(total_start, clock_type::now());
  std::vector<device_matrix_type> device_parts;
  device_parts.reserve(opts.parts);
  for (auto const &part : parts) {
    device_parts.emplace_back(raft::make_device_matrix<float, int64_t>(
        resources, part.rows, part.dim));
  }
  allocation.end_ms = relative_ms(total_start, clock_type::now());

  std::thread loader([&] {
    load_pipeline(parts, device_parts, copy_engine, total_start, metrics, state,
                  opts.mode == "local");
  });

  std::vector<index_type> owned_indices;
  std::vector<index_type *> index_ptrs;
  owned_indices.reserve(opts.parts);
  index_ptrs.reserve(opts.parts);

  try {
    for (auto const &part : parts) {
      auto wait_start = clock_type::now();
      {
        std::unique_lock lock(state.mutex);
        state.cv.wait(lock, [&] {
          return state.loaded[part.id] || state.error || state.cancel.load();
        });
      }
      metrics[part.id].build_wait_ms =
          elapsed_ms(wait_start, clock_type::now());
      rethrow_pipeline_error(state);
      if (state.cancel.load()) {
        throw std::runtime_error("Pipeline cancelled");
      }

      metrics[part.id].build.start_ms =
          relative_ms(total_start, clock_type::now());
      cuvs::neighbors::cagra::index_params build_params;
      build_params.attach_dataset_on_build = false;
      auto view = raft::make_device_matrix_view<const float, int64_t>(
          device_parts[part.id].data_handle(), part.rows, part.dim);
      owned_indices.emplace_back(
          cuvs::neighbors::cagra::build(resources, build_params, view));
      attach_owned_dataset(resources, owned_indices.back(),
                           std::move(device_parts[part.id]));
      raft::resource::sync_stream(resources);
      metrics[part.id].build.end_ms =
          relative_ms(total_start, clock_type::now());
      index_ptrs.push_back(&owned_indices.back());
      std::cout << "part " << part.id << " rows=" << part.rows << " load_ms="
                << metrics[part.id].load.end_ms - metrics[part.id].load.start_ms
                << " build_ms="
                << metrics[part.id].build.end_ms -
                       metrics[part.id].build.start_ms
                << '\n';
    }
  } catch (...) {
    state.cancel.store(true);
    state.cv.notify_all();
    if (loader.joinable()) {
      loader.join();
    }
    if (downloader.joinable()) {
      downloader.join();
    }
    throw;
  }

  loader.join();
  if (downloader.joinable()) {
    downloader.join();
  }
  rethrow_pipeline_error(state);

  interval merge;
  std::optional<index_type> final_index;
  if (opts.parts == 1) {
    final_index.emplace(std::move(owned_indices.front()));
  } else {
    merge.start_ms = relative_ms(total_start, clock_type::now());
    cuvs::neighbors::cagra::index_params merge_params;
    final_index.emplace(cuvs::neighbors::cagra::detail::merge_with_k4_scaffold(
        resources, merge_params, index_ptrs));
    raft::resource::sync_stream(resources);
    merge.end_ms = relative_ms(total_start, clock_type::now());
  }

  raft::resource::sync_stream(resources);
  bool valid = final_index->size() == rows &&
               final_index->dim() == source.dim &&
               final_index->graph_degree() == 64 &&
               final_index->graph().extent(0) == rows &&
               final_index->dataset().extent(0) == rows &&
               final_index->dataset().extent(1) == source.dim &&
               final_index->data().is_owning();
  if (!valid) {
    throw std::runtime_error("Final CAGRA index failed validity checks");
  }
  auto total_ms = elapsed_ms(total_start, clock_type::now());

  append_summary(opts, source, rows, prepare_ms, allocation, metrics, merge,
                 total_ms, valid);
  append_parts(opts, parts, metrics);
  std::cout << std::fixed << std::setprecision(3) << "RESULT mode=" << opts.mode
            << " build_path=" << opts.build_path << " parts=" << opts.parts
            << " run=" << opts.run << " rows=" << rows
            << " total_ms=" << total_ms
            << " merge_ms=" << merge.end_ms - merge.start_ms
            << " valid=" << valid << '\n';
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return run(parse_args(argc, argv));
  } catch (std::exception const &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
