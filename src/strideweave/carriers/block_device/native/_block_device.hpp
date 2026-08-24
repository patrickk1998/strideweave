#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <sys/types.h>

namespace strideweave::block_device {

using Index = long long;

struct Geometry {
    Index capacity_bytes;
    Index logical_block_size;
    Index physical_block_size;
};

enum class TransferStatus { complete, zero_completion, os_error };

struct TransferResult {
    TransferStatus status;
    Index completed_bytes;
    int error_number;

    static TransferResult complete(Index completed_bytes);
    static TransferResult zero_completion(Index completed_bytes);
    static TransferResult os_error(Index completed_bytes, int error_number);
};

struct CloseResult {
    int error_number;

    bool succeeded() const noexcept { return error_number == 0; }
};

class SystemCallError : public std::runtime_error {
public:
    SystemCallError(std::string operation, int error_number);

    int error_number() const noexcept { return error_number_; }

private:
    int error_number_;
};

class InvalidDeviceError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class UnsupportedPlatformError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class IoBackend {
public:
    virtual ~IoBackend() = default;

    virtual ssize_t read_at(int descriptor, void* destination, std::size_t byte_count,
                            off_t byte_offset) noexcept = 0;
    virtual ssize_t write_at(int descriptor, const void* source, std::size_t byte_count,
                             off_t byte_offset) noexcept = 0;
    virtual int close_descriptor(int descriptor) noexcept = 0;
    virtual void descriptor_opened(int descriptor) noexcept;
};

class Descriptor {
public:
    Descriptor(int descriptor, std::shared_ptr<IoBackend> io);
    ~Descriptor();

    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    Descriptor(Descriptor&& other) noexcept;
    Descriptor& operator=(Descriptor&& other) noexcept;

    int get() const noexcept { return descriptor_; }
    bool is_open() const noexcept { return descriptor_ >= 0; }
    CloseResult close() noexcept;

private:
    int descriptor_ = -1;
    std::shared_ptr<IoBackend> io_;
};

class BlockDeviceHandle {
public:
    BlockDeviceHandle(Descriptor descriptor, Geometry geometry,
                      std::shared_ptr<IoBackend> io);

    BlockDeviceHandle(const BlockDeviceHandle&) = delete;
    BlockDeviceHandle& operator=(const BlockDeviceHandle&) = delete;
    BlockDeviceHandle(BlockDeviceHandle&&) = delete;
    BlockDeviceHandle& operator=(BlockDeviceHandle&&) = delete;

    Index capacity_bytes() const noexcept { return geometry_.capacity_bytes; }
    Index logical_block_size() const noexcept { return geometry_.logical_block_size; }
    Index physical_block_size() const noexcept { return geometry_.physical_block_size; }
    bool is_open() const noexcept { return descriptor_.is_open(); }

    TransferResult read_into(std::uintptr_t destination, Index byte_offset,
                             Index byte_count);
    TransferResult write_from(std::uintptr_t source, Index byte_offset,
                              Index byte_count);
    TransferResult zero_fill(Index byte_offset, Index byte_count);
    CloseResult close() noexcept { return descriptor_.close(); }

private:
    Descriptor descriptor_;
    Geometry geometry_;
    std::shared_ptr<IoBackend> io_;
};

void validate_geometry(const Geometry& geometry);
void validate_byte_range(Index capacity_bytes, Index byte_offset, Index byte_count);
void validate_pointer_range(std::uintptr_t pointer, Index byte_count);
std::size_t index_as_size(Index value, const char* name);

Geometry inspect_platform_device(int descriptor);

std::shared_ptr<BlockDeviceHandle> open_block_device(const std::string& path);
std::shared_ptr<BlockDeviceHandle>
open_block_device_for_test(const std::string& path, Geometry geometry,
                           std::shared_ptr<IoBackend> io);

}  // namespace strideweave::block_device
