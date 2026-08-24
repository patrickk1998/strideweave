#include "_block_device.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace strideweave::block_device {
namespace {

class PosixIo final : public IoBackend {
public:
    ssize_t read_at(int descriptor, void* destination, std::size_t byte_count,
                    off_t byte_offset) noexcept override {
        return ::pread(descriptor, destination, byte_count, byte_offset);
    }

    ssize_t write_at(int descriptor, const void* source, std::size_t byte_count,
                     off_t byte_offset) noexcept override {
        return ::pwrite(descriptor, source, byte_count, byte_offset);
    }

    int close_descriptor(int descriptor) noexcept override {
        return ::close(descriptor);
    }
};

int open_read_write(const std::string& path) {
    if (path.find('\0') != std::string::npos) {
        throw std::invalid_argument(
            "block-device path must not contain embedded NUL characters");
    }

    int flags = O_RDWR;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif

    int descriptor = -1;
    do {
        descriptor = ::open(path.c_str(), flags);
    } while (descriptor < 0 && errno == EINTR);

    if (descriptor < 0) {
        throw SystemCallError("open", errno);
    }
    return descriptor;
}

off_t index_as_offset(Index value) {
    if (value < 0) {
        throw std::invalid_argument("byte position must be non-negative");
    }
    if constexpr (sizeof(Index) > sizeof(off_t)) {
        if (value > static_cast<Index>(std::numeric_limits<off_t>::max())) {
            throw std::overflow_error("byte position does not fit in off_t");
        }
    }
    return static_cast<off_t>(value);
}

Index checked_add(Index lhs, Index rhs, const char* name) {
    if (rhs > std::numeric_limits<Index>::max() - lhs) {
        throw std::overflow_error(std::string(name) + " overflows Index");
    }
    return lhs + rhs;
}

std::size_t syscall_chunk(Index remaining) {
    const auto max_request = static_cast<std::uintmax_t>(SSIZE_MAX);
    const auto remaining_unsigned = static_cast<std::uintmax_t>(remaining);
    const auto selected = std::min(remaining_unsigned, max_request);
    if (selected >
        static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("transfer chunk does not fit in size_t");
    }
    return static_cast<std::size_t>(selected);
}

template <typename Transfer>
TransferResult transfer_all(Index byte_count, Transfer&& transfer) {
    Index completed = 0;
    while (completed < byte_count) {
        const Index remaining = byte_count - completed;
        const std::size_t request = syscall_chunk(remaining);
        const ssize_t transferred = transfer(completed, request);
        if (transferred > 0) {
            const auto progress = static_cast<std::uintmax_t>(transferred);
            if (progress > static_cast<std::uintmax_t>(request)) {
                return TransferResult::os_error(completed, EIO);
            }
            completed = checked_add(completed, static_cast<Index>(transferred),
                                    "completed byte count");
            continue;
        }
        if (transferred == 0) {
            return TransferResult::zero_completion(completed);
        }

        const int error_number = errno;
        if (error_number == EINTR) {
            continue;
        }
        return TransferResult::os_error(completed, error_number);
    }
    return TransferResult::complete(completed);
}

std::shared_ptr<BlockDeviceHandle> make_handle(int raw_descriptor, Geometry geometry,
                                               std::shared_ptr<IoBackend> io) {
    io->descriptor_opened(raw_descriptor);
    Descriptor descriptor(raw_descriptor, io);
    validate_geometry(geometry);
    return std::make_shared<BlockDeviceHandle>(std::move(descriptor), geometry,
                                               std::move(io));
}

}  // namespace

TransferResult TransferResult::complete(Index completed_bytes) {
    return {TransferStatus::complete, completed_bytes, 0};
}

TransferResult TransferResult::zero_completion(Index completed_bytes) {
    return {TransferStatus::zero_completion, completed_bytes, 0};
}

TransferResult TransferResult::os_error(Index completed_bytes, int error_number) {
    return {TransferStatus::os_error, completed_bytes, error_number};
}

SystemCallError::SystemCallError(std::string operation, int error_number)
    : std::runtime_error(std::move(operation) + ": " + std::strerror(error_number)),
      error_number_(error_number) {}

void IoBackend::descriptor_opened(int) noexcept {}

Descriptor::Descriptor(int descriptor, std::shared_ptr<IoBackend> io)
    : descriptor_(descriptor), io_(std::move(io)) {
    if (descriptor_ < 0) {
        throw std::invalid_argument("descriptor must be open");
    }
    if (io_ == nullptr) {
        throw std::invalid_argument("descriptor I/O backend must be present");
    }
}

Descriptor::~Descriptor() {
    static_cast<void>(close());
}

Descriptor::Descriptor(Descriptor&& other) noexcept
    : descriptor_(std::exchange(other.descriptor_, -1)), io_(std::move(other.io_)) {}

Descriptor& Descriptor::operator=(Descriptor&& other) noexcept {
    if (this != &other) {
        static_cast<void>(close());
        descriptor_ = std::exchange(other.descriptor_, -1);
        io_ = std::move(other.io_);
    }
    return *this;
}

CloseResult Descriptor::close() noexcept {
    if (descriptor_ < 0) {
        return {0};
    }

    // POSIX leaves close(EINTR) state platform-dependent. Invalidate first and
    // never retry, because the integer may already name a recycled descriptor.
    const int descriptor = std::exchange(descriptor_, -1);
    if (io_->close_descriptor(descriptor) == 0) {
        return {0};
    }
    return {errno};
}

BlockDeviceHandle::BlockDeviceHandle(Descriptor descriptor, Geometry geometry,
                                     std::shared_ptr<IoBackend> io)
    : descriptor_(std::move(descriptor)), geometry_(geometry), io_(std::move(io)) {
    if (io_ == nullptr) {
        throw std::invalid_argument("block-device I/O backend must be present");
    }
}

TransferResult BlockDeviceHandle::read_into(std::uintptr_t destination,
                                            Index byte_offset, Index byte_count) {
    if (!descriptor_.is_open()) {
        throw std::runtime_error("block-device handle is closed");
    }
    validate_byte_range(geometry_.capacity_bytes, byte_offset, byte_count);
    validate_pointer_range(destination, byte_count);
    if (byte_count == 0) {
        return TransferResult::complete(0);
    }

    auto* const base = reinterpret_cast<unsigned char*>(destination);
    return transfer_all(byte_count, [&](Index completed, std::size_t request) {
        const std::size_t progress = index_as_size(completed, "completed byte count");
        const Index position = checked_add(byte_offset, completed, "byte position");
        return io_->read_at(descriptor_.get(), base + progress, request,
                            index_as_offset(position));
    });
}

TransferResult BlockDeviceHandle::write_from(std::uintptr_t source, Index byte_offset,
                                             Index byte_count) {
    if (!descriptor_.is_open()) {
        throw std::runtime_error("block-device handle is closed");
    }
    validate_byte_range(geometry_.capacity_bytes, byte_offset, byte_count);
    validate_pointer_range(source, byte_count);
    if (byte_count == 0) {
        return TransferResult::complete(0);
    }

    const auto* const base = reinterpret_cast<const unsigned char*>(source);
    return transfer_all(byte_count, [&](Index completed, std::size_t request) {
        const std::size_t progress = index_as_size(completed, "completed byte count");
        const Index position = checked_add(byte_offset, completed, "byte position");
        return io_->write_at(descriptor_.get(), base + progress, request,
                             index_as_offset(position));
    });
}

TransferResult BlockDeviceHandle::zero_fill(Index byte_offset, Index byte_count) {
    if (!descriptor_.is_open()) {
        throw std::runtime_error("block-device handle is closed");
    }
    validate_byte_range(geometry_.capacity_bytes, byte_offset, byte_count);
    if (byte_count == 0) {
        return TransferResult::complete(0);
    }

    static constexpr std::array<unsigned char, 64 * 1024> zeros{};
    return transfer_all(byte_count, [&](Index completed, std::size_t request) {
        const Index position = checked_add(byte_offset, completed, "byte position");
        const std::size_t selected = std::min(request, zeros.size());
        return io_->write_at(descriptor_.get(), zeros.data(), selected,
                             index_as_offset(position));
    });
}

void validate_geometry(const Geometry& geometry) {
    if (geometry.capacity_bytes <= 0) {
        throw InvalidDeviceError("block-device capacity must be positive");
    }
    if (geometry.logical_block_size <= 0) {
        throw InvalidDeviceError("logical block size must be positive");
    }
    if (geometry.physical_block_size <= 0) {
        throw InvalidDeviceError("physical block size must be positive");
    }
    if (geometry.capacity_bytes % geometry.logical_block_size != 0) {
        throw InvalidDeviceError(
            "block-device capacity must be a multiple of logical block size");
    }
    if (geometry.physical_block_size % geometry.logical_block_size != 0) {
        throw InvalidDeviceError(
            "physical block size must be a multiple of logical block size");
    }
    static_cast<void>(index_as_offset(geometry.capacity_bytes));
}

void validate_byte_range(Index capacity_bytes, Index byte_offset, Index byte_count) {
    if (byte_offset < 0) {
        throw std::invalid_argument("byte_offset must be non-negative");
    }
    if (byte_count < 0) {
        throw std::invalid_argument("byte_count must be non-negative");
    }
    if (byte_offset > capacity_bytes) {
        throw std::invalid_argument("byte_offset exceeds block-device capacity");
    }
    const Index endpoint = checked_add(byte_offset, byte_count, "byte endpoint");
    if (endpoint > capacity_bytes) {
        throw std::invalid_argument("byte range exceeds block-device capacity");
    }
    static_cast<void>(index_as_offset(byte_offset));
    static_cast<void>(index_as_offset(endpoint));
}

void validate_pointer_range(std::uintptr_t pointer, Index byte_count) {
    if (byte_count < 0) {
        throw std::invalid_argument("byte_count must be non-negative");
    }
    if (byte_count == 0) {
        return;
    }
    if (pointer == 0) {
        throw std::invalid_argument("pointer must be non-null");
    }
    const auto count = static_cast<std::uintmax_t>(byte_count);
    const auto available = static_cast<std::uintmax_t>(
        std::numeric_limits<std::uintptr_t>::max() - pointer);
    if (count > available) {
        throw std::overflow_error("pointer byte endpoint overflows uintptr_t");
    }
}

std::size_t index_as_size(Index value, const char* name) {
    if (value < 0) {
        throw std::invalid_argument(std::string(name) + " must be non-negative");
    }
    if constexpr (sizeof(Index) > sizeof(std::size_t)) {
        if (value > static_cast<Index>(std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(std::string(name) + " does not fit in size_t");
        }
    }
    return static_cast<std::size_t>(value);
}

std::shared_ptr<BlockDeviceHandle> open_block_device(const std::string& path) {
    auto io = std::make_shared<PosixIo>();
    const int descriptor = open_read_write(path);
    io->descriptor_opened(descriptor);
    Descriptor owned_descriptor(descriptor, io);
    const Geometry geometry = inspect_platform_device(descriptor);
    validate_geometry(geometry);
    return std::make_shared<BlockDeviceHandle>(std::move(owned_descriptor), geometry,
                                               std::move(io));
}

std::shared_ptr<BlockDeviceHandle>
open_block_device_for_test(const std::string& path, Geometry geometry,
                           std::shared_ptr<IoBackend> io) {
    if (io == nullptr) {
        throw std::invalid_argument("test I/O backend must be present");
    }
    const int descriptor = open_read_write(path);
    return make_handle(descriptor, geometry, std::move(io));
}

}  // namespace strideweave::block_device
