#include "_block_device.hpp"

#include <cerrno>
#include <cstdint>
#include <limits>

#include <sys/disk.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strideweave::block_device {
namespace {

Index unsigned_as_index(std::uint64_t value, const char* name) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<Index>::max())) {
        throw std::overflow_error(std::string(name) + " does not fit in Index");
    }
    return static_cast<Index>(value);
}

}  // namespace

Geometry inspect_platform_device(int descriptor) {
    struct stat status{};
    if (::fstat(descriptor, &status) != 0) {
        throw SystemCallError("fstat", errno);
    }
    if (!S_ISBLK(status.st_mode)) {
        throw InvalidDeviceError(
            "opened resource is not a buffered macOS block device");
    }

    std::uint32_t logical_block_size = 0;
    if (::ioctl(descriptor, DKIOCGETBLOCKSIZE, &logical_block_size) != 0) {
        throw SystemCallError("DKIOCGETBLOCKSIZE", errno);
    }
    std::uint64_t block_count = 0;
    if (::ioctl(descriptor, DKIOCGETBLOCKCOUNT, &block_count) != 0) {
        throw SystemCallError("DKIOCGETBLOCKCOUNT", errno);
    }
    std::uint32_t physical_block_size = 0;
    if (::ioctl(descriptor, DKIOCGETPHYSICALBLOCKSIZE, &physical_block_size) != 0) {
        throw SystemCallError("DKIOCGETPHYSICALBLOCKSIZE", errno);
    }

    if (logical_block_size != 0 &&
        block_count > std::numeric_limits<std::uint64_t>::max() / logical_block_size) {
        throw std::overflow_error("block-device capacity overflows uint64_t");
    }
    const std::uint64_t capacity = block_count * logical_block_size;
    return {unsigned_as_index(capacity, "block-device capacity"),
            static_cast<Index>(logical_block_size),
            static_cast<Index>(physical_block_size)};
}

}  // namespace strideweave::block_device
