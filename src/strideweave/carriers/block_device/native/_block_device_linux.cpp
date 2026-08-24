#include "_block_device.hpp"

#include <cerrno>
#include <cstdint>
#include <limits>

#include <linux/fs.h>
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
        throw InvalidDeviceError("opened resource is not a Linux block device");
    }

    std::uint64_t capacity = 0;
    if (::ioctl(descriptor, BLKGETSIZE64, &capacity) != 0) {
        throw SystemCallError("BLKGETSIZE64", errno);
    }
    int logical_block_size = 0;
    if (::ioctl(descriptor, BLKSSZGET, &logical_block_size) != 0) {
        throw SystemCallError("BLKSSZGET", errno);
    }
    int physical_block_size = 0;
    if (::ioctl(descriptor, BLKPBSZGET, &physical_block_size) != 0) {
        throw SystemCallError("BLKPBSZGET", errno);
    }

    return {unsigned_as_index(capacity, "block-device capacity"),
            static_cast<Index>(logical_block_size),
            static_cast<Index>(physical_block_size)};
}

}  // namespace strideweave::block_device
