#include "_block_device.hpp"

namespace strideweave::block_device {

Geometry inspect_platform_device(int) {
    throw UnsupportedPlatformError(
        "block-device access is supported only on Linux and macOS");
}

}  // namespace strideweave::block_device
