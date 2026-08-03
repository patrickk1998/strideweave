#include "_cpu_binary.hpp"
#include "_cpu_registry.hpp"
#include "ops/_ops.hpp"

namespace strideweave::carrier {
namespace {

class CpuLogicalNotOperation : public strideweave::operation::Operation {
public:
    py::object _forward(py::args inputs) override {
        if (py::len(inputs) != 1) {
            throw py::type_error("CPU logical_not requires a tensor");
        }
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        CpuTensorView view = cpu_tensor_view(tensor, "tensor");
        const CpuPlan plan =
            resolve_cpu_plan(executing_carrier_class(tensor), "logical_not",
                             cpu_dtype_object(view.carrier->cpu_dtype()));
        CpuTensorAllocation result =
            allocate_cpu_tensor(injective_layout_for(tensor), plan.output);
        {
            py::gil_scoped_release release;
            std::vector<Index> key(view.leaf_rank(), 0);
            for (Index i = 0; i < view.logical_size; ++i) {
                // logical_not is Float32-only.  NaN is nonzero (therefore
                // logically true), while either signed zero is false.
                result.view.write_bool_expanded(key,
                                                view.read_float_expanded(key) == 0.0f);
                view.cache->increment_key(key.data(), key.size());
            }
        }
        return make_tensor(std::move(result.carrier_object),
                           std::move(result.layout_object));
    }
    py::object backward(py::object) override { return py::make_tuple(); }
};

constexpr CpuKernelMetadata kMetadata{"logical_not", "cpu.logical_not", "default",
                                      "_CPULogicalNotOperation"};

}  // namespace

void register_cpu_logical_not(py::module_& module) {
    bind_and_register_cpu_operation<CpuLogicalNotOperation>(module, kMetadata);
}

}  // namespace strideweave::carrier
