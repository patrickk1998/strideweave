#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "_operation.hpp"

namespace py = pybind11;

namespace {

thread_local bool grad_enabled = true;

bool is_grad_enabled() {
    return grad_enabled;
}

void set_grad_enabled(bool enabled) {
    grad_enabled = enabled;
}

using strideweave::operation::Operation;

using Clock = std::chrono::steady_clock;
using EventId = std::uint64_t;
using Nanoseconds = std::chrono::nanoseconds;

struct RawProfilerEvent {
    RawProfilerEvent(EventId event_id, std::optional<EventId> event_parent_id,
                     std::string event_name, py::object event_carrier_type,
                     py::object event_implementation_type,
                     py::object event_input_shapes, std::int64_t event_start_time_ns,
                     unsigned long event_thread_id)
        : id(event_id), parent_id(event_parent_id), name(std::move(event_name)),
          carrier_type(std::move(event_carrier_type)),
          implementation_type(std::move(event_implementation_type)),
          input_shapes(std::move(event_input_shapes)),
          start_time_ns(event_start_time_ns), thread_id(event_thread_id) {}

    EventId id;
    std::optional<EventId> parent_id;
    std::string name;
    py::object carrier_type;
    py::object implementation_type;
    py::object input_shapes;
    std::int64_t start_time_ns;
    std::int64_t duration_ns = 0;
    std::int64_t self_time_ns = 0;
    unsigned long thread_id;
    bool succeeded = false;
};

py::tuple snapshot_input_shapes(py::args inputs) {
    py::object tensor = strideweave::operation::tensor_type();
    py::tuple shapes(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        py::handle input = inputs[i];
        if (!py::isinstance(input, tensor)) {
            shapes[i] = py::none();
            continue;
        }
        shapes[i] = input.attr("layout").attr("shape").attr("top_level");
    }
    return shapes;
}

std::int64_t nanoseconds_since_epoch(Clock::time_point time) {
    return std::chrono::duration_cast<Nanoseconds>(time.time_since_epoch()).count();
}

struct ProfilerFrame {
    ProfilerFrame(Clock::time_point frame_start,
                  std::optional<EventId> recorded_parent_id)
        : start(frame_start), nearest_recorded_id(recorded_parent_id) {}

    Clock::time_point start;
    std::int64_t child_time_ns = 0;
    std::optional<std::size_t> event_index;
    std::optional<EventId> nearest_recorded_id;
};

class RawProfilerSession;
thread_local RawProfilerSession* active_profiler = nullptr;
void recover_abandoned_profiler();

class RawProfilerSession {
public:
    RawProfilerSession(py::object carrier_types, bool record_shapes)
        : carrier_types_(std::move(carrier_types)), record_shapes_(record_shapes) {
        if (!carrier_types_.is_none()) {
            carrier_types_ =
                py::module_::import("builtins").attr("frozenset")(carrier_types_);
        }
    }

    void start(py::object session_object) {
        if (started_) {
            throw std::runtime_error("Profiler session may only be started once");
        }
        recover_abandoned_profiler();
        if (active_profiler != nullptr) {
            throw std::runtime_error("A profiler session is already active");
        }
        started_ = true;
        active_ = true;
        owner_thread_ = std::this_thread::get_id();
        active_session_keepalive_ = std::move(session_object);
        active_profiler = this;
    }

    void stop() {
        if (!active_) {
            throw std::runtime_error("Profiler session is not active");
        }
        if (owner_thread_ != std::this_thread::get_id()) {
            abandoned_.store(true, std::memory_order_release);
            throw std::runtime_error(
                "Profiler session must be stopped on its active thread");
        }
        if (active_profiler != this) {
            throw std::runtime_error("Profiler session is not active");
        }
        if (!frames_.empty()) {
            throw std::runtime_error(
                "Profiler session cannot stop during an operation");
        }
        active_profiler = nullptr;
        active_ = false;
        active_session_keepalive_ = py::none();
    }

    void abandon() noexcept {
        if (active_) {
            abandoned_.store(true, std::memory_order_release);
        }
    }

    bool is_active() const {
        return active_ && !abandoned_.load(std::memory_order_acquire);
    }

    bool is_abandoned() const { return abandoned_.load(std::memory_order_acquire); }

    bool can_recover_on_current_thread() const {
        return is_abandoned() && owner_thread_ == std::this_thread::get_id() &&
               frames_.empty();
    }

    py::object keepalive() const { return active_session_keepalive_; }

    void discard_abandoned_events() {
        active_ = false;
        events_.clear();
        active_session_keepalive_ = py::none();
    }

    py::tuple events() const {
        py::tuple result(events_.size());
        for (std::size_t i = 0; i < events_.size(); ++i) {
            result[i] = py::cast(events_[i]);
        }
        return result;
    }

    void begin(Operation& operation, py::args inputs) {
        const Clock::time_point start = Clock::now();
        const std::optional<EventId> parent_id =
            frames_.empty() ? std::nullopt : frames_.back().nearest_recorded_id;
        const py::object carrier_type = operation.dispatch_carrier_class();
        const bool selected =
            carrier_types_.is_none() ||
            PySet_Contains(carrier_types_.ptr(), carrier_type.ptr()) == 1;

        ProfilerFrame frame(start, parent_id);
        if (selected) {
            const EventId event_id = next_event_id_++;
            py::object operation_object =
                py::cast(&operation, py::return_value_policy::reference);
            py::object input_shapes = record_shapes_
                                          ? py::object(snapshot_input_shapes(inputs))
                                          : py::object(py::none());
            events_.emplace_back(
                event_id, parent_id, operation.operation_name_value(), carrier_type,
                py::type::of(operation_object), std::move(input_shapes),
                nanoseconds_since_epoch(start), PyThread_get_thread_ident());
            frame.event_index = events_.size() - 1;
            frame.nearest_recorded_id = event_id;
        }
        frames_.push_back(std::move(frame));
    }

    void finish(bool succeeded) {
        const Clock::time_point finish = Clock::now();
        if (frames_.empty()) {
            throw std::runtime_error("Profiler execution stack is empty");
        }
        ProfilerFrame frame = std::move(frames_.back());
        frames_.pop_back();
        const std::int64_t duration_ns =
            std::chrono::duration_cast<Nanoseconds>(finish - frame.start).count();
        if (frame.event_index.has_value()) {
            RawProfilerEvent& event = events_[*frame.event_index];
            event.duration_ns = duration_ns;
            event.self_time_ns =
                std::max<std::int64_t>(0, duration_ns - frame.child_time_ns);
            event.succeeded = succeeded;
        }
        if (!frames_.empty()) {
            frames_.back().child_time_ns += duration_ns;
        }
    }

private:
    py::object carrier_types_;
    bool record_shapes_;
    bool started_ = false;
    bool active_ = false;
    std::atomic<bool> abandoned_{false};
    std::thread::id owner_thread_;
    py::object active_session_keepalive_;
    EventId next_event_id_ = 0;
    std::vector<RawProfilerEvent> events_;
    std::vector<ProfilerFrame> frames_;
};

void recover_abandoned_profiler() {
    RawProfilerSession* abandoned = active_profiler;
    if (abandoned == nullptr || !abandoned->can_recover_on_current_thread()) {
        return;
    }

    py::object keepalive = abandoned->keepalive();
    active_profiler = nullptr;
    abandoned->discard_abandoned_events();
}

class PyOperation : public Operation {
public:
    using Operation::Operation;

    py::object _forward(py::args inputs) override {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(this, "_forward");
        if (!override) {
            throw py::type_error("Operation._forward must be implemented");
        }

        PyObject* result = PyObject_CallObject(override.ptr(), inputs.ptr());
        if (result == nullptr) {
            throw py::error_already_set();
        }
        return py::reinterpret_steal<py::object>(result);
    }

    py::object backward(py::object gradient) override {
        PYBIND11_OVERRIDE_PURE(py::object, Operation, backward, gradient);
    }
};

}  // namespace

py::object strideweave::operation::Operation::execute(py::args inputs) {
    recover_abandoned_profiler();
    RawProfilerSession* profiler = active_profiler;
    if (profiler == nullptr || profiler->is_abandoned() || !is_dispatched()) {
        py::object result = _forward(inputs);
        if (!py::isinstance(result, tensor_type())) {
            throw py::type_error("Operation._forward must return a Tensor");
        }
        return result;
    }

    profiler->begin(*this, inputs);
    try {
        py::object result = _forward(inputs);
        if (!py::isinstance(result, tensor_type())) {
            throw py::type_error("Operation._forward must return a Tensor");
        }
        profiler->finish(true);
        return result;
    } catch (...) {
        profiler->finish(false);
        throw;
    }
}

PYBIND11_MODULE(_operation, module) {
    module.doc() = "Native operation base class for strideweave autograd";

    module.def("is_grad_enabled", &is_grad_enabled);
    module.def("set_grad_enabled", &set_grad_enabled, py::arg("enabled"));

    py::class_<RawProfilerEvent>(module, "_RawProfilerEvent")
        .def_property_readonly("id",
                               [](const RawProfilerEvent& event) { return event.id; })
        .def_property_readonly(
            "parent_id", [](const RawProfilerEvent& event) { return event.parent_id; })
        .def_property_readonly("name",
                               [](const RawProfilerEvent& event) { return event.name; })
        .def_property_readonly(
            "carrier_type",
            [](const RawProfilerEvent& event) { return event.carrier_type; })
        .def_property_readonly(
            "implementation_type",
            [](const RawProfilerEvent& event) { return event.implementation_type; })
        .def_property_readonly(
            "input_shapes",
            [](const RawProfilerEvent& event) { return event.input_shapes; })
        .def_property_readonly(
            "start_time_ns",
            [](const RawProfilerEvent& event) { return event.start_time_ns; })
        .def_property_readonly(
            "duration_ns",
            [](const RawProfilerEvent& event) { return event.duration_ns; })
        .def_property_readonly(
            "self_time_ns",
            [](const RawProfilerEvent& event) { return event.self_time_ns; })
        .def_property_readonly(
            "thread_id", [](const RawProfilerEvent& event) { return event.thread_id; })
        .def_property_readonly(
            "succeeded", [](const RawProfilerEvent& event) { return event.succeeded; });

    py::class_<RawProfilerSession>(module, "_RawProfilerSession")
        .def(py::init<py::object, bool>(), py::arg("carrier_types") = py::none(),
             py::arg("record_shapes") = false)
        .def("start",
             [](py::object session_object) {
                 session_object.cast<RawProfilerSession&>().start(session_object);
             })
        .def("stop", &RawProfilerSession::stop)
        .def("_abandon", &RawProfilerSession::abandon)
        .def("events", &RawProfilerSession::events)
        .def_property_readonly("is_active", &RawProfilerSession::is_active);

    py::class_<Operation, PyOperation>(module, "Operation")
        .def(py::init<>())
        .def("forward", [](Operation& operation,
                           py::args inputs) { return operation.forward(inputs); })
        .def("_forward", [](Operation& operation,
                            py::args inputs) { return operation._forward(inputs); })
        .def("_execute_lowered",
             [](Operation& operation, py::args inputs) {
                 return operation.execute_lowered(inputs);
             })
        .def("backward", &Operation::backward, py::arg("gradient"))
        .def_property_readonly("ctx", &Operation::ctx)
        .def_property_readonly("_operation_name", &Operation::operation_name)
        .def_property_readonly("_dispatch_carrier_class",
                               &Operation::dispatch_carrier_class)
        .def("store_inputs", [](Operation& operation,
                                py::args inputs) { operation.store_inputs(inputs); })
        .def("inputs", &Operation::inputs)
        .def("input_versions", &Operation::input_versions)
        .def("validate_input_versions", &Operation::validate_input_versions);

    module.def(
        "_execute_lowered",
        [](Operation& operation, py::args inputs) {
            return operation.execute_lowered(inputs);
        },
        py::arg("operation"));
}
