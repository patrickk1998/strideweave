import importlib.util
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_train_mlp_cpu_example_loss_decreases():
    example = load_example("train_mlp_cpu")

    final = example.main(epochs=200, log_every=None)

    assert final < 0.1


def test_train_mlp_cpu_friendly_example_matches_raw_example():
    raw = load_example("train_mlp_cpu")
    ergonomic = load_example("train_mlp_cpu_friendly")

    assert ergonomic.main(epochs=200, log_every=None) == raw.main(
        epochs=200, log_every=None
    )
