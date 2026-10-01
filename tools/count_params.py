"""Count total parameters for each model config across all benchmarks.

Every benchmark ships a single parameterized composer at
``conf/experiment/<bench>/default.yaml`` and picks the model via the
``model/<bench>`` group. ``discover_experiments`` enumerates the
selectable model names from ``conf/model/<bench>/``; ``count_params``
accepts the resolved composer name plus any extra overrides (e.g.
``model/synthetic=cnp``). Uses the context-manager form of
``hydra.initialize_config_dir`` so ``GlobalHydra`` is cleared between
iterations (a second bare ``hydra.initialize`` call otherwise raises
``GlobalHydra is already initialized``).
"""

from __future__ import annotations

from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conf import _resolvers  # noqa: F401,E402  (registers eval/product/range)


_CONF_DIR = _REPO_ROOT / "conf"
_EXPERIMENT_DIR = _CONF_DIR / "experiment"
_MODEL_DIR = _CONF_DIR / "model"

# Every benchmark routes through a single parameterized experiment
# composer at ``conf/experiment/<bench>/default.yaml``; the model is
# selected at the CLI via ``model/<bench>=<name>``.
_PARAMETERIZED_BENCHMARKS = {
    "synthetic": "synthetic/default",
    "image": "image/default",
    "era5": "era5/default",
    "kolmogorov": "kolmogorov/default",
    "predprey": "predprey/default",
}


def count_params(experiment_name: str, extra_overrides: list[str] | None = None) -> int:
    """Compose ``+experiment=<experiment_name>`` (plus any
    ``extra_overrides``) and return the model's total parameter count.
    """
    overrides = [f"+experiment={experiment_name}", *(extra_overrides or [])]
    with initialize_config_dir(
        config_dir=str(_CONF_DIR), version_base="1.3"
    ):
        cfg = compose(config_name="config", overrides=overrides)
    model = instantiate(cfg.model)
    return sum(p.numel() for p in model.parameters())


def discover_experiments() -> dict[str, list[str]]:
    """Return ``{benchmark: [model_name, ...]}`` enumerated from
    ``conf/model/<benchmark>/<model>.yaml`` for each parameterized
    benchmark.
    """
    benchmarks: dict[str, list[str]] = {}
    for bench_dir in sorted(p for p in _EXPERIMENT_DIR.iterdir() if p.is_dir()):
        bench = bench_dir.name
        if bench in _PARAMETERIZED_BENCHMARKS:
            models = sorted(p.stem for p in (_MODEL_DIR / bench).glob("*.yaml"))
        else:
            models = sorted(p.stem for p in bench_dir.glob("*.yaml"))
        if models:
            benchmarks[bench] = models
    return benchmarks


def _resolve(benchmark: str, model_name: str) -> tuple[str, list[str]]:
    """Map a (benchmark, model_name) pair to a composer + override list."""
    if benchmark in _PARAMETERIZED_BENCHMARKS:
        return (
            _PARAMETERIZED_BENCHMARKS[benchmark],
            [f"model/{benchmark}={model_name}"],
        )
    return (f"{benchmark}/{model_name}", [])


if __name__ == "__main__":
    for benchmark, models in discover_experiments().items():
        print(f"\n{'=' * 50}")
        print(f"Benchmark: {benchmark}")
        print(f"{'=' * 50}")
        for model_name in models:
            try:
                experiment_name, extra = _resolve(benchmark, model_name)
                n = count_params(experiment_name, extra)
                print(f"  {model_name:<45} {n:>12,}")
            except Exception as e:
                print(f"  {model_name:<45} ERROR: {e}")
