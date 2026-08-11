## Causal Discovery Benchmarking Infrastructure

Contributors: @Gitanaskhan26

### Introduction

pgmpy currently ships several causal discovery algorithms (`PC`, `GES`, `HillClimbSearch`, `ChowLiu`, `TAN`, `ANM`, `TOPIC`, `ExpertInLoop`, `LLMPairwise`), all unified under a common `BaseCausalDiscovery` interface (`pgmpy/causal_discovery/_base.py`): every estimator exposes `.fit(X)`, populates a fitted `causal_graph_`, and exposes `.score(X=None, true_graph=None, metric=None)`. Separately, `pgmpy.metrics` now ships `BaseSupervisedMetric` / `BaseUnsupervisedMetric`, a `get_metrics(**tag_filters)` registry, and concrete metrics (`SHD`, `AdjacencyConfusionMatrix`, `OrientationConfusionMatrix`, `StructureScore`, `CorrelationScore`, `ImpliedCIs`, `FisherC`), each tagged with `requires_true_graph`, `requires_data`, and `lower_is_better`. The dataset side (this proposal's [Simulation Mixin proposal](../1_simulation_mixin/proposal.md)) has landed as `load_dataset()` / `list_datasets()` / `BaseSimulatedDataset`, with `has_ground_truth` / `is_simulated` tags.

All the pieces to *run* a causal discovery method and *score* it against a dataset already exist. What's missing is a way to compare many of them at once. Today that means a one-off script per experiment. [CausalEval#7](https://github.com/pgmpy/CausalEval/pull/7) (open, unmerged) is a concrete example: it hard-codes a single Linear Gaussian dataset, `n=1000`, exactly two algorithms (PC, GES) in a sequential loop, and — because it predates the `.fit()`/`.score()`/metrics refactor above — it's already written against a superseded API (`PC(data).estimate(...)`, a bare `SHD(true_dag, learned_dag)` call). It would need a rewrite regardless of this proposal. [CausalEval#13](https://github.com/pgmpy/CausalEval/issues/13), opened by @ankurankan, asks for exactly this capability: "a benchmark suite ... to test how well these methods can recover the true graph," and is still open.

CausalEval already solves a structurally similar problem for CI tests: `ci_benchmarks/` (`DGM.py` + `ci_benchmark.py`) sweeps data-generating mechanisms against CI tests via a hand-written dict registry (`DGP_REGISTRY`, `DGM_TO_CITESTS`) and fully sequential nested `for` loops, writing raw + summary CSVs consumed by `pgmpy.org/causalbench`. It predates the tag/`skbase`-object model that `datasets`, `metrics`, and `causal_discovery` have since converged on, and it doesn't parallelize.

This proposal is for `CausalDiscoveryBenchmark`: given a set of causal discovery estimators, datasets or simulators, metrics, and sample sizes, run every applicable combination, evaluate the requested metrics, and return one tidy results table. The first implementation uses local CPU parallelism through `joblib` and can write that table to CSV for the existing CausalEval workflow.

**Non-goals.** This proposal does not add new causal discovery algorithms, new datasets, or new metrics — it composes the ones that already exist (and the ones sibling proposals in this repo are adding). It also does not cover publishing results to `pgmpy.org/causalbench` (CausalEval's existing `web/` dashboard) — that's a natural follow-up, noted under Open Questions, but kept out of scope here so this proposal stays focused on the benchmarking class itself.

### Goals

* One declarative entry point that sweeps `estimators × datasets × n_samples`, evaluates the requested metrics on each fitted result, and returns a single results table.
* Reuse pgmpy's existing primitives end-to-end — `BaseCausalDiscovery.fit`, `pgmpy.metrics.get_metrics`, `load_dataset`/`list_datasets` — rather than re-implementing fitting, scoring, or dataset-loading logic.
* Parallel fitting on a single machine through `joblib`, without changing the estimators' public APIs.
* Tolerant of partial failure: one incompatible or crashing (estimator, dataset, metric) combination shouldn't abort the rest of the sweep.
* Produce a stable CSV export in addition to the in-memory results table, so downstream reporting and the existing CausalEval frontend can consume the same result format.
* Lands in `CausalEval`, alongside the existing `ci_benchmarks/`.

### References

* [CausalEval#13](https://github.com/pgmpy/CausalEval/issues/13) — "[ENH] Benchmarks for causal discovery algorithms on Linear Gaussian Data" (@ankurankan)
* [CausalEval#7](https://github.com/pgmpy/CausalEval/pull/7) — existing one-off PC/GES benchmark script; motivating example for this proposal
* [Simulation Mixin proposal](../1_simulation_mixin/proposal.md) — `load_dataset`/`BaseSimulatedDataset`, which this proposal's dataset resolution builds on directly
* `pgmpy/metrics/` (`_base.py`, `shd.py`, `adjacency_cm.py`, ...) — existing metric implementations this proposal wraps, not reimplements

---

### Proposed Solution

`CausalDiscoveryBenchmark` is configured once (estimators, datasets, metrics, sample sizes, and a few execution options) and run with `.run()`, which populates `.results_`. It is a long/tidy `DataFrame`: each scalar metric result has one row per `(dataset, n_samples, repeat_idx, estimator, metric, component)` combination. Scalar metrics use `component="value"`; dictionary-valued metrics contribute one row for each returned key. A `.summary()` method pivots this into the wide `estimator × dataset` view (averaging across repeats), and `.to_csv()` writes the raw long-format result table.

For each `(dataset, n_samples, repeat_idx)` triple, data is resolved once in the main process so every estimator sees the same draw. The resulting estimator tasks run in parallel. A task clones its estimator before fitting, which avoids fitted state leaking across tasks and keeps the design safe with either process or thread backends. When the outer benchmark is parallel, estimators that expose `n_jobs` run with `n_jobs=1` inside a task to avoid nested parallelism.

Each task fits once and evaluates every requested metric against that one fitted graph:

* A metric with `requires_true_graph=True` (e.g. `SHD`) that receives no ground truth will raise an error, recorded in the result row with `status="error"`. The user is responsible for supplying ground truth (via the dataset or `ground_truth={...}`) for any metric that requires it.
* A metric must support the fitted graph's type. For example, `StructureScore` supports `DAG`, while `PC` and `GES` return `PDAG` by default; a benchmark requiring `StructureScore` must request `return_type="dag"` for those estimators. A type mismatch is recorded as `status="error"`.
* If `train_test_split` is set, the estimator fits on the training split and `requires_data=True` metrics receive the held-out split. `requires_true_graph=True` metrics are unaffected because they compare graphs rather than data.

A failure while fitting is recorded once for that estimator task; a failure in one metric is recorded only for that metric and does not prevent the remaining metrics from being evaluated. There is no `"skipped"` status — every requested combination either produces a value (`"ok"`) or an error (`"error"`).

For execution, estimator tasks are dispatched with `joblib.Parallel`, which is already a dependency of pgmpy. The initial scope is local execution through `n_jobs`; distributed backends remain a possible extension once the result schema and task boundaries are proven locally.

This lives in `CausalEval` as `cd_benchmarks/`, a sibling to `ci_benchmarks/`, rather than in pgmpy core — it composes public pgmpy APIs and doesn't need to live inside the library itself, matching where the analogous CI-test benchmarking already lives.

### Alternative Solutions

**A. Results table shape: long, wide, or both?**
The two shapes sketched in the original guidance are actually two different things: a single-metric long table (`dataset, estimator, metric`) and a multi-metric wide table (`Algo, Dataset, M1, M2, M3`). Rather than pick one, `.results_` is long/tidy (one row per metric component — easiest to `groupby`/filter/plot, and the natural shape when metrics do not all apply to every dataset) and `.summary()` derives the wide pivot on demand. This also matches `ci_benchmarks`' existing raw-CSV-plus-summary-CSV convention, so it is not a new pattern for the repo.

**B. Distributed backends in the first implementation.**
Considered documenting Dask and Ray through joblib's pluggable backend immediately. Rejected for the initial implementation: local execution is enough to settle task serialization, resource use, and error handling. A later extension can document an optional joblib-compatible backend without making Dask or Ray a dependency of CausalEval.

**C. Fail-fast vs. per-task error capture.**
A sweep can contain a structurally invalid configuration (for example, an unknown dataset name) or a valid combination that fails only at runtime. The former should fail before tasks start. The latter should be captured in the affected result row, with an exception type and message. Metric evaluation is isolated further: one unsupported or failing metric must not discard the other metric values for the same fitted graph.

**D. Should `CausalDiscoveryBenchmark` itself subclass `BaseEstimator`/`BaseObject`?**
`BaseCausalDiscovery` subclasses sklearn's `BaseEstimator`; `BaseDataset`/`BaseSupervisedMetric`/`BaseUnsupervisedMetric` subclass `skbase.base.BaseObject`. Both exist so those components are *discoverable and swappable* (via `get_metrics()`/`all_objects()`-style registry lookups and `get_params()`/`set_params()`/tags). `CausalDiscoveryBenchmark` isn't itself a pluggable component — nothing needs to look it up by tag or clone it — so it's a plain class. It does still benefit from the same `repr()` convention (see Details), just without inheriting anything to get it.

**E. Scalar-only metrics vs. handling structured metric values.**
Not every metric in `pgmpy.metrics` returns a single number — `AdjacencyConfusionMatrix.evaluate()` returns a dictionary of values such as precision, recall, and F1. The runner stores these as separate long-format rows with the same metric name and different `component` values. `.summary()` then creates a column for each `metric.component` pair.

### Details of proposed solution

**File layout** (inside `CausalEval`):

```
CausalEval/
    ci_benchmarks/                 # existing, unchanged
        DGM.py
        ci_benchmark.py
    cd_benchmarks/                 # new
        __init__.py
        _base.py                   # CausalDiscoveryBenchmark, task resolution + execution
```

**Resolving datasets.** `datasets` accepts a string name (resolved through the existing `load_dataset` registry) or an already-constructed `BaseSimulatedDataset` instance (e.g. `AdditiveNoiseModel(n_nodes=10, edge_prob=0.3)`). The label for a string is its name and the label for an instance is its `repr()`.

For string datasets, each `(dataset, n_samples, repeat_idx)` combination receives its own `data_seed` from the benchmark's seed schedule. This seed is passed to `load_dataset(..., seed=data_seed)`, so every repeat draws an independent sample from the same data-generating process.

For initialized `BaseSimulatedDataset` instances, the simulator owns its own seed through its constructor. The benchmark calls `load_dataframe(n_samples=n_samples)` on the same object for every repeat. Whether this produces independent samples depends on the simulator's internal state. **Phase 1 does not inject the benchmark's `data_seed` into initialized simulators.** For reliable repeated sampling, use a registered dataset name (string) rather than a pre-constructed instance. Extending the simulator contract to accept an external seed for per-repeat variation is noted as a Phase 2 item.

```python
from pgmpy.datasets import load_dataset, list_datasets
from pgmpy.datasets._base import BaseSimulatedDataset, Dataset

def _resolve_dataset(dataset, n_samples, data_seed) -> tuple[str, Dataset]:
    """Load one (dataset, n_samples) combination, returning a (label, Dataset) pair."""
    if isinstance(dataset, str):
        if dataset not in list_datasets():
            raise ValueError(f"Unknown dataset name: {dataset!r}")
        return dataset, load_dataset(dataset, n_samples=n_samples, seed=data_seed)

    if isinstance(dataset, BaseSimulatedDataset):
        df = dataset.load_dataframe(n_samples=n_samples)
        return repr(dataset), Dataset(
            name=repr(dataset),
            data=df,
            expert_knowledge=None,
            ground_truth=dataset.load_ground_truth(),
            tags=dataset.get_tags(),
        )

    raise TypeError(
        f"datasets entries must be a registered name or a BaseSimulatedDataset "
        f"instance, got {type(dataset)}"
    )
```

**Resolving metrics.** The benchmark accepts metric instances only (e.g. `SHD()`, `SHD(edge_reverse_penalty=2)`). Passing an uninstantiated class raises a `TypeError`. Applicability is read from each metric's `requires_true_graph`, `requires_data`, and `supported_graph_types` tags.

```python
from pgmpy.metrics import BaseSupervisedMetric, BaseUnsupervisedMetric

def _resolve_metrics(metrics):
    for metric in metrics:
        if isinstance(metric, type):
            raise TypeError(
                f"metrics must contain metric instances, not classes. "
                f"Use {metric.__name__}() instead of {metric.__name__}."
            )
    if not all(isinstance(metric, (BaseSupervisedMetric, BaseUnsupervisedMetric)) for metric in metrics):
        raise TypeError("metrics must contain BaseSupervisedMetric or BaseUnsupervisedMetric instances")
    return list(metrics)
```

**Seed propagation.** The benchmark's `seed` drives two independent `numpy.random.SeedSequence` streams:

* **`data_seeds`**: one per `(dataset, n_samples, repeat_idx)` triple. Used by `_resolve_dataset` so every estimator in the same repeat sees the same data draw, and to create the same train/test split for those estimators. Different repeats receive different data seeds. Recorded in the `data_seed` results column.
* **`estimator_seeds`**: one per `(dataset, n_samples, repeat_idx, estimator)` task. Injected into the cloned estimator via `set_params(seed=estimator_seed)` if the estimator accepts a `seed` parameter. Recorded in the `estimator_seed` results column.

Both seeds are stored in `results_` so that any individual row can be reproduced from the CSV alone.

```python
import numpy as np

def _generate_seeds(seed_sequence, n):
    """Generate n independent integer seeds from a SeedSequence stream."""
    return [int(child.generate_state(1)[0]) for child in seed_sequence.spawn(n)]
```

**Running one task.** A task clones and fits one estimator, then evaluates each metric. It returns a list of long-format rows. The learned graph is passed directly to metrics rather than reconstructing from edges, preserving isolated nodes. This keeps estimator state local to the task, avoids nested parallelism, and lets one metric fail without hiding the other results:

```python
import time
from sklearn.base import clone
from sklearn.model_selection import train_test_split as sk_train_test_split

def _run_one_task(
    dataset_label, dataset, n_samples, repeat_idx,
    estimator, metrics, train_test_split,
    data_seed, estimator_seed, outer_n_jobs,
):
    base_row = {
        "dataset": dataset_label,
        "n_samples": n_samples,
        "n_samples_actual": dataset.data.shape[0],
        "repeat_idx": repeat_idx,
        "data_seed": data_seed,
        "estimator_seed": estimator_seed,
        "estimator": repr(estimator),
    }
    try:
        train_df, test_df = dataset.data, dataset.data
        if train_test_split is not None:
            train_df, test_df = sk_train_test_split(
                dataset.data, train_size=train_test_split, random_state=data_seed
            )

        fitted = clone(estimator)
        params = fitted.get_params(deep=False)
        if outer_n_jobs != 1 and "n_jobs" in params:
            fitted.set_params(n_jobs=1)
        if "seed" in params:
            fitted.set_params(seed=estimator_seed)
        start = time.perf_counter()
        fitted.fit(train_df)
        runtime_sec = time.perf_counter() - start
    except Exception as exc:
        return [
            {
                **base_row,
                "metric": None,
                "component": None,
                "value": None,
                "runtime_sec": None,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        ]

    rows = []
    for metric in metrics:
        name = metric.get_tag("name", repr(metric), raise_error=False)
        metric_row = {**base_row, "metric": name, "runtime_sec": runtime_sec}
        needs_truth = metric.get_tag("requires_true_graph", False, raise_error=False)
        try:
            if needs_truth:
                value = metric.evaluate(true_causal_graph=dataset.ground_truth, est_causal_graph=fitted.causal_graph_)
            else:
                value = metric.evaluate(X=test_df, causal_graph=fitted.causal_graph_)
            values = value.items() if isinstance(value, dict) else [("value", value)]
            rows.extend(
                {
                    **metric_row,
                    "component": key,
                    "value": val,
                    "status": "ok",
                    "error": None,
                }
                for key, val in values
            )
        except Exception as exc:
            rows.append(
                {
                    **metric_row,
                    "component": None,
                    "value": None,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return rows
```

**The `CausalDiscoveryBenchmark` class itself:**

```python
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pgmpy.causal_discovery._base import BaseCausalDiscovery

class CausalDiscoveryBenchmark:
    def __init__(
        self,
        estimators,
        datasets,
        metrics,
        n_samples=None,
        n_repeats: int = 1,
        ground_truth: dict | None = None,
        train_test_split: float | None = None,
        n_jobs: int = -1,
        show_progress: bool = True,
        seed: int = 42,
    ):
        if not all(isinstance(estimator, BaseCausalDiscovery) for estimator in estimators):
            raise TypeError("estimators must contain BaseCausalDiscovery instances")
        if not isinstance(n_repeats, int) or n_repeats < 1:
            raise ValueError(f"n_repeats must be a positive integer, got {n_repeats!r}")
        self.estimators = estimators
        self.datasets = datasets
        self.metrics = _resolve_metrics(metrics)
        self.n_samples = n_samples if isinstance(n_samples, list) else [n_samples]
        self.n_repeats = n_repeats
        self.ground_truth = ground_truth or {}
        self.train_test_split = train_test_split
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self.seed = seed

    def run(self, n_jobs=None, show_progress=None) -> "CausalDiscoveryBenchmark":
        n_jobs = self.n_jobs if n_jobs is None else n_jobs
        show_progress = self.show_progress if show_progress is None else show_progress

        # Build the grid of (dataset_spec, n_samples, repeat_idx) triples.
        data_specs = []
        for dataset_spec in self.datasets:
            for n in self.n_samples:
                for repeat_idx in range(self.n_repeats):
                    data_specs.append((dataset_spec, n, repeat_idx))

        # Two independent seed streams from the master seed.
        ss = np.random.SeedSequence(self.seed)
        data_ss, est_ss = ss.spawn(2)
        data_seeds = _generate_seeds(data_ss, len(data_specs))
        est_seeds = _generate_seeds(est_ss, len(data_specs) * len(self.estimators))

        # Resolve datasets and build task list.
        tasks = []
        est_seed_idx = 0
        for spec_idx, (dataset_spec, n, repeat_idx) in enumerate(data_specs):
            data_seed = data_seeds[spec_idx]
            label, dataset = _resolve_dataset(dataset_spec, n, data_seed)
            if dataset.ground_truth is None and label in self.ground_truth:
                dataset.ground_truth = self.ground_truth[label]
            for estimator in self.estimators:
                tasks.append(
                    (label, dataset, n, repeat_idx, estimator,
                     data_seed, est_seeds[est_seed_idx])
                )
                est_seed_idx += 1

        task_rows = Parallel(n_jobs=n_jobs, verbose=10 if show_progress else 0)(
            delayed(_run_one_task)(
                label, dataset, n, repeat_idx, estimator, self.metrics,
                self.train_test_split, data_seed, est_seed, n_jobs
            )
            for label, dataset, n, repeat_idx, estimator, data_seed, est_seed in tasks
        )
        self.results_ = pd.DataFrame([row for rows in task_rows for row in rows])
        return self

    def summary(self) -> pd.DataFrame:
        """Return a wide estimator-by-dataset view of successful metric values, averaged across repeats."""
        successful = self.results_.query("status == 'ok'").copy()
        successful["metric_component"] = successful["metric"] + "." + successful["component"]
        return successful.pivot_table(
            index=["estimator", "dataset", "n_samples"],
            columns="metric_component",
            values="value",
            aggfunc="mean",
        ).reset_index()

    def to_csv(self, path, **kwargs):
        """Write the raw long-format benchmark results to CSV."""
        self.results_.to_csv(path, index=False, **kwargs)
```

*(Imports are shown next to the code they support for readability. The implementation would collect them at the top of `cd_benchmarks/_base.py`.)*

**Results schema** (`.results_`, long format):

| column | meaning |
|---|---|
| `dataset` | dataset label — the string name, or `repr()` of a simulator instance |
| `n_samples` | the requested sweep value (grouping key; use this for e.g. a metric-vs-n_samples plot) |
| `n_samples_actual` | rows actually loaded — differs from `n_samples` only when a static dataset is smaller than requested (existing `load_dataset` warn-and-cap behavior) |
| `repeat_idx` | zero-based index identifying which repeat this row belongs to (0 when `n_repeats=1`) |
| `data_seed` | the seed used to generate the dataset for this repeat — shared by all estimators in the same `(dataset, n_samples, repeat_idx)` group |
| `estimator_seed` | the seed injected into this estimator via `set_params(seed=...)` — unique per `(dataset, n_samples, repeat_idx, estimator)` task |
| `estimator` | `repr()` of the estimator instance |
| `metric` | metric tag name, such as `SHD` or `adjacency_confusion_matrix` |
| `component` | `value` for scalar metrics, or a dictionary key such as `precision` |
| `value` | scalar result value |
| `runtime_sec` | wall-clock time for fitting the estimator |
| `status` | `"ok"` or `"error"` |
| `error_type` | exception class when the status is `"error"` |
| `error` | exception message when the status is `"error"` |

### User journeys with the solution

**1. Comparing algorithms on a simulated linear Gaussian dataset:**

```python
from pgmpy.causal_discovery import GES, PC
from pgmpy.metrics import SHD

benchmark = CausalDiscoveryBenchmark(
    estimators=[
        PC(ci_test="pearsonr", return_type="dag", max_cond_vars=5),
        GES(scoring_method="bic-g", return_type="dag"),
    ],
    datasets=["linear_gaussian_scm"],
    metrics=[SHD()],
).run()
benchmark.summary()
```

**2. Sweeping sample sizes**, to see how each method's accuracy changes with more data:

```python
from pgmpy.metrics import StructureScore

CausalDiscoveryBenchmark(
    estimators=[
        PC(ci_test="pearsonr", return_type="dag"),
        GES(scoring_method="bic-g", return_type="dag"),
    ],
    datasets=["linear_gaussian_scm"],
    metrics=[SHD(), StructureScore()],
    n_samples=[200, 500, 1000, 5000],
    n_repeats=10,
).run().results_
```

**3. A dataset with no ground truth, scored with an explicit override:**

```python
CausalDiscoveryBenchmark(
    estimators=[PC(ci_test="pearsonr", return_type="dag"), GES(return_type="dag")],
    datasets=["some_real_dataset_without_dagitty_ground_truth"],
    metrics=[SHD()],
    ground_truth={"some_real_dataset_without_dagitty_ground_truth": my_known_dag},
).run()
```

**4. An unsupervised metric with a train/test split** — fit on 70%, score fit quality on the held-out 30%:

```python
CausalDiscoveryBenchmark(
    estimators=[PC(ci_test="pearsonr", return_type="dag"), HillClimbSearch(scoring_method="bic-g")],
    datasets=["linear_gaussian_scm"],
    metrics=[StructureScore()],
    train_test_split=0.7,
).run()
```

**5. Averaging over repeated runs** — with `n_repeats=10`, the raw `results_` contains one row per repeat while `.summary()` averages across repeats:

```python
>>> benchmark = CausalDiscoveryBenchmark(
...     estimators=[PC(ci_test="pearsonr", return_type="dag")],
...     datasets=["linear_gaussian_scm"],
...     metrics=[SHD()],
...     n_samples=[500, 1000],
...     n_repeats=10,
... ).run()
>>> benchmark.results_[["dataset", "n_samples", "repeat_idx", "data_seed", "metric", "value"]].head()
     dataset              n_samples  repeat_idx  data_seed    metric  value
0    linear_gaussian_scm   500       0           291839481    SHD     4.0
1    linear_gaussian_scm   500       1           839104726    SHD     3.0
...
>>> benchmark.summary()  # averages across repeat_idx
```

**6. Persisting raw benchmark results:**

```python
benchmark.to_csv("results/linear_gaussian.csv")
```

### Open Questions

* **Per-task timeouts.** Some estimator/dataset combinations may hang rather than error (e.g. an exhaustive search on a large graph). Not addressed here.
* **Simulator seed contract for `n_repeats`.** Phase 1's `n_repeats` produces independent data draws only for string (registered) datasets, because `load_dataset(..., seed=data_seed)` creates a fresh simulator per call. Initialized `BaseSimulatedDataset` instances do not currently accept an external seed for per-repeat resampling. Phase 2 should extend the simulator contract (e.g. a `reseed(seed)` method or a factory/configuration API) so that `n_repeats` works reliably with initialized instances as well.
* **Distributed execution.** `joblib` supports pluggable backends (`joblib.parallel_config(backend="dask")`). The same `Parallel(n_jobs=...)` calls in `CausalDiscoveryBenchmark.run()` can dispatch tasks to a Dask or Ray cluster, but this requires: (a) installing and registering the backend (`pip install dask distributed`), (b) ensuring estimator, metric, and dataset objects are serializable across workers, and (c) integration testing before calling the backend supported. The initial implementation uses local CPU parallelism only. A Dask example:

    ```python
    from joblib import parallel_config
    from dask.distributed import Client

    client = Client("scheduler-address:8786")
    with parallel_config(backend="dask"):
        benchmark.run()
    ```

    This will be documented as a supported extension once serialization and an integration test are in place.
* **Publishing to `pgmpy.org/causalbench`.** The runner writes a stable CSV in Phase 1. Wiring that output into the `web/` dashboard remains a separate follow-up.
* **Module name.** `cd_benchmarks` mirrors the existing `ci_benchmarks` naming, but `causal_discovery_benchmarks` is more explicit — open to either.

### Rollout plan

* **Phase 1:** `CausalDiscoveryBenchmark` core — input validation, dataset resolution (string datasets only for reliable `n_repeats`), cloned local `joblib` tasks, `n_repeats` with `SeedSequence`-based dual seed streams (`data_seed` + `estimator_seed`), seed injection into estimators via `set_params`, `train_test_split`, ground-truth overrides, long-format `results_` with seed columns for reproducibility, `.summary()`, and `.to_csv()`. Tests cover scalar and dictionary metrics, missing ground truth, unsupported graph types, metric-level errors, isolated nodes, repeated runs, and static versus simulated datasets.
* **Phase 2:** Simulator seed contract for `n_repeats` with initialized instances. Per-task timeouts. Distributed-backend documentation with Dask/Ray integration test.
* **Phase 3:** Dashboard integration using the CSV output convention already used by `ci_benchmarks`.
