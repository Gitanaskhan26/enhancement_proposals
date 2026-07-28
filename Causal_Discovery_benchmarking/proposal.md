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

`CausalDiscoveryBenchmark` is configured once (estimators, datasets, metrics, sample sizes, and a few execution options) and run with `.run()`, which populates `.results_`. It is a long/tidy `DataFrame`: each scalar metric result has one row per `(dataset, n_samples, estimator, metric, component)` combination. Scalar metrics use `component="value"`; dictionary-valued metrics contribute one row for each returned key. A `.summary()` method pivots this into the wide `estimator × dataset` view, and `.to_csv()` writes the raw long-format result table.

For each `(dataset, n_samples)` pair, data is resolved once in the main process so every estimator sees the same draw. The resulting estimator tasks run in parallel. A task clones its estimator before fitting, which avoids fitted state leaking across tasks and keeps the design safe with either process or thread backends. When the outer benchmark is parallel, estimators that expose `n_jobs` run with `n_jobs=1` inside a task to avoid nested parallelism.

Each task fits once and evaluates every requested metric against that one fitted graph. Which metrics apply is resolved from metric tags and the fitted graph:

* A metric with `requires_true_graph=True` (e.g. `SHD`) only runs if a ground-truth graph is available, either from the dataset or through `ground_truth={label: dag, ...}`.
* A metric must support the fitted graph's type. For example, `StructureScore` supports `DAG`, while `PC` and `GES` return `PDAG` by default; a benchmark requiring `StructureScore` must request `return_type="dag"` for those estimators.
* If `train_test_split` is set, the estimator fits on the training split and `requires_data=True` metrics receive the held-out split. `requires_true_graph=True` metrics are unaffected because they compare graphs rather than data.

Metrics that do not apply are recorded as `status="skipped"` rows with a reason. A failure while fitting is recorded once for that estimator task; a failure in one metric is recorded only for that metric and does not prevent the remaining metrics from being evaluated.

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

**Resolving datasets.** `datasets` accepts a string name (resolved through the existing `load_dataset` registry) or an already-constructed `BaseSimulatedDataset` instance (e.g. `AdditiveNoiseModel(n_nodes=10, edge_prob=0.3)`). The latter is useful when configuration contains objects that cannot be represented as keyword arguments, such as a custom noise distribution or a fixed DAG. A string dataset uses the benchmark's `seed`; an initialized simulator owns its seed through its constructor. The label for a string is its name and the label for an instance is its `repr()`.

```python
from pgmpy.datasets import load_dataset, list_datasets
from pgmpy.datasets._base import BaseSimulatedDataset, Dataset

def _resolve_dataset(dataset, n_samples, seed) -> tuple[str, Dataset]:
    """Load one (dataset, n_samples) combination, returning a (label, Dataset) pair."""
    if isinstance(dataset, str):
        if dataset not in list_datasets():
            raise ValueError(f"Unknown dataset name: {dataset!r}")
        return dataset, load_dataset(dataset, n_samples=n_samples, seed=seed)

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

**Resolving metrics.** The benchmark accepts metric classes (`SHD`) or instances (`SHD(edge_reverse_penalty=2)`); classes are instantiated with defaults. Inputs are validated as pgmpy metric objects. Applicability is read from each metric's `requires_true_graph`, `requires_data`, and `supported_graph_types` tags.

```python
from pgmpy.metrics import BaseSupervisedMetric, BaseUnsupervisedMetric

def _resolve_metrics(metrics):
    resolved = [metric() if isinstance(metric, type) else metric for metric in metrics]
    if not all(isinstance(metric, (BaseSupervisedMetric, BaseUnsupervisedMetric)) for metric in resolved):
        raise TypeError("metrics must contain pgmpy metric classes or instances")
    return resolved
```

**Running one task.** A task clones and fits one estimator, then evaluates each metric separately. It returns a list of long-format rows. It passes the learned graph directly to metrics rather than reconstructing one from its edges, preserving isolated nodes. This keeps estimator state local to the task, avoids nested parallelism, and lets one metric fail without hiding the other results:

```python
import time
from sklearn.base import clone
from sklearn.model_selection import train_test_split as sk_train_test_split

def _run_one_task(dataset_label, dataset, n_samples, estimator, metrics, train_test_split, seed, outer_n_jobs):
    base_row = {
        "dataset": dataset_label,
        "n_samples": n_samples,
        "n_samples_actual": dataset.data.shape[0],
        "estimator": repr(estimator),
    }
    try:
        train_df, test_df = dataset.data, dataset.data
        if train_test_split is not None:
            train_df, test_df = sk_train_test_split(
                dataset.data, train_size=train_test_split, random_state=seed
            )

        fitted = clone(estimator)
        if outer_n_jobs != 1 and "n_jobs" in fitted.get_params(deep=False):
            fitted.set_params(n_jobs=1)
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
        supported_types = metric.get_tag("supported_graph_types", (), raise_error=False)
        if needs_truth and dataset.ground_truth is None:
            rows.append(
                {
                    **metric_row,
                    "component": None,
                    "value": None,
                    "status": "skipped",
                    "error": "no ground truth",
                }
            )
            continue
        if not isinstance(fitted.causal_graph_, supported_types):
            rows.append(
                {
                    **metric_row,
                    "component": None,
                    "value": None,
                    "status": "skipped",
                    "error": "unsupported graph type",
                }
            )
            continue
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
        ground_truth: dict | None = None,
        train_test_split: float | None = None,
        seed: int = 42,
        n_jobs: int = -1,
        show_progress: bool = True,
    ):
        if not all(isinstance(estimator, BaseCausalDiscovery) for estimator in estimators):
            raise TypeError("estimators must contain BaseCausalDiscovery instances")
        self.estimators = estimators
        self.datasets = datasets
        self.metrics = _resolve_metrics(metrics)
        self.n_samples = n_samples if isinstance(n_samples, list) else [n_samples]
        self.ground_truth = ground_truth or {}
        self.train_test_split = train_test_split
        self.seed = seed
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def run(self, n_jobs=None, show_progress=None) -> "CausalDiscoveryBenchmark":
        n_jobs = self.n_jobs if n_jobs is None else n_jobs
        show_progress = self.show_progress if show_progress is None else show_progress

        tasks = []
        for dataset_spec in self.datasets:
            for n in self.n_samples:
                label, dataset = _resolve_dataset(dataset_spec, n, self.seed)
                if dataset.ground_truth is None and label in self.ground_truth:
                    dataset.ground_truth = self.ground_truth[label]
                for estimator in self.estimators:
                    tasks.append((label, dataset, n, estimator))

        task_rows = Parallel(n_jobs=n_jobs, verbose=10 if show_progress else 0)(
            delayed(_run_one_task)(
                label, dataset, n, estimator, self.metrics, self.train_test_split, self.seed, n_jobs
            )
            for label, dataset, n, estimator in tasks
        )
        self.results_ = pd.DataFrame([row for rows in task_rows for row in rows])
        return self

    def summary(self) -> pd.DataFrame:
        """Return a wide estimator-by-dataset view of successful metric values."""
        successful = self.results_.query("status == 'ok'").copy()
        successful["metric_component"] = successful["metric"] + "." + successful["component"]
        return successful.pivot_table(
            index=["estimator", "dataset", "n_samples"],
            columns="metric_component",
            values="value",
            aggfunc="first",
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
| `estimator` | `repr()` of the estimator instance |
| `metric` | metric tag name, such as `SHD` or `adjacency_confusion_matrix` |
| `component` | `value` for scalar metrics, or a dictionary key such as `precision` |
| `value` | scalar result value |
| `runtime_sec` | wall-clock time for fitting the estimator |
| `status` | `"ok"`, `"skipped"`, or `"error"` |
| `error_type` | exception class when the status is `"error"` |
| `error` | skip reason or exception message when the status is not `"ok"` |

### User journeys with the solution

**1. Comparing algorithms on a simulated linear Gaussian dataset:**

```python
from pgmpy.causal_discovery import GES, PC
from pgmpy.datasets.linear_gaussian_scm import LinearGaussianSCM
from pgmpy.metrics import SHD

benchmark = CausalDiscoveryBenchmark(
    estimators=[
        PC(ci_test="pearsonr", return_type="dag", max_cond_vars=5),
        GES(scoring_method="bic-g", return_type="dag"),
    ],
    datasets=[LinearGaussianSCM(n_nodes=10, edge_prob=0.3, seed=42)],
    metrics=[SHD],
).run()
benchmark.summary()
```

**2. Sweeping sample sizes**, to see how each method's accuracy changes with more data:

```python
CausalDiscoveryBenchmark(
    estimators=[
        PC(ci_test="pearsonr", return_type="dag"),
        GES(scoring_method="bic-g", return_type="dag"),
    ],
    datasets=[LinearGaussianSCM(n_nodes=10, edge_prob=0.3)],
    metrics=[SHD, StructureScore],
    n_samples=[200, 500, 1000, 5000],
).run().results_
```

**3. A dataset with no ground truth, scored with an explicit override:**

```python
CausalDiscoveryBenchmark(
    estimators=[PC(ci_test="pearsonr", return_type="dag"), GES(return_type="dag")],
    datasets=["some_real_dataset_without_dagitty_ground_truth"],
    metrics=[SHD],
    ground_truth={"some_real_dataset_without_dagitty_ground_truth": my_known_dag},
).run()
```

**4. An unsupervised metric with a train/test split** — fit on 70%, score fit quality on the held-out 30%:

```python
CausalDiscoveryBenchmark(
    estimators=[PC(ci_test="pearsonr", return_type="dag"), HillClimbSearch(scoring_method="bic-g")],
    datasets=[LinearGaussianSCM(n_nodes=8, edge_prob=0.3, seed=42)],
    metrics=[StructureScore],
    train_test_split=0.7,
).run()
```

**5. Reading off a skipped metric** — one dataset has no ground truth while another runs cleanly:

```python
>>> benchmark.results_[["dataset", "metric", "status", "error"]]
     dataset               metric  status    error
0    LinearGaussianSCM(...) SHD    ok        None
1    some_dataset_no_gt    SHD     skipped   no ground truth
```

**6. Persisting raw benchmark results:**

```python
benchmark.to_csv("results/linear_gaussian.csv")
```

### Open Questions

* **Repeats across stochastic simulators.** A single draw from a simulator is noisy; averaging over several seeds per `(dataset, estimator, n_samples)` cell would make comparisons more reliable, similar to `ci_benchmarks`' existing `n_repeats`. Phase 1 uses the benchmark `seed` for string dataset generation and train/test splitting; initialized simulators and stochastic estimator settings remain explicit user configuration. A future `n_repeats` API should define a per-task seed schedule.
* **Per-task timeouts.** Some estimator/dataset combinations may hang rather than error (e.g. an exhaustive search on a large graph). Not addressed here.
* **Distributed execution.** Dask or Ray can be considered after local execution is tested. They should remain optional joblib-compatible backends rather than dependencies of CausalEval.
* **Publishing to `pgmpy.org/causalbench`.** The runner writes a stable CSV in Phase 1. Wiring that output into the `web/` dashboard remains a separate follow-up.
* **Module name.** `cd_benchmarks` mirrors the existing `ci_benchmarks` naming, but `causal_discovery_benchmarks` is more explicit — open to either.

### Rollout plan

* **Phase 1:** `CausalDiscoveryBenchmark` core — input validation, dataset resolution, cloned local `joblib` tasks, `train_test_split`, ground-truth overrides, long-format `results_`, `.summary()`, and `.to_csv()`. Tests cover scalar and dictionary metrics, missing ground truth, unsupported graph types, metric-level errors, isolated nodes, and static versus simulated datasets.
* **Phase 2:** Repeated runs with an explicit seed schedule and per-task timeouts.
* **Phase 3:** Optional distributed-backend documentation and dashboard integration using the CSV output convention already used by `ci_benchmarks`.
