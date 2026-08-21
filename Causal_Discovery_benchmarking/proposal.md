## Causal Discovery Benchmarking Infrastructure

Contributors: @Gitanaskhan26

### Introduction

pgmpy currently ships several causal discovery algorithms (`PC`, `GES`, `HillClimbSearch`, `ChowLiu`, `TAN`, `ANM`, `TOPIC`, `ExpertInLoop`, `LLMPairwise`), all unified under a common `BaseCausalDiscovery` interface (`pgmpy/causal_discovery/_base.py`): every estimator exposes `.fit(X)`, populates a fitted `causal_graph_`, and exposes `.score(X=None, true_graph=None, metric=None)`. Separately, `pgmpy.metrics` now ships `BaseSupervisedMetric` / `BaseUnsupervisedMetric`, a `get_metrics(**tag_filters)` registry, and concrete metrics (`SHD`, `AdjacencyConfusionMatrix`, `OrientationConfusionMatrix`, `StructureScore`, `CorrelationScore`, `ImpliedCIs`, `FisherC`), each tagged with `requires_true_graph`, `requires_data`, and `lower_is_better`. The dataset side (this proposal's [Simulation Mixin proposal](../1_simulation_mixin/proposal.md)) has landed as `load_dataset()` / `list_datasets()` / `BaseSimulatedDataset`, with `has_ground_truth` / `is_simulated` tags.

All the pieces to *run* a causal discovery method and *score* it against a dataset already exist. What's missing is a way to compare many of them at once. Today that means a one-off script per experiment. [CausalEval#7](https://github.com/pgmpy/CausalEval/pull/7) (open, unmerged) is a concrete example: it hard-codes a single Linear Gaussian dataset, `n=1000`, exactly two algorithms (PC, GES) in a sequential loop, and — because it predates the `.fit()`/`.score()`/metrics refactor above — it's already written against a superseded API (`PC(data).estimate(...)`, a bare `SHD(true_dag, learned_dag)` call). It would need a rewrite regardless of this proposal. [CausalEval#13](https://github.com/pgmpy/CausalEval/issues/13), opened by @ankurankan, asks for exactly this capability: "a benchmark suite ... to test how well these methods can recover the true graph," and is still open.

CausalEval already solves a structurally similar problem for CI tests: `ci_benchmarks/` (`DGM.py` + `ci_benchmark.py`) sweeps data-generating mechanisms against CI tests via a hand-written dict registry (`DGP_REGISTRY`, `DGM_TO_CITESTS`) and fully sequential nested `for` loops, writing raw + summary CSVs consumed by `pgmpy.org/causalbench`. It predates the tag/`skbase`-object model that `datasets`, `metrics`, and `causal_discovery` have since converged on, and it doesn't parallelize. A generalized benchmarking infrastructure inside pgmpy itself could serve as the common foundation for benchmarking causal discovery, CI tests, causal identification, and eventually causal estimation.

This proposal is for a new `pgmpy.benchmarking` module, starting with `CausalDiscoveryBenchmark`: given a set of causal discovery estimators, datasets, metrics, and sample sizes, run every applicable combination, evaluate the requested metrics, and return one tidy results table. The implementation uses `joblib.Parallel` for execution and is designed from the start so that the same code scales from a laptop to a compute cluster by switching the `joblib` backend.

**Non-goals.** This proposal does not add new causal discovery algorithms, new datasets, or new metrics — it composes the ones that already exist (and the ones sibling proposals in this repo are adding). It also does not cover publishing results to `pgmpy.org/causalbench` — that's a natural follow-up, noted under Open Questions.

### Goals

* One declarative entry point that sweeps `estimators × datasets × n_samples × n_repeats`, evaluates the requested metrics on each fitted result, and returns a single results table.
* Reuse pgmpy's existing primitives end-to-end — `BaseCausalDiscovery.fit`, `pgmpy.metrics`, `load_dataset`/`list_datasets` — rather than re-implementing fitting, scoring, or dataset-loading logic.
* Parallel execution through `joblib`, scaling from a single machine to a compute cluster via `joblib`'s pluggable backend API (Dask, Ray) without any changes to the benchmark class itself.
* Tolerant of partial failure: one incompatible or crashing (estimator, dataset, metric) combination shouldn't abort the rest of the sweep. Errors are recorded per-task with full exception details for debugging.
* Produce a stable CSV export in addition to the in-memory results table, so downstream reporting can consume the same result format.
* Lands in pgmpy within a new `benchmarking` module, designed so the same base infrastructure can later support CI test benchmarks, causal identification benchmarks, and causal estimation benchmarks.

### References

* [CausalEval#13](https://github.com/pgmpy/CausalEval/issues/13) — "[ENH] Benchmarks for causal discovery algorithms on Linear Gaussian Data" (@ankurankan)
* [CausalEval#7](https://github.com/pgmpy/CausalEval/pull/7) — existing one-off PC/GES benchmark script; motivating example for this proposal
* [Simulation Mixin proposal](../1_simulation_mixin/proposal.md) — `load_dataset`/`BaseSimulatedDataset`, which this proposal's dataset resolution builds on directly
* `pgmpy/metrics/` (`_base.py`, `shd.py`, `adjacency_cm.py`, ...) — existing metric implementations this proposal wraps, not reimplements

---

### Proposed Solution

`CausalDiscoveryBenchmark` is configured once (estimators, datasets, metrics, sample sizes, and execution options) and run with `.run()`, which populates `.results_`. It is a long/tidy `DataFrame`: one row per `(dataset, n_samples, repeat_idx, estimator, metric)` combination. A `.summary()` method pivots this into the wide `estimator × dataset` view (averaging across repeats), and `.to_csv()` writes the raw long-format result table.

For each `(dataset, n_samples, repeat_idx)` triple, data is resolved once so every estimator sees the same draw. The resulting estimator tasks run in parallel. A task clones its estimator before fitting, which avoids fitted state leaking across tasks and keeps the design safe with either process or thread backends. When the outer benchmark is parallel, estimators that expose `n_jobs` run with `n_jobs=1` inside a task to avoid nested parallelism.

Each task fits once and evaluates every requested metric against that one fitted graph:

* A metric with `requires_true_graph=True` (e.g. `SHD`) that receives no ground truth will record `status="error"` with the full exception message for debugging. The user is responsible for supplying ground truth (via the dataset or `ground_truth={...}`) for any metric that requires it.
* A metric must support the fitted graph's type. For example, `StructureScore` supports `DAG`, while `PC` and `GES` return `PDAG` by default; a benchmark requiring `StructureScore` must request `return_type="dag"` for those estimators. A type mismatch is recorded as `status="error"` with `error_type` and `error` columns containing the exception class and message.
* If `train_test_split` is set, the estimator fits on the training split and `requires_data=True` metrics receive the held-out split. `requires_true_graph=True` metrics are unaffected because they compare graphs rather than data.

A failure while fitting is recorded once for that estimator task with `status="error"`, `error_type` (exception class name), and `error` (exception message) columns so the user can diagnose which estimator/dataset combination failed and why. A failure in one metric is recorded only for that metric and does not prevent the remaining metrics from being evaluated. There is no `"skipped"` status — every requested combination either produces a value (`"ok"`) or an error (`"error"`).

**Execution and distributed backends.** Tasks are dispatched with `joblib.Parallel`. The benchmark exposes a `backend` parameter (default `"loky"`) that maps directly to `joblib`'s pluggable backend system. Switching from local execution to a Dask or Ray cluster requires only changing this parameter and setting up the cluster externally — no changes to the benchmark class or its task functions. This design follows the same pattern that `sklearn.model_selection.GridSearchCV` uses to scale from a laptop to a cluster.

To make this work across distributed workers, the design enforces two constraints:

1. **Tasks receive lightweight specifications, not heavy data objects.** Data is resolved inside each worker by calling `load_dataset(name, n_samples=..., seed=data_seed)` rather than serializing DataFrames from the main process. For DataFrame inputs without a registry name, the data is passed directly (suitable for local backends; for cluster backends the user is responsible for ensuring workers can access the data).
2. **All task inputs (estimators, metrics) must be picklable.** pgmpy's `BaseCausalDiscovery` estimators inherit from `sklearn.BaseEstimator`, which is already picklable. Metrics inherit from `skbase.base.BaseObject`, also picklable.

This lives in pgmpy within a new `benchmarking` module.

### Alternative Solutions

**A. Results table shape: long, wide, or both?**
The two shapes sketched in the original guidance are actually two different things: a single-metric long table (`dataset, estimator, metric`) and a multi-metric wide table (`Algo, Dataset, M1, M2, M3`). Rather than pick one, `.results_` is long/tidy (one row per metric — easiest to `groupby`/filter/plot, and the natural shape when metrics do not all apply to every dataset) and `.summary()` derives the wide pivot on demand.

**B. Distributed backend choice.**
Evaluated Optuna (shared-database workers), Ray Tune (centralized driver with cloudpickle), scikit-learn/joblib (pluggable backends), benchopt (subprocess + disk caching), W&B Sweeps (pull-based agents), and MLflow (decoupled tracking). The `joblib` pluggable backend approach (scikit-learn pattern) is the best fit: it adds no new dependencies for local execution, reuses pgmpy's existing `joblib` dependency, and transparently scales to Dask/Ray clusters through the same `Parallel(n_jobs=...)` API. The main design constraint it imposes — task inputs must be picklable — is already satisfied by pgmpy's sklearn-compatible estimators and skbase metrics.

**C. Fail-fast vs. per-task error capture.**
A sweep can contain a structurally invalid configuration (for example, an unknown dataset name) or a valid combination that fails only at runtime. The former should fail before tasks start. The latter should be captured in the affected result row, with an exception type and message. Metric evaluation is isolated further: one unsupported or failing metric must not discard the other metric values for the same fitted graph.

**D. Should `CausalDiscoveryBenchmark` itself subclass `BaseEstimator`/`BaseObject`?**
`BaseCausalDiscovery` subclasses sklearn's `BaseEstimator`; `BaseDataset`/`BaseSupervisedMetric`/`BaseUnsupervisedMetric` subclass `skbase.base.BaseObject`. Both exist so those components are *discoverable and swappable* (via `get_metrics()`/`all_objects()`-style registry lookups and `get_params()`/`set_params()`/tags). `CausalDiscoveryBenchmark` isn't itself a pluggable component — nothing needs to look it up by tag or clone it — so it's a plain class.

**E. Metric return types.**
The metrics API is being updated so that each metric returns a single float or an array. The benchmark stores each scalar value in its own row; array-valued metrics produce one row per element. The `component` column is not needed — the metric name itself (or an indexed suffix for array elements) identifies the value.

### Details of proposed solution

**File layout** (inside `pgmpy`):

```
pgmpy/
    benchmarking/
        __init__.py
        _base.py                   # CausalDiscoveryBenchmark, task resolution + execution
    tests/
        test_benchmarking/
            __init__.py
            test_causal_discovery.py
```

**Resolving datasets.** `datasets` accepts several input types:

1. **String name** — resolved through the existing `load_dataset` registry. Each `(dataset, n_samples, repeat_idx)` receives its own `data_seed`, producing independent samples per repeat.
2. **`BaseSimulatedDataset` instance** — e.g. `AdditiveNoiseModel(n_nodes=10, edge_prob=0.3)`. The simulator owns its seed through its constructor. Phase 1 does not inject the benchmark's `data_seed` into initialized simulators.
3. **`Dataset` object** — used directly. Useful when the user has already loaded data with specific preprocessing.
4. **`pd.DataFrame`** — wrapped in a `Dataset` with no ground truth. Suitable for unsupervised metrics or when ground truth is supplied via the `ground_truth` parameter.

The label for a string is its name; the label for a `Dataset` or `BaseSimulatedDataset` is its `repr()`; the label for a DataFrame is `"DataFrame_i(n_rows×n_cols)"` where `i` is its position in the `datasets` list (to disambiguate DataFrames that share the same shape).

See `prototype.py` for the full `_resolve_dataset` implementation.

**Resolving metrics.** The benchmark accepts metric instances only (e.g. `SHD()`, `SHD(edge_reverse_penalty=2)`). Passing an uninstantiated class raises a `TypeError` with a message telling the user to add `()`. Applicability is read from each metric's `requires_true_graph`, `requires_data`, and `supported_graph_types` tags.

See `prototype.py` for the full `_resolve_metrics` implementation.

**Seed propagation.** The benchmark's `seed` drives two independent `numpy.random.SeedSequence` streams:

* **`data_seeds`**: one per `(dataset, n_samples, repeat_idx)` triple. Used by `_resolve_dataset` so every estimator in the same repeat sees the same data draw, and different repeats see different draws. Recorded in the `data_seed` results column.
* **`estimator_seeds`**: one per `(dataset, n_samples, repeat_idx, estimator)` task. Injected into the cloned estimator via `set_params(seed=estimator_seed)` if the estimator accepts a `seed` parameter. Recorded in the `estimator_seed` results column.

Both seeds are stored in `results_` so that any individual row can be reproduced from the CSV alone.

**Running one task.** A task clones and fits one estimator, then evaluates each metric. It returns a list of long-format rows. The learned graph is passed directly to metrics rather than reconstructing from edges, preserving isolated nodes. A fit failure records `status="error"` with the full exception type and message. A metric failure records `status="error"` for that metric only — other metrics still run.

See `prototype.py` for the full `_run_one_task` implementation.

**The `CausalDiscoveryBenchmark` class.** Constructor validates types, stores configuration, and resolves metrics. `.run(n_jobs=None, backend=None)` builds the task grid, generates seed streams, resolves datasets, and dispatches tasks via `joblib.Parallel` with the configured backend. `.summary()` pivots successful results into a wide table, averaging across repeats. `.to_csv()` writes the raw long-format results.

See `prototype.py` for the full class implementation.

**Results schema** (`.results_`, long format):

| column | meaning |
|---|---|
| `dataset` | dataset label — string name, `repr()` of a simulator/Dataset, or `"DataFrame_i(n×m)"` |
| `n_samples` | the requested sweep value (grouping key) |
| `n_samples_actual` | rows actually loaded — differs from `n_samples` when a static dataset is smaller |
| `repeat_idx` | zero-based repeat index (0 when `n_repeats=1`) |
| `data_seed` | seed for data generation — shared by all estimators in the same `(dataset, n_samples, repeat_idx)` group |
| `estimator_seed` | seed injected into this estimator via `set_params(seed=...)` — unique per task |
| `estimator` | `repr()` of the estimator instance |
| `metric` | metric name (e.g. `SHD`) |
| `value` | scalar float result |
| `runtime_sec` | wall-clock time for fitting the estimator |
| `status` | `"ok"` or `"error"` |
| `error_type` | exception class name when `status="error"` |
| `error` | exception message when `status="error"` |

### User journeys with the solution

**1. Comparing algorithms on a registered dataset:**

```python
from pgmpy.benchmarking import CausalDiscoveryBenchmark
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

```
   estimator                                              dataset              n_samples  SHD
0  GES(scoring_method='bic-g', return_type='dag')         linear_gaussian_scm  None       2.0
1  PC(ci_test='pearsonr', return_type='dag', ...)         linear_gaussian_scm  None       4.0
```

**2. Sweeping sample sizes with repeated runs:**

```python
from pgmpy.metrics import StructureScore

benchmark = CausalDiscoveryBenchmark(
    estimators=[
        PC(ci_test="pearsonr", return_type="dag"),
        GES(scoring_method="bic-g", return_type="dag"),
    ],
    datasets=["linear_gaussian_scm"],
    metrics=[SHD(), StructureScore()],
    n_samples=[200, 500, 1000, 5000],
    n_repeats=10,
).run()
benchmark.results_
```

```
    dataset              n_samples  n_samples_actual  repeat_idx  data_seed    estimator_seed  estimator                         metric           value  runtime_sec  status  error_type  error
0   linear_gaussian_scm  200        200               0           291839481    3847261094       PC(ci_test='pearsonr', ...)       SHD              6.0  0.34         ok      None        None
1   linear_gaussian_scm  200        200               0           291839481    3847261094       PC(ci_test='pearsonr', ...)       structure_score  -4521.2  0.34     ok      None        None
2   linear_gaussian_scm  200        200               0           291839481    1650283947       GES(scoring_method='bic-g', ...)  SHD              5.0  0.52         ok      None        None
3   linear_gaussian_scm  200        200               0           291839481    1650283947       GES(scoring_method='bic-g', ...)  structure_score  -4380.7  0.52     ok      None        None
4   linear_gaussian_scm  200        200               1           839104726    2019384756       PC(ci_test='pearsonr', ...)       SHD              7.0  0.31         ok      None        None
..  ...                  ...        ...               ...         ...          ...              ...                               ...              ...  ...          ...     ...         ...
```

```python
benchmark.summary()
```

```
   estimator                                              dataset              n_samples  SHD   structure_score
0  GES(scoring_method='bic-g', return_type='dag')         linear_gaussian_scm  200        4.1   -4312.5
1  GES(scoring_method='bic-g', return_type='dag')         linear_gaussian_scm  500        2.3   -10842.1
2  GES(scoring_method='bic-g', return_type='dag')         linear_gaussian_scm  1000       1.2   -21503.8
3  GES(scoring_method='bic-g', return_type='dag')         linear_gaussian_scm  5000       0.4   -107218.4
4  PC(ci_test='pearsonr', return_type='dag')              linear_gaussian_scm  200        5.8   -4487.3
5  PC(ci_test='pearsonr', return_type='dag')              linear_gaussian_scm  500        3.4   -10923.6
6  PC(ci_test='pearsonr', return_type='dag')              linear_gaussian_scm  1000       2.1   -21612.0
7  PC(ci_test='pearsonr', return_type='dag')              linear_gaussian_scm  5000       0.8   -107345.2
```

**3. Using a DataFrame directly (no ground truth):**

```python
import pandas as pd

df = pd.read_csv("my_observational_data.csv")

CausalDiscoveryBenchmark(
    estimators=[PC(ci_test="pearsonr", return_type="dag")],
    datasets=[df],
    metrics=[StructureScore()],
).run()
```

**4. Ground-truth override for a dataset without one:**

```python
CausalDiscoveryBenchmark(
    estimators=[PC(ci_test="pearsonr", return_type="dag"), GES(return_type="dag")],
    datasets=["some_real_dataset"],
    metrics=[SHD()],
    ground_truth={"some_real_dataset": my_known_dag},
).run()
```

**5. Running on a Dask cluster:**

```python
from dask.distributed import Client

client = Client("scheduler-address:8786")

benchmark = CausalDiscoveryBenchmark(
    estimators=[PC(ci_test="pearsonr", return_type="dag"), GES(return_type="dag")],
    datasets=["linear_gaussian_scm"],
    metrics=[SHD()],
    n_samples=[500, 1000, 5000],
    n_repeats=20,
).run(backend="dask")
```

**6. Inspecting errors from failed estimator/dataset combinations:**

```python
>>> benchmark.results_.query("status == 'error'")[["estimator", "dataset", "error_type", "error"]]
     estimator      dataset             error_type        error
42   PC(...)        large_graph_dataset  MemoryError       Unable to allocate 8.00 GiB...
```

**7. Persisting and resuming results:**

```python
benchmark.to_csv("results/linear_gaussian.csv")
```

### Open Questions

* **Per-task timeouts.** Some estimator/dataset combinations may hang rather than error (e.g. an exhaustive search on a large graph). Not addressed here.
* **Simulator seed contract for `n_repeats`.** Phase 1's `n_repeats` produces independent data draws only for string (registered) datasets, because `load_dataset(..., seed=data_seed)` creates a fresh simulator per call. Initialized `BaseSimulatedDataset` instances do not currently accept an external seed for per-repeat resampling. Phase 2 should extend the simulator contract (e.g. a `reseed(seed)` method) so that `n_repeats` works reliably with initialized instances as well.
* **Distributed execution prerequisites.** `joblib`'s pluggable backend API (`joblib.parallel_config(backend="dask")`) lets the same `Parallel(n_jobs=...)` calls dispatch to Dask or Ray workers. The benchmark is designed for this — tasks receive lightweight specs, estimators/metrics are picklable. However, cluster execution still requires: (a) backend installation (`pip install dask distributed`), (b) serialization testing for all estimator/metric combinations, and (c) an integration test. These will be validated in Phase 2.
* **Checkpointing.** Long-running benchmarks (hours to days) would benefit from disk-based checkpointing so that crashed runs can resume without re-running completed tasks. This is a Phase 2 feature, following the `benchopt` pattern of appending results to disk after each task.
* **CI test / causal identification benchmarks.** The `CausalDiscoveryBenchmark` design is intentionally structured so that a similar class can be created for CI tests (replacing the hand-written `CausalEval/ci_benchmarks/` loops), causal identification, and causal estimation. The shared infrastructure (grid construction, seed management, parallel dispatch, error isolation, results collection) could be factored into a `BaseBenchmark` when the second benchmark class is added. Phase 1 does not add this base class — it would be premature abstraction before the second use case exists.
* **Publishing to `pgmpy.org/causalbench`.** The runner writes a stable CSV. Wiring that output into a dashboard remains a separate follow-up.

### Rollout plan

* **Phase 1:** `CausalDiscoveryBenchmark` in `pgmpy/benchmarking/`. Input validation and dataset resolution (strings, DataFrames, `Dataset` objects, `BaseSimulatedDataset` instances). Cloned `joblib` tasks with configurable `backend`. `n_repeats` with `SeedSequence`-based dual seed streams (`data_seed` + `estimator_seed`). Seed injection into estimators via `set_params`. `train_test_split`. Ground-truth overrides. Long-format `results_` with seed columns and full error details. `.summary()` and `.to_csv()`. Tests cover scalar and array metrics, missing ground truth, unsupported graph types, metric-level errors, isolated nodes, repeated runs, DataFrames, and static versus simulated datasets.
* **Phase 2:** Serialization test suite for distributed backends. Disk-based checkpointing. Simulator seed contract for `n_repeats` with initialized instances. Per-task timeouts. `BaseBenchmark` factoring when a second benchmark class (CI tests or causal identification) is added.
* **Phase 3:** Dashboard integration. CI test benchmark migration from CausalEval.
