## Causal Discovery Benchmarkrking Infrastructure

Contributors: @Gitanaskhan26

### Introduction

pgmpy currently ships several causal discovery algorithms (`PC`, `GES`, `HillClimbSearch`, `TreeSearch`/`ChowLiu`/`TAN`, `ANM`, `TOPIC`, `ExpertInLoop`, `LLMPairwise`), all unified under a common `BaseCausalDiscovery` interface (`pgmpy/causal_discovery/_base.py`): every estimator exposes `.fit(X)`, populates a fitted `causal_graph_`, and exposes `.score(X=None, true_graph=None, metric=None)`. Separately, `pgmpy.metrics` now ships `BaseSupervisedMetric` / `BaseUnsupervisedMetric`, a `get_metrics(**tag_filters)` registry, and concrete metrics (`SHD`, `AdjacencyConfusionMatrix`, `OrientationConfusionMatrix`, `StructureScore`, `CorrelationScore`, `ImpliedCIs`, `FisherC`), each tagged with `requires_true_graph`, `requires_data`, and `lower_is_better`. The dataset side (this proposal's [Simulation Mixin proposal](../1_simulation_mixin/proposal.md)) has landed as `load_dataset()` / `list_datasets()` / `BaseSimulatedDataset`, with `has_ground_truth` / `is_simulated` tags.

All the pieces to *run* a causal discovery method and *score* it against a dataset already exist. What's missing is a way to compare many of them at once. Today that means a one-off script per experiment. [CausalEval#7](https://github.com/pgmpy/CausalEval/pull/7) (open, unmerged) is a concrete example: it hard-codes a single Linear Gaussian dataset, `n=1000`, exactly two algorithms (PC, GES) in a sequential loop, and — because it predates the `.fit()`/`.score()`/metrics refactor above — it's already written against a superseded API (`PC(data).estimate(...)`, a bare `SHD(true_dag, learned_dag)` call). It would need a rewrite regardless of this proposal. [CausalEval#13](https://github.com/pgmpy/CausalEval/issues/13), opened by @ankurankan, asks for exactly this capability: "a benchmark suite ... to test how well these methods can recover the true graph," and is still open.

CausalEval already solves a structurally similar problem for CI tests: `ci_benchmarks/` (`DGM.py` + `ci_benchmark.py`) sweeps data-generating mechanisms against CI tests via a hand-written dict registry (`DGP_REGISTRY`, `DGM_TO_CITESTS`) and fully sequential nested `for` loops, writing raw + summary CSVs consumed by `pgmpy.org/causalbench`. It predates the tag/`skbase`-object model that `datasets`, `metrics`, and `causal_discovery` have since converged on, and it doesn't parallelize.

This proposal is for `CausalDiscoveryBenchmark`: given a set of causal discovery estimators, a set of datasets/simulators, a set of metrics, and a set of sample sizes, run every applicable combination, evaluate the requested metrics, and return one tidy results table — parallelized across CPU cores by default, and across a cluster without any cluster-specific code in pgmpy or CausalEval.

**Non-goals.** This proposal does not add new causal discovery algorithms, new datasets, or new metrics — it composes the ones that already exist (and the ones sibling proposals in this repo are adding). It also does not cover publishing results to `pgmpy.org/causalbench` (CausalEval's existing `web/` dashboard) — that's a natural follow-up, noted under Open Questions, but kept out of scope here so this proposal stays focused on the benchmarking class itself.

### Goals

* One declarative entry point that sweeps `estimators × datasets × n_samples`, evaluates the requested metrics on each fitted result, and returns a single results table.
* Reuse pgmpy's existing primitives end-to-end — `BaseCausalDiscovery.fit`, `pgmpy.metrics.get_metrics`, `load_dataset`/`list_datasets` — rather than re-implementing fitting, scoring, or dataset-loading logic.
* Parallel by default on a single machine; scalable to a cluster without adding a hard dependency on a specific distributed-computing framework.
* Tolerant of partial failure: one incompatible or crashing (estimator, dataset, metric) combination shouldn't abort the rest of the sweep.
* Lands in `CausalEval`, alongside the existing `ci_benchmarks/`.

### References

* [CausalEval#13](https://github.com/pgmpy/CausalEval/issues/13) — "[ENH] Benchmarks for causal discovery algorithms on Linear Gaussian Data" (@ankurankan)
* [CausalEval#7](https://github.com/pgmpy/CausalEval/pull/7) — existing one-off PC/GES benchmark script; motivating example for this proposal
* [Simulation Mixin proposal](../1_simulation_mixin/proposal.md) — `load_dataset`/`BaseSimulatedDataset`, which this proposal's dataset resolution builds on directly
* `pgmpy/metrics/` (`_base.py`, `shd.py`, `adjacency_cm.py`, ...) — existing metric implementations this proposal wraps, not reimplements

---

### Proposed Solution

`CausalDiscoveryBenchmark` is configured once (estimators, datasets, metrics, sample sizes, and a few execution options) and run with `.run()`, which populates `.results_` — a long/tidy `DataFrame` with one row per `(dataset, n_samples, estimator, metric)` combination. A `.summary()` method pivots this into the wide `estimator × dataset` view sketched in the original guidance, with one column per metric.

The unit of parallel work is one `(dataset, n_samples, estimator)` triple: load/simulate the data once, call `.fit()` once, then evaluate *every* requested metric against that single fit (metric evaluation is cheap relative to structure search, so there's no reason to refit per metric). Which metrics apply to a given dataset is resolved automatically from tags already on the metric classes and the dataset:

* A metric with `requires_true_graph=True` (e.g. `SHD`) only runs if the dataset has `has_ground_truth=True`, or the caller supplied an explicit override via `ground_truth={label: dag, ...}`.
* A metric with `requires_data=True` (e.g. `StructureScore`) always runs, since it doesn't need ground truth — it scores the fitted graph against held-out data.
* If `train_test_split` is set, the estimator fits on the training split and `requires_data=True` metrics are evaluated against the held-out split; `requires_true_graph=True` metrics are unaffected, since they compare graphs, not data.

Combinations that don't apply (e.g. an `SHD` request against a dataset with no ground truth and no override) are dropped up front with a logged warning rather than silently producing `NaN`s or raising mid-sweep.

For execution, tasks are dispatched with `joblib.Parallel` — already a hard dependency of pgmpy, and already used the same way elsewhere in the codebase (`_TreeSearchMixin._get_weights`). This gets multi-core execution for free with `n_jobs`, and cluster execution for free too: joblib supports pluggable backends, so wrapping the call in `with joblib.parallel_config(backend="dask"): benchmark.run()` (after connecting a `dask.distributed.Client`) or `backend="ray"` (after `ray.util.joblib.register_ray()`) routes the exact same tasks to a cluster, with zero cluster-specific code in this proposal. Each task catches its own exceptions, so a single estimator crashing or timing out on one dataset shows up as a `status`/`error` cell rather than killing the run.

This lives in `CausalEval` as `cd_benchmarks/`, a sibling to `ci_benchmarks/`, rather than in pgmpy core — it composes public pgmpy APIs and doesn't need to live inside the library itself, matching where the analogous CI-test benchmarking already lives.

### Alternative Solutions

**A. Results table shape: long, wide, or both?**
The two shapes sketched in the original guidance are actually two different things: a single-metric long table (`dataset, estimator, metric`) and a multi-metric wide table (`Algo, Dataset, M1, M2, M3`). Rather than pick one, `.results_` is long/tidy (one row per metric — easiest to `groupby`/filter/plot, and the natural shape when metrics don't all apply to every dataset) and `.summary()` derives the wide pivot on demand. This also matches `ci_benchmarks`' existing raw-CSV-plus-summary-CSV convention, so it isn't a new pattern for the repo.

**B. Hard dependency on Dask/Ray vs. joblib's pluggable backend.**
Considered adding `dask.distributed` or `ray` directly and branching on an explicit `backend=` argument. Rejected: it forces a choice of cluster framework onto everyone who installs CausalEval, even those who only ever run locally, and pgmpy already has a working precedent (`_TreeSearchMixin`) for joblib-only parallelism. Depending only on `joblib.Parallel` and letting users bring their own joblib-compatible backend (`loky` locally by default, `dask`/`ray` for a cluster, both of which register themselves as joblib backends) gets cluster scaling without CausalEval ever importing a distributed-computing library itself.

**C. Fail-fast vs. per-task error capture.**
A sweep this size will occasionally hit a combination that doesn't make sense (an `SHD` request with no ground truth anywhere) or one that crashes at runtime (a solver that diverges on a particular sample). Failing the entire `.run()` for one bad cell is unfriendly for something meant to run tens or hundreds of combinations unattended. Chosen approach: validate structurally invalid configuration eagerly, before any task starts (e.g. an `estimators` entry that isn't a `BaseCausalDiscovery`, a dataset string `list_datasets()` doesn't recognize) — that should fail loudly and immediately — but catch exceptions *within* each dispatched task and record them in a `status` column, so a single flaky combination doesn't take down a multi-hour run.

**D. Should `CausalDiscoveryBenchmark` itself subclass `BaseEstimator`/`BaseObject`?**
`BaseCausalDiscovery` subclasses sklearn's `BaseEstimator`; `BaseDataset`/`BaseSupervisedMetric`/`BaseUnsupervisedMetric` subclass `skbase.base.BaseObject`. Both exist so those components are *discoverable and swappable* (via `get_metrics()`/`all_objects()`-style registry lookups and `get_params()`/`set_params()`/tags). `CausalDiscoveryBenchmark` isn't itself a pluggable component — nothing needs to look it up by tag or clone it — so it's a plain class. It does still benefit from the same `repr()` convention (see Details), just without inheriting anything to get it.

**E. Scalar-only metrics vs. handling structured metric values.**
Not every metric in `pgmpy.metrics` returns a single number — `AdjacencyConfusionMatrix.evaluate()` returns a `Dict[str, float]` (precision, recall, F1, ...), not a scalar. Assuming scalars-only would either break on that metric or require a separate code path for it. Instead, `_run_one_task` flattens dict-valued metric results into one column per key (`AdjacencyConfusionMatrix.precision`, `AdjacencyConfusionMatrix.recall`, ...), so the wide `.summary()` table stays fully tabular regardless of which metrics are requested.

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

**Resolving datasets.** `datasets` accepts a string name (resolved through the existing `load_dataset` registry) or an already-constructed `BaseSimulatedDataset` instance (e.g. `AdditiveNoiseModel(n_nodes=10, edge_prob=0.3)`), which matters because some simulator configuration — a custom noise distribution, a fixed DAG — can't always be round-tripped through a string-plus-kwargs call. The label for a string is the name itself; the label for an instance is its `repr()`, which both `BaseDataset` (via `skbase.base.BaseObject`) and `BaseCausalDiscovery` (via sklearn's `BaseEstimator`) already produce for free, showing only non-default constructor arguments — I checked this empirically against `skbase.base.BaseObject` and it matches sklearn's behavior exactly (e.g. `AdditiveNoiseModel(n_nodes=10)`, not every default field). That's also exactly the format the original guidance's example table already assumed (`PC(ci_test='chi_square')`), so no separate labeling scheme is needed for estimators either.

```python
from pgmpy.datasets import load_dataset, list_datasets, BaseDataset
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

**Resolving metrics.** Accepts metric classes (`SHD`) or instances (`SHD(edge_reverse_penalty=2)`) — classes are instantiated with defaults. Applicability is read straight off each metric's own tags (`requires_true_graph`) rather than an `isinstance(metric, BaseSupervisedMetric)` check — the two are equivalent for pgmpy's built-in metrics, but tags also work for a metric that mixes both behaviors, and match how `BaseCausalDiscovery.score()` already resolves metrics internally:

```python
def _resolve_metrics(metrics):
    return [m() if isinstance(m, type) else m for m in metrics]

def _applicable_metrics(metrics, has_ground_truth: bool):
    applicable, skipped = [], []
    for metric in metrics:
        needs_truth = metric.get_tag("requires_true_graph", False, raise_error=False)
        if needs_truth and not has_ground_truth:
            name = metric.get_tag("name", repr(metric), raise_error=False)
            skipped.append((name, "no ground truth available"))
        else:
            applicable.append(metric)
    return applicable, skipped
```

**Running one task.** Fits once, times it, evaluates every applicable metric against that one fit, flattens dict-valued results, and never lets an exception escape:

```python
import time
from sklearn.model_selection import train_test_split as sk_train_test_split

def _run_one_task(dataset_label, dataset, n_samples, estimator, metrics, train_test_split, seed):
    row = {
        "dataset": dataset_label,
        "n_samples": n_samples,
        "n_samples_actual": dataset.data.shape[0],
        "estimator": repr(estimator),
        "status": "ok",
    }
    try:
        has_gt = dataset.ground_truth is not None
        applicable, skipped = _applicable_metrics(metrics, has_gt)
        for name, reason in skipped:
            row[name] = None  # metric not applicable to this dataset; see `status`
        if skipped:
            row["status"] = f"skipped {len(skipped)} metric(s): " + "; ".join(
                f"{n} ({r})" for n, r in skipped
            )

        train_df, test_df = dataset.data, dataset.data
        if train_test_split is not None:
            train_df, test_df = sk_train_test_split(
                dataset.data, train_size=train_test_split, random_state=seed
            )

        start = time.perf_counter()
        fitted = estimator.fit(train_df)
        row["runtime_sec"] = time.perf_counter() - start

        for metric in applicable:
            name = metric.get_tag("name", repr(metric), raise_error=False)
            if metric.get_tag("requires_true_graph", False, raise_error=False):
                value = metric.evaluate(true_causal_graph=dataset.ground_truth, est_causal_graph=fitted.causal_graph_)
            else:
                value = metric.evaluate(X=test_df, causal_graph=fitted.causal_graph_)
            if isinstance(value, dict):
                for k, v in value.items():
                    row[f"{name}.{k}"] = v
            else:
                row[name] = value
    except Exception as exc:  # one bad combination must not sink the whole sweep
        row["status"] = f"error: {exc}"
    return row
```

**The `CausalDiscoveryBenchmark` class itself:**

```python
import pandas as pd
from joblib import Parallel, delayed

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

        rows = Parallel(n_jobs=n_jobs, verbose=10 if show_progress else 0)(
            delayed(_run_one_task)(
                label, dataset, n, estimator, self.metrics, self.train_test_split, self.seed
            )
            for label, dataset, n, estimator in tasks
        )
        self.results_ = pd.DataFrame(rows)
        return self

    def summary(self) -> pd.DataFrame:
        """Wide estimator x dataset view, one column per metric (the Algo/Dataset/M1/M2/M3 shape)."""
        value_cols = [
            c for c in self.results_.columns
            if c not in {"dataset", "n_samples", "n_samples_actual", "estimator", "status", "runtime_sec"}
        ]
        return self.results_.pivot_table(
            index=["estimator", "dataset", "n_samples"], values=value_cols, aggfunc="first"
        ).reset_index()
```

*(Imports shown next to the code they support, for readability here; in the actual module they'd be collected at the top of `cd_benchmarks/_base.py`. `BaseCausalDiscovery` and `BaseSimulatedDataset` aren't currently exported from their packages' `__init__.py` `__all__` — I imported them from `pgmpy.causal_discovery._base` / `pgmpy.datasets._base` directly above and checked that this resolves correctly against current `dev`. The same is true one level down: concrete simulators like `AdditiveNoiseModel`/`LinearGaussianSCM` aren't re-exported from `pgmpy.datasets` either, only from their own submodules — fine when going through `load_dataset("anm")`, but it's the exact "initialized dataset object" path this proposal needs, so it's worth exporting alongside `BaseCausalDiscovery`/`BaseSimulatedDataset`. All three are one-line, separate fixes worth raising regardless of this proposal.)*

**Results schema** (`.results_`, long format):

| column | meaning |
|---|---|
| `dataset` | dataset label — the string name, or `repr()` of a simulator instance |
| `n_samples` | the requested sweep value (grouping key; use this for e.g. a metric-vs-n_samples plot) |
| `n_samples_actual` | rows actually loaded — differs from `n_samples` only when a static dataset is smaller than requested (existing `load_dataset` warn-and-cap behavior) |
| `estimator` | `repr()` of the estimator instance |
| `runtime_sec` | wall-clock time for `.fit()` |
| `<metric name>` | one column per scalar metric; dict-valued metrics get `<metric name>.<key>` instead |
| `status` | `"ok"`, `"skipped ..."` (metric inapplicable to this dataset), or `"error: ..."` (caught exception) |

**Cluster execution** requires no code change — only a context around `.run()`:

```python
from dask.distributed import Client
import joblib

client = Client("scheduler-address:8786")
with joblib.parallel_config(backend="dask"):
    benchmark.run()
```

### User journeys with the solution

**1. The original sketch, with the real PC parameter name** (the guidance's example uses `max_cond_set`, which doesn't exist on `PC` — the actual keyword is `max_cond_vars`):

```python
from pgmpy.causal_discovery import PC
from pgmpy.datasets.anm import AdditiveNoiseModel  # not yet re-exported from pgmpy.datasets, see note below
from pgmpy.metrics import SHD

benchmark = CausalDiscoveryBenchmark(
    estimators=[PC(max_cond_vars=5, ci_test="chi_square"), PC(ci_test="pillai")],
    datasets=["hitters", AdditiveNoiseModel()],
    metrics=[SHD],
).run()
benchmark.summary()
```

**2. Sweeping sample sizes**, to see how each method's accuracy changes with more data:

```python
CausalDiscoveryBenchmark(
    estimators=[PC(ci_test="chi_square"), GES()],
    datasets=[LinearGaussianSCM(n_nodes=10, edge_prob=0.3)],
    metrics=[SHD, StructureScore],
    n_samples=[200, 500, 1000, 5000],
).run().results_
```

**3. A dataset with no ground truth, scored with an explicit override:**

```python
CausalDiscoveryBenchmark(
    estimators=[PC(), GES()],
    datasets=["some_real_dataset_without_dagitty_ground_truth"],
    metrics=[SHD],
    ground_truth={"some_real_dataset_without_dagitty_ground_truth": my_known_dag},
).run()
```

**4. An unsupervised metric with a train/test split** — fit on 70%, score fit quality on the held-out 30%:

```python
CausalDiscoveryBenchmark(
    estimators=[PC(), HillClimbSearch()],
    datasets=["hitters"],
    metrics=[StructureScore],
    train_test_split=0.7,
).run()
```

**5. Reading off a partial failure** — one combination that had no ground truth, alongside one that ran cleanly:

```python
>>> benchmark.results_[["dataset", "estimator", "status"]]
     dataset               estimator                  status
0    hitters               PC(ci_test='chi_square')    ok
1    some_dataset_no_gt    PC(ci_test='chi_square')    skipped 1 metric(s): SHD (no ground truth available)
```

**6. Same sweep, run on a Dask cluster** instead of local cores — identical benchmark definition, only the execution context changes (shown above under Cluster execution).

### Open Questions

* **Repeats across stochastic simulators.** A single draw from a simulator is noisy; averaging over several seeds per `(dataset, estimator, n_samples)` cell would make comparisons more reliable, similar to `ci_benchmarks`' existing `n_repeats`. Left out of this proposal to keep it focused on the core sweep-and-collect mechanics; a natural `n_repeats` follow-up.
* **Per-task timeouts.** Some estimator/dataset combinations may hang rather than error (e.g. an exhaustive search on a large graph). Not addressed here.
* **Publishing to `pgmpy.org/causalbench`.** `ci_benchmarks` already has a CSV output + `web/` dashboard convention; wiring `.results_`/`.summary()` into that pipeline is a reasonable follow-up once the core class is settled.
* **Module name.** `cd_benchmarks` mirrors the existing `ci_benchmarks` naming, but `causal_discovery_benchmarks` is more explicit — open to either.

### Rollout plan

* **Phase 1:** `CausalDiscoveryBenchmark` core — dataset/metric/estimator resolution, `joblib`-parallel `_run_one_task` execution, long-format `results_`, single-machine only. Tests against `PC`/`GES` on `LinearGaussianSCM`/`AdditiveNoiseModel`.
* **Phase 2:** `.summary()` wide pivot, `train_test_split`, `ground_truth` override, dict-valued metric flattening.
* **Phase 3:** Documented cluster example (Dask), and CSV/dashboard integration matching `ci_benchmarks`' existing output convention.