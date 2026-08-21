"""
CausalDiscoveryBenchmark — prototype implementation.

This file contains the reference implementation for the proposal. It is not
meant to be run directly; it shows the intended code structure and logic for
the ``pgmpy/benchmarking/_base.py`` module described in proposal.md.
"""

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.model_selection import train_test_split as sk_train_test_split

from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.datasets import list_datasets, load_dataset
from pgmpy.datasets._base import BaseDataset, BaseSimulatedDataset, Dataset
from pgmpy.metrics import BaseSupervisedMetric, BaseUnsupervisedMetric



# Dataset resolution
def _resolve_dataset(dataset, n_samples, data_seed, dataset_idx=0):
    """Resolve a single dataset specification into a (label, Dataset) pair.

    Parameters
    ----------
    dataset : str | BaseSimulatedDataset | Dataset | pd.DataFrame
        The dataset specification.
    n_samples : int | None
        Number of samples to generate/subsample.
    data_seed : int
        Seed for reproducible data generation.
    dataset_idx : int
        Position of this dataset in the user's ``datasets`` list.
        Used to disambiguate DataFrames that happen to share the
        same shape.

    Returns
    -------
    tuple[str, Dataset]
        A (label, Dataset) pair.
    """
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

    if isinstance(dataset, Dataset):
        return dataset.name, dataset

    if isinstance(dataset, pd.DataFrame):
        label = f"DataFrame_{dataset_idx}({dataset.shape[0]}x{dataset.shape[1]})"
        return label, Dataset(
            name=label,
            data=dataset,
            expert_knowledge=None,
            ground_truth=None,
            tags=None,
        )

    raise TypeError(
        f"datasets entries must be a registered name (str), BaseSimulatedDataset, "
        f"Dataset, or pd.DataFrame, got {type(dataset)}"
    )



# Metric resolution
def _resolve_metrics(metrics):
    """Validate that all metrics are initialized instances.

    Parameters
    ----------
    metrics : list
        List of metric instances.

    Returns
    -------
    list
        Validated list of metric instances.

    Raises
    ------
    TypeError
        If any metric is a class instead of an instance, or is not a
        BaseSupervisedMetric/BaseUnsupervisedMetric instance.
    """
    for metric in metrics:
        if isinstance(metric, type):
            raise TypeError(
                f"metrics must contain metric instances, not classes. "
                f"Use {metric.__name__}() instead of {metric.__name__}."
            )
    if not all(
        isinstance(metric, (BaseSupervisedMetric, BaseUnsupervisedMetric))
        for metric in metrics
    ):
        raise TypeError(
            "metrics must contain BaseSupervisedMetric or "
            "BaseUnsupervisedMetric instances"
        )
    return list(metrics)


# Seed generation
def _generate_seeds(seed_sequence, n):
    """Generate n independent integer seeds from a SeedSequence stream."""
    return [int(child.generate_state(1)[0]) for child in seed_sequence.spawn(n)]


# Single-task execution


def _run_one_task(
    dataset_label,
    dataset,
    n_samples,
    repeat_idx,
    estimator,
    metrics,
    train_test_split,
    data_seed,
    estimator_seed,
    outer_n_jobs,
):
    """Clone, fit, and evaluate one (estimator, dataset) task.

    Returns a list of result-row dicts (one per metric, or one error row
    if fitting itself fails).
    """
    base_row = {
        "dataset": dataset_label,
        "n_samples": n_samples,
        "n_samples_actual": dataset.data.shape[0],
        "repeat_idx": repeat_idx,
        "data_seed": data_seed,
        "estimator_seed": estimator_seed,
        "estimator": repr(estimator),
    }

    # Fit
    try:
        train_df, test_df = dataset.data, dataset.data
        if train_test_split is not None:
            train_df, test_df = sk_train_test_split(
                dataset.data,
                train_size=train_test_split,
                random_state=data_seed,
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
                "value": None,
                "runtime_sec": None,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        ]

    # Evaluate metrics
    rows = []
    for metric in metrics:
        name = metric.get_tag("name", repr(metric), raise_error=False)
        metric_row = {**base_row, "metric": name, "runtime_sec": runtime_sec}
        needs_truth = metric.get_tag(
            "requires_true_graph", False, raise_error=False
        )

        try:
            if needs_truth:
                value = metric.evaluate(
                    true_causal_graph=dataset.ground_truth,
                    est_causal_graph=fitted.causal_graph_,
                )
            else:
                value = metric.evaluate(
                    X=test_df, causal_graph=fitted.causal_graph_
                )

            # Handle scalar or array return values.
            if isinstance(value, (int, float, np.integer, np.floating)):
                rows.append(
                    {
                        **metric_row,
                        "value": float(value),
                        "status": "ok",
                        "error_type": None,
                        "error": None,
                    }
                )
            elif isinstance(value, (np.ndarray, list, tuple)):
                for i, v in enumerate(
                    value.flat if isinstance(value, np.ndarray) else value
                ):
                    rows.append(
                        {
                            **metric_row,
                            "metric": f"{name}.{i}",
                            "value": float(v),
                            "status": "ok",
                            "error_type": None,
                            "error": None,
                        }
                    )
            else:
                rows.append(
                    {
                        **metric_row,
                        "value": float(value),
                        "status": "ok",
                        "error_type": None,
                        "error": None,
                    }
                )

        except Exception as exc:
            rows.append(
                {
                    **metric_row,
                    "value": None,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    return rows

# ---------------------------------------------------------------------------
# CausalDiscoveryBenchmark
# ---------------------------------------------------------------------------


class CausalDiscoveryBenchmark:
    """Sweep causal discovery estimators over datasets and metrics.

    Parameters
    ----------
    estimators : list of BaseCausalDiscovery
        Estimator instances to benchmark.
    datasets : list of str | BaseSimulatedDataset | Dataset | pd.DataFrame
        Dataset specifications.
    metrics : list of BaseSupervisedMetric | BaseUnsupervisedMetric
        Metric instances (not classes).
    n_samples : int | list[int] | None
        Sample sizes to sweep.
    n_repeats : int
        Number of repeated runs per (dataset, n_samples) pair.
    ground_truth : dict | None
        Manual ground-truth overrides keyed by dataset label.
    train_test_split : float | None
        If set, fraction of data used for training.
    n_jobs : int
        Number of parallel jobs (default -1 = all CPUs).
    backend : str
        joblib backend name (default "loky"). Use "dask" or "ray"
        for cluster execution.
    show_progress : bool
        Whether to show progress output.
    seed : int
        Master seed for reproducibility.
    """

    def __init__(
        self,
        estimators,
        datasets,
        metrics,
        n_samples=None,
        n_repeats=1,
        ground_truth=None,
        train_test_split=None,
        n_jobs=-1,
        backend="loky",
        show_progress=True,
        seed=42,
    ):
        if not all(
            isinstance(est, BaseCausalDiscovery) for est in estimators
        ):
            raise TypeError(
                "estimators must contain BaseCausalDiscovery instances"
            )
        if not isinstance(n_repeats, int) or n_repeats < 1:
            raise ValueError(
                f"n_repeats must be a positive integer, got {n_repeats!r}"
            )

        self.estimators = estimators
        self.datasets = datasets
        self.metrics = _resolve_metrics(metrics)
        self.n_samples = (
            n_samples if isinstance(n_samples, list) else [n_samples]
        )
        self.n_repeats = n_repeats
        self.ground_truth = ground_truth or {}
        self.train_test_split = train_test_split
        self.n_jobs = n_jobs
        self.backend = backend
        self.show_progress = show_progress
        self.seed = seed

    def run(self, n_jobs=None, backend=None, show_progress=None):
        """Execute the benchmark sweep.

        Parameters
        ----------
        n_jobs : int, optional
            Override the instance's n_jobs.
        backend : str, optional
            Override the instance's joblib backend.
        show_progress : bool, optional
            Override the instance's show_progress.

        Returns
        -------
        self
            The benchmark instance with ``results_`` populated.
        """
        n_jobs = self.n_jobs if n_jobs is None else n_jobs
        backend = self.backend if backend is None else backend
        show_progress = (
            self.show_progress if show_progress is None else show_progress
        )

        # Build the grid of (dataset_spec, n_samples, repeat_idx) triples.
        data_specs = []
        for ds_idx, dataset_spec in enumerate(self.datasets):
            for n in self.n_samples:
                for repeat_idx in range(self.n_repeats):
                    data_specs.append((dataset_spec, n, repeat_idx, ds_idx))

        # Two independent seed streams from the master seed.
        ss = np.random.SeedSequence(self.seed)
        data_ss, est_ss = ss.spawn(2)
        data_seeds = _generate_seeds(data_ss, len(data_specs))
        est_seeds = _generate_seeds(
            est_ss, len(data_specs) * len(self.estimators)
        )

        # Resolve datasets and build task list.
        tasks = []
        est_seed_idx = 0
        for spec_idx, (dataset_spec, n, repeat_idx, ds_idx) in enumerate(data_specs):
            data_seed = data_seeds[spec_idx]
            label, dataset = _resolve_dataset(dataset_spec, n, data_seed, ds_idx)
            if dataset.ground_truth is None and label in self.ground_truth:
                dataset.ground_truth = self.ground_truth[label]
            for estimator in self.estimators:
                tasks.append(
                    (
                        label,
                        dataset,
                        n,
                        repeat_idx,
                        estimator,
                        data_seed,
                        est_seeds[est_seed_idx],
                    )
                )
                est_seed_idx += 1

        with parallel_config(backend=backend):
            task_rows = Parallel(
                n_jobs=n_jobs, verbose=10 if show_progress else 0
            )(
                delayed(_run_one_task)(
                    label,
                    dataset,
                    n,
                    repeat_idx,
                    estimator,
                    self.metrics,
                    self.train_test_split,
                    data_seed,
                    est_seed,
                    n_jobs,
                )
                for (
                    label,
                    dataset,
                    n,
                    repeat_idx,
                    estimator,
                    data_seed,
                    est_seed,
                ) in tasks
            )

        self.results_ = pd.DataFrame(
            [row for rows in task_rows for row in rows]
        )
        return self

    def summary(self):
        """Wide estimator-by-dataset view, averaged across repeats."""
        successful = self.results_.query("status == 'ok'").copy()
        return successful.pivot_table(
            index=["estimator", "dataset", "n_samples"],
            columns="metric",
            values="value",
            aggfunc="mean",
        ).reset_index()

    def to_csv(self, path, **kwargs):
        """Write the raw long-format benchmark results to CSV."""
        self.results_.to_csv(path, index=False, **kwargs)