from __future__ import annotations
import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Sequence
import h5py
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from scipy import sparse
from torch.utils.data import DataLoader, Dataset
from pascient.data.batch_mappers import BatchMapper
from pascient.data.data_samplers import BaseSampler
from baseline.age.adapters.pascient_loss import LOSS_CONTRACT, loss_contract

OFFICIAL_COMMIT = "e1c645fa4c43546fd4f03c948c170429034556d0"

AGE_LABEL = "age"

DONOR_LABEL = "donor_id"

def official_worker_init_fn(worker_id: int) -> None:
    """Released PaSCient worker hook, without importing its TileDB data module."""

    print(f"Worker {worker_id} is initialized")


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


@dataclass(frozen=True)
class DonorSelection:
    """One official dynamic sample of cells from one donor."""

    donor_id: str
    group_key: str
    row_indices: tuple[int, ...]

class DonorCSRDataset(Dataset):
    """Select cells exactly as released ``scDataset.__getitem__`` does."""

    def __init__(
        self,
        donor_ids: Sequence[str],
        donor_to_group: dict[str, str],
        donor_n_cells: dict[str, int],
        sample_size: int,
    ) -> None:
        self.donor_ids = list(donor_ids)
        self.donor_to_group = dict(donor_to_group)
        self.donor_n_cells = dict(donor_n_cells)
        self.sample_size = int(sample_size)
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")

    def __len__(self) -> int:
        return len(self.donor_ids)

    def __getitem__(self, index: int) -> DonorSelection:
        donor_id = self.donor_ids[index]
        n_cells = self.donor_n_cells[donor_id]
        if n_cells <= 0:
            raise ValueError(f"donor {donor_id} has no cells")
        # As in released PaSCient: sampling without replacement; donors below
        # sample_size contribute every cell in random order.
        selected_n = min(n_cells, self.sample_size)
        rows = random.sample(range(n_cells), selected_n)
        return DonorSelection(
            donor_id=donor_id,
            group_key=self.donor_to_group[donor_id],
            row_indices=tuple(rows),
        )

class _StoreMetadata:
    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path.resolve()
        if not self.store_path.is_file():
            raise FileNotFoundError(self.store_path)
        with h5py.File(self.store_path, "r") as handle:
            store_format = handle.attrs.get("format")
            if isinstance(store_format, bytes):
                store_format = store_format.decode("utf-8")
            if store_format != "pascient_donor_csr_v1":
                raise ValueError(f"unsupported PaSCient store format: {self.store_path}")
            if bool(handle.attrs.get("labels_present", True)):
                raise ValueError("PaSCient expression stores must contain no age labels")
            self.donor_ids = decode_strings(handle["donor_ids"][:])
            self.genes = decode_strings(handle["genes"][:])
            if len(self.donor_ids) != len(set(self.donor_ids)):
                raise ValueError("duplicate donor IDs in PaSCient store")
            if len(self.genes) != len(set(self.genes)):
                raise ValueError("duplicate genes in PaSCient store")
            self.donor_to_group: dict[str, str] = {}
            self.donor_n_cells: dict[str, int] = {}
            donor_group = handle["donors"]
            for index, donor_id in enumerate(self.donor_ids):
                key = str(index)
                if key not in donor_group:
                    raise ValueError(f"missing donor group {key} for {donor_id}")
                group = donor_group[key]
                stored_donor = group.attrs["donor_id"]
                if isinstance(stored_donor, bytes):
                    stored_donor = stored_donor.decode("utf-8")
                if str(stored_donor) != donor_id:
                    raise ValueError(f"donor group/order mismatch at {index}")
                n_cells = int(group.attrs["n_cells"])
                indptr = group["indptr"]
                if len(indptr) != n_cells + 1:
                    raise ValueError(f"invalid CSR indptr for donor {donor_id}")
                self.donor_to_group[donor_id] = key
                self.donor_n_cells[donor_id] = n_cells

def _read_dense_batch(store_path: Path, selections: Sequence[DonorSelection]) -> torch.Tensor:
    with h5py.File(store_path, "r") as handle:
        donor_group = handle["donors"]
        n_genes = len(handle["genes"])
        total_rows = sum(len(selection.row_indices) for selection in selections)
        dense_batch = np.empty((total_rows, n_genes), dtype=np.float32)
        cursor = 0
        for selection in selections:
            group = donor_group[selection.group_key]
            n_cells = int(group.attrs["n_cells"])
            matrix = sparse.csr_matrix(
                (
                    group["data"][:].astype(np.float32, copy=False),
                    group["indices"][:].astype(np.int32, copy=False),
                    group["indptr"][:].astype(np.int32, copy=False),
                ),
                shape=(n_cells, n_genes),
            )
            rows = np.asarray(selection.row_indices, dtype=np.int64)
            dense = matrix[rows, :].toarray().astype(np.float32, copy=False)
            dense_batch[cursor : cursor + len(rows)] = dense
            cursor += len(rows)
    return torch.from_numpy(dense_batch)

def _make_mapper(
    donor_ids: Sequence[str],
    age_mapping: dict[float, float],
    sample_size: int,
    target_sum: float,
) -> BatchMapper:
    mapper = BatchMapper(
        pad_size=sample_size,
        sample_labels=[[DONOR_LABEL, "categorical"], [AGE_LABEL, "continuous"]],
        cell_labels=[],
        return_ontology=False,
        return_index=False,
    )
    mapper.set_dict(
        {
            DONOR_LABEL: {donor_id: index for index, donor_id in enumerate(donor_ids)},
            AGE_LABEL: dict(age_mapping),
        }
    )
    mapper.create_ancestry_matrix(None)
    mapper.set_augmentations({})
    mapper.set_normalization(lognorm=True, target_sum=target_sum)
    return mapper

class _CSRDataModuleBase(LightningDataModule):
    def __init__(
        self,
        store_path: str,
        sample_size: int = 1500,
        batch_size: int = 32,
        num_workers: int = 16,
        target_sum: float = 10000.0,
        persistent_workers: bool = True,
        multiprocessing_context: str | None = None,
        prefetch_factor: int = 1,
    ) -> None:
        super().__init__()
        self.store = _StoreMetadata(Path(store_path))
        self.sample_size = int(sample_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.target_sum = float(target_sum)
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        self.multiprocessing_context = multiprocessing_context
        self.prefetch_factor = int(prefetch_factor)
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")

    @property
    def num_genes(self) -> int:
        return len(self.store.genes)

    def _dataset(self, donor_ids: Sequence[str]) -> DonorCSRDataset:
        return DonorCSRDataset(
            donor_ids=donor_ids,
            donor_to_group=self.store.donor_to_group,
            donor_n_cells=self.store.donor_n_cells,
            sample_size=self.sample_size,
        )

    def _sampler(self, donor_ids: Sequence[str], shuffle: bool) -> BaseSampler:
        frame = pd.DataFrame({"study::::sample": list(donor_ids)})
        int2sample = {index: donor_id for index, donor_id in enumerate(donor_ids)}
        return BaseSampler(
            data_df=frame,
            int2sample=int2sample,
            bsz=self.batch_size,
            shuffle=shuffle,
        )

    def _loader(
        self,
        dataset: DonorCSRDataset,
        sampler: BaseSampler,
        collate_fn,
    ) -> DataLoader:
        loader_kwargs = {
            "dataset": dataset,
            "num_workers": self.num_workers,
            "collate_fn": collate_fn,
            "pin_memory": True,
            "batch_sampler": sampler,
            "persistent_workers": self.persistent_workers,
            "multiprocessing_context": self.multiprocessing_context,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["worker_init_fn"] = official_worker_init_fn
        return DataLoader(
            **loader_kwargs,
        )

    def _collate(
        self,
        selections: Sequence[DonorSelection],
        mapper: BatchMapper,
        age_by_donor: dict[str, float],
    ):
        counts = _read_dense_batch(self.store.store_path, selections)
        frames: list[pd.DataFrame] = []
        for sample_index, selection in enumerate(selections):
            n_cells = len(selection.row_indices)
            frames.append(
                pd.DataFrame(
                    {
                        DONOR_LABEL: [selection.donor_id] * n_cells,
                        AGE_LABEL: [float(age_by_donor[selection.donor_id])] * n_cells,
                        "sample_id_batch": [sample_index] * n_cells,
                        "view_id": [0] * n_cells,
                    }
                )
            )
        metadata = pd.concat(frames, ignore_index=True)
        return mapper((counts, metadata))

class CIMAPaSCientDataModule(_CSRDataModuleBase):
    """Five-fold CIMA adapter with a train-fold-only continuous scaler."""

    def __init__(
        self,
        store_path: str,
        split_path: str,
        fold: int,
        repeat: int = 0,
        contract_path: str | None = None,
        expected_donors: int = 417,
        sample_size: int = 1500,
        batch_size: int = 32,
        num_workers: int = 16,
        target_sum: float = 10000.0,
        persistent_workers: bool = True,
        multiprocessing_context: str | None = None,
        prefetch_factor: int = 1,
        age_loss_contract: str | None = None,
    ) -> None:
        if age_loss_contract not in (None, LOSS_CONTRACT):
            raise ValueError("Unknown PaSCient age-loss contract")
        super().__init__(
            store_path=store_path,
            sample_size=sample_size,
            batch_size=batch_size,
            num_workers=num_workers,
            target_sum=target_sum,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
        )
        self.split_path = Path(split_path).resolve()
        split = pd.read_csv(self.split_path, sep="\t")
        required = {"donor_id", "age", "fold", "repeat"}
        if not required.issubset(split.columns):
            raise ValueError(f"split table missing columns {sorted(required - set(split.columns))}")
        split = split.loc[split["repeat"].eq(int(repeat)), list(required)].copy()
        split["donor_id"] = split["donor_id"].astype(str)
        if len(split) != expected_donors or split["donor_id"].duplicated().any():
            raise ValueError(
                f"repeat {repeat} must contain {expected_donors} unique CIMA donors"
            )
        if set(split["donor_id"]) != set(self.store.donor_ids):
            missing_store = sorted(set(split["donor_id"]) - set(self.store.donor_ids))
            unexpected_store = sorted(set(self.store.donor_ids) - set(split["donor_id"]))
            raise ValueError(
                "CIMA store/split donor mismatch: "
                f"missing_store={missing_store[:5]}, unexpected_store={unexpected_store[:5]}"
            )
        fold_domain = set(split["fold"].astype(int))
        if fold_domain != set(range(5)):
            raise ValueError(f"CIMA split fold domain must be 0-4; found {sorted(fold_domain)}")
        if int(fold) not in fold_domain:
            raise ValueError(f"fold {fold} is absent from repeat {repeat}")

        train = split.loc[~split["fold"].eq(int(fold))].copy()
        validation = split.loc[split["fold"].eq(int(fold))].copy()
        # The released module constructs int2sample from pandas groupby keys,
        # which are sorted by default.
        self.train_donor_ids = sorted(train["donor_id"].tolist())
        self.validation_donor_ids = sorted(validation["donor_id"].tolist())
        if set(self.train_donor_ids) & set(self.validation_donor_ids):
            raise ValueError("CIMA train/validation donor leakage")

        self.age_by_donor = dict(
            zip(split["donor_id"], split["age"].astype(float), strict=True)
        )
        train_ages = train["age"].astype(float).to_numpy()
        train_weights = np.asarray(
            [self.store.donor_n_cells[donor] for donor in train["donor_id"]],
            dtype=np.int64,
        )
        n_training_cells = int(train_weights.sum())
        self.age_mean = float(np.average(train_ages, weights=train_weights))
        # PaSCient's released BatchMapper calls pandas Series.std(), whose
        # default is sample SD (ddof=1), over its per-cell metadata frame.
        # Reproduce that cell weighting, but fit on the training fold only.
        weighted_ss = float(np.sum(train_weights * (train_ages - self.age_mean) ** 2))
        self.age_std = float(np.sqrt(weighted_ss / (n_training_cells - 1)))
        if not np.isfinite(self.age_std) or self.age_std <= 0:
            raise ValueError("training-fold age standard deviation is not positive")
        age_mapping = {
            float(age): (float(age) - self.age_mean) / self.age_std
            for age in split["age"].astype(float).unique()
        }
        self.train_mapper = _make_mapper(
            self.store.donor_ids, age_mapping, self.sample_size, self.target_sum
        )
        self.validation_mapper = _make_mapper(
            self.store.donor_ids, age_mapping, self.sample_size, self.target_sum
        )
        self.train_dataset = self._dataset(self.train_donor_ids)
        self.validation_dataset = self._dataset(self.validation_donor_ids)
        self.train_sampler = self._sampler(self.train_donor_ids, shuffle=True)
        self.validation_sampler = self._sampler(self.validation_donor_ids, shuffle=False)

        if contract_path is not None:
            contract = {
                "schema_version": "official_pascient_cima_fold_v1",
                "official_repo_commit": OFFICIAL_COMMIT,
                "official_entrypoint": "pascient.scripts.train.train",
                "store_path": str(self.store.store_path),
                "store_manifest": str(self.store.store_path.with_suffix(".manifest.json")),
                "split_path": str(self.split_path),
                "repeat": int(repeat),
                "fold": int(fold),
                "train_donors": self.train_donor_ids,
                "validation_donors": self.validation_donor_ids,
                "donor_leakage": 0,
                "n_genes": self.num_genes,
                "sample_size": self.sample_size,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "prefetch_factor": self.prefetch_factor,
                "normalization": "official BatchMapper log1p(CP10K)",
                "age_scaler": {
                    "fit_scope": "CIMA training fold only",
                    "mean": self.age_mean,
                    "std": self.age_std,
                    "ddof": 1,
                    "weighting": "one repeated age value per training cell, as released BatchMapper.init_mapping",
                    "n_training_cells": n_training_cells,
                },
                "released_mse_shape_behavior": loss_contract() if age_loss_contract else {
                    "preserved": True,
                    "true_before_loss": "[N,1]",
                    "true_inside_MSELossViews": "[N,1,1]",
                    "prediction_per_view": "[N,1]",
                    "broadcast_result": "[N,N,1]",
                },
                "labels_present_in_expression_store": False,
            }
            destination = Path(contract_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    def train_dataloader(self) -> DataLoader:
        return self._loader(
            self.train_dataset,
            self.train_sampler,
            partial(
                self._collate,
                mapper=self.train_mapper,
                age_by_donor=self.age_by_donor,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader(
            self.validation_dataset,
            self.validation_sampler,
            partial(
                self._collate,
                mapper=self.validation_mapper,
                age_by_donor=self.age_by_donor,
            ),
        )

class CIMAFullPaSCientDataModule(_CSRDataModuleBase):
    """All-CIMA fit for the external model.

    PaSCient's trainer requires a validation loader; the training cohort is
    used as a logging-only validation loader. External inference uses the
    epoch-four checkpoint.
    """

    def __init__(
        self,
        store_path: str,
        split_path: str,
        repeat: int = 0,
        contract_path: str | None = None,
        expected_donors: int = 417,
        sample_size: int = 1500,
        batch_size: int = 32,
        num_workers: int = 16,
        target_sum: float = 10000.0,
        persistent_workers: bool = True,
        multiprocessing_context: str | None = None,
        prefetch_factor: int = 1,
        age_loss_contract: str | None = None,
    ) -> None:
        if age_loss_contract not in (None, LOSS_CONTRACT):
            raise ValueError("Unknown PaSCient age-loss contract")
        super().__init__(
            store_path=store_path,
            sample_size=sample_size,
            batch_size=batch_size,
            num_workers=num_workers,
            target_sum=target_sum,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
        )
        self.split_path = Path(split_path).resolve()
        split = pd.read_csv(self.split_path, sep="\t")
        required = {"donor_id", "age", "fold", "repeat"}
        if not required.issubset(split.columns):
            raise ValueError(f"split table missing columns {sorted(required - set(split.columns))}")
        split = split.loc[split["repeat"].eq(int(repeat)), list(required)].copy()
        split["donor_id"] = split["donor_id"].astype(str)
        if len(split) != expected_donors or split["donor_id"].duplicated().any():
            raise ValueError(
                f"repeat {repeat} must contain {expected_donors} unique CIMA donors"
            )
        if set(split["donor_id"]) != set(self.store.donor_ids):
            raise ValueError("full-CIMA store/split donor mismatch")
        if set(split["fold"].astype(int)) != set(range(5)):
            raise ValueError("full-CIMA split must retain the frozen fold domain 0-4")

        self.train_donor_ids = sorted(split["donor_id"].tolist())
        self.validation_donor_ids = list(self.train_donor_ids)
        self.age_by_donor = dict(
            zip(split["donor_id"], split["age"].astype(float), strict=True)
        )
        ages = split["age"].astype(float).to_numpy()
        weights = np.asarray(
            [self.store.donor_n_cells[donor] for donor in split["donor_id"]],
            dtype=np.int64,
        )
        n_training_cells = int(weights.sum())
        self.age_mean = float(np.average(ages, weights=weights))
        weighted_ss = float(np.sum(weights * (ages - self.age_mean) ** 2))
        self.age_std = float(np.sqrt(weighted_ss / (n_training_cells - 1)))
        if not np.isfinite(self.age_std) or self.age_std <= 0:
            raise ValueError("full-CIMA age standard deviation is not positive")
        age_mapping = {
            float(age): (float(age) - self.age_mean) / self.age_std
            for age in split["age"].astype(float).unique()
        }
        self.train_mapper = _make_mapper(
            self.store.donor_ids, age_mapping, self.sample_size, self.target_sum
        )
        self.validation_mapper = _make_mapper(
            self.store.donor_ids, age_mapping, self.sample_size, self.target_sum
        )
        self.train_dataset = self._dataset(self.train_donor_ids)
        self.validation_dataset = self._dataset(self.validation_donor_ids)
        self.train_sampler = self._sampler(self.train_donor_ids, shuffle=True)
        self.validation_sampler = self._sampler(self.validation_donor_ids, shuffle=False)

        if contract_path is not None:
            contract = {
                "schema_version": "official_pascient_cima_full_v1",
                "official_repo_commit": OFFICIAL_COMMIT,
                "official_entrypoint": "pascient.scripts.train.train",
                "store_path": str(self.store.store_path),
                "store_manifest": str(self.store.store_path.with_suffix(".manifest.json")),
                "split_path": str(self.split_path),
                "repeat": int(repeat),
                "training_donors": self.train_donor_ids,
                "external_model_fit_scope": "all 417 CIMA donors",
                "logging_validation_donors": self.validation_donor_ids,
                "logging_validation_only": True,
                "checkpoint_selected_by_validation": False,
                "n_genes": self.num_genes,
                "sample_size": self.sample_size,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "prefetch_factor": self.prefetch_factor,
                "normalization": "official BatchMapper log1p(CP10K)",
                "age_scaler": {
                    "fit_scope": "all 417 CIMA donors only",
                    "mean": self.age_mean,
                    "std": self.age_std,
                    "ddof": 1,
                    "weighting": "one repeated age value per CIMA cell",
                    "n_training_cells": n_training_cells,
                },
                "released_mse_shape_behavior": loss_contract() if age_loss_contract else {
                    "preserved": True,
                    "true_before_loss": "[N,1]",
                    "true_inside_MSELossViews": "[N,1,1]",
                    "prediction_per_view": "[N,1]",
                    "broadcast_result": "[N,N,1]",
                },
                "labels_present_in_expression_store": False,
            }
            destination = Path(contract_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    def train_dataloader(self) -> DataLoader:
        return self._loader(
            self.train_dataset,
            self.train_sampler,
            partial(
                self._collate,
                mapper=self.train_mapper,
                age_by_donor=self.age_by_donor,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader(
            self.validation_dataset,
            self.validation_sampler,
            partial(
                self._collate,
                mapper=self.validation_mapper,
                age_by_donor=self.age_by_donor,
            ),
        )

class PaSCientPredictionDataModule(_CSRDataModuleBase):
    """Target-cohort adapter; the age field is an unused placeholder."""

    def __init__(
        self,
        store_path: str,
        donor_ids: Sequence[str] | None = None,
        sample_size: int = 1500,
        batch_size: int = 32,
        num_workers: int = 16,
        target_sum: float = 10000.0,
        persistent_workers: bool = True,
        multiprocessing_context: str | None = None,
        prefetch_factor: int = 1,
    ) -> None:
        super().__init__(
            store_path=store_path,
            sample_size=sample_size,
            batch_size=batch_size,
            num_workers=num_workers,
            target_sum=target_sum,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
        )
        selected = self.store.donor_ids if donor_ids is None else list(map(str, donor_ids))
        if len(selected) != len(set(selected)):
            raise ValueError("prediction donor list contains duplicates")
        missing = sorted(set(selected) - set(self.store.donor_ids))
        if missing:
            raise ValueError(f"prediction donors missing from expression store: {missing[:5]}")
        self.donor_ids = selected
        self.int2donor = {index: donor for index, donor in enumerate(self.store.donor_ids)}
        self.dummy_age_by_donor = {donor: 0.0 for donor in self.store.donor_ids}
        self.mapper = _make_mapper(
            self.store.donor_ids, {0.0: 0.0}, self.sample_size, self.target_sum
        )
        self.dataset = self._dataset(self.donor_ids)
        self.sampler = self._sampler(self.donor_ids, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(
            self.dataset,
            self.sampler,
            partial(
                self._collate,
                mapper=self.mapper,
                age_by_donor=self.dummy_age_by_donor,
            ),
        )


