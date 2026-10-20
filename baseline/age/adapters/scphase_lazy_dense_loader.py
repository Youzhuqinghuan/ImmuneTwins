from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset

STORAGE_CONTRACT = "scphase_per_donor_lazy_dense_float32_v1"

MASTER_SCHEMA = "scphase_lazy_dense_master_v1"


def _manifest_path(master_path: Path) -> Path:
    return master_path.with_suffix(".manifest.json")

def validate_master(
    master_path: Path,
    *,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Load a descriptor-only master and validate every donor shard."""

    master_path = master_path.resolve()
    manifest_path = _manifest_path(master_path)
    if not master_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"scPhase lazy master or manifest is absent: {master_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MASTER_SCHEMA:
        raise ValueError("scPhase master does not use the lazy-dense schema")
    if manifest.get("storage_contract") != STORAGE_CONTRACT:
        raise ValueError("scPhase master does not use the frozen lazy-dense storage contract")
    if expected_role is not None and manifest.get("preprocessing_role") != expected_role:
        raise ValueError(
            f"scPhase master role mismatch: expected {expected_role}, "
            f"found {manifest.get('preprocessing_role')}"
        )
    if expected_role == "external_apply" and manifest.get("labels_present") is not False:
        raise ValueError("external scPhase manifest must declare labels_present=false")

    with master_path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("schema") != MASTER_SCHEMA:
        raise ValueError("scPhase master pickle has the wrong schema")
    if payload.get("storage_contract") != STORAGE_CONTRACT:
        raise ValueError("scPhase master pickle has the wrong storage contract")

    refs = payload.get("data_list_dense")
    labels = np.asarray(payload.get("labels"))
    groups = np.asarray(payload.get("groups"))
    sample_ids = [str(value) for value in payload.get("sample_ids", [])]
    if not isinstance(refs, list) or not refs:
        raise ValueError("scPhase lazy master has no donor shard descriptors")
    if len(refs) != len(labels) or len(refs) != len(groups) or len(refs) != len(sample_ids):
        raise ValueError("scPhase master arrays have inconsistent donor counts")
    if len(set(sample_ids)) != len(sample_ids) or any(not donor for donor in sample_ids):
        raise ValueError("scPhase master donor IDs are blank or duplicated")
    metadata = payload.get("adapter_metadata", {})
    if expected_role == "external_apply":
        if not np.isnan(labels.astype(np.float64)).all():
            raise ValueError("external scPhase master labels must all be NaN sentinels")
        if metadata.get("target_labels_accessed") is not False:
            raise ValueError("external scPhase master does not prove label-free construction")
        if metadata.get("target_statistics_fitted") is not False:
            raise ValueError("external scPhase master fitted target statistics")

    expected_genes = int(manifest["n_genes"])
    observed_cells = 0
    observed_bytes = 0
    for index, (ref, donor) in enumerate(zip(refs, sample_ids)):
        if not isinstance(ref, dict):
            raise ValueError(f"scPhase shard descriptor {index} is not a mapping")
        required = {"path",  "n_cells", "n_genes", "dtype", "donor_id", "bytes"}
        if not required.issubset(ref):
            raise ValueError(
                f"scPhase shard descriptor {index} fields differ from the frozen contract"
            )
        if str(ref["donor_id"]) != donor:
            raise ValueError(f"scPhase shard descriptor donor mismatch at index {index}")
        if int(ref["n_genes"]) != expected_genes or ref["dtype"] != "float32":
            raise ValueError(f"scPhase shard {donor} has an invalid feature/dtype contract")
        path = Path(ref["path"])
        if not path.is_absolute() or not path.is_file():
            raise FileNotFoundError(f"scPhase shard is absent or not absolute: {path}")
        if path.stat().st_size != int(ref["bytes"]):
            raise ValueError(f"scPhase shard byte size changed: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (int(ref["n_cells"]), expected_genes)
        if array.shape != expected_shape or array.dtype != np.dtype("float32"):
            raise ValueError(
                f"scPhase shard {donor} shape/dtype mismatch: "
                f"found {array.shape}/{array.dtype}, expected {expected_shape}/float32"
            )
        del array
        observed_cells += int(ref["n_cells"])
        observed_bytes += int(ref["bytes"])

    if observed_cells != int(manifest["n_cells"]):
        raise ValueError("scPhase donor shards do not cover the declared number of cells")
    if observed_bytes != int(manifest["total_shard_bytes"]):
        raise ValueError("scPhase donor shard bytes do not match the manifest")
    return payload

def load_data(config: dict[str, Any]):
    """Drop-in replacement for ``scphase.data_loader.load_data``."""

    contract = config.get("adapter_contract")
    if not isinstance(contract, dict) or contract.get("storage_contract") != STORAGE_CONTRACT:
        raise ValueError("scPhase config lacks the lazy-dense loader contract")
    master_path = Path(config["path_params"].get("_master_pickle_file", ""))
    payload = validate_master(master_path)
    metadata = payload.get("adapter_metadata", {})
    if metadata.get("storage_contract") != STORAGE_CONTRACT:
        raise ValueError("scPhase master metadata does not authorize the lazy-dense loader")
    return (
        payload["data_list_dense"],
        np.asarray(payload["labels"]),
        np.asarray(payload["groups"]).astype(int),
        [str(value) for value in payload["sample_ids"]],
    )

class AttnMoE_Dataset(Dataset):
    """Released Dataset interface with no resident or persistent-worker cache."""

    def __init__(self, DataList, DataLabel, DataBatch, SampleIDs):
        super().__init__()
        self.DataList = DataList
        self.DataLabel = DataLabel
        self.DataBatch = DataBatch
        self.SampleIDs = SampleIDs
        if not (
            len(self.DataList)
            == len(self.DataLabel)
            == len(self.DataBatch)
            == len(self.SampleIDs)
        ):
            raise ValueError("scPhase lazy Dataset arrays have inconsistent lengths")

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        ref = self.DataList[index]
        # The released fold code builds SampleIDs from the pre-fold array. SampleID
        # is not used by the model or loss, so the released behaviour is kept.
        mmap = np.load(Path(ref["path"]), mmap_mode="r", allow_pickle=False)
        expected_shape = (int(ref["n_cells"]), int(ref["n_genes"]))
        if mmap.shape != expected_shape or mmap.dtype != np.dtype("float32"):
            raise ValueError(f"scPhase shard changed after runtime preflight: {ref['path']}")

        # A writable contiguous copy avoids a read-only NumPy/Torch alias.
        dense = np.array(mmap, dtype=np.float32, order="C", copy=True)
        del mmap
        if (
            not np.isfinite(dense).all()
            or float(dense.max(initial=-np.inf)) > 10.0
            or float(dense.min(initial=np.inf)) < -10.0
        ):
            raise ValueError(f"invalid scPhase scaled values in donor {ref['donor_id']}")
        data = torch.from_numpy(dense)
        return data, self.DataLabel[index], self.DataBatch[index], self.SampleIDs[index]

    def __len__(self):
        return len(self.DataLabel)

def sparse_collate_fn(batch):
    """Numerically equivalent batch-size-one dense collate without another copy."""

    if len(batch) != 1:
        raise ValueError("the frozen scPhase config requires batch_size=1")
    data, label, batch_label, sample_id = batch[0]
    if data.is_sparse or data.dtype != torch.float32 or data.ndim != 2:
        raise ValueError("scPhase lazy loader must emit a dense float32 donor matrix")
    return (
        data.unsqueeze(0),
        torch.as_tensor([label]),
        torch.as_tensor([batch_label]),
        [sample_id],
    )
