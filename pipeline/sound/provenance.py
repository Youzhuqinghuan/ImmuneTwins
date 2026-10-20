from __future__ import annotations
from paths import expand_paths
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
import pandas as pd

V2_PROTOCOL_SCHEMA = "sound_vaccine_immunetwin_protocol_2"

V2_TRAIN_SCHEMA = "sound_vaccine_immunetwin_runtime_2"

V2_COMPLETE_SCHEMA = "sound_vaccine_immunetwin_complete_2"


V2_ARMS = ("current", "w", "gene", "full")

V2_PROTOCOL_SCOPES = {
    "full": V2_ARMS,
    "current_only": ("current",),
}

V2_TASKS = (
    "t1_y1_d0_to_y1_d7",
    "t2_d7_to_d90",
    "t3_y2_d0_to_y2_d90",
)

V2_METHODS = {
    "current": "ImmuneTwin-current",
    "w": "ImmuneTwin-w",
    "gene": "ImmuneTwin-gene",
    "full": "ImmuneTwin-full",
}

V2_REQUIRED_ARTIFACTS = frozenset(
    {
        "checkpoint",
        "best_validation_checkpoint",
        "predictions",
        "observed_targets",
    }
)

V2_LINEAGE_INPUT_NAMES = (
    "lineage_gene_summary",
    "lineage_gene_summary_audit",
)


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)

def expected_v2_job_ids(scope: str = "full") -> tuple[str, ...]:
    """Return the job grid for one protocol scope."""

    try:
        arms = V2_PROTOCOL_SCOPES[scope]
    except KeyError as error:
        raise ValueError(f"unknown v2 protocol scope: {scope}") from error
    jobs: list[str] = []
    for arm in arms:
        tasks = V2_TASKS if arm == "current" else V2_TASKS[1:]
        for task in tasks:
            for fold in range(5):
                jobs.append(f"{arm}__{task}__outer{fold}_seed42")
    expected_count = 45 if scope == "full" else 15
    if len(jobs) != expected_count or len(set(jobs)) != expected_count:  # pragma: no cover
        raise AssertionError(
            f"v2 {scope} vaccine grid must contain exactly {expected_count} unique jobs"
        )
    return tuple(jobs)

def current_dependency_identifier(
    arm: str, task: str, outer_fold: int, *, seed: int = 42
) -> str | None:
    """Return the matched current job required by one history job."""

    if arm == "current":
        return None
    if arm not in V2_ARMS or task not in V2_TASKS[1:] or outer_fold not in range(5):
        raise ValueError("invalid history job identity for current dependency")
    if seed != 42:
        raise ValueError("v2 current dependencies require seed 42")
    return f"current__{task}__outer{outer_fold}_seed42"

def staged_execution_contract(scope: str = "full") -> dict[str, object]:
    """Return the two-stage execution graph defined by the protocol."""

    expected = expected_v2_job_ids(scope)
    current = [identifier for identifier in expected if identifier.startswith("current__")]
    history = [identifier for identifier in expected if not identifier.startswith("current__")]
    return {
        "barrier": "all_current_jobs_complete_before_any_history_job",
        "current_jobs": current,
        "history_jobs": history,
        "history_dependency": "same_task_same_outer_fold_current",
        "selection_checkpoint": "best_validation.pt",
        "final_checkpoint": "checkpoint.pt",
        "clinical_history": "off",
    }


def _inside(path: Path, root: Path, *, field: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise RuntimeError(f"{field} escapes the v2 job directory: {resolved}") from error
    return resolved

def _v2_root(path: str | Path, *, field: str, must_exist: bool = True) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve(strict=must_exist)
    if "vaccine_immunetwin_v2" not in resolved.parts:
        raise ValueError(f"{field} must be isolated below vaccine_immunetwin_v2")
    if "vaccine_immunetwin_v1" in resolved.parts:
        raise ValueError(f"{field} cannot point inside the diagnostic v1 root")
    return resolved

def file_records(
    values: Mapping[str, str | Path], *, verify_files: bool = True
) -> list[dict[str, object]]:
    """Build sorted named file records for a protocol-manifest section."""

    records: list[dict[str, object]] = []
    for name, raw_path in sorted(values.items()):
        path = Path(raw_path).resolve(strict=verify_files)
        records.append(
            {
                "name": str(name),
                "path": str(path),
            }
        )
    return records


def build_v2_protocol_payload(
    *,
    formal_output_root: str | Path,
    scored_output_root: str | Path,
    configuration: Mapping[str, str | Path],
    inputs: Mapping[str, str | Path],
    evaluation_contract: str | Path,
    sealed_baseline_audit: str | Path,
    job_coverage: Mapping[str, tuple[int, int]],
    protocol_scope: str = "full",
    paper_methods: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a manifest for the job grid.

    ``job_coverage`` lists the identifiers in the requested scope, because
    held-out unit counts are fold-specific.
    """

    expected_ids = expected_v2_job_ids(protocol_scope)
    expected = set(expected_ids)
    allowed_arms = set(V2_PROTOCOL_SCOPES[protocol_scope])
    resolved_methods = dict(V2_METHODS)
    if paper_methods is not None:
        if set(paper_methods) != allowed_arms:
            raise ValueError(
                "paper-method overrides must contain exactly the protocol arms"
            )
        overrides = {str(key): str(value).strip() for key, value in paper_methods.items()}
        if any(not value for value in overrides.values()) or len(
            set(overrides.values())
        ) != len(overrides):
            raise ValueError("paper-method overrides must be nonempty and unique")
        resolved_methods.update(overrides)
    if set(job_coverage) != expected:
        missing = sorted(expected - set(job_coverage))
        extra = sorted(set(job_coverage) - expected)
        raise ValueError(
            f"v2 {protocol_scope} job coverage must contain exactly {len(expected)} jobs; "
            f"missing={missing}, extra={extra}"
        )
    formal = _v2_root(formal_output_root, field="formal_output_root", must_exist=False)
    scored = _v2_root(scored_output_root, field="scored_output_root", must_exist=False)
    configuration_records = file_records(configuration)
    input_records = file_records(inputs)
    jobs: list[dict[str, object]] = []
    for identifier in expected_ids:
        arm, tail = identifier.split("__", 1)
        task, fold_seed = tail.rsplit("__outer", 1)
        fold = int(fold_seed.split("_seed", 1)[0])
        units, donors = job_coverage[identifier]
        jobs.append(
            {
                "identifier": identifier,
                "arm": arm,
                "task": task,
                "outer_fold": fold,
                "seed": 42,
                "paper_method": resolved_methods[arm],
                "expected_units": int(units),
                "expected_donors": int(donors),
                "depends_on": current_dependency_identifier(arm, task, fold),
            }
        )
    contract = Path(evaluation_contract).resolve(strict=True)
    baseline = Path(sealed_baseline_audit).resolve(strict=True)
    payload: dict[str, Any] = {
        "status": "pass",
        "schema_version": V2_PROTOCOL_SCHEMA,
        "protocol_scope": protocol_scope,
        "seed": 42,
        "formal_output_root": str(formal),
        "scored_output_root": str(scored),
        "configuration": {
            "files": configuration_records,
        },
        "inputs": {"files": input_records},
        "evaluation_contract": {"path": str(contract)},
        "sealed_baseline": {"path": str(baseline)},
        "staged_execution": staged_execution_contract(protocol_scope),
        "jobs": jobs,
    }
    if paper_methods is not None:
        payload["paper_methods"] = {
            arm: resolved_methods[arm] for arm in V2_PROTOCOL_SCOPES[protocol_scope]
        }
    if any(job["arm"] in {"gene", "full"} for job in jobs):
        _validate_lineage_protocol_inputs(payload, verify_files=True)
    return payload

def write_v2_protocol_manifest(
    path: str | Path, payload: Mapping[str, Any]
) -> "V2Protocol":
    """Atomically write a newly built manifest and immediately verify it."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen v2 protocol: {destination}")
    atomic_json(destination, payload)
    return load_v2_protocol(destination, verify_files=True)

@dataclass(frozen=True)
class V2JobRecord:
    identifier: str
    arm: str
    task: str
    outer_fold: int
    seed: int
    paper_method: str
    expected_units: int
    expected_donors: int
    depends_on: str | None = None

@dataclass(frozen=True)
class V2Protocol:
    path: Path
    formal_output_root: Path
    scored_output_root: Path
    evaluation_contract: Path
    sealed_baseline_audit: Path
    jobs: Mapping[str, V2JobRecord]
    payload: Mapping[str, Any]
    scope: str = "full"

def _validate_file_section(
    payload: Mapping[str, Any], name: str, *, verify_files: bool
) -> None:
    section = payload.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"v2 protocol lacks mapping section {name!r}")
    records = section.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"v2 protocol section {name!r} has no file records")
    observed_names: set[str] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name}.files[{index}] is not a mapping")
        record_name = str(raw.get("name", ""))
        if not record_name or record_name in observed_names:
            raise ValueError(f"{name} contains an empty or duplicate file name")
        observed_names.add(record_name)
        path = Path(str(raw.get("path", ""))).resolve(strict=verify_files)
        if verify_files and not path.is_file():
            raise ValueError(f"v2 protocol input is not a file: {path}")

def _named_file_records(payload: Mapping[str, Any], section_name: str) -> dict[str, dict[str, str]]:
    section = payload.get(section_name)
    if not isinstance(section, Mapping) or not isinstance(section.get("files"), list):
        raise ValueError(f"v2 protocol lacks file section {section_name!r}")
    return {
        str(record["name"]): {
            "path": str(Path(str(record["path"])).resolve()),
        }
        for record in section["files"]
        if isinstance(record, Mapping)
    }

def _validate_lineage_protocol_inputs(payload: Mapping[str, Any], *, verify_files: bool) -> None:
    records = _named_file_records(payload, "inputs")
    missing = set(V2_LINEAGE_INPUT_NAMES) - set(records)
    if missing:
        raise ValueError(f"v2 protocol lacks full five-lineage inputs: {sorted(missing)}")
    summary = Path(records["lineage_gene_summary"]["path"])
    audit_path = Path(records["lineage_gene_summary_audit"]["path"])
    if not verify_files:
        return
    audit = expand_paths(json.loads(audit_path.read_text(encoding="utf-8")))
    expected = {
        "status": "pass",
        "schema_version": "sound_immunetwin_all_cell_lineage_gene_summary_1",
        "mode": "full",
        "development_only": False,
        "all_available_cells": True,
        "cross_visit_gene_selection": False,
        "n_visits": 489,
    }
    for name, value in expected.items():
        if audit.get(name) != value:
            raise RuntimeError(f"lineage summary audit field {name!r} is invalid")
    if audit.get("lineages") != ["B", "CD4_T", "CD8_T", "NK", "Myeloid"]:
        raise RuntimeError("lineage summary audit does not contain the frozen five lineages")
    if audit.get("coverage", {}).get("all_visits_have_all_five_lineages") is not True:
        raise RuntimeError("lineage summary audit lacks complete five-lineage visit coverage")
    output = audit.get("outputs", {}).get("lineage_gene_summary_h5")
    if not isinstance(output, Mapping):
        raise RuntimeError("lineage summary audit lacks output identity")
    if Path(str(output.get("path", ""))).resolve() != summary.resolve():
        raise RuntimeError("lineage summary audit path differs from frozen H5")

def load_v2_protocol(path: str | Path, *, verify_files: bool = True) -> V2Protocol:
    """Load and verify a protocol manifest."""

    protocol_path = Path(path).resolve(strict=True)
    payload = expand_paths(json.loads(protocol_path.read_text(encoding="utf-8")))
    if not isinstance(payload, Mapping):
        raise ValueError("v2 protocol manifest must be a JSON mapping")
    if payload.get("status") != "pass" or payload.get("schema_version") != V2_PROTOCOL_SCHEMA:
        raise ValueError("v2 protocol manifest schema/status is invalid")
    if int(payload.get("seed", -1)) != 42:
        raise ValueError("v2 protocol seed must be 42")
    scope = str(payload.get("protocol_scope", "full"))
    expected_ids = expected_v2_job_ids(scope)
    allowed_arms = set(V2_PROTOCOL_SCOPES[scope])
    raw_methods = payload.get("paper_methods")
    if raw_methods is None:
        protocol_methods = {arm: V2_METHODS[arm] for arm in allowed_arms}
    else:
        if not isinstance(raw_methods, Mapping) or set(map(str, raw_methods)) != allowed_arms:
            raise ValueError("v2 paper_methods must contain exactly the protocol arms")
        protocol_methods = {
            str(arm): str(method).strip() for arm, method in raw_methods.items()
        }
        if any(not value for value in protocol_methods.values()) or len(
            set(protocol_methods.values())
        ) != len(protocol_methods):
            raise ValueError("v2 paper_methods must be nonempty and unique")

    formal_root = _v2_root(
        payload.get("formal_output_root", ""), field="formal_output_root", must_exist=False
    )
    scored_root = _v2_root(
        payload.get("scored_output_root", ""), field="scored_output_root", must_exist=False
    )
    if formal_root == scored_root or formal_root in scored_root.parents or scored_root in formal_root.parents:
        raise ValueError("formal and scored v2 roots must be separate sibling subtrees")

    _validate_file_section(
        payload, "configuration", verify_files=verify_files
    )
    _validate_file_section(payload, "inputs", verify_files=verify_files)
    if allowed_arms & {"gene", "full"}:
        _validate_lineage_protocol_inputs(payload, verify_files=verify_files)

    if payload.get("staged_execution") != staged_execution_contract(scope):
        raise ValueError("v2 protocol staged-execution contract is invalid")

    evaluation = payload.get("evaluation_contract")
    baseline = payload.get("sealed_baseline")
    if not isinstance(evaluation, Mapping) or not isinstance(baseline, Mapping):
        raise ValueError("v2 protocol needs evaluation_contract and sealed_baseline records")
    evaluation_path = Path(str(evaluation.get("path", ""))).resolve(strict=verify_files)
    baseline_path = Path(str(baseline.get("path", ""))).resolve(strict=verify_files)

    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError("v2 protocol jobs must be a list")
    jobs: dict[str, V2JobRecord] = {}
    for raw in raw_jobs:
        if not isinstance(raw, Mapping):
            raise ValueError("v2 job record must be a mapping")
        arm, task = str(raw.get("arm", "")), str(raw.get("task", ""))
        fold, seed = int(raw.get("outer_fold", -1)), int(raw.get("seed", -1))
        identifier = str(raw.get("identifier", ""))
        expected_identifier = f"{arm}__{task}__outer{fold}_seed{seed}"
        if identifier != expected_identifier or identifier in jobs:
            raise ValueError(f"invalid or duplicate v2 job identifier: {identifier}")
        if arm not in allowed_arms or task not in V2_TASKS or fold not in range(5) or seed != 42:
            raise ValueError(f"invalid v2 job identity: {identifier}")
        if arm != "current" and task == V2_TASKS[0]:
            raise ValueError("history arms cannot train a separate T1 model")
        expected_units, expected_donors = (
            int(raw.get("expected_units", 0)),
            int(raw.get("expected_donors", 0)),
        )
        if expected_units < 1 or expected_donors < 1 or expected_donors > expected_units:
            raise ValueError(f"invalid expected held-out coverage for {identifier}")
        method = str(raw.get("paper_method", ""))
        if method != protocol_methods[arm]:
            raise ValueError(f"paper method does not match arm for {identifier}")
        depends_on = raw.get("depends_on")
        expected_dependency = current_dependency_identifier(arm, task, fold, seed=seed)
        if depends_on != expected_dependency:
            raise ValueError(f"current dependency differs from staged contract: {identifier}")
        jobs[identifier] = V2JobRecord(
            identifier=identifier,
            arm=arm,
            task=task,
            outer_fold=fold,
            seed=seed,
            paper_method=method,
            expected_units=expected_units,
            expected_donors=expected_donors,
            depends_on=depends_on,
        )
    if set(jobs) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(jobs))
        extra = sorted(set(jobs) - set(expected_ids))
        raise ValueError(
            f"v2 {scope} protocol must contain exactly {len(expected_ids)} jobs; "
            f"missing={missing}, extra={extra}"
        )

    configuration_records = payload["configuration"]["files"]
    input_records = payload["inputs"]["files"]
    payload = {
        name: payload[name]
        for name in (
            "status", "schema_version", "seed", "formal_output_root", "scored_output_root",
            "staged_execution",
        )
    } | {
        "protocol_scope": scope,
        "paper_methods": protocol_methods,
        "configuration": {"files": [
            {"name": record["name"], "path": record["path"]}
            for record in configuration_records
        ]},
        "inputs": {"files": [
            {"name": record["name"], "path": record["path"]}
            for record in input_records
        ]},
        "evaluation_contract": {"path": str(evaluation_path)},
        "sealed_baseline": {"path": str(baseline_path)},
        "jobs": [vars(job) for job in jobs.values()],
    }
    return V2Protocol(
        path=protocol_path,
        formal_output_root=formal_root,
        scored_output_root=scored_root,
        evaluation_contract=evaluation_path,
        sealed_baseline_audit=baseline_path,
        jobs=jobs,
        payload=payload,
        scope=scope,
    )

@dataclass(frozen=True)
class ValidatedV2Run:
    output: Path
    run_path: Path
    complete_path: Path
    manifest_path: Path
    run: Mapping[str, Any]
    prediction_manifest: pd.DataFrame

def _artifact_path(
    artifacts: Mapping[str, Any], name: str, output: Path
) -> Path:
    record = artifacts.get(name)
    if not isinstance(record, Mapping):
        raise RuntimeError(f"v2 run lacks required artifact record {name!r}")
    return _inside(Path(str(record.get("path", ""))), output, field=f"artifact {name}")

def _read_unique_table(path: Path, required: Iterable[str], *, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    missing = set(required) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label} lacks required columns: {sorted(missing)}")
    key = ["paper_task", "donor_id", "season", "outer_fold"]
    if frame.duplicated(key).any():
        raise RuntimeError(f"{label} duplicates a task-donor-season-fold key")
    return frame

def _validate_prediction_chain(
    *, output: Path, run: Mapping[str, Any], artifacts: Mapping[str, Any], job: V2JobRecord
) -> tuple[Path, pd.DataFrame]:
    observed_path = _artifact_path(artifacts, "observed_targets", output)
    manifest_path = _artifact_path(artifacts, "predictions", output)
    keys = ["paper_task", "donor_id", "season", "outer_fold"]
    observed = _read_unique_table(
        observed_path, set(keys) | {"observed_path", "observed_cells"},
        label="observed-target manifest",
    )
    final = _read_unique_table(
        manifest_path,
        set(keys) | {"source_path", "predicted_path", "observed_path", "prediction_uses_test_future"},
        label="prediction manifest",
    )
    if len(final) != job.expected_units or final["donor_id"].astype(str).nunique() != job.expected_donors:
        raise RuntimeError(f"held-out coverage differs from the v2 job: {job.identifier}")
    for frame, label in ((observed, "observed"), (final, "final")):
        if not frame["paper_task"].astype(str).eq(job.task).all():
            raise RuntimeError(f"{label} manifest task differs from {job.identifier}")
        if not pd.to_numeric(frame["outer_fold"], errors="raise").astype(int).eq(job.outer_fold).all():
            raise RuntimeError(f"{label} manifest fold differs from {job.identifier}")
    if final["prediction_uses_test_future"].astype(str).str.casefold().isin(
        {"true", "1", "yes"}
    ).any():
        raise RuntimeError("a v2 held-out prediction declares test-future use")
    observed_index = observed.set_index(keys).sort_index()
    final_index = final.set_index(keys).sort_index()
    if not observed_index.index.equals(final_index.index):
        raise RuntimeError("observed-target and final prediction support differ")
    for _, row in final.iterrows():
        for role in ("source", "predicted", "observed"):
            _inside(Path(str(row[f"{role}_path"])), output, field=f"{role} path")
    if (pd.to_numeric(observed["observed_cells"], errors="raise") < 1).any():
        raise RuntimeError("observed target artifact reports no cells")
    for identity in final_index.index:
        if Path(final_index.loc[identity, "observed_path"]).resolve() != Path(
            observed_index.loc[identity, "observed_path"]
        ).resolve():
            raise RuntimeError("final manifest observed path differs from observed-target manifest")
    if run.get("heldout_target_used_for_training_or_adaptation") is not False:
        raise RuntimeError("run audit does not exclude held-out targets from training/adaptation")
    return manifest_path, final

def _validate_external_record(
    record: object,
    *,
    expected_path: Path,
    label: str,
) -> None:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"v2 run lacks {label} dependency record")
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    if path != expected_path.resolve(strict=True):
        raise RuntimeError(f"{label} dependency differs from the parent/input")

def _validate_history_dependency(
    *, protocol: V2Protocol, job: V2JobRecord, run: Mapping[str, Any]
) -> None:
    observed = run.get("current_model_dependency")
    if job.arm == "current":
        if observed is not None:
            raise RuntimeError("current job must not declare a current-model dependency")
        return
    if job.depends_on is None:
        raise RuntimeError("history job lacks its protocol current dependency")
    parent_job = protocol.jobs[job.depends_on]
    parent_output = (
        protocol.formal_output_root
        / "current"
        / parent_job.task
        / f"outer{parent_job.outer_fold}_seed42"
    )
    parent = validate_completed_v2_run(
        parent_output,
        protocol=protocol,
        job_identifier=parent_job.identifier,
    )
    parent_artifacts = parent.run.get("artifacts")
    if not isinstance(parent_artifacts, Mapping):
        raise RuntimeError("matched current run lacks artifact records")
    selection_path = _artifact_path(
        parent_artifacts, "best_validation_checkpoint", parent.output
    )
    final_path = _artifact_path(parent_artifacts, "checkpoint", parent.output)
    if not isinstance(observed, Mapping):
        raise RuntimeError("history run lacks current_model_dependency")
    if observed.get("inner_selection_never_uses_outer_trained_current") is not True:
        raise RuntimeError("history inner selection is not isolated from outer current refit")
    _validate_external_record(
        observed.get("selection_inner_fit_checkpoint"),
        expected_path=selection_path,
        label="current inner-selection checkpoint",
    )
    _validate_external_record(
        observed.get("final_outer_training_checkpoint"),
        expected_path=final_path,
        label="current outer-training checkpoint",
    )

def _validate_lineage_dependency(
    *, protocol: V2Protocol, job: V2JobRecord, run: Mapping[str, Any]
) -> None:
    observed = run.get("lineage_gene_summary_dependency")
    if job.arm not in {"gene", "full"}:
        if observed is not None:
            raise RuntimeError("non-gene job must not declare a lineage-gene dependency")
        return
    records = _named_file_records(protocol.payload, "inputs")
    if not isinstance(observed, Mapping):
        raise RuntimeError("gene/full run lacks lineage_gene_summary_dependency")
    for run_name, protocol_name in (
        ("summary", "lineage_gene_summary"),
        ("builder_audit", "lineage_gene_summary_audit"),
    ):
        frozen = records[protocol_name]
        _validate_external_record(
            observed.get(run_name),
            expected_path=Path(frozen["path"]),
            label=f"lineage {run_name}",
        )

def validate_completed_v2_run(
    output: str | Path,
    *,
    protocol: V2Protocol,
    job_identifier: str,
) -> ValidatedV2Run:
    """Validate the structure and coverage of one task-by-fold job."""

    job = protocol.jobs.get(job_identifier)
    if job is None:
        raise RuntimeError(f"job is absent from frozen v2 protocol: {job_identifier}")
    expected_output = (
        protocol.formal_output_root
        / job.arm
        / job.task
        / f"outer{job.outer_fold}_seed42"
    ).resolve()
    output_path = Path(output).resolve()
    if output_path != expected_output:
        raise RuntimeError(f"v2 job directory differs from protocol: {output_path} != {expected_output}")
    _v2_root(output_path, field="v2 job output", must_exist=True)

    complete_path = output_path / "_COMPLETE.json"
    run_path = output_path / "run.json"
    if not complete_path.is_file() or not run_path.is_file():
        raise RuntimeError(f"v2 job lacks atomic completion chain: {output_path}")
    complete = expand_paths(json.loads(complete_path.read_text(encoding="utf-8")))
    required_complete = {
        "status": "pass",
        "schema_version": V2_COMPLETE_SCHEMA,
    }
    for field, expected in required_complete.items():
        if complete.get(field) != expected:
            raise RuntimeError(f"v2 completion field {field!r} is invalid: {complete_path}")

    if "job_identifier" in complete and complete["job_identifier"] != job.identifier:
        raise RuntimeError("completion job identifier differs from protocol")
    if "run_json" in complete and Path(complete["run_json"]).resolve() != run_path.resolve():
        raise RuntimeError("completion run path differs from the job directory")
    run = expand_paths(json.loads(run_path.read_text(encoding="utf-8")))
    required_run = {
        "status": "pass",
        "schema_version": V2_TRAIN_SCHEMA,
        "arm": job.arm,
        "paper_method": job.paper_method,
        "paper_task": job.task,
        "outer_fold": job.outer_fold,
        "seed": 42,
        "task_outer_fold_independent_model": True,
        "cells_treated_as_independent_n": False,
        "vectors_pooled_across_outer_folds": False,
    }
    for field, expected in required_run.items():
        if run.get(field) != expected:
            raise RuntimeError(f"v2 run field {field!r} is invalid: {run_path}")

    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("v2 run lacks its artifact map")
    required_artifacts = set(V2_REQUIRED_ARTIFACTS)
    if job.arm != "current":
        required_artifacts.update(
            {"selection_current_initialization", "final_current_initialization"}
        )
    if job.arm in {"gene", "full"}:
        required_artifacts.update(
            {"selection_lineage_gene_parameters", "final_lineage_gene_parameters"}
        )
    if not required_artifacts.issubset(artifacts):
        raise RuntimeError(
            f"v2 run artifact map lacks {sorted(required_artifacts - set(artifacts))}"
        )
    for name in required_artifacts:
        _artifact_path(artifacts, str(name), output_path)
    checkpoint_path = _artifact_path(artifacts, "checkpoint", output_path)
    if checkpoint_path.name != "checkpoint.pt":
        raise RuntimeError("v2 checkpoint artifact must be checkpoint.pt")
    best_path = _artifact_path(artifacts, "best_validation_checkpoint", output_path)
    if best_path.name != "best_validation.pt":
        raise RuntimeError("v2 inner checkpoint artifact must be best_validation.pt")
    if run.get("clinical_mode") != "off" or run.get("clinical_information_used") is not False:
        raise RuntimeError("formal v2 primary run must keep clinical history off")
    if job.arm != "current" and run.get("history_training_mode") != "residual_from_current":
        raise RuntimeError("history run is not residualized from matched current")
    if job.arm in {"gene", "full"} and run.get("gene_prior_mode") != "lineage":
        raise RuntimeError("gene/full run did not use the five-lineage gene prior")
    _validate_history_dependency(protocol=protocol, job=job, run=run)
    _validate_lineage_dependency(protocol=protocol, job=job, run=run)
    manifest_path, manifest = _validate_prediction_chain(
        output=output_path, run=run, artifacts=artifacts, job=job
    )
    if int(run.get("test_units", -1)) != job.expected_units:
        raise RuntimeError("run test-unit count differs from frozen v2 coverage")
    if int(run.get("test_donors", -1)) != job.expected_donors:
        raise RuntimeError("run test-donor count differs from frozen v2 coverage")
    return ValidatedV2Run(
        output=output_path,
        run_path=run_path,
        complete_path=complete_path,
        manifest_path=manifest_path,
        run=run,
        prediction_manifest=manifest,
    )
