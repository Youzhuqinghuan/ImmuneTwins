from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence
from pipeline.sound.provenance import build_v2_protocol_payload, load_v2_protocol, validate_completed_v2_run, write_v2_protocol_manifest

PAPER_METHOD = "ImmuneTwin-ProjectedSet"

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser

def _named_paths(payload: Mapping[str, Any], section: str) -> dict[str, Path]:
    raw = payload.get(section, {}).get("files", [])
    if not isinstance(raw, list):
        raise ValueError(f"parent protocol section {section!r} is invalid")
    result: dict[str, Path] = {}
    for record in raw:
        if not isinstance(record, Mapping):
            raise ValueError(f"parent protocol section {section!r} is malformed")
        name = str(record.get("name", ""))
        if not name or name in result:
            raise ValueError(f"duplicate or empty name in parent {section!r}")
        result[name] = Path(str(record.get("path", ""))).resolve(strict=True)
    return result

def _artifact(run: Mapping[str, Any], name: str) -> Path:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("parent run lacks artifact records")
    record = artifacts.get(name)
    if not isinstance(record, Mapping):
        raise RuntimeError(f"parent run lacks {name!r}")
    return Path(str(record.get("path", ""))).resolve(strict=True)

def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    parent = load_v2_protocol(args.parent_protocol, verify_files=False)
    if parent.scope != "current_only" or len(parent.jobs) != 15:
        raise RuntimeError("ProjectedSet requires the 15-job SetTransport protocol")

    configuration = _named_paths(parent.payload, "configuration")
    configuration["model_and_training"] = args.config.resolve(strict=True)
    inputs = _named_paths(parent.payload, "inputs")
    inputs["set_transport_parent_protocol"] = args.parent_protocol.resolve(strict=True)

    coverage: dict[str, tuple[int, int]] = {}
    for identifier, job in sorted(parent.jobs.items()):
        coverage[identifier] = (job.expected_units, job.expected_donors)
        parent_output = (
            parent.formal_output_root
            / "current"
            / job.task
            / f"outer{job.outer_fold}_seed42"
        )
        completed = validate_completed_v2_run(
            parent_output,
            protocol=parent,
            job_identifier=identifier,
        )
        prefix = f"{job.task}_outer{job.outer_fold}"
        inputs[f"{prefix}_set_transport_best_validation"] = _artifact(
            completed.run, "best_validation_checkpoint"
        )
        inputs[f"{prefix}_set_transport_checkpoint"] = _artifact(
            completed.run, "checkpoint"
        )
        pilot = (
            args.pilot_root.resolve(strict=True)
            / job.task
            / f"outer{job.outer_fold}_seed42"
        )
        inputs[f"{prefix}_projected_set_pilot_summary"] = (
            pilot / "summary.json"
        ).resolve(strict=True)
        inputs[f"{prefix}_projected_set_pilot_checkpoint"] = (
            pilot / "best.pt"
        ).resolve(strict=True)
        inputs[f"{prefix}_projected_set_pilot_history"] = (
            pilot / "history.tsv"
        ).resolve(strict=True)

    payload = build_v2_protocol_payload(
        formal_output_root=args.formal_root,
        scored_output_root=args.scored_root,
        configuration=configuration,
        inputs=inputs,
        evaluation_contract=parent.evaluation_contract,
        sealed_baseline_audit=parent.sealed_baseline_audit,
        job_coverage=coverage,
        protocol_scope="current_only",
        paper_methods={"current": PAPER_METHOD},
    )
    protocol = write_v2_protocol_manifest(args.output, payload)
    print(protocol.path)

if __name__ == "__main__":
    main()


def prepare_main():
    """Build a task plan pointing to new output directories."""
    import json
    from paths import ROOT
    from data.assets import default_assets_path
    from pipeline.sound.provenance import file_records
    parser = argparse.ArgumentParser(description=prepare_main.__doc__)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--formal-root',type=Path,required=True)
    parser.add_argument('--scored-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    for path in (args.formal_root,args.scored_root,args.output):
        path.resolve().relative_to(ROOT)
    source = load_v2_protocol(args.template,verify_files=False)
    payload = json.loads(json.dumps(source.payload))
    payload['formal_output_root'] = str(args.formal_root.resolve())
    payload['scored_output_root'] = str(args.scored_root.resolve())
    groups = {
        'configuration':{'assets':default_assets_path(),'model_and_training':args.config,
                         'current_evaluation':ROOT/'config/sound_current_evaluation.yaml'},
    }
    for name,files in groups.items():
        records = file_records(files)
        payload[name] = {'files':records}
    contract = ROOT/'config/sound_current_evaluation.yaml'
    payload['evaluation_contract'] = {'path':str(contract)}
    result = write_v2_protocol_manifest(args.output,payload)
    print(json.dumps({'protocol':str(args.output.resolve()),'jobs':len(result.jobs)}))
