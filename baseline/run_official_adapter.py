from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import pandas as pd
import yaml
from data.assets import default_assets_path

METHOD_SCRIPTS = {
    "scgen": "11_run_scgen_baseline.py",
    "cellot": "12_run_cellot_baseline.py",
    "trvae": "13_run_trvae_baseline.py",
}

RUNNER_ROOT = Path(__file__).resolve().parent / "vaccine_runners"

GROUPS = ("vaccinated", "updated")

PROTOCOL_SCHEMA = "sound_vaccine_baseline_v2"

TASK_CONTRACTS = {
    "t1_y1_d0_to_y1_d7": {
        "group": "vaccinated",
        "source_day": 0,
        "target_day": 7,
        "counts": {"flu_year_1": 83, "flu_year_2": 0},
    },
    "t2_d7_to_d90": {
        "group": "updated",
        "source_day": 7,
        "target_day": 90,
        "counts": {"flu_year_1": 83, "flu_year_2": 80},
    },
    "t3_y2_d0_to_y2_d90": {
        "group": "vaccinated",
        "source_day": 0,
        "target_day": 90,
        "counts": {"flu_year_1": 0, "flu_year_2": 75},
    },
}

def _validate_vaccine_v2_contract(episodes: pd.DataFrame) -> str:
    required = {
        "episode_id",
        "task",
        "donor_id",
        "source_sample",
        "target_sample",
        "target_day",
        "vaccinated",
        "season",
        "outer_fold",
        "interval_eligible",
    }
    missing = required - set(episodes.columns)
    if missing:
        raise RuntimeError(f"vaccine episode contract lacks {sorted(missing)}")
    if episodes["episode_id"].duplicated().any():
        raise RuntimeError("vaccine episode contract duplicates episode_id")
    if not episodes["task"].eq("conditional").all():
        raise RuntimeError("vaccine episode contract contains a non-conditional row")
    if not episodes["vaccinated"].astype(str).str.casefold().isin(
        {"true", "1", "yes"}
    ).all():
        raise RuntimeError("vaccine episode contract contains a non-vaccine row")
    if not episodes["interval_eligible"].astype(str).str.casefold().isin(
        {"true", "1", "yes"}
    ).all():
        raise RuntimeError("vaccine episode contract contains an ineligible interval")
    days = pd.to_numeric(episodes["target_day"], errors="raise").astype(int)
    if set(days) != {7, 90}:
        raise RuntimeError("vaccine episode contract must contain D7 and D90")
    keyed = episodes.assign(_target_day=days)
    if keyed.duplicated(["donor_id", "season", "_target_day"]).any():
        raise RuntimeError("vaccine episode contract duplicates donor-season-day")
    units = keyed.groupby(["donor_id", "season"], sort=False)
    if not units["_target_day"].nunique().eq(2).all():
        raise RuntimeError("vaccine episode contract contains an incomplete triplet")
    if not units["source_sample"].nunique().eq(1).all():
        raise RuntimeError("D7 and D90 do not share the same D0 sample")
    counts = keyed.loc[keyed["_target_day"].eq(7), "season"].value_counts()
    observed = {
        "flu_year_1": int(counts.get("flu_year_1", 0)),
        "flu_year_2": int(counts.get("flu_year_2", 0)),
    }
    required_task_columns = {
        "paper_task",
        "paper_source_day",
        "paper_target_day",
        "both_years_complete",
    }
    missing_task = required_task_columns - set(keyed.columns)
    if missing_task:
        raise RuntimeError(f"vaccine_v2 contract lacks {sorted(missing_task)}")
    task_values = keyed["paper_task"].dropna().astype(str).unique()
    if len(task_values) != 1:
        raise RuntimeError("vaccine_v2 episode contract must contain one paper task")
    task_id = task_values[0]
    expected = TASK_CONTRACTS.get(task_id)
    if expected is None:
        raise RuntimeError(f"unknown vaccine_v2 paper task {task_id}")
    source_days = pd.to_numeric(keyed["paper_source_day"], errors="raise")
    target_days = pd.to_numeric(keyed["paper_target_day"], errors="raise")
    if not source_days.eq(int(expected["source_day"])).all():
        raise RuntimeError(f"{task_id} has an invalid paper source day")
    if not target_days.eq(int(expected["target_day"])).all():
        raise RuntimeError(f"{task_id} has an invalid paper target day")
    if observed != expected["counts"]:
        raise RuntimeError(f"{task_id} strict-unit counts changed: {observed}")
    if task_id == "t3_y2_d0_to_y2_d90" and not keyed[
        "both_years_complete"
    ].astype(str).str.casefold().isin({"true", "1", "yes"}).all():
        raise RuntimeError("T3 contains a donor without complete two-year history")
    return task_id


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)

def _adapter_episodes(
    episodes: pd.DataFrame,
    group: str,
    target_day: int,
) -> tuple[pd.DataFrame, int, dict[tuple[str, int], str]]:
    if group not in GROUPS:
        raise ValueError(f"unsupported vaccine_v2 group {group!r}")
    if target_day not in {7, 90} or (group == "updated" and target_day != 90):
        raise ValueError("vaccinated requires Day 7 or 90; updated requires Day 90")
    eligible = episodes["interval_eligible"].astype(str).str.casefold().eq("true")
    vaccinated = episodes["vaccinated"].astype(str).str.casefold().isin(
        {"true", "1", "yes"}
    )
    selected = episodes.loc[
        eligible
        & episodes["task"].eq("conditional")
        & vaccinated
        & episodes["target_day"].astype(float).eq(float(target_day))
    ].copy()
    internal_day = int(target_day)
    if group == "updated":
        day7 = episodes.loc[
            eligible
            & episodes["task"].eq("conditional")
            & vaccinated
            & episodes["target_day"].astype(float).eq(7.0)
        ].copy()
        if day7["source_sample"].duplicated().any():
            raise RuntimeError(
                "one vaccination baseline maps to multiple Day 7 observations"
            )
        day7_for_baseline = dict(
            zip(
                day7["source_sample"].astype(str),
                day7["target_sample"].astype(str),
            )
        )
        selected["update_source_sample"] = selected[
            "source_sample"
        ].astype(str).map(day7_for_baseline)
        if selected["update_source_sample"].isna().any():
            raise RuntimeError("T2 contains a D90 row without its observed D7 state")
    if selected.empty:
        raise RuntimeError(f"no vaccine_v2 episodes for official group {group}")
    target_column = f"sample_D{internal_day}"
    complete_column = f"complete_D0_D{internal_day}"
    season = selected["season"].map({"flu_year_1": 1, "flu_year_2": 2})
    if season.isna().any():
        raise RuntimeError("vaccine_v2 adapter contains a non-Flu season")
    adapter = pd.DataFrame(
        {
            "donor_id": selected["donor_id"].astype(str),
            "season": season.astype(int),
            "vaccinated": selected["vaccinated"],
            "sample_D0": (
                selected["update_source_sample"].astype(str)
                if group == "updated"
                else selected["source_sample"].astype(str)
            ),
            target_column: selected["target_sample"].astype(str),
            complete_column: True,
        }
    )
    lookup = dict(
        zip(
            zip(selected["donor_id"].astype(str), season.astype(int)),
            selected["episode_id"].astype(str),
        )
    )
    if len(lookup) != len(selected):
        raise RuntimeError(f"{group} contains duplicate donor prediction episodes")
    return adapter, internal_day, lookup

def _prepare_adapter_root(
    *,
    adapter_root: Path,
    adapter_episodes: pd.DataFrame,
    runtime_root: Path,
) -> None:
    adapter_root.mkdir(parents=True, exist_ok=True)
    with (adapter_root / ".contract.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _atomic_table(
            adapter_episodes,
            adapter_root / "data/episode_manifests/sound_flu_episodes.tsv",
        )
        split_root = adapter_root / "config/splits"
        split_root.mkdir(parents=True, exist_ok=True)
        for name in (
            "sound_outer_groupkfold5_donors.tsv",
            "sound_inner_groupkfold4_donors.tsv",
        ):
            source = runtime_root / "config/splits" / name
            destination = split_root / name
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)

def _load_runner(runtime_root: Path, method: str):
    """Load the runner code against the local upstream checkouts."""

    runtime_root = runtime_root.resolve(strict=True)
    script = RUNNER_ROOT / METHOD_SCRIPTS[method]
    dependency_names = (
        "common",
        "formal_latent_reader",
        "trusted_torch_checkpoint",
    )
    missing = object()
    previous_modules = {
        name: sys.modules.get(name, missing) for name in dependency_names
    }
    sys.path.insert(0, str(RUNNER_ROOT))
    try:
        for name in dependency_names:
            sys.modules.pop(name, None)
        common_path = RUNNER_ROOT / "common.py"
        common_specification = importlib.util.spec_from_file_location(
            "common", common_path
        )
        if (
            common_specification is None
            or common_specification.loader is None
        ):
            raise ImportError(f"cannot load runner support {common_path}")
        common = importlib.util.module_from_spec(common_specification)
        sys.modules["common"] = common
        common_specification.loader.exec_module(common)
        # Upstream checkouts and environments are located under runtime_root.
        common.ROOT = runtime_root

        specification = importlib.util.spec_from_file_location(
            f"_immunetwins_official_{method}", script
        )
        if specification is None or specification.loader is None:
            raise ImportError(f"cannot load official runner {script}")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(RUNNER_ROOT):
            del sys.path[0]
        else:
            try:
                sys.path.remove(str(RUNNER_ROOT))
            except ValueError:
                pass
        for name, previous in previous_modules.items():
            sys.modules.pop(name, None)
            if previous is not missing:
                sys.modules[name] = previous
    return module

def _canonicalize(
    *,
    output: Path,
    group: str,
    target_day: int,
    episode_lookup: dict[tuple[str, int], str],
    episode_contract: Path,
    paper_task: str,
    runtime_root: Path,
    runner_source: Path,
    runner_command: list[str],
) -> None:
    run_path = output / "run.json"
    manifest_path = output / "predictions.tsv"
    if not run_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("official runner did not materialize its contract")
    run = json.loads(run_path.read_text())
    if run.get("cell_scope") != "all_cells":
        raise RuntimeError("official vaccine runner did not use the all-cell contract")
    manifest = pd.read_csv(manifest_path, sep="\t")
    if "season" not in manifest:
        raise RuntimeError("official predictions lack season")
    raw_season = manifest["season"].astype(str).str.strip()
    season_number = pd.to_numeric(
        raw_season.str.replace(r"^flu_year_", "", regex=True),
        errors="coerce",
    )
    if (
        season_number.isna().any()
        or not season_number.eq(season_number.round()).all()
        or not season_number.isin({1, 2}).all()
    ):
        invalid = sorted(raw_season.loc[
            season_number.isna()
            | ~season_number.eq(season_number.round())
            | ~season_number.isin({1, 2})
        ].unique())
        raise RuntimeError(
            f"official vaccine_v2 output has invalid season values: {invalid}"
        )
    season_number = season_number.astype(int)
    keys = list(
        zip(
            manifest["donor_id"].astype(str),
            season_number,
        )
    )
    canonical = pd.Series(
        [episode_lookup.get(key) for key in keys], index=manifest.index
    )
    if canonical.isna().any():
        missing = manifest.loc[canonical.isna(), "donor_id"].astype(str).tolist()
        raise RuntimeError(f"official predictions contain unknown donors: {missing}")
    manifest["episode_id"] = canonical
    manifest["task"] = "conditional"
    manifest["vaccinated"] = True
    manifest["season"] = season_number.map(
        {1: "flu_year_1", 2: "flu_year_2"}
    )
    manifest["target_day"] = int(target_day)
    manifest["source_day"] = 7 if group == "updated" else 0
    manifest["paper_task"] = paper_task
    _atomic_table(manifest, manifest_path)
    run.update(
        {
            "v6_task_group": group,
            "v6_task": "conditional",
            "v6_vaccinated": True,
            "v6_target_day": int(target_day),
            "v6_episode_contract": True,
            "source_observation_day": 7 if group == "updated" else 0,
            "same_person_observation_used": group == "updated",
            "digital_twin_update_comparator": group == "updated",
            "paper_protocol_schema": PROTOCOL_SCHEMA,
            "paper_task": paper_task,
            "cell_scope": "all_cells",
            "episode_contract": str(episode_contract.resolve()),
            "manifest": str(manifest_path.resolve()),
            "n_test_distributions": int(len(manifest)),
            "runner_source": str(runner_source.resolve()),
            "runtime_asset_root": str(runtime_root.resolve()),
            "runner_argv": runner_command,
        }
    )
    run_path.write_text(json.dumps(run, indent=2) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one official all-cell baseline on a Sound vaccine task"
        )
    )
    parser.add_argument("--assets", type=Path, default=default_assets_path())
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHOD_SCRIPTS), required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--target-day", type=int, choices=(7, 90), required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--paper-protocol-schema",
        choices=(PROTOCOL_SCHEMA,),
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("official baselines require explicit CUDA_VISIBLE_DEVICES")

    assets = yaml.safe_load(args.assets.read_text())
    episodes = pd.read_csv(args.episodes, sep="\t")
    paper_task = _validate_vaccine_v2_contract(episodes)
    task_contract = TASK_CONTRACTS[paper_task]
    if args.group != task_contract["group"]:
        raise RuntimeError(
            f"{paper_task} requires group {task_contract['group']}, got {args.group}"
        )
    if args.target_day != task_contract["target_day"]:
        raise RuntimeError(
            f"{paper_task} requires target day {task_contract['target_day']}, "
            f"got {args.target_day}"
        )
    adapter_episodes, internal_day, lookup = _adapter_episodes(
        episodes, args.group, args.target_day
    )
    runtime_root = Path(
        assets.get(
            "vaccine_baseline_runtime",
            Path(assets["project_root"]) / "vaccine_baseline_runtime",
        )
    ).resolve(strict=True)
    adapter_root = (
        Path(assets["artifact_root"])
        / "official_adapter"
        / args.group
        / f"day{int(args.target_day)}"
    )
    _prepare_adapter_root(
        adapter_root=adapter_root,
        adapter_episodes=adapter_episodes,
        runtime_root=runtime_root,
    )
    mapping = json.loads(
        Path(
            assets["sound"]["latent_maps"].format(fold=args.outer_fold)
        ).read_text()
    )
    scales = json.loads(
        Path(
            assets["sound"]["metric_maps"].format(fold=args.outer_fold)
        ).read_text()
    )
    runner_source = RUNNER_ROOT / METHOD_SCRIPTS[args.method]
    module = _load_runner(runtime_root, args.method)
    module.ROOT = adapter_root
    command = [
        str(runner_source),
        "--fold",
        str(args.outer_fold),
        "--seed",
        str(args.seed),
        "--target-day",
        str(internal_day),
        "--source-day",
        "7" if args.group == "updated" else "0",
        "--model",
        f"{args.method}_{args.group}",
        "--output-dir",
        str(args.output),
        "--training-latent-h5",
        mapping["training"]["all"],
        "--test-latent-h5",
        mapping["outer_test"]["all"],
    ]
    if args.method in {"cellot", "trvae"}:
        command.extend(["--scale-npz", scales["all"]])
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "method": args.method,
                    "paper_task": paper_task,
                    "outer_fold": args.outer_fold,
                    "seed": args.seed,
                    "cell_scope": "all_cells",
                    "runtime_root": str(runtime_root),
                    "runner": str(runner_source),
                    "training_latent_h5": mapping["training"]["all"],
                    "test_latent_h5": mapping["outer_test"]["all"],
                    "scale_npz": (
                        scales["all"]
                        if args.method in {"cellot", "trvae"}
                        else None
                    ),
                    "adapter_episode_rows": int(len(adapter_episodes)),
                    "command": command,
                },
                indent=2,
            )
        )
        return
    previous = sys.argv
    try:
        sys.argv = command
        module.main()
    finally:
        sys.argv = previous
    _canonicalize(
        output=args.output,
        group=args.group,
        target_day=args.target_day,
        episode_lookup=lookup,
        episode_contract=args.episodes,
        paper_task=paper_task,
        runtime_root=runtime_root,
        runner_source=runner_source,
        runner_command=command,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "method": args.method,
                "group": args.group,
                "target_day": args.target_day,
                "outer_fold": args.outer_fold,
                "cell_scope": "all_cells",
                "paper_task": paper_task,
                "seed": args.seed,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
