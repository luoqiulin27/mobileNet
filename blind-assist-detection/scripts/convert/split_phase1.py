"""
Split a converted detection dataset into train/val/test by session.

The splitter keeps every session in exactly one split to avoid leakage, while
searching for a split that preserves class coverage in validation and test.
It is intended for SANPO-style filenames:

    {session_id}_{frame_number}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    stems: list[str]
    class_counts: list[int]

    @property
    def image_count(self) -> int:
        return len(self.stems)


@dataclass
class SplitState:
    stems: dict[str, list[str]]
    sessions: dict[str, list[str]]
    class_counts: dict[str, list[int]]
    image_counts: dict[str, int]


@dataclass
class SplitCandidate:
    state: SplitState
    score: float
    seed: int
    class_warnings: list[str] = field(default_factory=list)


def extract_session_id(stem: str) -> str:
    last_underscore = stem.rfind("_")
    if last_underscore < 0:
        raise ValueError(f"Invalid filename format, no '_' found: {stem}")
    return stem[:last_underscore]


def read_classes(data_dir: Path) -> list[str]:
    classes_path = data_dir / "configs" / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found: {classes_path}")
    with open(classes_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def parse_label_counts(label_path: Path, num_classes: int) -> list[int]:
    counts = [0 for _ in range(num_classes)]
    if not label_path.exists():
        return counts
    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            if 0 <= class_id < num_classes:
                counts[class_id] += 1
    return counts


def scan_stems(data_dir: Path) -> tuple[list[str], list[str], int]:
    img_dir = data_dir / "images" / "all"
    lbl_dir = data_dir / "labels" / "all"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    image_stems = sorted([path.stem for path in img_dir.glob("*.png")])
    paired_stems: list[str] = []
    missing_label_count = 0
    for stem in image_stems:
        if (lbl_dir / f"{stem}.txt").exists():
            paired_stems.append(stem)
        else:
            missing_label_count += 1
    return image_stems, paired_stems, missing_label_count


def collect_sessions(data_dir: Path, stems: list[str], num_classes: int) -> list[SessionInfo]:
    lbl_dir = data_dir / "labels" / "all"
    grouped_stems: dict[str, list[str]] = {}
    grouped_counts: dict[str, list[int]] = {}

    for stem in stems:
        session_id = extract_session_id(stem)
        grouped_stems.setdefault(session_id, []).append(stem)
        if session_id not in grouped_counts:
            grouped_counts[session_id] = [0 for _ in range(num_classes)]
        label_counts = parse_label_counts(lbl_dir / f"{stem}.txt", num_classes)
        grouped_counts[session_id] = [
            grouped_counts[session_id][idx] + label_counts[idx]
            for idx in range(num_classes)
        ]

    sessions: list[SessionInfo] = []
    for session_id in sorted(grouped_stems):
        stems_for_session = sorted(grouped_stems[session_id])
        sessions.append(
            SessionInfo(
                session_id=session_id,
                stems=stems_for_session,
                class_counts=grouped_counts[session_id],
            )
        )
    return sessions


def empty_state(num_classes: int) -> SplitState:
    return SplitState(
        stems={name: [] for name in SPLIT_NAMES},
        sessions={name: [] for name in SPLIT_NAMES},
        class_counts={name: [0 for _ in range(num_classes)] for name in SPLIT_NAMES},
        image_counts={name: 0 for name in SPLIT_NAMES},
    )


def add_session(state: SplitState, split_name: str, session: SessionInfo) -> None:
    state.stems[split_name].extend(session.stems)
    state.sessions[split_name].append(session.session_id)
    state.image_counts[split_name] += session.image_count
    state.class_counts[split_name] = [
        state.class_counts[split_name][idx] + session.class_counts[idx]
        for idx in range(len(session.class_counts))
    ]


def clone_state_with_session(
    state: SplitState,
    split_name: str,
    session: SessionInfo,
) -> SplitState:
    cloned = SplitState(
        stems={name: list(state.stems[name]) for name in SPLIT_NAMES},
        sessions={name: list(state.sessions[name]) for name in SPLIT_NAMES},
        class_counts={name: list(state.class_counts[name]) for name in SPLIT_NAMES},
        image_counts=dict(state.image_counts),
    )
    add_session(cloned, split_name, session)
    return cloned


def split_targets(total_images: int, train_ratio: float, val_ratio: float) -> dict[str, float]:
    return {
        "train": total_images * train_ratio,
        "val": total_images * val_ratio,
        "test": total_images * max(0.0, 1.0 - train_ratio - val_ratio),
    }


def class_targets(
    total_class_counts: list[int],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[float]]:
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": max(0.0, 1.0 - train_ratio - val_ratio),
    }
    return {
        split: [count * ratio for count in total_class_counts]
        for split, ratio in ratios.items()
    }


def minimum_for_class(total: int, requested: int) -> int:
    if total <= 0:
        return 0
    # Keep very rare classes represented without making the search impossible.
    return min(requested, max(1, total // 20))


def score_state(
    state: SplitState,
    total_images: int,
    total_class_counts: list[int],
    image_targets: dict[str, float],
    target_class_counts: dict[str, list[float]],
    min_val_bboxes: int,
    min_test_bboxes: int,
    partial: bool,
) -> float:
    score = 0.0

    for split_name in SPLIT_NAMES:
        target = max(1.0, image_targets[split_name])
        image_error = abs(state.image_counts[split_name] - target) / target
        score += image_error * 6.0

    for split_name in SPLIT_NAMES:
        for class_id, total_count in enumerate(total_class_counts):
            if total_count <= 0:
                continue
            target = max(1.0, target_class_counts[split_name][class_id])
            actual = state.class_counts[split_name][class_id]
            score += abs(actual - target) / target

    if partial:
        return score

    for split_name, requested_min in (("val", min_val_bboxes), ("test", min_test_bboxes)):
        for class_id, total_count in enumerate(total_class_counts):
            minimum = minimum_for_class(total_count, requested_min)
            actual = state.class_counts[split_name][class_id]
            if actual < minimum:
                score += ((minimum - actual) / max(1, minimum)) * 250.0

    for split_name in SPLIT_NAMES:
        if state.image_counts[split_name] == 0:
            score += 1000.0

    assigned = sum(state.image_counts.values())
    if assigned != total_images:
        score += abs(total_images - assigned) * 100.0

    return score


def score_state_with_extra_session(
    state: SplitState,
    split_to_extend: str,
    session: SessionInfo,
    total_images: int,
    total_class_counts: list[int],
    image_targets: dict[str, float],
    target_class_counts: dict[str, list[float]],
) -> float:
    """Score a possible assignment without copying large split lists."""
    score = 0.0

    for split_name in SPLIT_NAMES:
        actual_images = state.image_counts[split_name]
        if split_name == split_to_extend:
            actual_images += session.image_count
        target = max(1.0, image_targets[split_name])
        score += (abs(actual_images - target) / target) * 6.0

    for split_name in SPLIT_NAMES:
        for class_id, total_count in enumerate(total_class_counts):
            if total_count <= 0:
                continue
            actual = state.class_counts[split_name][class_id]
            if split_name == split_to_extend:
                actual += session.class_counts[class_id]
            target = max(1.0, target_class_counts[split_name][class_id])
            score += abs(actual - target) / target

    return score


def build_candidate(
    sessions: list[SessionInfo],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_val_bboxes: int,
    min_test_bboxes: int,
) -> SplitCandidate:
    num_classes = len(sessions[0].class_counts) if sessions else 0
    total_images = sum(session.image_count for session in sessions)
    total_class_counts = [
        sum(session.class_counts[class_id] for session in sessions)
        for class_id in range(num_classes)
    ]
    image_targets = split_targets(total_images, train_ratio, val_ratio)
    target_class_counts = class_targets(total_class_counts, train_ratio, val_ratio)

    rng = random.Random(seed)
    ordered_sessions = list(sessions)
    rng.shuffle(ordered_sessions)

    # Put sessions carrying rare classes earlier, with randomness as a tie-breaker.
    rarity_weights = [
        (1.0 / count) if count > 0 else 0.0
        for count in total_class_counts
    ]
    ordered_sessions.sort(
        key=lambda session: (
            -sum(
                session.class_counts[idx] * rarity_weights[idx]
                for idx in range(num_classes)
            ),
            -session.image_count,
            rng.random(),
        )
    )

    state = empty_state(num_classes)
    for session in ordered_sessions:
        best_split = None
        best_score = None
        for split_name in SPLIT_NAMES:
            trial_score = score_state_with_extra_session(
                state,
                split_name,
                session,
                total_images,
                total_class_counts,
                image_targets,
                target_class_counts,
            )
            if best_score is None or trial_score < best_score:
                best_score = trial_score
                best_split = split_name
        add_session(state, best_split or "train", session)

    final_score = score_state(
        state,
        total_images,
        total_class_counts,
        image_targets,
        target_class_counts,
        min_val_bboxes,
        min_test_bboxes,
        partial=False,
    )
    return SplitCandidate(state=state, score=final_score, seed=seed)


def choose_best_split(
    sessions: list[SessionInfo],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    search_trials: int,
    min_val_bboxes: int,
    min_test_bboxes: int,
) -> SplitCandidate:
    if not sessions:
        raise ValueError("No sessions available to split")

    best: SplitCandidate | None = None
    for offset in range(max(1, search_trials)):
        candidate = build_candidate(
            sessions=sessions,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed + offset,
            min_val_bboxes=min_val_bboxes,
            min_test_bboxes=min_test_bboxes,
        )
        if best is None or candidate.score < best.score:
            best = candidate

    if best is None:
        raise RuntimeError("Unable to build a split candidate")
    return best


def verify_no_leak(state: SplitState) -> tuple[bool, dict[str, int]]:
    train_set = set(state.stems["train"])
    val_set = set(state.stems["val"])
    test_set = set(state.stems["test"])
    train_sessions = set(state.sessions["train"])
    val_sessions = set(state.sessions["val"])
    test_sessions = set(state.sessions["test"])

    details = {
        "frame_train_val_leak": len(train_set & val_set),
        "frame_train_test_leak": len(train_set & test_set),
        "frame_val_test_leak": len(val_set & test_set),
        "session_train_val_leak": len(train_sessions & val_sessions),
        "session_train_test_leak": len(train_sessions & test_sessions),
        "session_val_test_leak": len(val_sessions & test_sessions),
    }
    return all(value == 0 for value in details.values()), details


def class_coverage_warnings(
    class_names: list[str],
    total_class_counts: list[int],
    split_class_counts: dict[str, list[int]],
    min_val_bboxes: int,
    min_test_bboxes: int,
) -> list[str]:
    warnings: list[str] = []
    for split_name, requested_min in (("val", min_val_bboxes), ("test", min_test_bboxes)):
        for class_id, class_name in enumerate(class_names):
            total_count = total_class_counts[class_id]
            minimum = minimum_for_class(total_count, requested_min)
            count = split_class_counts[split_name][class_id]
            if total_count > 0 and count == 0:
                warnings.append(f"{class_name} has 0 bboxes in {split_name}")
            elif 0 < count < minimum:
                warnings.append(
                    f"{class_name} in {split_name} has {count} bboxes; target minimum is {minimum}"
                )
    return warnings


def write_split_files(meta_dir: Path, state: SplitState) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        list_path = meta_dir / f"{split_name}.txt"
        with open(list_path, "w", encoding="utf-8") as file:
            for stem in sorted(state.stems[split_name]):
                file.write(f"{stem}\n")
        print(f"[Split] wrote {list_path} ({len(state.stems[split_name])} samples)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split converted detection data into train/val/test by session"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent / "data" / "sanpo_obstacle_8class"
        ),
        help="Converted dataset root",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=500)
    parser.add_argument("--min-val-bboxes", type=int, default=50)
    parser.add_argument("--min-test-bboxes", type=int, default=50)
    parser.add_argument(
        "--limit-stems",
        type=int,
        default=0,
        help="Debug mode: limit stems after preserving whole sessions",
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        print("[ERROR] ratios must satisfy train > 0, val > 0, train + val < 1")
        return 1

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] data directory not found: {data_dir}")
        return 1

    t_start = time.time()
    class_names = read_classes(data_dir)
    image_stems, paired_stems, missing_labels = scan_stems(data_dir)
    stems_to_use = paired_stems

    if args.limit_stems > 0:
        grouped: dict[str, list[str]] = {}
        for stem in paired_stems:
            grouped.setdefault(extract_session_id(stem), []).append(stem)
        limited: list[str] = []
        for session_id in sorted(grouped):
            limited.extend(sorted(grouped[session_id]))
            if len(limited) >= args.limit_stems:
                break
        stems_to_use = limited
        print(f"[Split] debug mode: using {len(stems_to_use)} stems")

    duplicate_stems = len(stems_to_use) - len(set(stems_to_use))
    sessions = collect_sessions(data_dir, stems_to_use, len(class_names))
    total_class_counts = [
        sum(session.class_counts[class_id] for session in sessions)
        for class_id in range(len(class_names))
    ]

    print(
        f"[Split] images={len(image_stems)}, labels={len(paired_stems)}, "
        f"sessions={len(sessions)}, missing_labels={missing_labels}"
    )
    print(f"[Split] searching {args.search_trials} split candidates...")

    candidate = choose_best_split(
        sessions=sessions,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        search_trials=args.search_trials,
        min_val_bboxes=args.min_val_bboxes,
        min_test_bboxes=args.min_test_bboxes,
    )
    state = candidate.state
    leak_passed, leak_details = verify_no_leak(state)

    warnings = class_coverage_warnings(
        class_names=class_names,
        total_class_counts=total_class_counts,
        split_class_counts=state.class_counts,
        min_val_bboxes=args.min_val_bboxes,
        min_test_bboxes=args.min_test_bboxes,
    )

    meta_dir = data_dir / "meta"
    write_split_files(meta_dir, state)

    elapsed = time.time() - t_start
    report = {
        "data_dir": str(data_dir),
        "strategy": "session_stratified_search",
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
        "seed": args.seed,
        "selected_seed": candidate.seed,
        "search_trials": args.search_trials,
        "score": round(candidate.score, 4),
        "min_val_bboxes": args.min_val_bboxes,
        "min_test_bboxes": args.min_test_bboxes,
        "total_images": len(image_stems),
        "total_labels": len(paired_stems),
        "total_sessions": len(sessions),
        "train_sessions": len(state.sessions["train"]),
        "val_sessions": len(state.sessions["val"]),
        "test_sessions": len(state.sessions["test"]),
        "train_images": len(state.stems["train"]),
        "val_images": len(state.stems["val"]),
        "test_images": len(state.stems["test"]),
        "missing_label_files": missing_labels,
        "duplicate_stems": duplicate_stems,
        "leak_check_passed": leak_passed,
        "leak_details": leak_details,
        "class_counts": {
            class_names[idx]: total_class_counts[idx]
            for idx in range(len(class_names))
        },
        "split_class_counts": {
            split_name: {
                class_names[idx]: state.class_counts[split_name][idx]
                for idx in range(len(class_names))
            }
            for split_name in SPLIT_NAMES
        },
        "warnings": warnings,
        "elapsed_seconds": round(elapsed, 1),
    }
    report_path = meta_dir / "split_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"[Split] complete in {elapsed:.1f}s, selected_seed={candidate.seed}, score={candidate.score:.4f}")
    for split_name in SPLIT_NAMES:
        print(
            f"[Split] {split_name}: {len(state.sessions[split_name])} sessions, "
            f"{len(state.stems[split_name])} images"
        )
    print(f"[Split] leak check: {'PASS' if leak_passed else 'FAIL'}")
    for warning in warnings:
        print(f"[WARN] {warning}")
    print(f"[Split] wrote {report_path}")
    return 0 if leak_passed else 1


if __name__ == "__main__":
    sys.exit(main())
