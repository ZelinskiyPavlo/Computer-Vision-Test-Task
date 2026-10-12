"""Split one CVAT "Ultralytics YOLO" export (one clip) into train/val by frame ranges.

The frames are consecutive video frames, so a random split would put near-identical neighbours
into both train and val and inflate val metrics. Instead, val is one or more contiguous frame
ranges, and `gap` frame numbers on each side of every range are dropped entirely.

All numbers (range start/end, gap) are the frame numbers from the filenames (frame_000200.png ->
200), not image indices. The exports keep every 2nd frame, so `--gap 20` drops 10 images per side.
"""

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS_DIR = ROOT / "raw_data" / "splits"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
FRAME_RE = re.compile(r"frame_(\d+)")


def frame_number(path: Path) -> int:
    match = FRAME_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Can't parse a frame number from {path.name}")
    return int(match.group(1))


def parse_range(text: str) -> tuple[int, int]:
    """'200:300' -> (200, 300), inclusive on both ends."""
    try:
        start, end = (int(x) for x in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected START:END, got {text!r}")
    if start > end:
        raise argparse.ArgumentTypeError(f"start > end in {text!r}")
    return start, end


def assign(frame: int, val_ranges: list[tuple[int, int]], gap: int) -> str:
    """Return 'val', 'gap' or 'train' for one frame number. Val wins over another range's gap."""
    if any(start <= frame <= end for start, end in val_ranges):
        return "val"
    if any(start - gap <= frame < start or end < frame <= end + gap for start, end in val_ranges):
        return "gap"
    return "train"


def span(frames: list[int]) -> str:
    return f"{min(frames)}..{max(frames)}" if frames else "-"


def split_task(task_dir: Path, val_ranges: list[tuple[int, int]], gap: int, out_root: Path = SPLITS_DIR) -> Path:
    """Copy one export into <out_root>/<task>/{train,val}/{images,labels}. Returns the task output dir."""
    images_dir = task_dir / "images" / "train"
    labels_dir = task_dir / "labels" / "train"
    images = sorted(
        (p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=frame_number,
    )
    image_stems = {p.stem for p in images}
    orphan_labels = sorted(p.name for p in labels_dir.glob("*.txt") if p.stem not in image_stems)

    frames = [frame_number(p) for p in images]
    for start, end in val_ranges:
        if start < frames[0] or end > frames[-1]:
            print(f"WARNING: val range {start}:{end} goes outside the clip span {frames[0]}..{frames[-1]}")
        if not any(start <= f <= end for f in frames):
            print(f"WARNING: val range {start}:{end} matches no frames")

    # Wipe old output so a re-run with different ranges can't leave stale files behind.
    out_dir = out_root / task_dir.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True)
        (out_dir / split / "labels").mkdir(parents=True)

    assigned: dict[str, list[int]] = {"train": [], "gap": [], "val": []}
    background = []
    for img, frame in zip(images, frames):
        split = assign(frame, val_ranges, gap)
        assigned[split].append(frame)
        if split == "gap":
            continue
        shutil.copy2(img, out_dir / split / "images" / img.name)
        label = labels_dir / f"{img.stem}.txt"
        out_label = out_dir / split / "labels" / label.name
        if label.exists():
            shutil.copy2(label, out_label)
        else:
            # No label file: keep the image as a background (no objects) example.
            out_label.write_text("")
            background.append(img.stem)

    shutil.copy2(task_dir / "data.yaml", out_dir / "data.yaml")

    print(f"{task_dir.name}: {len(images)} images, val ranges {val_ranges or 'none'}, gap {gap}")
    for name in ("train", "gap", "val"):
        note = " (dropped)" if name == "gap" else ""
        print(f"  {name:5}: {len(assigned[name]):4} images, frames {span(assigned[name])}{note}")
    if background:
        print(f"  background (no label file, written empty): {', '.join(background)}")
    if orphan_labels:
        print(f"  WARNING: labels without an image, skipped: {', '.join(orphan_labels)}")
    print(f"  -> {out_dir}")
    return out_dir