"""Merge one split (train or val) of several split_dataset.py outputs into a single YOLO dataset.

Every CVAT export uses the same filenames (frame_000000.png, ...), so files are renamed to
<task>_<stem> (task_5_frame_000200.png) to avoid collisions.
"""

import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
MERGED_DIR = ROOT / "input" / "train"


def load_names(input_dir: Path) -> dict:
    return yaml.safe_load((input_dir / "data.yaml").read_text())["names"]


def merge_split(split: str, inputs: list[Path], out_dir: Path = MERGED_DIR) -> Path:
    """Copy <input>/<split>/{images,labels} of every input into <out_dir>. Returns the data.yaml path."""
    names = load_names(inputs[0])
    for input_dir in inputs[1:]:
        if load_names(input_dir) != names:
            raise ValueError(f"Class names differ: {inputs[0]} {names} vs {input_dir} {load_names(input_dir)}")

    out_images = out_dir / "images" / split
    out_labels = out_dir / "labels" / split
    for d in (out_images, out_labels):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    total_images = total_labels = 0
    for input_dir in inputs:
        images = sorted((input_dir / split / "images").glob("*"))
        labels = sorted((input_dir / split / "labels").glob("*.txt"))
        for src, dst_dir in [(p, out_images) for p in images] + [(p, out_labels) for p in labels]:
            shutil.copy2(src, dst_dir / f"{input_dir.name}_{src.name}")
        print(f"  {input_dir.name}: {len(images)} images, {len(labels)} labels")
        total_images += len(images)
        total_labels += len(labels)
    print(f"{split}: {total_images} images, {total_labels} labels -> {out_images.parent.parent}")
    if total_images != total_labels:
        print("WARNING: image and label counts differ")

    data_yaml = out_dir / "data.yaml"
    # No "path" key: Ultralytics then uses the folder of data.yaml as the dataset root, so the
    # dataset can be moved or zipped to another machine. A relative "path" would resolve against the cwd.
    data = {
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
    return data_yaml