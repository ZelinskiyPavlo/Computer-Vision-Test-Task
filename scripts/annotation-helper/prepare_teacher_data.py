"""
Merge CVAT "Ultralytics YOLO Detection 1.0" exports (one per clip, exported
with "Save images") into one dataset for fine-tuning the labeling teacher.
 
- Keeps each clip in its own folder: CVAT names frames frame_000000.jpg in
  every task, so merging into one folder would overwrite files.
- Splits train/val by clip, not by frame: neighbouring frames are near-copies,
  so a random frame split puts the "same" image in train and val.
- Keeps every --step-th frame to cut that redundancy.
- Forces class 0 and clips boxes to the image: Ultralytics drops the whole
  image if any label value is outside [-0.01, 1.01].
 
Usage:
    python prepare_teacher_data.py clip1_export clip2_export clip3_export \
        --val clip3_export --step 5 --out teacher_data
    yolo detect train model=yolo11x.pt data=teacher_data/data.yaml imgsz=1280 flipud=0.5
"""
 
import argparse
import shutil
from pathlib import Path
 
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
def clip_label_line(line):
    parts = line.split()
    if len(parts) != 5:
        return None
    cx, cy, w, h = map(float, parts[1:])
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:  # box entirely outside the frame
        return None
    return f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
 
 
def convert_clip(src, dst, step):
    images = sorted(p for p in (src / "images").rglob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"{src}: no images found - export with 'Save images' enabled")
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    kept = boxes = dropped = 0
    for img in images[::step]:
        label = src / "labels" / img.relative_to(src / "images").with_suffix(".txt")
        raw = label.read_text().splitlines() if label.exists() else []  # missing = no vehicles
        lines = [clip_label_line(l) for l in raw if l.strip()]
        dropped += sum(l is None for l in lines)
        lines = [l for l in lines if l is not None]
        shutil.copy2(img, dst / "images" / img.name)
        (dst / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        kept += 1
        boxes += len(lines)
    print(f"{src.name}: {kept}/{len(images)} frames, {boxes} boxes, {dropped} boxes dropped")
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path, help="unzipped CVAT export folders, one per clip")
    ap.add_argument("--val", nargs="*", type=Path, default=[], help="clip folder(s) to hold out for validation")
    ap.add_argument("--step", type=int, default=5, help="keep every N-th frame")
    ap.add_argument("--name", default="car", help="class name; 'car' lets Ultralytics reuse COCO's car head row")
    ap.add_argument("--out", type=Path, default=Path("teacher_data"))
    args = ap.parse_args()
 
    val = {p.resolve() for p in args.val}
    train_dirs, val_dirs = [], []
    for i, clip in enumerate(args.clips):
        name = f"clip{i}_{clip.name}"
        convert_clip(clip, args.out / name, args.step)
        (val_dirs if clip.resolve() in val else train_dirs).append(f"{name}/images")
    if not val_dirs:  # Ultralytics requires a val set; fall back to the last clip
        val_dirs.append(train_dirs.pop())
        print(f"no --val given, holding out {val_dirs[0]}")
    if not train_dirs:
        raise SystemExit("all clips ended up in val - leave at least one for training")
 
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        f"train: [{', '.join(train_dirs)}]\n"
        f"val: [{', '.join(val_dirs)}]\n"
        f"names:\n  0: {args.name}\n"
    )
    print(f"wrote {args.out / 'data.yaml'}")
 
 
if __name__ == "__main__":
    main()