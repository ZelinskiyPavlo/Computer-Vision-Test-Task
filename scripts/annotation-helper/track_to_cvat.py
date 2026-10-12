"""
Pre-annotate a video with tracks for CVAT.
 
detector + ByteTrack/BoT-SORT (Ultralytics)  ->  CVAT for video 1.1 XML
 
Only keyframes are written. CVAT linearly interpolates a track between
consecutive keyframes, so a short detector miss (<= --max-gap frames) shows up
in CVAT as an interpolated box you can inspect/fix. A longer miss is closed
with an outside=1 keyframe, so the track is hidden until the vehicle is
detected again.
 
Frame step: tracking runs on every video frame, then only the frames that
exist in the CVAT task (start, start+step, ...) are written, numbered as
absolute video frames - the numbering CVAT expects on import. Pass the same
--frame-step / --start-frame / --stop-frame you used when creating the task.
 
Usage (COCO-pretrained model; classes 2/3/5/7 = car/motorcycle/bus/truck):
    python track_to_cvat.py highway.mp4 --model yolo11x.pt --classes 2 3 5 7 --frame-step 2
Then in CVAT: task -> Actions -> Upload annotations -> "CVAT 1.1" -> annotations.xml
"""
 
import argparse
from collections import defaultdict
from functools import partial
from xml.sax.saxutils import quoteattr
 
import numpy as np
 
 
def nested_keep(xyxy, conf, thr):
    """Greedy suppression on intersection-over-smaller-box instead of IoU.
 
    IoU-based NMS misses nested boxes: a box covering 40% of a bigger one has
    IoU 0.4 with it, below the usual 0.5-0.7 thresholds. Here a box is dropped
    when more than `thr` of the smaller of the two lies inside the other and
    the other has higher confidence. Returns sorted indices to keep.
    """
    order = np.argsort(-conf)
    b = xyxy[order]
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    iw = np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None)
    cover = iw * ih / np.maximum(np.minimum(area[:, None], area[None, :]), 1e-9)
    suppressed = np.zeros(len(b), dtype=bool)
    keep = []
    for i in range(len(b)):
        if not suppressed[i]:
            keep.append(order[i])
            suppressed |= cover[i] > thr
    return np.sort(np.array(keep, dtype=int))
 
 
def suppress_nested(predictor, thr):
    """Ultralytics callback: filter each frame's detections before the tracker sees them."""
    import torch
 
    for r in predictor.results:
        if r.boxes is None or len(r.boxes) < 2:
            continue
        keep = nested_keep(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), thr)
        if len(keep) == len(r.boxes):
            continue
        idx = torch.as_tensor(keep, device=r.boxes.data.device)
        feats = getattr(r, "feats", None)  # ReID features (BoT-SORT with_reid) must stay aligned
        r.update(boxes=r.boxes.data[idx])
        if feats is not None:
            r.feats = feats[idx]
 
 
def run_tracker(args):
    from ultralytics import YOLO
 
    model = YOLO(args.model)
    classes = [c for c in (args.classes or []) if c in model.names]
    if args.classes and not classes:
        # e.g. default COCO ids 2/3/5/7 with a fine-tuned single-class teacher
        print(f"--classes {args.classes} not in model {model.names}; keeping all classes")
    if args.nested_thr < 1:
        # Callbacks run in registration order; model.track() appends the tracker's own
        # on_predict_postprocess_end callback after this one, so filtering happens first.
        model.add_callback("on_predict_postprocess_end", partial(suppress_nested, thr=args.nested_thr))
    tracks = defaultdict(list)  # track_id -> [(frame, x1, y1, x2, y2), ...]
    n_frames = 0
    # Track every video frame (no vid_stride): half the motion between frames makes
    # association easier, and vid_stride=2 would decode frames 1,3,5,... not 0,2,4,...
    stream = model.track(
        source=args.video,
        stream=True,
        tracker=args.tracker,
        imgsz=args.imgsz,
        conf=args.conf,  # keep low: ByteTrack's 2nd association stage needs low-score boxes
        iou=args.iou,  # NMS IoU threshold
        classes=classes or None,  # None = keep all classes
        # NMS is per-class by default, so one car can survive as both "car" and "truck"
        # and spawn two tracks; class-agnostic NMS keeps only the best box per object
        agnostic_nms=not args.per_class_nms,
        verbose=False,
    )
    cbs = model.callbacks["on_predict_postprocess_end"]
    mine = [i for i, cb in enumerate(cbs) if getattr(cb, "func", None) is suppress_nested]
    trk = [i for i, cb in enumerate(cbs) if getattr(getattr(cb, "func", None), "__module__", "") == "ultralytics.trackers.track"]
    if mine and trk and mine[0] > trk[0]:
        raise RuntimeError("nested-box filter would run after the tracker; check Ultralytics callback order")
    for frame, r in enumerate(stream):
        n_frames = frame + 1
        res = r.boxes
        if res is None or res.id is None:
            continue
        h, w = r.orig_shape
        for tid, (x1, y1, x2, y2) in zip(res.id.int().tolist(), res.xyxy.tolist()):
            x1, x2 = max(0.0, x1), min(float(w), x2)
            y1, y2 = max(0.0, y1), min(float(h), y2)
            if x2 > x1 and y2 > y1:
                tracks[tid].append((frame, x1, y1, x2, y2))
    return tracks, n_frames
 
 
def to_task_frames(tracks, n_frames, start, step, stop=None):
    """Keep only the video frames present in the CVAT task, renumbered 0..n_task-1."""
    stop = n_frames - 1 if stop is None else min(stop, n_frames - 1)
    n_task = (stop - start) // step + 1
    out = {}
    for tid, dets in tracks.items():
        kept = [((f - start) // step, *box) for f, *box in dets if start <= f <= stop and (f - start) % step == 0]
        if kept:
            out[tid] = kept
    return out, n_task
 
 
def split_segments(dets, max_gap):
    """Split one track's detections where more than max_gap frames are missing."""
    seg = [dets[0]]
    for d in dets[1:]:
        if d[0] - seg[-1][0] - 1 > max_gap:
            yield seg
            seg = []
        seg.append(d)
    yield seg
 
 
def box_xml(frame, x1, y1, x2, y2, outside=0):
    return (
        f'    <box frame="{frame}" keyframe="1" outside="{outside}" occluded="0" '
        f'xtl="{x1:.2f}" ytl="{y1:.2f}" xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0"/>'
    )
 
 
def write_cvat_xml(tracks, n_frames, path, label, max_gap, min_len, kf_step, to_abs=lambda f: f):
    """tracks/n_frames/max_gap/min_len are in task frames; to_abs maps them to video frame numbers."""
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>"]
    out_id = 0
    for dets in tracks.values():
        if len(dets) < min_len:  # short-lived IDs are usually flicker / false positives
            continue
        dets = sorted(dets)
        lines.append(f'  <track id="{out_id}" label={quoteattr(label)} source="auto">')
        for seg in split_segments(dets, max_gap):
            keys = seg[::kf_step]
            if keys[-1] is not seg[-1]:
                keys.append(seg[-1])
            lines.extend(box_xml(to_abs(d[0]), *d[1:]) for d in keys)
            last = seg[-1]
            if last[0] + 1 < n_frames:  # without this the track would persist to the end
                lines.append(box_xml(to_abs(last[0] + 1), *last[1:], outside=1))
        lines.append("  </track>")
        out_id += 1
    lines.append("</annotations>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_id
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11x.pt", help="COCO-pretrained detection weights")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml, botsort.yaml or a custom yaml")
    ap.add_argument("--imgsz", type=int, default=1280, help="go higher (e.g. 1920) if small cars are missed")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--classes", type=int, nargs="*", default=[2, 3, 5, 7],
                    help="COCO ids: 2 car, 3 motorcycle, 5 bus, 7 truck; pass nothing to keep all")
    ap.add_argument("--per-class-nms", action="store_true", help="disable class-agnostic NMS")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (Ultralytics default 0.7); lower = stricter")
    ap.add_argument("--nested-thr", type=float, default=0.8,
                    help="drop a box if more than this share of the smaller box overlaps a more confident box; 1 = off")
    ap.add_argument("--label", default="vehicle", help="must match the label name in the CVAT task")
    ap.add_argument("--frame-step", type=int, default=1, help="CVAT task frame step")
    ap.add_argument("--start-frame", type=int, default=0, help="CVAT task start frame")
    ap.add_argument("--stop-frame", type=int, default=None, help="CVAT task stop frame (default: last)")
    ap.add_argument("--max-gap", type=int, default=15, help="missing task frames CVAT should interpolate over")
    ap.add_argument("--min-len", type=int, default=5, help="drop tracks detected in fewer task frames")
    ap.add_argument("--kf-step", type=int, default=1, help="keep every N-th detection as a keyframe")
    ap.add_argument("--out", default="annotations.xml")
    args = ap.parse_args()
 
    tracks, n_frames = run_tracker(args)
    start, step = args.start_frame, args.frame_step
    tracks, n_task = to_task_frames(tracks, n_frames, start, step, args.stop_frame)
    n = write_cvat_xml(tracks, n_task, args.out, args.label, args.max_gap, args.min_len, args.kf_step,
                       to_abs=lambda f: start + f * step)
    print(f"{n} tracks over {n_task} task frames ({n_frames} video frames) -> {args.out}")
 
 
if __name__ == "__main__":
    main()
 