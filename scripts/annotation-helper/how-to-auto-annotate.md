# How to auto-annotate videos for CVAT

## 1. Pre-annotate with a default COCO model

```bash
python track_to_cvat.py highway.mp4 --model yolo11x.pt --frame-step 2 --out annotations.xml
```

- `--classes` defaults to COCO `2 3 5 7` (car, motorcycle, bus, truck).
- `--frame-step` / `--start-frame` / `--stop-frame` must match the settings of the CVAT task.
- `--label` must match the label name in the CVAT task (default `vehicle`).
- If small/far cars are missed, raise `--imgsz` (default 1280).

In CVAT: **task → Actions → Upload annotations → "CVAT 1.1" → `annotations.xml`**, then fix the
boxes by hand.

## 2. Fine-tune the model on clips you have already labeled

1. Export each labeled CVAT task as **"Ultralytics YOLO Detection 1.0"** with **Save images** on,
   then unzip each export into its own folder.
2. Build the dataset. Hold out a whole clip for validation; a random frame split leaks
   near-duplicate frames into val.

   ```bash
   python prepare_teacher_data.py task_5 task_6 task_7 --val task_6 --step 5 --out teacher_data
   ```

   This keeps every 5th frame, forces a single class `car`, clips boxes to the image and writes
   `teacher_data/data.yaml`.
3. Train:

   ```bash
   yolo detect train model=yolo11x.pt data=teacher_data/data.yaml imgsz=1080 epochs=60 flipud=0.5
   ```

   Weights are saved to `runs/detect/train*/weights/best.pt`.

## 3. Pre-annotate new videos with the fine-tuned model

```bash
python track_to_cvat.py new_clip.mp4 --model runs/detect/train/weights/best.pt --classes --frame-step 2
```

`--classes` with no values keeps all classes, which is what you want for the single-class
fine-tuned model. Upload the XML to CVAT the same way as in step 1.
