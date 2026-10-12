# Single command, which works for arbitrary resolution
Note: required `ffmpeg`

Crop to square (works with any resolution):
```
ffmpeg -i input.mp4 -vf "crop='min(iw\,ih)':'min(iw\,ih)'" -c:v libx264 -crf 17 -preset fast -c:a copy intermediate_cropped.mp4
```

(optional) Resize to 640x640:
```
ffmpeg -i intermediate_cropped.mp4 -vf "scale=640:640" -c:v libx264 -crf 23 -preset medium -c:a copy output.mp4
```