# Cricket Ball Detection & Tracking

## Overview

This project focuses on detecting the cricket ball’s centroid in each frame of a video and tracking its trajectory. All videos are recorded from a single, fixed camera. The repository includes all code, outputs, annotations, and documentation required for fully reproducible inference.

---

## Repository Structure

```
yolo_assignment/  
├─ 25_nov_2025/  
├─ annotations/  
├─ code/  
│  ├─ main.py  
│  ├─ train.ipynb 
│  └─ utils/  
│     ├─ detector.py  
│     └─ draw_utils.py  
├─ results/  
├─ bash           (bash.sh)  
├─ README         (README.md)  
├─ requirements   (requirements.txt)  
└─ 
```

---

## Output Formats

### CSV Annotation Format

Each CSV contains one row per frame:

```
frame_index,x,y,visible
0,512.3,298.1,1
1,518.7,305.4,1
2,-1,-1,0
```

* **frame_index**: Frame index
* **x, y**: Detected centroid coordinates
* **visible**: 1 = ball detected, 0 = not visible

Outputs are saved to: `annotations/`

### Processed Videos

Each output video contains:

* Detected centroid on every frame
* Trajectory overlay showing movement history

Outputs are saved to: `results/`

---

## Quick Reproduction

From the project root (`yolo_assignment/`), run:

```
bash bash.sh
```

This script automatically:

* Runs `code/main.py` for each input video
* Saves processed videos to `results/`
* Saves CSV annotations to `annotations/`

All outputs included in this repository were generated using this script. Re-run only if needed.

