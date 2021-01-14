# SORT-Live-Streaming
This is MOT task with live streaming.

## Requirements
### Python Packages
* Flask
* pytorch
* imutils
* filterpy

### Others
* ffmpeg

## Main Files
* SORT-tracking reference (https://github.com/cfotache/pytorch_objectdetecttrack)
* object_tracking.py : load camera and track objects
* models.py: create module and YOLOv3 model
* sort.py: detect and tracking

## Usages
* camera tracking on web
```python object_tracker.py --ip 0.0.0.0 --port 8000```
* Access web [127.0.0.1:8000](127.0.0.1:8000).
