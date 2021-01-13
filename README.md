# SORT-Live-Streaming
This is MOT task with live streaming.

## HW4 Grading
1. Base Model (40%): video/audio live steaming
2. Function 1 (30%): video live streaming with object tracking
3. Function 2 (10%): Providing a user interface to specify which object to track
4. Report (20%)
5. Bonus (10%): supporting multiple bitrates OR trick mode (users can perform
pause/resume/rewind)

## TODO
To create live streaming with HTTP Live Streaming (HLS), you will need to
* Use webcam to capture/encode live video
* Use your deep learning network model to do segmentation or object tracking
* Use ffmpeg to transcode the video into .ts
* Use Stream segmenter or file segmenter to generate a series of small media files (.ts) and an index
file (.m3u8)
* To support various bitrates, you can use VariantPlaylistCreator to generate master index file
(.m3u8)
* Create HTML file to play the video (or with controls such as play, pause)
* View the results in a browser that supports HTML5

## Requirements
### Python Packages
* Flask
* pytorch


### Others

* ffmpeg

## Main Files
* SORT-tracking(https://github.com/cfotache/pytorch_objectdetecttrack)
* object_tracking.py : load camera and track objects
* models.py: create module and YOLOv3 model
* sort.py: detect and tracking

## Usages
* camera tracking on web
```python object_tracker.py --ip 0.0.0.0 --port 8000```
* Access web [127.0.0.1:8000](127.0.0.1:8000).
