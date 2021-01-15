# SORT-Live-Streaming
This is MOT task with live streaming.

## Workflow
### Live STreaming and Tracking
* Set up the server and camera live streaming.
* Use SORT model to do multiple objects tracking and save frame as jpg.
* Each 50 frames do transcode:
  * Collect all frames and output .mp4 through FFMPEG.
  * Transcode .mp4 to .ts and .m3u8 through FFMPEG.
  * Remove the last line (#EXT-X-ENDLIST) in .m3u8 file to ensure continuous streaming.

### Specific Object Tracking
* Use XMLHttpRequest to GET all bbox list from server and show on html selectionbox.
* Use XMLHttpRequest to POST the submit from user selection to server and display the specific bbox.

## Requirements
### Python Packages
* Flask
* pytorch
* imutils
* filterpy

### Others
* ffmpeg

## Main Files
* [SORT-tracking](https://github.com/cfotache/pytorch_objectdetecttrack)
* [Flask and JPG stream](https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/)
* object_tracking.py : setup server, load camera, and display tracking result
* models.py: create module and YOLOv3 model
* sort.py: detect and tracking

## Usages
* camera tracking on web
```python object_tracker.py --ip 0.0.0.0 --port 8000```
* Access web [127.0.0.1:8000](127.0.0.1:8000).
* Select specific bbox id from selectionbox.
