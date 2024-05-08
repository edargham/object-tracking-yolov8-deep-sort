# Object Tracking using YOLOv8 and Deep Sort.
Repository forked from: [computervisioneng/object-tracking-yolov8-deep-sort](https://github.com/computervisioneng/object-tracking-yolov8-deep-sort)

## Deep Sort
You can download deep sort feature extraction model [here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp).

## Docker
You can build this project as a docker container.

```bash
$ docker build . -t <image-name>
```

Run it using this command.

```bash
$ docker run --rm -it --gpus all --net=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY <image-name> --source <source-path>
```
This project is intended to complement the ExpansionNetV2 serving fork I built [here](https://github.com/edargham/ExpansionNetV2). You can send the tracked object for image captioning by following the build and run process of the other project, and here using:

```bash
docker run --rm -it --gpus all --net=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -v <host-out-dir>:/<container-out-dir> -e DISPLAY=$DISPLAY <image-name> --source <source-path> --output /<container-out-dir> --caption_url <expansion-net-serving-url> --port <expansion-net-serving-port>
```

## License
This project is licensed under the GNU General Public License (GPL) version 3.0. 

The object-tracking-yolov8-deep-sort project incorporates modifies and uses the following open source software components, which are distributed under their respective licenses:

- Yolov8 Object Detection with Ultralytics: [GitHub repository](https://github.com/ultralytics/ultralytics)
- Deep Sort Object Tracking: [GitHub repository](https://github.com/nwojke/deep_sort)

Please refer to the individual repositories for the specific license terms and conditions of each component.

You can find the full text of the GNU General Public License (GPL) version 3.0 [here](https://www.gnu.org/licenses/gpl-3.0.en.html).
