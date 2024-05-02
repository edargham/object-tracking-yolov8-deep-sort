import os
import random

import cv2
from ultralytics import YOLO

from tracker import Tracker

video_path = os.path.join('.', 'data', 'Test.mp4')
# video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# cap_out = cv2.VideoWriter(
#   video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), 
#   cap.get(cv2.CAP_PROP_FPS),
#   (frame.shape[1], frame.shape[0])
# )

model = YOLO("yolov8s.pt")
classes = model.names

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5

ret, frame = cap.read()
while ret:

  results = model(frame)
  for result in results:
    detections = []
    for r in result.boxes.data.tolist():
      x1, y1, x2, y2, score, class_id = r
      x1 = int(x1)
      x2 = int(x2)
      y1 = int(y1)
      y2 = int(y2)
      class_id = int(class_id)
      if score > detection_threshold:
        detections.append([x1, y1, x2, y2, score, class_id])

    tracker.update(frame, detections)
    if tracker.tracks is not None:
      for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        
        track_id = track.track_id
        class_id = track.class_id

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

        cv2.rectangle(frame, (int(x1-2), int(y1 - 15)), (int(x1 + 150), int(y1)), (colors[track_id % len(colors)]), -1)
        cv2.putText(frame, f'id: {track_id}', (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'class: {classes[class_id]}', (int(x1 + 60), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

  # cap_out.write(frame)
  cv2.imshow('Footage', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  ret, frame = cap.read()

cap.release()
# cap_out.release()
cv2.destroyAllWindows()
