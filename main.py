import os
import argparse
import random
import uuid
import multiprocessing

import textwrap
import numpy as np
import cv2
import requests
from ultralytics import YOLO

from tracker import Tracker
from deep_sort.deep_sort.track import Track
import base64


def process_track(track: Track, frame: np.ndarray):
  x, y, w, h = track.to_tlwh()
  if frame is not None and frame.shape[0] != 0 and frame.shape[1] != 0:
    crop = frame[int(y):int(y+h), int(x):int(x+w)]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
      return

    _, encoded_image = cv2.imencode('.jpg', crop)
    crop_base64 = base64.b64encode(encoded_image).decode()

    try:
      endpoint_url = f'{caption_url}:{caption_url_port}/predictions/expansion-net-v2/'
      payload = {'image': crop_base64}
      response = requests.post(endpoint_url, json=payload)
      
      if response.status_code == 200:
        print('image sent successfully')
        caption = response.content.decode()
        print('Caption', caption)
        wrapped_text = textwrap.wrap(caption, width=35)
        if not os.path.exists(out_dir):
          os.makedirs(out_dir)

        uuid_str = str(uuid.uuid4())
        os.makedirs(os.path.join(out_dir, uuid_str), exist_ok=True)
        
        crop_path = os.path.join(out_dir, uuid_str, f'image.jpg')
        cv2.imwrite(crop_path, crop)

        y = 0

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (255, 0, 0)
        (_, line_height), _ = cv2.getTextSize('A', font, font_scale, thickness)
        
        for line in wrapped_text:
          y += line_height + 10
        
        crop = cv2.putText(crop, caption, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Image', crop)

        caption_path = os.path.join(out_dir, uuid_str, f'caption.txt')
        with open(caption_path, 'w') as f:
          f.write(caption)
      else:
        print('Failed to send crop. status code:', response.status_code)
    except Exception as e:
      print('Failed to send crop:', e)
def on_delete(track: Track, frame: np.ndarray):
  #p = multiprocessing.Process(target=process_track, args=(track, frame))
  #p.start()
  process_track(track, frame)
  
def main(source: str):
  cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
  if not cap.isOpened():
    print("Failed to open video source")
    return
  
  model = YOLO("yolov8n.pt")
  model.classes = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17]
  classes = model.names

  tracker = Tracker()
  colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

  detection_threshold = 0.75

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
         
        if score > detection_threshold and class_id in model.classes:
          detections.append([x1, y1, x2, y2, score, class_id])

      tracker.update(frame, detections, on_delete=on_delete)
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

    cv2.imshow('Footage', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    ret, frame = cap.read()

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  global out_dir
  global caption_url
  global caption_url_port

  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=str, default='./data/Test2.mp4', help='source')
  parser.add_argument('--output', type=str, default='./output', help='output')
  parser.add_argument('--caption_url', type=str, default='http://0.0.0.0')
  parser.add_argument('--port', type=int, default=8080)

  args = parser.parse_args()
  out_dir = args.output
  caption_url = args.caption_url if args.caption_url.startswith('http://') or args.caption_url.startswith('https://') else f'http://{args.caption_url}'
  caption_url_port = args.port

  main(args.source)
