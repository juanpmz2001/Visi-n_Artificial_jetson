from importlib.resources import path
from time import time
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

#Sólo es necesario cambiar el path del modelo entrenado 
model = torch.hub.load('ultralytics/yolov5', 'custom', path = r'./trained_model_yolom_crop/content/yolov5/runs/train/exp5/weights/best.pt', force_reload=True)

cap = cv2.VideoCapture(0)
while cap .isOpened():
    start = time()
    ret,frame = cap.read()
    result = model(frame)
    cv2.imshow('screen', np.squeeze(result.render()))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('screen', cv2.WND_PROP_VISIBLE) < 1:
        break
    end = time()
    fps = 1/(end-start)
    print(fps)
cap.release()
cv2.destroyAllWindows()