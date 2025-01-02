import numpy as np
import cv2
from PIL import Image
import streamlit as st
import torch
from ultralytics import YOLO


yolo_classes = 'yolo.txt'  # Make sure this points to your class names file

# YOLOv3 detection function
def yolo_detection_single_image_v3(image_np, yolo_classes, yolo_cfg='yolov3.cfg', yolo_weights='yolov3.weights'):
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    classes = []
    with open(yolo_classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getUnconnectedOutLayersNames()
    height, width, _ = image_np.shape
    blob = cv2.dnn.blobFromImage(image_np, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (255, 0, 0)
            cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image_np, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    return image_np


# YOLOv5 detection function
def yolo_detection_single_image_v5(image_np, yolo_weights='yolov5x.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    detections = results.pred[0]
    class_ids = detections[:, 5].int().tolist()
    confidences = detections[:, 4].tolist()
    boxes = detections[:, :4].tolist()

    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        label = model.names[class_id]
        confidence_text = f"{confidence:.2f}"
        color = (255, 0, 0)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_np, f"{label} {confidence_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image_np


# YOLOv8 detection function
def yolo_detection_single_image_v8(image_np, yolo_weights='yolov8x.pt'):
    model = YOLO(yolo_weights)  # Load the model
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = model(image_rgb)  # Run inference

    # Access the results
    detections = results[0].boxes  # Access boxes directly
    class_ids = detections.cls.int().tolist()  # Class IDs
    confidences = detections.conf.float().tolist()  # Confidence scores
    boxes = detections.xyxy.numpy().tolist()  # Bounding boxes

    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        label = model.names[class_id]  # Get class name
        confidence_text = f"{confidence:.2f}"
        color = (255, 0, 0)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_np, f"{label} {confidence_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image_np


    
# Streamlit app layout
st.title('YOLO Object Detection')
model_choice = st.selectbox("Choose YOLO version:", ["YOLOv3", "YOLOv5", "YOLOv8"])

uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)  # Convert to NumPy array

    if model_choice == "YOLOv3":
        processed_image = yolo_detection_single_image_v3(image_np, yolo_classes)
    elif model_choice == "YOLOv5":
        processed_image = yolo_detection_single_image_v5(image_np)
    elif model_choice == "YOLOv8":
        processed_image = yolo_detection_single_image_v8(image_np)

    # Display original and processed images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)

    with col2:
        st.image(processed_image, caption="Detection Result", use_column_width=True)
