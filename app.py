from flask import Flask, render_template, Response
import cv2
import numpy as np
from PIL import Image, ImageDraw

app = Flask(__name__, static_url_path='/static')

# Load YOLOv3 model
yolo_net = cv2.dnn.readNet("/Volumes/My Files/Projects/Real_time_animal_detection_counting_using_CNN/yolov3.weights", "/Volumes/My Files/Projects/Real_time_animal_detection_counting_using_CNN/yolov3.cfg")

# Load class labels
yolo_classes = []
with open('/Volumes/My Files/Projects/Real_time_animal_detection_counting_using_CNN/coco.names', 'r') as f:
    yolo_classes = f.read().splitlines()

# Exclude the class label "person" if it exists
if "person" in yolo_classes:
    yolo_classes.remove("person")   

def perform_yolo_detection(img):
    height, width, _ = img.shape  # Get image dimensions

    # Preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo_net.setInput(blob)

    # Get YOLO output
    output_layers_names = yolo_net.getUnconnectedOutLayersNames()
    layer_outputs = yolo_net.forward(output_layers_names)

    # Parse YOLO output
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indexes = np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

           
    # Draw bounding boxes and labels on the image
    font = cv2.FONT_HERSHEY_SIMPLEX  # Change font type020
    thickness = 3  # Thickness of the rectangle border
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(yolo_classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        # color = tuple(int(c) for c in colors[i])
        color = (0, 0, 255)
        # Draw outer rectangle using OpenCV (unchanged)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)

        # Calculate text size for dynamic positioning (unchanged)
        (label_width, label_height), baseline = cv2.getTextSize(label + " " + confidence, font, 0.5, 2)

        # Draw inner rectangle using PIL for rounded corners
        pil_img = Image.fromarray(img)  # Convert OpenCV image to PIL image
        draw = ImageDraw.Draw(pil_img)
        draw.rounded_rectangle((x + 3, y - 20, x + label_width + 3, y + 2), radius=5, fill=color, outline=color, width=3)
        img = np.array(pil_img)  # Convert back to OpenCV image

        # Draw text using OpenCV (unchanged)
        cv2.putText(img, label + " " + confidence, (x, y - 5), font, 0.5, (255, 255, 255), 1)

    height, width, _ = img.shape  # Get image dimensions

    # Preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo_net.setInput(blob)

    # Get YOLO output
    output_layers_names = yolo_net.getUnconnectedOutLayersNames()
    layer_outputs = yolo_net.forward(output_layers_names)

    # Initialize dictionaries to store counts and coordinates of objects
    object_counts = {}
    object_coordinates = {}

    # Parse YOLO output
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.7:
                label = str(yolo_classes[class_id])

                # Calculate coordinates of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                # Store coordinates of the object
                object_coordinates.setdefault(label, []).append((x, y, w, h))

                # Count occurrences of the object class
                object_counts[label] = object_counts.get(label, 0) + 1

    # Draw bounding boxes and labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 3
    colors = np.random.uniform(0, 255, size=(len(object_counts), 3))

    for label, count in object_counts.items():
        for coords in object_coordinates[label]:
            x, y, w, h = coords
            color = (0, 0, 255)  # Red color for bounding boxes
            #cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
            #cv2.putText(img, label + " " + str(count), (x, y - 5), font, 0.5, (255, 255, 255), 1)
    for label, count in object_counts.items():
        cv2.putText(img, label + ": " + str(count), (20, 80 + 30 * list(object_counts.keys()).index(label)), font, 1, (0, 0, 255), 2)
    
    return img





def gen_frames():  
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Perform YOLO object detection on the frame
        yolo_detected_frame = perform_yolo_detection(frame)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', yolo_detected_frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame in bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)





