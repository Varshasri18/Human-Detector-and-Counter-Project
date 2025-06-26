import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Get the names of all layers in the network
layer_names = net.getLayerNames()

# Get indices of the output layers
output_layers_indices = net.getUnconnectedOutLayers()

# Extract names of the output layers
output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_and_count(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            for obj in detection:
                # Check dimensions of obj
                if len(obj) == 85:  # 80 class scores + 5 bounding box values = 85
                    obj = np.array(obj)
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and class_id == 0:  # 0 is the class ID for "person"
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        w = int(obj[2] * width)
                        h = int(obj[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    person_count = len(indexes)

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = f"Person {i+1}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(frame, f"Total Persons: {person_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Real-time Human Detection", frame)

def start_detection(video_source=0):
    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detect_and_count(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detection()
