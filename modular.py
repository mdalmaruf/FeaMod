import cv2
import numpy as np

# Rule 1: Reusability
def load_model():
    return cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Rule 1: Reusability
def load_image(image_path):
    return cv2.imread(image_path)

# Rule 2: Cohesion
def process_image(image):
    return cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Rule 2: Cohesion
def detect_objects(model, blob):
    model.setInput(blob)
    return model.forward()

# Rule 4: Conditional Segmentation and Rule 5: Loop Abstraction
def get_confidences_and_boxes(detections):
    confidences = []
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            confidences.append(confidence)
            boxes.append(box.astype("int"))
    return confidences, boxes

# Rule 2: Cohesion
def calculate_mean_confidence(confidences):
    return np.mean(confidences)

# Rule 7: Error Handling
def display_mean_confidence(image_path, mean_confidence):
    try:
        print(f"Mean confidence for image {image_path}: {mean_confidence}")
    except Exception as e:
        print(f"Error: {e}")

# Rule 2: Cohesion
def display_detections(image, boxes, confidences):
    for i in range(len(boxes)):
        (startX, startY, endX, endY) = boxes[i]
        text = "{:.2f}%".format(confidences[i] * 100)
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Output", image)
    cv2.waitKey(0)

# Rule 6: Loose Coupling
def main(image_path):
    model = load_model()
    image = load_image(image_path)
    blob = process_image(image)
    detections = detect_objects(model, blob)
    confidences, boxes = get_confidences_and_boxes(detections)
    mean_confidence = calculate_mean_confidence(confidences)
    display_detections(image, boxes, confidences)
    display_mean_confidence(image_path, mean_confidence)

main("input.jpg")