import cv2
import numpy as np

# Load model
model = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Load image
image_path = "input.jpg"
image = cv2.imread(image_path)

# Process image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Detect objects
model.setInput(blob)
detections = model.forward()

# Display detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        text = "{:.2f}%".format(confidence * 100)
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

# Calculate mean confidence
mean_confidence = np.mean([detections[0, 0, i, 2] for i in range(detections.shape[2])])

try:
    # Display mean confidence
    print(f"Mean confidence for image {image_path}: {mean_confidence}")
except Exception as e:
    print(f"Error: {e}")

cv2.imshow("Output", image)
cv2.waitKey(0)