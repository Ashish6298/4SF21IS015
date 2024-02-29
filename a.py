import cv2

# Load the pre-trained EAST text detection model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# Load the input image
image = cv2.imread("ana.jpg")
orig = image.copy()
(H, W) = image.shape[:2]

# Pre-process the image
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

# Set the blob as input to the network and forward pass
net.setInput(blob)
(scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

# Define the minimum confidence score to filter weak text detections
min_confidence = 0.5

# Loop over the detections
for i in range(0, scores.shape[2]):
    # Extract the confidence score and geometrical data for the text region
    score = scores[0, 0, i, 0]
    if score < min_confidence:
        continue
    box = geometry[0, 0, i, :].flatten()
    (startX, startY, endX, endY) = box.astype("int")

    # Scale the bounding box coordinates based on the image dimensions
    startX = int(startX * W)
    startY = int(startY * H)
    endX = int(endX * W)
    endY = int(endY * H)

    # Draw the bounding box on the image
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Display the output image with text detections
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
