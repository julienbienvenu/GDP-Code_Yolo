import cv2
import numpy as np
import deep_sort
import os

# Load the video
video_path = "path/to/your/video.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize the Deep SORT tracker
model_filename = "deep_sort/mars-small128.pb"
encoder = deep_sort.build_tracker(model_filename, depth=16)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run object detection on the frame
    # Replace this with your favorite object detection method
    boxes, scores, classes, num = run_detection(frame)

    # Convert the detections to a numpy array
    detections = np.array(
        [
            [x, y, x + w, y + h, s]
            for (x, y, w, h, s) in zip(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores)
        ]
    )

    # Update the tracker
    tracker = encoder.update(frame, detections)

    # Draw the tracking boxes on the frame
    for track in tracker:
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2)
    
    # Display the updated frame
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
