from cv2 import cv2

# Video file
video = cv2.VideoCapture("")

# For using webcam
# webcam = cv2.VideoCapture(0)

# Pre-trained car and pedestrain classifier
car_tracker_file = "car-detector.xml"
pedestrain_tracker_file = "pedestrians-detector.xml"

# Create classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrain_tracker = cv2.CascadeClassifier(pedestrain_tracker_file)


# Run forever until the car stops
while True:

    # Read the current frame
    read_successful, frame = video.read()

    if read_successful:
        # Must convert to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars and pedestrains
    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrains = pedestrain_tracker.detectMultiScale(grayscale_frame)

    # Draw rectangle around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)

    # Draw rectangle around the pedestrains
    for (x, y, w, h) in pedestrains:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)

    # Display the image
    cv2.imshow('Car Detection', frame)
    key = cv2.waitKey(1)

    # Stop if G is pressed
    if key == 71 or key == 103:
        break


# Release the video capture object
video.release()

print("Code Completed")
