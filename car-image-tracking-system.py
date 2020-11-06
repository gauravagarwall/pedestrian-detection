from cv2 import cv2

# Img file
# img_file = "Address of file"

# Pre-trained car classifier
classifier_file = "car-detector.xml"

# Create an opencv image
img = cv2.imread(img_file)


# Convert to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect cars
cars = car_tracker.detectMultiScale(grayscale_img)

# print(cars)

# Draw rectangle around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)

# Display the image
cv2.imshow('Car Detection', img)
cv2.waitKey()


print("Code Completed")
