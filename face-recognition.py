import cv2
import pathlib
import os
import argparse

"""
parser = argparse.ArgumentParser()

parser.add_argument("path")
parser.add_argument("-f", help="specify features")
parser.add_argument("-s", help="specify source")
parser.add_argument("-i", help="input path")
args = parser.parse_args()


print(args.f)
print(args.s)
print(args.i)

target_source = pathlib.Path(args.path)
if not target_source.exists():
    print("Error - target file does not exist.\n")
    raise SystemExit(1)
"""
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

cascade_eyes_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_eye.xml"

classifier = cv2.CascadeClassifier(str(cascade_path))
eye_classifier = cv2.CascadeClassifier(str(cascade_eyes_path))


cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        gray, 
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    eyes = eye_classifier.detectMultiScale(
        gray, 
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 255), 2)
    
    for (x, y, width, height) in eyes:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (100, 200, 0), 2)

    cv2.imshow("Detections", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
