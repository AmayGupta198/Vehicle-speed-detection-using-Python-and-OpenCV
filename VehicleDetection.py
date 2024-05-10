import cv2  # Import OpenCV for image and video processing.
from SpeedDetection import *  # Import a custom SpeedDetection class (Assuming it's provided in a separate file).
import numpy as np
end = 0  # Initialize a variable to track the end of the program.

#Creater Tracker Object
tracker = EuclideanDistTracker()

# Open a video capture from a file (Modify the file path as needed).
cap = cv2.VideoCapture("TrafficRecord/traffic4.mp4")

# Set the frame rate and width for video processing.
f = 25
w = int(1000/(f-1))


# Object Detection using a background subtractor
object_detector = cv2.createBackgroundSubtractorMOG2(history=None,varThreshold=None)

# Define various kernels for image processing.
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8)
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernal_e = np.ones((5,5),np.uint8)

while True:
    ret,frame = cap.read()  # Read a frame from the video.
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)   # Resize the frame to half its size.
    height,width,_ = frame.shape  # Get the height and width of the frame.


    # Extract the Region of Interest (ROI) from the frame.
    roi = frame[50:540,200:960]

    #Applying MASKING METHOD for vehicle detection
    fgmask = fgbg.apply(roi)
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)
    e_img = cv2.erode(mask2, kernal_e)

    # Find contours in the processed image to detect objects.
    contours,_ = cv2.findContours(e_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # THRESHOLD to detect objects
        if area > 1000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
            detections.append([x,y,w,h])

    #Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id

        # Check if the speed of the object is within the speed limit.
        if(tracker.getsp(id)<tracker.limit()):
            cv2.putText(roi,str(id)+" "+str(tracker.getsp(id)),(x,y-15), cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:
            cv2.putText(roi,str(id)+ " "+str(tracker.getsp(id)),(x, y-15),cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 165, 255), 3)

        s = tracker.getsp(id)

        # Capture vehicle data and check for speed violations.
        if (tracker.f[id] == 1 and s != 0):
            tracker.capture(roi, x, y, h, w, s, id)

    # DRAW LINES for reference points.
    cv2.line(roi, (0, 410), (960, 410), (0, 0, 255), 2)
    cv2.line(roi, (0, 430), (960, 430), (0, 0, 255), 2)

    cv2.line(roi, (0, 235), (960, 235), (0, 0, 255), 2)
    cv2.line(roi, (0, 255), (960, 255), (0, 0, 255), 2)


    # Display the processed frame.
    #cv2.imshow("Mask",mask2)
    #cv2.imshow("Erode", e_img)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(w-10)

    # Check for the "Esc" key to exit the program.
    if key==27:
        tracker.end()
        end=1
        break

# Close the video capture and destroy any open windows.
if(end!=1):
    tracker.end()

cap.release()
cv2.destroyAllWindows()