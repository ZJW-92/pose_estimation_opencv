
#import packages
import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Take video input for pose detection
cap = cv2.VideoCapture("sample1.mp4")

# Take live camera input for pose detection
# cap = cv2.VideoCapture(0)

# Read each frame/image from capture object
while True:
    ret, img = cap.read()
# Resize image/frame so we can accommodate it on our screen
    img = cv2.resize(img, (600, 400))

    results = pose.process(img)
# Draw the detected pose on original video
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((0,0,255), 2, 2)
                           )
# Display pose on original video/live stream
    cv2.imshow("Pose Estimation", img)

# Extract and draw pose on plain white image
    h, w, c = img.shape   # get shape of original frame
    opImg = np.zeros([h, w, c])  # create blank image with original frame size
    opImg.fill(0)  # set black background. put 255 if you want to make it blank

# Draw extracted pose on black white image
    mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255,0, 0), 2, 2),
                           mp_draw.DrawingSpec((0,0,255), 2, 2)
                           )
# Display extracted pose on blank images
    cv2.imshow("Extracted Pose", opImg)

    # Print all landmarks
    print(results.pose_landmarks)

    cv2.waitKey(1)
