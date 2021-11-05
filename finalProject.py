import math
import cv2
import numpy as np
import time as t
import mediapipe as mp
import matplotlib.pyplot as plt
from djitellopy import tello
import KeyPressModule as kp
import cvzone
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImageAttendance'
images = []
classNames2 = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames2.append(os.path.splitext(cl)[0])


def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open("Attendance.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

        print(myDataList)


encodeListKnown = findEncoding(images)

classNames = []
# 사물 감지를 위한 object dataSet
classFile = "Resources/coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
configPath = 'Resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'Resources/frozen_inference_graph.pb'

# 네트워크 불러오기  opencv 딥러닝 실행하기 위해서는
net = cv2.dnn_DetectionModel(weightPath, configPath)

# 임계값 설정
thres = 0.65
nmsThres = 0.2

# set dnn_detectionModel
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))

me = tello.Tello()
# 배터리 표시
me.connect()
# 드론 위도우 창에 보여주기 위해 stream
print(me.get_battery())

me.streamoff()
me.streamon()
me.send_rc_control(0, 0, 0, 0)
# 화면 사이즈 조정

cap = cv2.VideoCapture(0)  # 위치 조정해복;

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.2, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

# Specify a size of the figure.
plt.figure(figsize=[10, 10])


def faceRecongition(img):
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    # 저장한 사진과 webcam 사진을 비교하기
    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        # 가장 작은 distance가 매치하는것
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames2[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    return img


def detectObject(image, thres, nmsThres):
    classIds, confs, bbox = net.detect(image, confThreshold=thres, nmsThreshold=nmsThres)
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cvzone.cornerRect(image, box)
            # putText object name            class name start 1   conf는 0.xx 이므로 곱하기 100 하고 소수점 2자리까지 반올림
            if (classNames[classId - 1] == 'person'):
                continue
            cv2.putText(image, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                        # x:0 y:0 에서  +10 +30
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 255, 0), 2)
    #           for coconames object detection

    except:
        pass
    return image


def detectPose(image, pose, display=True):
    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

        # Also Plot the Pose landmarks in 3D.
        # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # Otherwise
    else:

        # Return the output image and the found landmarks.
        return output_image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle


def classifyPose(landmarks, output_image, display=False):
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)

    # Calculate the required angles.
    # ----------------------------------------------------------------------------------------------------------------

    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    # Get the angle between the right shoulder, elbow and wrist points.
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    # Get the angle between the left elbow, shoulder and hip points.
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points.
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points.
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # ----------------------------------------------------------------------------------------------------------------

    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    # ----------------------------------------------------------------------------------------------------------------

    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        # left_elbow_angle 165~195        right_elbow_angle  165 ~195
        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
            # left_shoulder_angle  80~110   right_shoulder_angle  80 ~110
            # Check if it is the warrior II pose.
            # ----------------------------------------------------------------------------------------------------------------

            # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                # Check if the other leg is bended at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose'

                    # ----------------------------------------------------------------------------------------------------------------

            # Check if it is the T pose.
            # ----------------------------------------------------------------------------------------------------------------

            # Check if both legs are straight
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                # Specify the label of the pose that is tree pose.
                label = 'T Pose'

    # ----------------------------------------------------------------------------------------------------------------

    # Check if it is the tree pose.
    # ----------------------------------------------------------------------------------------------------------------

    # Check if one leg is straight
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

        # Check if the other leg is bended at the required angle.
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:
            # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'

    # ----------------------------------------------------------------------------------------------------------------

    # Check if suha pose
    if left_shoulder_angle > 170 and left_shoulder_angle < 190 and right_shoulder_angle < 190 and right_shoulder_angle > 170:
        label = 'hands up'
        cv2.imwrite(f'Resources/Images/{t.time()}hands_up.jpg', output_image)

    # ----------------------------------------------------------------------------------------------------------------

    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)

        # Write the label on the output image.



    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2)
     # Check if the resultant image is specified to be displayed.
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    else:

        # Return the output image and the classified label.
        return output_image, label


# ----------------------------------------------------------------------------------------------------------------


# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)  # height
camera_video.set(4, 960)  # width

# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    # ok, frame = camera_video.read()
    frame = me.get_frame_read().frame
    #add
    frame2 = me.get_frame_read().frame

    # Check if frame is not read properly.
    # if not ok:
    #     # Continue to the next iteration to read the next frame and ignore the empty camera frame.
    #     continue

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    frame2 = cv2.flip(frame2, 1)
    # Get the width and height of the frame
    frame_height, frame_width, _ = frame.shape
    frame2_height, frame2_width, _ = frame2.shape
    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    frame2 = cv2.resize(frame2, (int(frame2_width * (640 / frame2_height)), 640))
    # frame = threading.Thread(target=detectObject, args=(frame, thres, nmsThres))
    # frame.start()
    # frame = threading.Thread(target=faceRecongition, args=(frame))
    # frame.start()

    frame=detectObject(frame,0.65,0.2)
    frame=faceRecongition(frame)

    # Perform Pose landmark detection.
    frame2, landmarks = detectPose(frame2, pose_video, display=False)

    # Check if the landmarks are detected.
    if landmarks:
        # Perform the Pose Classification.
         frame2, _ = classifyPose(landmarks, frame2, display=False)

    cv2.imshow('object Classification', frame)
    cv2.imshow('Pose Classification', frame2)
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed.
    if (k == 27):
        # Break the loop.
        break

# Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllWindows()
