import cv2
import numpy as np
from djitellopy import tello
import time
import KeyPressModule as kp
import cvzone


# 임계값 설정
thres = 0.65
nmsThres = 0.2 #0.2

classNames = []
# 사물 감지를 위한 object dataSet
classFile = "Resources/coco.names"
#classNames 에 하나씩 읽어오기
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
print(classNames)
configPath = 'Resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'Resources/frozen_inference_graph.pb'

#네트워크 불러오기  opencv 딥러닝 실행하기 위해서는
net = cv2.dnn_DetectionModel(weightPath, configPath)

#set dnn_detectionModel
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
#opencv BGR를 RGB로 교체
net.setInputSwapRB(True)

#키보드 설정
kp.init()
#tello 연결
me = tello.Tello()
#배터리 표시
me.connect()
# 드론 위도우 창에 보여주기 위해 stream
print(me.get_battery())

me.streamoff()
me.streamon()

# 이륙하기
# me.takeoff()
# 속도 초기값 설정
# me.send_rc_control(0, 0, 0, 0)
# time.sleep(2)

#화면 사이즈 조정
w, h = 720, 680
#훈련을 통해 거리유지의 최적의 값을 찾아냄
fbRange = [5000, 8000]
pid = [0.4, 0.4, 0]
pError = 0
cap = cv2.VideoCapture(0) #위치 조정해복;

def findFace(img):
    #classifier 이용해서 분류
    faceCasecade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #얼굴 검출 #1.2 와 8이 최고!
    faces = faceCasecade.detectMultiScale(imgGray, 1.2, 8)

    # center x ,y
    myFaceListC = []
    myFaceListArea = []
    #물체 검출
    #confThreshold에 임계값 넣어주기  nmsThreshold
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cvzone.cornerRect(img, box)
            #putText object name            class name start 1   conf는 0.xx 이므로 곱하기 100 하고 소수점 2자리까지 반올림
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                        #x:0 y:0 에서  +10 +30
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 255, 0), 2)
    #           for coconames object detection
    except:
        pass

    #face 의 좌표정보를 받고 이용
    for (x, y, w, h) in faces:
        #얼굴 위치 표시
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        #중앙점 표시
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    #
    if len(myFaceListArea) != 0:

        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackFace(info, w, pid, pError):
    # 얼굴 영역 벗어나면 드론 위치 수정하기
    area = info[1]
    x, y = info[0]
    fb = 0

    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -30, 30))
        #5000~8000 사이일떄 유지
    if area > fbRange[0] and area < fbRange[1]:

        fb = 0
        print("between",area)

        #8000 이상일떄 얼굴이 커지므로 드론 back
    elif area > fbRange[1]:  # 8000
        fb = -20
        print("over", area)
        time.sleep(0.5)
        me.send_rc_control(0, fb, 0, speed)


        #5000 이하일떄 얼굴이 작아지므로 드론 forward
    elif area < fbRange[0] and area != 0:  # 5000
        fb = 20
        print("under", area)
        time.sleep(0.5)
        me.send_rc_control(0, fb, 0, speed)




    if x == 0:
        speed = 0
        error = 0
    # me.send_rc_control(0, fb, 0, speed)

    return error

#키보드로 속도 조절
def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if kp.getKey("LEFT"): lr = -speed
    elif kp.getKey("RIGHT"): lr = speed

    if kp.getKey("UP"): fb = speed
    elif kp.getKey("DOWN"): fb = -speed

    if kp.getKey("w"): ud = speed
    elif kp.getKey("s"): ud = -speed

    if kp.getKey("a"): yv = -speed
    elif kp.getKey("d"): yv = speed

    if kp.getKey("q"):
        me.land()


    if kp.getKey("t"): me.takeoff()

    if kp.getKey("z"):
        cv2.imwrite(f'Resources/Images/{time.time()}.jpg',img)


    return [lr, fb, ud, yv]

# cap = cv2.VideoCapture(0) #위치 조정해복;

while True:
    # _, img = cap.read()

    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    img, info = findFace(img)
    pError = trackFace(info, w, pid, pError)
    # print("center",info[0],"Area",info[1])
    cv2.imshow("Output", img)

    if cv2.waitKey(1) & kp.getKey("q"):
        me.land()
        cv2.destroyAllWindows()
        break
