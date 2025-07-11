import mediapipe as mp
import cv2 as cv
#Mediapipe Object
mp_hands=mp.solutions.hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5)
#Capturing video from frame
s=cv.VideoCapture(0)
s.set(cv.CAP_PROP_FRAME_HEIGHT,480)
s.set(cv.CAP_PROP_FRAME_WIDTH,480)
try:
    while True :
        has_frame,frame=s.read()
        if not has_frame :break
        frame=cv.flip(frame,1)                     #flipping frame horizontally
        rgb_frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)            #mediapipe supports rgb format
        result=mp_hands.process(rgb_frame)

        
        if result.multi_hand_landmarks:              #drawing mediapipe landmarks
            for hand_landmarks in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)     #landmarks drawn on frame  
                

        cv.imshow('Hand Landmarks',frame)

        if cv.waitKey(1) & 0xFF==27: 
            break 

finally:
    s.release()
    cv.destroyAllWindows()