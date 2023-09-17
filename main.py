import cv2
import mediapipe as md
import numpy as np
import time

import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from scipy import signal

patient_name = input("Patient Name: ")
current_date = datetime.now()

filename = patient_name + '_' +  current_date.strftime("%m_%d_%Y")
print(filename)
# init empty csv
emptydf = pd.DataFrame()

emptydf.to_csv(f"{filename}.csv", sep=',')

md_drawing = md.solutions.drawing_utils
md_drawing_style = md.solutions.drawing_styles
md_pose = md.solutions.pose

plt.ion()
count = 0

position = None


cap = cv2.VideoCapture(1)
# find camera frame dimensions
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


def detect_angle_3(p1: list, p2: list, p3: list):
    # return angle between p1[x, y] p2 and p3 in degrees within 180 degrees
    radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(
        p1[1] - p2[1],
        p1[0] - p2[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


counter = 0
steps = 0

position = "None"

first_1 = 1

first_2 = 1

last_right_x = 0
last_right_y = 0

last_left_x = 0
last_left_y = 0

time_eclipsed_stance_time = 0
time_start_stance_time = 0

y_threshold = 0

column_names = ["step_time", "steps"]
rowdf = pd.DataFrame(columns=['index','step_time','steps'])

testdf = pd.DataFrame(columns =['Right_y','Left_y','Threshold'])
filename2 = filename + 'Kinematics'
testdf.to_csv(f"{filename2}.csv", sep=',')

with md_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        image1 = cv2.flip(frame, 1)
        image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        result = pose.process(image)

        image1.flags.writeable = True

        try:
            landmarks = result.pose_landmarks.landmark

            for i in range(0, len(landmarks)):
                if 23 <= i <= 28:
                    if landmarks[i].visibility <= 0.01:
                        for j in range(0, len(landmarks)):
                            landmarks[j].x = 0
                            landmarks[j].y = 0
                            landmarks[j].visibility = 0


                    pass

                else:
                    landmarks[i].visibility = 0



            l_shoulder = [landmarks[11].x,
                          landmarks[11].y]
            l_elbow = [landmarks[13].x,
                       landmarks[13].y]
            l_wrist = [landmarks[15].x,
                       landmarks[15].y]

            r_shoulder = [landmarks[12].x,
                          landmarks[12].y]
            r_elbow = [landmarks[14].x,
                       landmarks[14].y]
            r_wrist = [landmarks[16].x,
                       landmarks[16].y]



            r_hip = [landmarks[24].x, landmarks[24].y]
            r_knee = [landmarks[26].x, landmarks[26].y]
            r_ankle = [landmarks[28].x, landmarks[28].y]


            l_hip = [landmarks[23].x, landmarks[23].y]
            l_knee = [landmarks[25].x, landmarks[25].y]
            l_ankle = [landmarks[27].x, landmarks[27].y]

            r_ankle_1 = [landmarks[28].x, landmarks[28].y]
            r_ankle_2 = [landmarks[30].x, landmarks[30].y]
            r_ankle_3 = [landmarks[32].x, landmarks[32].y]

            l_ankle_1 = [landmarks[27].x, landmarks[27].y]
            l_ankle_2 = [landmarks[29].x, landmarks[29].y]
            l_ankle_3 = [landmarks[31].x, landmarks[31].y]


            #print (landmarks[27])

            l_back_angle = detect_angle_3(l_shoulder, l_hip, l_ankle)
            r_back_angle = detect_angle_3(r_shoulder, r_hip, r_ankle)

            left_leg_angle = detect_angle_3(l_hip, l_knee, l_ankle)
            right_leg_angle = detect_angle_3(r_hip, r_knee, r_ankle)


            left_ankle_angle = detect_angle_3(l_ankle_1, l_ankle_2, l_ankle_3)
            right_ankle_angle = detect_angle_3(r_ankle_1, r_ankle_2, r_ankle_3)


            # show angle on image and make animation for angle detection

            right = "{:.2f}".format(right_leg_angle)
            left = "{:.2f}".format(left_leg_angle)



            angle = 0
            back = 0
            # if landmarks[14].z > landmarks[13].z:
            #     angle = right_angle
            #     back = r_back_angle
            # else:
            #     angle = left_angle
            #     back = l_back_angle
            #print (landmarks[32].x)
            #print("31",landmarks[31])
            #print("32",landmarks[32])
            #print("Landmarks: ",landmarks[31].y, landmarks[32].y)
            #print("Threshold:", y_threshold)
            col2_names = ['Right_y','Left_y','Threshold']
            my2dict = {"Right_y": landmarks[27].y, "Left_y": landmarks[28].y,
                       "Threshold": y_threshold}
            testdf = pd.DataFrame([my2dict], columns=col2_names, index=[0])

            testdf.to_csv(f"{filename2}.csv", sep=',', header=None, mode='a')

            print(position)




            if (first_1):
                if (((last_right_x) >= landmarks[27].x-0.005) or (last_right_x <= landmarks[28].x+0.005) or ((last_left_x) >= landmarks[27].x-0.005) or (last_left_x <= landmarks[28].x+0.005)):
                    # If we detect the tip of foot in the same spot for 15 counts, it counts as a step
                    if (count > 15):
                        #Increment Steps
                        steps+=1
                        y_threshold = max(landmarks[27].y, landmarks[28].y)
                        print(y_threshold)
                        start_time = time.time()
                        position = "Down"
                        first_1 = 0
                    else:
                        count+=1

            if (position == "Down" and (y_threshold-0.05 > landmarks[27].y or y_threshold-0.05 > landmarks[28].y)):
                stance_time = time.time() - start_time
                #y_threshold = max(landmarks[27].y, landmarks[28].y)
                print("Stance Time:",stance_time)
                position = "Up"

            if (position == "Up" and (y_threshold-0.05 < landmarks[27].y and y_threshold-0.05 < landmarks[28].y)):
                steps+=1
                step_time = round((time.time() - start_time), 2)
                print("Step Time:",step_time)
                y_threshold = max(landmarks[27].y, landmarks[28].y)
                start_time = time.time()
                position = "Down"

                column_names = ["step_time", "steps"]
                mydict = {"step_time": step_time, "steps": steps}
                rowdf = pd.DataFrame([mydict], columns=column_names, index=[0])
                #rowdf = rowdf.append([mydict], ignore_index=True)
                # datarow = {"step_time": time_eclipsed_step_time, "index": steps}

                # df = df.append(datarow, ignore_index=True)
                # testdf = pd.DataFrame([my2dict], columns=col2_names, index=[0])

                rowdf.to_csv(f"{filename}.csv", sep=',', header=None, mode='a')





                # print("JHuygwbougbearou")
                # if (first_2):
                #     first_2 = 0
                # else:
                #     time_eclipsed_stance_time = time.time() - time_start_stance_time
                # #If not same as last postion, start timer
                # time_start_stance_time = time.time()
                # print(time_eclipsed_stance_time)

            #print(landmarks[31].x,landmarks[31].y,landmarks[31].z)

            #Determining the last value for the tip of each leg, so we can compare it in the next loop
            last_right_x = landmarks[32].x
            last_right_y = landmarks[32].y
            last_left_x = landmarks[31].x
            last_left_y = landmarks[31].y

                # if l_back_angle < 150 or r_back_angle < 150:
                #     # put text to say that back is not straight
                #     cv2.putText(image, "Back is not straight",
                #                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                #                 cv2.LINE_AA)
                #print(counter)
            #print(f"right: {landmarks[14].z}, left: {landmarks[13].z}, back: {r_back_angle} angle: {right_leg_angle}")


        except:
            pass

        md_drawing.draw_landmarks(image1, result.pose_landmarks,
                                  md_pose.POSE_CONNECTIONS,
                                  md_drawing.DrawingSpec(
                                      color=(245, 117, 66), thickness=2,
                                      circle_radius=2),
                                  md_drawing.DrawingSpec(
                                      color=(245, 66, 230), thickness=2,
                                      circle_radius=2)
                                  )
        if counter >= 0:
            cv2.putText(image1, "Steps: " + str(steps), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image1, "Count: " + str(count), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Video", image1)
        key = cv2.waitKey(1)
        if key == ord("q"):
            rowdf.to_csv("testercsv1.csv", sep=',', header=None, mode='a')
            break


#testdf.to_csv("test123.csv", sep=',')
#rowdf.to_csv("testercsv.csv", sep=',')

cap.release()
cv2.destroyAllWindows()
