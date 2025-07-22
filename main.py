import cv2
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

# 動画を保存する用のモジュール
# fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
# video = cv2.VideoWriter('demo.mp4', fourcc, 20.0, (640, 360))

# 静止画像の場合：
# with mp_pose.Pose(
#     static_image_mode=True, min_detection_confidence=0.5) as pose:
#   file_list = ["datas/1.jpg"]
#   for idx, file in enumerate(file_list):
#     image = cv2.imread(file)
#     image_height, image_width, _ = image.shape
#     # 処理する前にBGR画像をRGBに変換
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     if not results.pose_landmarks:
#       continue
#     print(
#         f'Nose coordinates: ('
#         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
#         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
#     )
#     # 画像にポーズのランドマークを描画
#     annotated_image = image.copy()
#     # upper_body_onlyがTrueの時
#     # 以下の描画にはmp_pose.UPPER_BODY_POSE_CONNECTIONSを使用
#     mp_drawing.draw_landmarks(
#        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#     cv2.imwrite('./tmp/annotated_image' + str(idx) + '.png', annotated_image)

# Webカメラ入力の場合：
cap = cv2.VideoCapture("./datas/mv2.mov")
with mp_pose.Pose(
   min_detection_confidence=0.5,
   min_tracking_confidence=0.5) as pose:
   count = 0
   while cap.isOpened():
      success, image = cap.read()
      if not success:
         print("Ignoring empty camera frame.")
         # ビデオをロードする場合は、「continue」ではなく「break」を使用してください
         break
      shape = image.shape

      # 後で自分撮りビューを表示するために画像を水平方向に反転し、BGR画像をRGBに変換
      # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      # パフォーマンスを向上させるには、オプションで、参照渡しのためにイメージを書き込み不可としてマーク
      image.flags.writeable = False
      results = pose.process(image) 

      # 画像にポーズアノテーションを描画
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      # mp_drawing.draw_landmarks(
      #     image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      if results.pose_landmarks is not None:
         print(results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[23])

         pt1 = results.pose_landmarks.landmark[0] # 鼻
         pt2 = results.pose_landmarks.landmark[11] # 肩
         pt3 = results.pose_landmarks.landmark[23] # 腰
         pt1 = (int(pt1.x*shape[1]), int(pt1.y*shape[0]))
         pt2 = (int(pt2.x*shape[1]), int(pt2.y*shape[0]))
         pt3 = (int(pt3.x*shape[1]), int(pt3.y*shape[0]))

         cv2.circle(image, pt1, 5, (255,0,0), -1)
         cv2.circle(image, pt2, 5, (0,255,0), -1)
         cv2.circle(image, pt3, 5, (0,0,255), -1)

         cv2.line(image, pt2, pt1, (128, 128, 0), 3)
         cv2.line(image, pt2, pt3, (128, 0, 128), 3)
         cv2.line(image, pt3, pt1, (0, 128, 128), 3)

         pt1 = np.array(pt1, dtype=np.float32)
         pt2 = np.array(pt2, dtype=np.float32)
         pt3 = np.array(pt3, dtype=np.float32)

         vec_base = np.array([1,0])
         vec1_2 = pt1 - pt2
         vec1_3 = pt1 - pt3
         vec2_3 = pt2 - pt3

         # 符号
         sign1_2 = 1 
         sign1_3 = 1 
         sign2_3 = 1 
         # 写真と実際のx-y軸は，y軸の向きが反転しているので，補正する．
         vec1_2[1] = -vec1_2[1]
         vec1_3[1] = -vec1_3[1]
         vec2_3[1] = -vec2_3[1]

         if vec1_2[1] < 0:
            sign1_2 = -1
         if vec1_3[1] < 0:
            sign1_3 = -1
         if vec2_3[1] < 0:
            sign2_3 = -1

         theta1_2 = sign1_2 * math.acos(np.dot(vec1_2, vec_base) / (np.linalg.norm(vec_base,ord=2)*np.linalg.norm(vec1_2,ord=2)))
         theta1_3 = sign1_3 * math.acos(np.dot(vec1_3, vec_base) / (np.linalg.norm(vec_base,ord=2)*np.linalg.norm(vec1_3,ord=2)))
         theta2_3 = sign2_3 * math.acos(np.dot(vec2_3, vec_base) / (np.linalg.norm(vec_base,ord=2)*np.linalg.norm(vec2_3,ord=2)))

         print(np.rad2deg(theta1_2), np.rad2deg(theta1_3), np.rad2deg(theta2_3))
         cv2.putText(image, "waist -> head : {:.3f}".format(np.rad2deg(theta1_3)), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0))
         cv2.putText(image, "waist -> shoulder : {:.3f}".format(np.rad2deg(theta2_3)), (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0))
         cv2.putText(image, "shoulder -> head : {:.3f}".format(np.rad2deg(theta1_2)), (20, 110), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0))
      else:
         cv2.putText(image, "waist -> head : None", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0))
         cv2.putText(image, "waist -> shoulder : None", (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0))
         cv2.putText(image, "shoulder -> head : None", (20, 110), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0))

      cv2.imshow('MediaPipe Pose', image)
      # サンプルとして丁度いいのが206フレーム目以降なので，それまでは流す
      if count <= 206: 
         if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      else:
         if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      # print(count)
      count += 1

#       video.write(image)

# video.release()
cap.release()