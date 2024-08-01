import mediapipe as mp
import cv2
import math
import time


#Mediapipe Face Mesh 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

#화면 및 카메라
cap = cv2.VideoCapture(0)
cv2.namedWindow("Eye Tracking", cv2.WINDOW_NORMAL)

screen_width, screen_height = 640,480
cv2.resizeWindow("Eye Tracing", screen_width, screen_height)

#점 초기 위치 설정
point_x, point_y = screen_width // 2, screen_height //2

#눈 깜빡임 임계값
blink_threshold = 0.01

#선택지 위치
left_button_pos = (0,screen_height //2)
right_button_pos = (screen_width - 120, screen_height // 2)
up_button_pos = (screen_width // 2 - 55, 35)
down_button_pos = (screen_width // 2 - 50, screen_height - 15)
button_width, button_height = 120, 65

# "SELECTED" 표시를 위한 변수
selected = False
selected_start_time = None  # "SELECTED" 시작 시간

# 프레임 속도 측정 변수
fps = 0
prev_frame_time = 0

#버튼 그리기 함수
def draw_button(image, text, position, color):
    cv2.rectangle(image, (position[0], position[1] - 35), (position[0] + button_width, position[1] + 10), color, cv2.FILLED)
    cv2.putText(image, text, (position[0] + 5, position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # BGR 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 왼쪽 눈 상단과 하단 랜드마크 인덱스: 159, 145
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]

            # 오른쪽 눈 상단과 하단 랜드마크 인덱스: 386, 374
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]

            # 눈 세로 길이 계산
            left_eye_vertical_length = math.hypot(left_eye_top.x - left_eye_bottom.x, left_eye_top.y - left_eye_bottom.y)
            right_eye_vertical_length = math.hypot(right_eye_top.x - right_eye_bottom.x, right_eye_top.y - right_eye_bottom.y)

            # 눈 깜빡임 감지
            blink_detected = left_eye_vertical_length < blink_threshold or right_eye_vertical_length < blink_threshold
            if blink_detected and (
                    (left_button_pos[0] <= point_x <= left_button_pos[0] + button_width and left_button_pos[1] - 35 <= point_y <= left_button_pos[1] + 10) or
                    (right_button_pos[0] <= point_x <= right_button_pos[0] + button_width and right_button_pos[1] - 35 <= point_y <= right_button_pos[1] + 10) or
                    (up_button_pos[0] <= point_x <= up_button_pos[0] + button_width and up_button_pos[1] - 35 <= point_y <= up_button_pos[1] + 10) or
                    (down_button_pos[0] <= point_x <= down_button_pos[0] + button_width and down_button_pos[1] - 35 <= point_y <= down_button_pos[1] + 10)):
                selected = True
                selected_start_time = time.time()  # 현재 시간 기록

            # 버튼 UI 개선
            draw_button(image, "turn left", left_button_pos, (0, 255, 0))  # 화면 왼쪽에 배치
            draw_button(image, "turn right", right_button_pos, (0, 0, 255))  # 화면 오른쪽에 배치
            draw_button(image, "leg Up", up_button_pos, (200, 0, 0))  # 화면 위쪽에 배치
            draw_button(image, "leg Down", down_button_pos, (300, 300, 0))  # 화면 아래쪽에 배치

            # 눈동자 위치 추적
            left_eye_landmarks = [face_landmarks.landmark[i] for i in range(159, 144, -1)]  # 왼쪽 눈 랜드마크
            right_eye_landmarks = [face_landmarks.landmark[i] for i in range(386, 374, -1)]  # 오른쪽 눈 랜드마크

            left_eye_center = (left_eye_landmarks[0].x, left_eye_landmarks[0].y)  # 왼쪽 눈 중심
            right_eye_center = (right_eye_landmarks[0].x, right_eye_landmarks[0].y)  # 오른쪽 눈 중심

            # 눈동자 위치로부터 눈 주변 랜드마크까지의 상대적인 거리 계산
            left_eye_distance_x = left_eye_landmarks[4].x - left_eye_center[0]
            left_eye_distance_y = left_eye_landmarks[4].y - left_eye_center[1]
            right_eye_distance_x = right_eye_center[0] - right_eye_landmarks[4].x
            right_eye_distance_y = right_eye_center[1] - right_eye_landmarks[4].y

            # 눈동자 이동 방향 결정
            if abs(left_eye_distance_x) > abs(right_eye_distance_x):
                horizontal_direction = "Look Left" if left_eye_distance_x > 0 else "Look Right"
            else:
                horizontal_direction = "Look Left" if right_eye_distance_x < 0 else "Look Right"

            if abs(left_eye_distance_y) > abs(right_eye_distance_y):
                vertical_direction = "Look Up" if left_eye_distance_y > 0 else "Look Down"
            else:
                vertical_direction = "Look Up" if right_eye_distance_y < 0 else "Look Down"

            # 화면 중심을 기준으로 점 이동
            if horizontal_direction == "Look Left":
                point_x -= 5  # 민감도를 높이기 위해 이동 거리를 줄임
            elif horizontal_direction == "Look Right":
                point_x += 5  # 민감도를 높이기 위해 이동 거리를 줄임

            if vertical_direction == "Look Up":
                point_y -= 5  # 민감도를 높이기 위해 이동 거리를 줄임
            elif vertical_direction == "Look Down":
                point_y += 5  # 민감도를 높이기 위해 이동 거리를 줄임

            # 화면 중심을 기준으로 십자 모양 이동 제한
            if abs(point_x - screen_width // 2) > abs(point_y - screen_height // 2):
                point_y = screen_height // 2
            else:
                point_x = screen_width // 2

            # 화면 끝까지만 이동 가능하도록 제한
            point_x = max(0, min(point_x, screen_width))
            point_y = max(0, min(point_y, screen_height))

            # 방향 텍스트 표시
            cv2.putText(image, horizontal_direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, vertical_direction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 점 그리기
            cv2.circle(image, (point_x, point_y), 5, (0, 255, 0), -1)

    # "SELECTED" 텍스트 표시
    if selected:
        elapsed_time = time.time() - selected_start_time
        if elapsed_time < 2:
            # 2초 동안 화면 중앙에 표시
            cv2.putText(image, "SELECTED", (screen_width // 2 - 105, screen_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        else:
            # 이후 왼쪽 구석에 작게 표시
            cv2.putText(image, "MODE: SELECTED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 프레임 속도 계산
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # 프레임 속도 텍스트 표시
    cv2.putText(image, f'FPS: {int(fps)}', (10, screen_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 이미지 화면에 표시
    cv2.imshow("Eye Tracking", image)

    # 'q' 키를 눌러 프로그램 종료
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
