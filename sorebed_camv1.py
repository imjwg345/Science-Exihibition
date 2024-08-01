import cv2

# 카메라 캡처 객체 생성, 인덱스 0은 기본 카메라를 의미
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        print("프레임을 수신할 수 없습니다. (스트림 끝?)")
        break
    
    # 프레임을 화면에 표시
    cv2.imshow('Camera', frame)
    
    # 'q' 키를 누르면 루프 탈출
    if cv2.waitKey(1) == ord('q'):
        break

# 모든 리소스 해제
cap.release()
cv2.destroyAllWindows()
