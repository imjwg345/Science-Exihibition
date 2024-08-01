import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import threading
import subprocess

class EyeTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Tracking with MediaPipe")

        # 프레임 설정
        self.image_frame = tk.Frame(self.root, bg='white', width=400, height=800)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.camera_frame = tk.Frame(self.root, bg='white', width=400, height=800)
        self.camera_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 이미지 캔버스 설정
        self.canvas = tk.Canvas(self.image_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 카메라 캔버스 설정
        self.camera_canvas = tk.Canvas(self.camera_frame, bg='white')
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)

        # 메뉴바 추가
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image)

        self.image_on_canvas = None
        self.canvas.bind("<Button-1>", self.draw_point)

        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1)
        self.mp_drawing = mp.solutions.drawing_utils

        # 카메라 초기화
        self.cap = cv2.VideoCapture(0)
        self.update_camera()

    def open_image(self):
        # 이미지 파일 열기 대화상자 표시
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        # 이미지 열기
        self.image = Image.open(file_path)
        self.image.thumbnail((800, 800))  # 큰 이미지는 캔버스 크기에 맞게 조정
        self.tk_image = ImageTk.PhotoImage(self.image)

        # 이미지 캔버스에 표시
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def draw_point(self, event):
        if self.image_on_canvas:
            x, y = event.x, event.y
            radius = 3
            self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="red")

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            # MediaPipe를 이용한 얼굴 랜드마크 추적
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    for landmark in landmarks.landmark:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        # 랜드마크에 점을 찍습니다.
                        if landmark.visibility > 0.5:
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    # 눈의 랜드마크 좌표 가져오기
                    left_eye = [landmarks.landmark[i] for i in [33, 160, 158, 133]]  # 왼쪽 눈 랜드마크
                    right_eye = [landmarks.landmark[i] for i in [362, 385, 387, 263]]  # 오른쪽 눈 랜드마크

                    # 눈의 중심 좌표를 구합니다.
                    left_eye_center = self.get_eye_center(left_eye)
                    right_eye_center = self.get_eye_center(right_eye)

                    # 시선 방향을 표시합니다.
                    self.draw_eye_direction(frame, left_eye_center, right_eye_center)

            # OpenCV 프레임을 PIL 이미지로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.camera_image = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=self.camera_image)
            self.camera_canvas.config(width=self.camera_image.width(), height=self.camera_image.height())

        self.root.after(10, self.update_camera)

    def draw_eye_direction(self, frame, left_eye_center, right_eye_center):
        if self.image_on_canvas:
            # 시선 방향을 나타낼 수 있는 위치를 계산합니다.
            image_x = (left_eye_center[0] / self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * self.tk_image.width()
            image_y = (left_eye_center[1] / self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * self.tk_image.height()

            # 화면 중심을 기준으로 십자 모양 이동 제한
            canvas_center_x = self.tk_image.width() // 2
            canvas_center_y = self.tk_image.height() // 2

            if abs(image_x - canvas_center_x) > abs(image_y - canvas_center_y):
                image_y = canvas_center_y
            else:
                image_x = canvas_center_x

            # 이미지를 업데이트하여 시선 방향을 표시합니다.
            self.canvas.create_oval(image_x-5, image_y-5, image_x+5, image_y+5, fill="blue")

    def get_eye_center(self, eye_landmarks):
        x_coords = [landmark.x * self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) for landmark in eye_landmarks]
        y_coords = [landmark.y * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) for landmark in eye_landmarks]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

def run_another_script():
    subprocess.run(['python', 'sorebad_camv1.py'])

if __name__ == "__main__":
    root = tk.Tk()
    
    app = EyeTrackingApp(root)
    
    # 다른 스크립트를 별도의 스레드에서 실행
    script_thread = threading.Thread(target=run_another_script)
    script_thread.start()
    
    root.mainloop()
