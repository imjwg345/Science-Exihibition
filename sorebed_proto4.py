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
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils

        # 카메라 초기화
        self.cap = cv2.VideoCapture(0)
        self.screen_width, self.screen_height = 640, 480  # 화면 크기
        self.point_x, self.point_y = self.screen_width // 2, self.screen_height // 2  # 점 초기 위치 (화면 중심)
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
                    # 왼쪽 눈동자 중심 좌표
                    left_eye_center = landmarks.landmark[468]
                    # 오른쪽 눈동자 중심 좌표
                    right_eye_center = landmarks.landmark[473]

                    # 화면 크기를 고려하여 좌표 변환
                    left_eye_x = int(left_eye_center.x * frame.shape[1])
                    left_eye_y = int(left_eye_center.y * frame.shape[0])
                    right_eye_x = int(right_eye_center.x * frame.shape[1])
                    right_eye_y = int(right_eye_center.y * frame.shape[0])

                    # 두 눈의 중심 좌표
                    eye_center_x = (left_eye_x + right_eye_x) // 2
                    eye_center_y = (left_eye_y + right_eye_y) // 2

                    # 화면 중심을 기준으로 점 이동
                    self.point_x = eye_center_x
                    self.point_y = eye_center_y

                    # 이미지 위에 점 그리기
                    if self.image_on_canvas:
                        canvas_x = int((self.point_x / self.screen_width) * self.tk_image.width())
                        canvas_y = int((self.point_y / self.screen_height) * self.tk_image.height())
                        self.canvas.create_oval(canvas_x-5, canvas_y-5, canvas_x+5, canvas_y+5, fill="blue")

            # OpenCV 프레임을 PIL 이미지로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.camera_image = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=self.camera_image)
            self.camera_canvas.config(width=self.camera_image.width(), height=self.camera_image.height())

        self.root.after(10, self.update_camera)

def run_another_script():
    subprocess.run(['python', 'sorebad_camv1.py'])

if __name__ == "__main__":
    root = tk.Tk()
    
    app = EyeTrackingApp(root)
    
    # 다른 스크립트를 별도의 스레드에서 실행
    script_thread = threading.Thread(target=run_another_script)
    script_thread.start()
    
    root.mainloop()



