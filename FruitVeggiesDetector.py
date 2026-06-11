import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load your local model weights
MODEL_PATH = "trained_fruits_veggies_model.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")

class ModernYoloApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit & Veggie AI Detector")
        self.root.geometry("750x650")
        self.root.configure(bg="#121212")  # Deep dark background
        
        # Protocol for clicking the 'X' close button safely
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # State variables
        self.cap = None
        self.is_running = False

        # ---------------------------------------------------------
        # APP STYLING CONFIGS
        # ---------------------------------------------------------
        self.bg_color = "#121212"
        self.card_color = "#1e1e1e"
        self.accent_blue = "#2563eb"
        self.accent_green = "#059669"
        self.accent_red = "#dc2626"
        self.text_main = "#f4f4f5"
        self.text_muted = "#a1a1aa"
        
        # ---------------------------------------------------------
        # LAYER 1: MAIN MENU FRAME
        # ---------------------------------------------------------
        self.menu_frame = tk.Frame(self.root, bg=self.bg_color)
        self.menu_frame.pack(fill="both", expand=True)

        # Title
        self.title_label = tk.Label(
            self.menu_frame, 
            text="Fruit & Veggie AI Detector", 
            font=("Segoe UI", 20, "bold"), 
            bg=self.bg_color, 
            fg=self.text_main
        )
        self.title_label.pack(pady=(80, 10))

        self.subtitle_label = tk.Label(
            self.menu_frame, 
            text="Select an input source to begin real-time detection", 
            font=("Segoe UI", 11), 
            bg=self.bg_color, 
            fg=self.text_muted
        )
        self.subtitle_label.pack(pady=(0, 50))

        # Buttons Container
        btn_container = tk.Frame(self.menu_frame, bg=self.bg_color)
        btn_container.pack()

        btn_style = {
            "font": ("Segoe UI", 12, "bold"),
            "fg": self.text_main,
            "activeforeground": self.text_main,
            "bd": 0,
            "relief": "flat",
            "height": 2,
            "width": 22,
            "cursor": "hand2"
        }

        self.webcam_btn = tk.Button(
            btn_container, text="Use Live Webcam", 
            bg=self.accent_blue, activebackground="#1d4ed8",
            command=self.start_webcam, **btn_style
        )
        self.webcam_btn.grid(row=0, column=0, padx=15)

        self.upload_btn = tk.Button(
            btn_container, text="Upload Video File", 
            bg=self.accent_green, activebackground="#047857",
            command=self.start_video, **btn_style
        )
        self.upload_btn.grid(row=0, column=1, padx=15)

        # ---------------------------------------------------------
        # LAYER 2: STREAM / DETECTION FRAME (Initially Hidden)
        # ---------------------------------------------------------
        self.stream_frame = tk.Frame(self.root, bg=self.bg_color)

        # Video Frame Presentation Label
        self.video_display = tk.Label(self.stream_frame, bg=self.card_color, bd=0)
        self.video_display.pack(pady=20, padx=20, fill="both", expand=True)

        # Stop Button
        self.stop_btn = tk.Button(
            self.stream_frame, text="Stop & Return to Menu", 
            bg=self.accent_red, activebackground="#b91c1c",
            command=self.stop_stream, **btn_style
        )
        self.stop_btn.pack(pady=(0, 20))

    # ---------------------------------------------------------
    # CORE LOGIC & CORE LOOPS
    # ---------------------------------------------------------
    def switch_view(self, to_stream=True):
        """Swaps UI panels smoothly without launching extra windows"""
        if to_stream:
            self.menu_frame.pack_forget()
            self.stream_frame.pack(fill="both", expand=True)
        else:
            self.stream_frame.pack_forget()
            self.menu_frame.pack(fill="both", expand=True)

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not access your webcam device.")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_running = True
        self.switch_view(to_stream=True)
        self.update_frame_loop()

    def start_video(self):
        file_path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if not file_path:
            return  # User cancelled selection
            
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open the selected video file.")
            return

        self.is_running = True
        self.switch_view(to_stream=True)
        self.update_frame_loop()

    def update_frame_loop(self):
        """The non-blocking recursive loop updating frames on the same window"""
        if self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Apply your exact accuracy configurations
                results = model.predict(frame, conf=0.25, verbose=False)
                annotated_frame = results[0].plot()

                # Process BGR frame from OpenCV to RGB for Tkinter
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Resize smoothly to fit nicely in our modern layout container if needed
                # (Maintains your full native resolution processing, but scales presentation)
                img = Image.fromarray(rgb_frame)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                
                img_tk = ImageTk.PhotoImage(image=img)

                # Push image onto the label element
                self.video_display.configure(image=img_tk)
                self.video_display.image = img_tk

                # Trigger the next frame processing cycle in 10ms
                self.root.after(10, self.update_frame_loop)
            else:
                # Video ended or frame failed to read
                self.stop_stream()

    def stop_stream(self):
        """Cleans up resources and returns cleanly to main dashboard"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Clear image container
        self.video_display.configure(image="")
        self.switch_view(to_stream=False)

    def on_closing(self):
        """Ensures webcam gets released cleanly if user kills the window"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

# Execution Block
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernYoloApp(root)
    root.mainloop()