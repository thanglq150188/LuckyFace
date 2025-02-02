import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import threading
from typing import Optional
from ..camera.capture import CameraManager 
from ..vlms.analyzer import FaceAnalyzer
from ..core.config import Settings
from ..core.models import AnalysisResult
import cv2

class LuckyFaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.settings = Settings()
        self.camera = CameraManager(self.settings.camera_id)
        self.analyzer = FaceAnalyzer(self.settings)
        
        self.title("LuckyFace")
        self.geometry("1200x800")
        
        self.setup_ui()
        self.setup_camera()

    def setup_ui(self):
        """Setup the UI components."""
        # Main container
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Camera frame
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="")
        self.camera_label.pack(expand=True)
        
        # Analysis frame
        self.analysis_frame = ctk.CTkFrame(self)
        self.analysis_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.analysis_text = ctk.CTkTextbox(self.analysis_frame, width=400, height=600)
        self.analysis_text.pack(padx=20, pady=20)
        
        # Capture button
        self.capture_btn = ctk.CTkButton(
            self, text="Capture & Analyze", command=self.capture_and_analyze
        )
        self.capture_btn.grid(row=1, column=0, columnspan=2, pady=20)

    def setup_camera(self):
        """Initialize and start the camera."""
        if not self.camera.start():
            self.show_error("Failed to start camera")
            return
            
        self.update_camera()

    def update_camera(self):
        """Update the camera preview."""
        ret, frame = self.camera.get_frame()
        if ret and frame is not None:
            # Ensure frame is the correct type
            frame = np.array(frame, dtype=np.uint8)
            
            # Convert color space
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(rgb_frame)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=image)
            
            # Update label
            self.camera_label.configure(image=photo)
            self.camera_label._image = photo
        
        self.after(10, self.update_camera)

    def capture_and_analyze(self):
        """Capture frame and perform analysis."""
        ret, frame = self.camera.get_frame()
        if not ret and frame is not None:
            self.show_error("Failed to capture frame")
            return
            
        face = self.camera.detect_face(frame)
        if face is None:
            self.show_error("No face detected")
            return
            
        # Process in background
        threading.Thread(
            target=self._process_face,
            args=(face,),
            daemon=True
        ).start()

    def _process_face(self, face_image: np.ndarray):
        """Process the captured face in background."""
        try:
            from ..camera.utils import preprocess_image, resize_image
            
            # Resize face image to a standard size
            resized_face = resize_image(face_image, (300, 300))
            
            # Convert to PIL Image for analysis
            pil_image = preprocess_image(resized_face)
            result = self.analyzer.analyze_face(pil_image)
            
            self.analysis_text.delete(1.0, ctk.END)
            self.analysis_text.insert(ctk.END, result['analysis'])
            
        except Exception as e:
            self.show_error(f"Analysis failed: {str(e)}")

    def show_error(self, message: str):
        """Show error message."""
        self.analysis_text.delete(1.0, ctk.END)
        self.analysis_text.insert(ctk.END, f"Error: {message}")

if __name__ == "__main__":
    app = LuckyFaceApp()
    app.mainloop()