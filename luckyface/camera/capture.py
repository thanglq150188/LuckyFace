import cv2
import cv2.data
import numpy as np
from typing import Optional, Tuple


class CameraManager:
    
    def __init__(self, camera_id: int = 0) -> None:
        self.camera_id = camera_id
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    
    def start(self) -> bool:
        """Start the camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        return self.cap.isOpened()

    def stop(self) -> None:
        """Stop the camera capture."""
        if self.cap:
            self.cap.release()

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a frame from the camera."""
        if not self.cap:
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and return the face region from the frame."""
        from .utils import enhance_image
        
        # Enhance image for better face detection
        enhanced_frame = enhance_image(frame)
        
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
            
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Add some padding
        padding = int(w * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        face_region = frame[y:y+h, x:x+w]
        return face_region
    

def test_camera_manager():
    """Test the CameraManager class functionality."""
    # Initialize the camera manager
    cam = CameraManager(camera_id=0)
    
    # Test camera start
    if not cam.start():
        print("Failed to start camera")
        return
    
    print("Camera started successfully")
    
    try:
        while True:
            # Get frame
            ret, frame = cam.get_frame()
            if not ret or frame is None:
                print("Failed to get frame")
                break
                
            # Try to detect face
            face = cam.detect_face(frame)
            
            # Draw rectangle around detected face
            if face is not None:
                # Get original frame dimensions
                height, width = frame.shape[:2]
                
                # Draw text indicating face detected
                cv2.putText(
                    frame,
                    "Face Detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Show face detection result in separate window
                cv2.imshow("Detected Face", face)
            
            # Display the original frame
            cv2.imshow("Camera Test", frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        
    finally:
        # Cleanup
        cam.stop()
        cv2.destroyAllWindows()
        print("Camera stopped and windows closed")
        

if __name__ == "__main__":
    print("Starting camera test...")
    print("Press 'q' to quit")
    test_camera_manager()