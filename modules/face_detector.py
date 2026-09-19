import cv2
import mediapipe as mp

class FaceMeshDetector:
    """
    A class to detect face landmarks using MediaPipe Face Mesh.
    """
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=True, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    def find_face_mesh(self, img, draw=True):
        """
        Finds face mesh landmarks in the given image.
        
        Args:
            img: OpenCV BGR image.
            draw: Boolean indicating whether to draw the mesh on the image.
            
        Returns:
            img: Image with drawn landmarks (if draw is True).
            faces: List of detected faces, where each face is a list of (x, y) pixel coordinates of landmarks.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)
        
        faces = []
        h, w, c = img.shape
        
        if self.results.multi_face_landmarks:
            for face_lms in self.results.multi_face_landmarks:
                if draw:
                    # Draw tesselation
                    self.mp_draw.draw_landmarks(
                        image=img,
                        landmark_list=face_lms,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.draw_spec,
                        connection_drawing_spec=self.mp_draw.DrawingSpec(thickness=1, color=(200, 200, 200))
                    )
                    # Draw contours (eyes, brows, face oval, etc.)
                    self.mp_draw.draw_landmarks(
                        image=img,
                        landmark_list=face_lms,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.draw_spec,
                        connection_drawing_spec=self.mp_draw.DrawingSpec(thickness=1, color=(0, 255, 0))
                    )
                    # Draw iris landmarks (if refined)
                    if self.refine_landmarks:
                        self.mp_draw.draw_landmarks(
                            image=img,
                            landmark_list=face_lms,
                            connections=self.mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=self.draw_spec,
                            connection_drawing_spec=self.mp_draw.DrawingSpec(thickness=1, color=(255, 0, 0))
                        )
                
                face = []
                for lm in face_lms.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face.append((cx, cy))
                faces.append(face)
                
        return img, faces
