import cv2
import numpy as np

class RadarGun:
    def __init__(self):
        self.prev_frame = None
        self.prev_points = None
        self.focal_length = 3.6  # Focal length of the camera lens in mm
        self.known_width = 150   # Known width of the head to be measured in mm

    def calculate_speed(self, pixel_width):
        # Calculate the speed based on the pixel width of the head
        speed = (self.focal_length * self.known_width) / pixel_width
        return speed

    def measure_speed(self, frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If it's the first frame or the previous points are not available, detect the face
        if self.prev_frame is None or self.prev_points is None:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                self.prev_points = np.array([[x + w // 2, y + h // 2]], dtype=np.float32)
                self.prev_frame = gray

        # Calculate the optical flow
        if self.prev_frame is not None and self.prev_points is not None:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, self.prev_points, None)

            if next_points is not None:
                # Select only the good points for tracking
                good_points = next_points[status[:, 0] == 1]
                prev_good_points = self.prev_points[status[:, 0] == 1]

                # Calculate the distance traveled by the head
                if len(good_points) > 0 and len(prev_good_points) > 0:
                    pixel_distance = np.linalg.norm(good_points - prev_good_points, axis=1).sum()
                    speed = self.calculate_speed(pixel_distance)

                    # Update the previous points with the current good points
                    self.prev_points = good_points.reshape(-1, 1, 2).astype(np.float32)

                    return speed

        # Update the previous frame
        self.prev_frame = gray.copy()
        self.prev_points = None

        return None


# Example usage
rg = RadarGun()
cap = cv2.VideoCapture(0)  # Use the appropriate camera index if not the default

while True:
    ret, frame = cap.read()

    if not ret:
        break

    speed = rg.measure_speed(frame)
    if speed is not None:
        label = f"Speed: {speed:.2f} mm/s"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw a bounding box around the head
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Radar Gun", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
