import cv2
import mediapipe as mp
import time

class PoseDetector:
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
    
    def process_frame(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

            # Extract coordinates of the left elbow (landmark index 13)
            left_elbow = results.pose_landmarks.landmark[13]
            h, w, c = img.shape
            left_elbow_x, left_elbow_y = int(left_elbow.x * w), int(left_elbow.y * h)

            # Print the coordinates of the left elbow
            print("Left Elbow Coordinates:", left_elbow_x, left_elbow_y)

            # Draw a circle at the left elbow position
            cv2.circle(img, (left_elbow_x, left_elbow_y), 10, (0, 255, 0), cv2.FILLED)

    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        pTime = 0

        while True:
            success, img = cap.read()

            if not success:
                print("Error reading frame from webcam")
                break

            self.process_frame(img)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == 27:  
                break

        cap.release()
        cv2.destroyAllWindows()
