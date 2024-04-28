import numpy as np
import cv2

class FrameIterator():
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)

    def frame_generator(self):
        # Skip the first 1200 frames
        for _ in range(1200):
            self.cap.read()

        frame_count = 0
        # Define a generator that yields frames from the video
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_count % 25 == 0:
                yield frame
            frame_count += 1
        self.cap.release()

    def main(self):
        i = 0
        for frame in self.frame_generator():
            cv2.imwrite('/home/u000000/Desktop/github_2_Semester/mini_project_odomity/src/output/frame_' + str(i) + '.png', frame)
            i+=1

fi = FrameIterator('/home/u000000/Desktop/Large-scaleDronePerception/11_Visual_Odometry_miniproject/DJI_0199.MOV')
fi.main()