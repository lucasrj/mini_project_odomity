import numpy as np
import cv2
import sys

class FrameIterator():
    def __init__(self, filename,output_dir):
        self.cap = cv2.VideoCapture(filename)
        self.output_dir = output_dir

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
            cv2.imwrite(self.output_dir + "%02d" % i+ '.jpg', frame)
            i+=1

if len(sys.argv) < 3 :
    print("Usage: command [imput file] [output dir] ")
    sys.exit(1)


fi = FrameIterator(sys.argv[1],sys.argv[2])
fi.main()