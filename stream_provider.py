import os
import cv2
from threading import Thread, Lock
import time

class Stream:
    def __init__(self, src=0, iw=640, ih=480):
        #self.stream = cv2.VideoCapture(src, vcode)
        self.src = src
        if isinstance(src, int):
            platform = os.name
            if platform == 'nt':
                self.vcodec = cv2.CAP_DSHOW
                self.start_stream()
            elif platform == "posix":
                self.vcodec = cv2.CAP_ANY
                self.start_stream()

            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, iw)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, ih)
            
            self.streamWidth = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.streamHeight = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        elif isinstance(src, str):
            self.vcodec = cv2.CAP_FFMPEG
            self.start_stream()
            self.streamWidth = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.streamHeight = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            raise RuntimeError("no")
        
        print("Stream size -> width : {}, height : {}".format(self.streamWidth, self.streamHeight))

        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start_stream(self):
        self.stream = cv2.VideoCapture(self.src, self.vcodec)

    def start(self):
        if self.started:
            print("Already start")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.stream.read()
            if grabbed:
                self.read_lock.acquire()
                self.grabbed, self.frame = grabbed, frame.copy()
                self.read_lock.release()
            else:
                if self.stream is not None and self.stream.isOpened == False:
                    print("stream have trouble")
                    self.stream.release()
                    time.sleep(3)
                self.start_stream()



    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        # frame = cv2.flip(frame, 1)
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        self.stream.release()
        # self.thread.join()

    def getStreamSize(self):
        return (self.streamWidth, self.streamHeight)

    def __exit__(self, exc_type, exc_value, traceback):
        print("camera release")
        self.stream.release()
