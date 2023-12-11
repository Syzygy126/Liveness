import os
import cv2
import numpy as np
import stream_provider as sp
from face_tools import Face_Helper

if __name__ == "__main__":

    ''' 
    Please modify according to the situation
    1. image_size : size of source image. if source is camera, should be specified
    2. input_size : Size of input to model inference.
    3. stream_src : The source can be camera index, RTSP path,
    '''

    #Size format (width, height)
    image_size = (1920,1080) 
    input_size = (640, 360)  # Model input size, image will resize to input size, input size can be changed freely
    stream_src = 0   
    stream = sp.Stream(src=stream_src, iw=image_size[0], ih=image_size[1])
    stream.start()

    image_size = stream.getStreamSize() #After opening the stream, get the image size


    fh = Face_Helper(image_size=image_size, input_size=input_size, 
                     detect_threshold=0.75,
                     detect_weight_path="weights/yunet.onnx")

    win_name = "demo"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    tm = cv2.TickMeter() # OpenCv tool to calculate time

    while True:
        tm.reset()
        tm.start()

        img_ori = stream.read()

        #Use smaller sized images as model input
        img_input = fh.img_processing(img_ori.copy())

        #Model output 
        faces = fh.detect(img_input)[1]

        img_show = img_ori.copy()

        if faces is not None:            
            # The index just represents the number in the array and has no meaning of tracking.
            for index, face in enumerate(faces):
                # Convert coordinates from input size to original image size
                face_ori = fh.rescale2ori(face)

                # draw face box, and can add text from "info_txt" 
                img_show = fh.draw_one_face(image=img_show, result=face_ori, info_txt=str(index))
        
        tm.stop()

        #cv2.putText(img_show, "FPS : {:.0f}".format(tm.getFPS()), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

        cv2.imshow(win_name, img_show)

        key = cv2.waitKeyEx(1)

        if key != -1:
            if key == 113:  # q
                break

    stream.stop()
    cv2.destroyAllWindows()
