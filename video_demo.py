import os
import cv2
import numpy as np
from face_tools import Face_Helper

if __name__ == "__main__":

    ''' 
    Please modify parameter according to the situation
    1. image_size : size of source image. After the program is launched, the size will be set according to the video
    2. input_size : Size of input to model inference.
    3. stream_src : The source is video path, you can put video in project's video folder,

    '''
    stream_src = "test.avi"
    stream = cv2.VideoCapture(stream_src)
    image_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = stream.get(cv2.CAP_PROP_FRAME_COUNT)

    #Size format (width, height)
    image_size = (image_width, image_height)
    input_size = (640, 360)  # Model input size, in script, will resize image to input_size, input size can be changed freely

    fh = Face_Helper(image_size=image_size, input_size=input_size, 
                     detect_threshold=0.75,
                     detect_weight_path="yunet.onnx")

    win_name = "demo"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name,1280,720)
    
    while stream.isOpened():

        ret, img_ori = stream.read()

        if not ret:
            break

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

                # draw face box, and can add text from parameter(info_txt) 
                img_show = fh.draw_one_face(image=img_show, result=face_ori, info_txt=str(index))

        cv2.imshow(win_name, img_show)

        key = cv2.waitKeyEx(1)

        if key != -1:
            if key == 113:  # q
                break

    print("The video has finished playing")
    stream.release()
    cv2.destroyAllWindows()
