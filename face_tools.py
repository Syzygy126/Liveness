import cv2
import numpy as np
import math

def check_input_size(img_ori, max_size=1200):
    h, w ,c = img_ori.shape
    fr = max_size / w if w > h else max_size / h
    img_resize = cv2.resize(img_ori.copy(),None, fx=fr, fy=fr)
    rh, rw, rc = img_resize.shape
    return img_resize, (w,h), (rw, rh)
    
class Face_Helper:
    """
        Detect faces and compare similarity
        input_size = (w,h)
    """

    def __init__(self,
                image_size=(640, 480), 
                input_size=(640, 480),
                detect_threshold=0.75,
                nms_threshold=0.15,
                detect_weight_path="yunet.onnx"):
        
        self.image_size = image_size
        self.input_size = input_size

        # Check whether CUDA exists 
        is_cuda = True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False
        print("Cuda is available : {} ".format(is_cuda))

        backend_id = cv2.dnn.DNN_BACKEND_CUDA if is_cuda else cv2.dnn.DNN_BACKEND_DEFAULT
        target_id = cv2.dnn.DNN_TARGET_CUDA if is_cuda else cv2.dnn.DNN_TARGET_CPU

        # face detector
        self.detector = cv2.FaceDetectorYN.create(
            detect_weight_path, "", self.input_size, detect_threshold, nms_threshold, 1000,
                backend_id=backend_id, target_id=target_id)

    def img_processing(self, image):
        image = cv2.resize(image, self.input_size)
        return image
    
    def detect(self, input_image):
        detect_results = self.detector.detect(input_image)
        return detect_results
    
    def reset_image_parm(self, new_image_size, new_input_size):
        '''
        If the size of the image is different and you need to change the model input parameters, 
        you can call this to modify
        '''
        self.image_size = new_image_size
        self.input_size = new_input_size
        self.detector.setInputSize(self.input_size)

    def rescale2ori(self, face):
        '''
        Convert face coordinates from input size to original image size

        return np.float32
        '''
        face_ori = face.copy()

        face_ori[0:-1:2] = face[0:-1:2] / self.input_size[0] * self.image_size[0]
        face_ori[1:-1:2] = face[1:-1:2] / self.input_size[1] * self.image_size[1]

        return face_ori
    
    def rescale2norm(self, face):
        '''
        Convert face coordinates from input size to [0,1]

        return np.float32
        '''
        face_norm = face.copy()

        face_norm[0:-1:2] = face[0:-1:2] / self.input_size[0]
        face_norm[1:-1:2] = face[1:-1:2] / self.input_size[1]

        return face_norm
    
    def optimal_font_dims(self, imageShape, font_scale=1e-3, thickness_scale=2e-3):
        h, w, c = imageShape
        font_scale *= min(w, h)
        thickness = math.ceil(min(w, h) * thickness_scale)
        return font_scale, thickness
    
    def find_largest_face(self,faces):
        '''
        From all faces, find the one with the largest area
        return face
        '''
        max_index = np.argmax([face[2] * face[3] for face in faces])
        return faces[max_index]
    
    def draw_one_face(self, image, result, score, info_txt=None, draw_point=False):
        coords = result[:-1].astype(np.int32)

        font_scale, thickness = self.optimal_font_dims(imageShape=image.shape)
        if score > 0.5:
            cv2.rectangle(image, (coords[0], coords[1]),
                        (coords[0] + coords[2], coords[1] + coords[3]), (0,255,0), thickness)
        else:
            cv2.rectangle(image, (coords[0], coords[1]),
                        (coords[0] + coords[2], coords[1] + coords[3]), (0,0,255), thickness)
        if draw_point:
            cv2.circle(image, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(image, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(image, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(image, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(image, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

        if info_txt is not None:
            if score > 0.5:
                cv2.putText(image, info_txt, (coords[0], coords[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)
            else:
                cv2.putText(image, info_txt, (coords[0], coords[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), thickness)
                

        return image
    
    
    def draw_largest_face(self, image, detect_results, threshold=0.5, info_txt=None, draw_point=False):
        largest_face = self.find_largest_face(detect_results)

        if largest_face is not None:
            score = largest_face[-1]
            if score > threshold:
                image = self.draw_one_face(image, largest_face, score, info_txt, draw_point)

        return image

