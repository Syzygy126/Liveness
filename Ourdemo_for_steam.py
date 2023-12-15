import cv2
import numpy as np
from face_tools import Face_Helper
from datetime import datetime

# Load the liveness detection model
liveness_net = cv2.dnn.readNetFromONNX("model_test/liveness_detection_model/model_1E2.onnx")

def preprocess_face(face_image, target_size=(224, 224)):
    # Resize and normalize the face image
    face_image = cv2.resize(face_image, target_size)
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB) 
    face_image = face_image.astype("float32") / 255.0
    face_image = np.transpose(face_image, (2, 0, 1))  # Image Convert to NCHW
    face_image = np.expand_dims(face_image, axis=0)
    return face_image
    
def detect_liveness(face_image):
    # Perform inference on the face image
    liveness_net.setInput(face_image)
    preds = liveness_net.forward()
    return preds
    
def classify_face(real_score, threshold=0.5):
    return "Real" if real_score > threshold else "Fake"

def get_score_label(real_score):
    return f"Real Score: {real_score:.2f}"

stream = cv2.VideoCapture(0)
image_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frame_count = stream.get(cv2.CAP_PROP_FRAME_COUNT)


image_size = (image_width, image_height)
input_size = (640, 360)  
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#output_filename = f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
#out = cv2.VideoWriter(output_filename, fourcc, 20.0, image_size)

fh = Face_Helper(image_size=image_size, input_size=input_size, 
                     detect_threshold=0.75,
                     detect_weight_path="model_test/yunet.onnx")


win_name = "demo"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name,1200, 720)  # can be changed freely
while stream.isOpened():
    ret, img_ori = stream.read()
    if not ret:
        break

    img_input = fh.img_processing(img_ori.copy())
    faces = fh.detect(img_input)[1]

    if faces is not None:
        for index, face in enumerate(faces):
            face_ori = fh.rescale2ori(face)
            (x, y, w, h) = [int(v) for v in face_ori[:4]]
            face_roi = img_ori[y:y+h, x:x+w]

            if face_roi.size == 0:
                continue

            preprocessed_face = preprocess_face(face_roi)
            preds = detect_liveness(preprocessed_face)
            real_score = preds[0][0]  

            face_label = classify_face(real_score, threshold=0.5)
            score_label = get_score_label(real_score)

            info_txt = f"Face {index}: {face_label}, {score_label}"
            img_ori = fh.draw_largest_face(img_ori, face_ori, real_score, info_txt)
            #print(f"face_label{face_label}, {score_label}")
            
    
    cv2.imshow(win_name, img_ori)
    #out.write(img_ori)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

stream.release()
#out.release()
cv2.destroyAllWindows()
