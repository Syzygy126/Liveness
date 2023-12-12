import cv2
import numpy as np
from face_tools import Face_Helper

# Load the liveness detection model
liveness_net = cv2.dnn.readNetFromONNX("liveness_detection_model/model_1D2.onnx")

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

    
stream_src = "/Users/syzygy/Documents/Liveness Detection/Test_Videos/Normal_Speed/User01_0.5to1m.mp4"
stream = cv2.VideoCapture(stream_src)
image_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frame_count = stream.get(cv2.CAP_PROP_FRAME_COUNT)


image_size = (image_width, image_height)
input_size = (640, 360)  

real_count = 0
fake_count = 0
fh = Face_Helper(image_size=image_size, input_size=input_size, 
                     detect_threshold=0.75,
                     detect_weight_path="yunet.onnx")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('./1E2_output/01_normal_nomask.mp4', fourcc, 20.0, (image_width, image_height))

win_name = "demo"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name,1200,720)  # can be changed freely
while stream.isOpened():
    ret, img_ori = stream.read()
    print(type(img_ori))
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
            img_ori = fh.draw_one_face(img_ori, face_ori, real_score,info_txt)
            
            if face_label == "Real":
                real_count += 1
            else:
                fake_count += 1
    #out.write(img_ori)
    cv2.imshow(win_name, img_ori)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
stream.release()
#out.release()
cv2.destroyAllWindows()
print(f"model = 1E, real:fake = {real_count} : {fake_count}")