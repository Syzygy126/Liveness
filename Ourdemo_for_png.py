import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX('/Users/syzygy/Documents/Liveness Detection/model_test/liveness_detection_model/model_1e.onnx')

for i in range(1,100):
    img = cv2.imread(f"/Users/syzygy/Documents/Liveness Detection/model_test/VideoFrame/frame_00{i:02d}.png")

    img = cv2.resize(img, (224, 224))  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = img.astype("float") / 255.0
    img = np.transpose(img, (2, 0, 1))  # Let Image Convert to NCHW
    img = np.expand_dims(img, axis=0)

    net.setInput(img)
    out = net.forward()
    print(out) 

    predicted_score = out[0][0]
    if predicted_score > 0.5:
        predicted_label = 'real'
    else:
        predicted_label = 'fake'

    print(f'Frame: {i} Predicted label: {predicted_label}')