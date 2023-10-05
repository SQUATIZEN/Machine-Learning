import cv2
import numpy as np

# Muat model YOLOv4
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Baca gambar
img = cv2.imread('ban_motor.jpg')
height, width, _ = img.shape

# Lakukan deteksi objek dengan YOLO
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

# Temukan kontur alur ban
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5: # ambil deteksi dengan confidence di atas 0.5
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            # Gambar kotak di sekitar alur ban
            cv2.rectangle(img, (center_x, center_y), (center_x+w, center_y+h), (255, 0, 0), 2)

# Tampilkan gambar
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()