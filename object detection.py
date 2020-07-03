import cv2
import numpy as np
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array


model = ResNet50(weights='imagenet')
camera = cv2.VideoCapture(0)
# If possible control the FPS to constrain the labels appearances
# camera.set(cv2.CAP_PROP_FPS, 4)

while True:
    (grabbed, frame) = camera.read()
    # frame = imutils.resize(frame, width=256)

    image = cv2.resize(frame, (224, 224))
    image = img_to_array(image)
    image = imagenet_utils.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    P = imagenet_utils.decode_predictions(preds)[0]

    for i in range(3):
        prob = str(np.round(P[i][2], 2))
        cv2.putText(frame, "{} : {}".format(P[i][1], prob), (10, 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imshow("Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
