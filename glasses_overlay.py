from imutils.video import WebcamVideoStream
from imutils import face_utils
import imutils
import cv2
import numpy as np
import json
import dlib


# Overlays an image
def overlay_image(bg, fg, fgMask, coords):
    (sH, sW) = fg.shape[:2]
    (x, y) = coords
    # the overlay should be the same width and height as the input
    # image and be totally blank *except* for the foreground which
    # we add to the overlay via array slicing
    overlay = np.zeros(bg.shape, dtype="uint8")
    overlay[y:y + sH, x:x + sW] = fg
    # the alpha channel, which controls *where* and *how much*
    # transparency a given region has, should also be the same
    # width and height as the input image, but will contain only
    # the foreground mask
    alpha = np.zeros(bg.shape[:2], dtype="uint8")
    alpha[y:y + sH, x:x + sW] = fgMask
    alpha = np.dstack([alpha] * 3)
    # perform alpha blending to merge the foreground, background,
    # and alpha channel together
    output = alpha_blend(overlay, bg, alpha)

    return output

def alpha_blend(fg, bg, alpha):
    # convert the foreground, background, and alpha layers from
    # unsigned 8-bit integers to floats, making sure to scale the
    # alpha layer to the range [0, 1]
    fg = fg.astype("float")
    bg = bg.astype("float")
    alpha = alpha.astype("float") / 255
    # perform alpha blending
    fg = cv2.multiply(alpha, fg)
    bg = cv2.multiply(1 - alpha, bg)
    # add the foreground and background to obtain the final output
    # image
    output = cv2.add(fg, bg)
    
    return output.astype("uint8")

# Variables
config = json.loads(open("config.json".read()))
glasses = cv2.imread(config["sunglasses"])
glassesMask = cv2.imread(config["sunglasses_mask"])
detector = cv2.dnn.readNetFromCaffe(config["face_detector_prototxt"], config["face_detector_weights"])
predictor = dlib.shape_predictor(config["landmark_predictor"])
vs = WebcamVideoStream(src=0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    frame = cv2.flip(frame, 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF 
    
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()