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