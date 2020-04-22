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
config = json.loads(open("config.json").read())
sgOrig = cv2.imread(config["sunglasses"])
sgMaskOrig = cv2.imread(config["sunglasses_mask"])
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(config["landmark_predictor"])
debug = True
vs = WebcamVideoStream(src=0).start()

# Preprocess mask
sgMaskOrig = cv2.cvtColor(sgMaskOrig, cv2.COLOR_BGR2GRAY)
sgMaskOrig = cv2.threshold(sgMaskOrig, 0, 255, cv2.THRESH_BINARY)[1]

while True:
    frame = vs.read()
    # frame = imutils.resize(frame, width=600)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray, 0)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()       

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # grab the indexes of the facial landmarks for the left and right
        # eye, respectively, then extract (x, y)-coordinates for each eye
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye center
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # rotate the sunglasses image by our computed angle, ensuring the
        # sunglasses will align with how the head is tilted
        # glasses = imutils.rotate_bound(sgOrig, angle)

        # the sunglasses shouldn't be the *entire* width of the face and
        # ideally should just cover the eyes -- here we'll do a quick
        # approximation and use 90% of the face width for the sunglasses
        # width
        sgW = int(w * 0.9)
        sgH = int(h * 0.2)
        glasses = cv2.resize(sgOrig, (sgW, sgH))
        

        # our sunglasses contain transparency (the bottom parts, underneath
        # the lenses and nose) so in order to achieve that transparency in
        # the output image we need a mask which we'll use in conjunction with
        # alpha blending to obtain the desired result -- here we're binarizing
        # our mask and performing the same image processing operations as
        # above        
        # glassesMask = imutils.rotate_bound(sgMaskOrig, angle)
        glassesMask = cv2.resize(sgMaskOrig, (sgW, sgH))

        output = overlay_image(frame, glasses, glassesMask, (x, y + int(h*0.2)))

        if debug:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 1)

            # Draw landmarks in red
            for (x, y) in shape:
                if (x,y) not in leftEyePts and (x,y) not in rightEyePts:
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

            # Draw landmarks of the eyes in green
            for (x,y) in leftEyePts:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            for (x,y) in rightEyePts:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            
    if len(faces) > 0:
        cv2.imshow("Output", output)

    if debug:
        cv2.imshow("Frame", frame)
        
    key = cv2.waitKey(1) & 0xFF 
    
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()