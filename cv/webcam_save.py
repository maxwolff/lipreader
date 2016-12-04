import cv2, os

os.chdir('/Users/boraerden/Google Drive/classes/4 Senior/Seni 1/cs221/cs 221 project/code/saved_frames')


faceCascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.13.1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))


count = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cropped_face = frame[y:y+h, x:x+w]
        # cv2.imwrite("frame%010d.jpg" % count, cropped_face)
        out.write(cropped_face)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1

# When everything is done, release the capture
video_capture.release()
out.release()
cv2.destroyAllWindows()