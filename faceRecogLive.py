import cv2
import numpy as np

#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Benedict", "Vivek", "Yashraj", "Robert", "Amit Sir", "Mark", "Chris"]

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('E:\Documents\Projects\College\FaceRecognition\opencv-files\haarcascade_frontalface_alt.xml')
    #face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize = (30,30) )

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    grays=[]
    for face in faces:
        (x, y, w, h) = face
        grays.append(gray[y:y+w, x:x+h])
    #return only the face part of the image
    return grays, faces
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read('E:\Documents\Projects\College\FaceRecognition\\recog.xml')
face_recognizer.read('E:\Documents\Projects\College\FaceRecognition\\recog.yaml')


#function to draw rectangle on image
#according to given (x, y) coordinates and
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#function to draw text on give image starting from
#passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the
#subject
def predict(test_img):
    #make a copy of the image as we don't want to change original image
    img = test_img.copy()
    #detect face from the image
    faces, rects = detect_face(img)
    try:
        for face,rect in zip(faces,rects):
        #predict the image using our face recognizer
            label= face_recognizer.predict(face)
            #get name of respective label returned by face recognizer
            label_text = subjects[label[0]]

            #draw a rectangle around face detected
            draw_rectangle(img, rect)
            #draw name of predicted person
            draw_text(img, label_text, rect[0], rect[1]-5)
    except:
        pass
    return img

print("Predicting images...")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
#load test images
    #test_img1 = cv2.imread()

    #perform a prediction
    predicted_img1 = predict(frame)
    print("Prediction complete")

    #display both images
    cv2.imshow("frame",predicted_img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cv2.waitKey(0)
cv2.destroyAllWindows()
