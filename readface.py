import cv2

img = cv2.imread('location of image', 1)
#print(img)
#print(img.shape)


#resize = cv2.resize(img, (int(img.shape[1]*10), int(img.shape[0]*10)))
#cv2.imshow('Mr.T', img)

#cv2.waitKey(2000)

#cv2.destroyAllWindows()

face_cascade = cv2.CascadeClassifier('location of haarcascade_frontalface_default.xml')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

#print(type(faces))
#print(faces)

for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),3)

resized = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))
cv2.imshow('Gray', resized)
cv2.waitKey(0)