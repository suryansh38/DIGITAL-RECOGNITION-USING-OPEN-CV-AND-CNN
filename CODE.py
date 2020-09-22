import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
from keras.optimizers import Adam
from keras.datasets import mnist

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
X_train=np.float32(X_train)/255
X_test=np.float32(X_test)/255

Y_train=keras.utils.to_categorical(Y_train,num_classes=10)
Y_test=keras.utils.to_categorical(Y_test,num_classes=10)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_train.shape)


model=Sequential()

model.add(Conv2D(64,kernel_size=(5,5),input_shape=(28,28),padding="same",activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))

model.add(Conv2D(128,kernel_size=(5,5),padding="same",activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))

model.add(Flatten())
model.add(Dense(84,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10,activation="softmax"))

model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(X_train,Y_train,batch_size=256,epochs=5,verbose=1,validation_data=(X_test,Y_test))
score=model.evaluate(X_test,Y_test,batch_size=256,verbose=0)
print("accuracy: ",score[1])

flag=1

while(1):

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    ret,frame = cap.read()

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    low = np.array([250])
    high = np.array([255])

    image_mask = cv2.inRange(frame,low,high)

    output1 = cv2.bitwise_and(frame,frame,mask = image_mask)
    output2 = output1
    wt,ht = output2.shape

    for i in range(wt):
        for j in range(ht):
            output2[i,j]=0
    while ret:
        ret,image = cap.read()
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        low = np.array([250])
        high = np.array([255])

        image_mask = cv2.inRange(frame,low,high)
        output1 = cv2.bitwise_and(frame,frame,mask = image_mask)
        cv2.rectangle(output2,(100,100),(400,400),(255,255,255),2)
        cv2.rectangle(image,(100,100),(400,400),(255,255,255),2)
        output2 = cv2.bitwise_or(output2,output1,output2)
        show=cv2.flip(image,1)
        cv2.imshow('kjfknekn',output2)
        cv2.imshow('faltu',show)
        if cv2.waitKey(1)==27:
            break
        elif cv2.waitKey(1)==ord("q"):
            flag=0
            break
    if flag==0:
        break
    img = output2[100:400,100:400]
    img=cv2.flip(img,1)
    cap.release()
    cv2.destroyAllWindows()

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img = cv2.erode(img, k ,iterations =1)
    cv2.imshow('ewjj',img)
    img = cv2.dilate(img,k,iterations = 1 )
    out=cv2.resize(img,(28,28),cv2.INTER_CUBIC)
    cv2.imshow("out",out)
    print(out.shape)
    cv2.imshow('image',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    y=np.array([out])
    y=y.reshape((1,28,28))

    cv2.imwrite("/home/the_confused_1/Desktop/output_of_finalcode.png",img)

    result=model.predict(y)
    print(np.argmax(result))
    if flag==0:
        break
    if cv2.waitKey(1)==ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
