import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

capture = cv2.VideoCapture(0)
while True:
    ret, img = capture.read()
    img3 = cv2.imread(r"")
    print (type(img))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 135,255,cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(gray, 30, 100)
    t = []
    he,wi,dim = img.shape
    img2 = img.reshape(he*wi,dim)

    contours, __ = cv2.findContours(thresh, cv2.THRESH_BINARY, cv2.CHAIN_APPROX_NONE)

   
    kmeans = KMeans(n_clusters=4).fit(img2)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    print(kmeans.cluster_centers_.shape)
    centroid = np.array([254.96338379, 254.97917064, 254.96566591,203.02751685,  71.16432755,  63.30843187,36.89086374,  27.87558466, 235.27081384,77.68192008, 176.2862983,   35.95582223])
    cent = centroid.reshape(4,3)
    kmeans.cluster_centers_ = cent
    print("cluster :", kmeans.cluster_centers_)

   
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        img3 = cv2.resize(img3, (w,h))
        box = img[y:y+h, x:x+w]
       
        height, width, z = box.shape
        box2 = box.reshape(height*width, z)
      
        midpoint = (int((x+x+w)/2),int((y+y+h)/2))
       
        mid = img[midpoint[1], midpoint[0]]
        mid2 = mid.reshape(1,3)
        prediction = kmeans.predict(mid2)
        print("mid :", mid2)
        print("predi :", prediction)
        if (prediction==0):
            print("white")
            cv2.putText(img, "white", (int(w) + 3, int(h + 3)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
        elif (prediction==1):
            print("blue")
            cv2.putText(img, "blue", (x+ 3, y +3), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
            img[y:y+h, x:x+w] = img3
        elif (prediction==3):
            print("green")
            cv2.putText(img, "green", (x+ 3, y +3), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))  
        else:
            print("red")
            cv2.putText(img, "red", (x+ 3, y +3), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0),4)
        
    
    print("centroid : ", kmeans.cluster_centers_)
    print("box2 :", box2[0])
   
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
