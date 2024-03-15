author = "skyfly"
email = "tianxianghuang@whut.edu.cn"

import cv2
from pyzbar.pyzbar import decode
import webbrowser


img = cv2.imread("qrcode_1.png")
cap = cv2.VideoCapture(0)

data = ['link']

while True:
    success ,img = cap.read()
    QR_code = decode(img)
    #print(QR_code)

    for QR in QR_code:
        QR_data = QR.data.decode('utf-8')

        if(QR_data != data[-1]):
            data.append(QR_data)







            webbrowser.open(QR_data)
            print(data)

        point = QR.rect
        #print(point)

        #画矩形框和添加文字
        cv2.rectangle(img,(point[0],point[1]),(point[0]+point[2],point[1]+point[3]),(100,0,100),3)


        cv2.putText(img,QR_data,(point[0],point[1]-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.6,(100,0,0),1)


    cv2.imshow("output",img)

    if cv2.waitKey(1) & 0xFF == 27:
        break