import cv2             #opencv
import urllib.request  #para abrir y leer URL
import numpy as np
import matplotlib.pyplot as plt
     
#============================= PROGRAMA DE CLASIFICACION DE OBJETOS USANDO DROIDCAM =============================

# Iniciamos nuestras variables a utilizar
NombreVentana = 'CAMARA DEL MOVIL'             #Nombre de la ventana
cv2.namedWindow(NombreVentana,cv2.WINDOW_AUTOSIZE)   #Actualizar el nombre en la ventana
classNames = []                                #Arreglo que contendra los nombres de los objetos detectados
classFile = 'Objetos.names'                    #Archivo donde estaran los nombres de los objetos que se peuden clasificar

# recorremos los nombres de los objetos a clasificar
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Archivo de los datos que ya fueron entrenados para el reconocimiento de imagenes
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Representamos una API de alto nivel para redes de detección de objetos
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Capturamos la camara a utilizar del latop
captura = cv2.VideoCapture(2)

# Bucle para clasificar nuestros objetos
while(True):
    grabbed,img=captura.read()  #nos devolvera una imagen de la camara que se esta usando
    if not grabbed:             #verificar si se esta grabando con la camara
        break
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # rotamos la imagen para que se adecue al movil en cuanto a la rotacion
    
    classIds, confs, bbox = net.detect(img,confThreshold=0.5) # detectamos la imagen 
    print(classIds,bbox)
    if len(classIds) != 0:      #si los puntos calves de la imagen es 0 e spor que no tiene contorno la imagen
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
             cv2.rectangle(img,box,color=(255,0,128),thickness = 3)  #mostramos en rectangulo el objetivo
             cv2.putText(img, classNames[classId-1], (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2) #mostramos el nombre del objeto
    cv2.imshow(NombreVentana,img) # mostramos la imagen
    #esperamos a que se presione ESC para terminar el programa
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
# destrumos la ventana donde se msotraba la camara del celular
cv2.destroyAllWindows()
