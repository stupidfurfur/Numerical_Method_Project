import sys
import os
import cv2
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLabel,QHBoxLayout,QApplication, QWidget, QPushButton, QFileDialog)
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import numpy as np
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        # Update the layout
        self.img1 = None
        self.quantized_image = None
        
    def initUI(self):
        # box's color:gray
        style_box = '''
            background:#bbb;
            border:1px solid #bbb; 
        '''
        # button's color:white
        style_btn = '''
            background:#fff;
            border:1px solid #bbb
        '''
        self.setWindowTitle('Numerical Method Project by E94096013') #set title name
        self.setGeometry(50, 50, 1500, 1000)     #position
        
        # 2 boxes in the background with the color gray
        self.hbox1 = QWidget(self)
        self.hbox1.setGeometry(320,90,1000,800)
        self.hbox1.setStyleSheet(style_box)
        self.h_layout = QHBoxLayout(self.hbox1)
        
        # Labels on top of the buttons corresponding to 2 different parts
        # (Image Processing / Image Smoothing)
        
        # Labels represent image name (default = None)
        self.label3 = QLabel(self)  
        self.label3.setText('No Image')
        self.label3.setGeometry(QtCore.QRect(100,210, 180, 32))

        self.label4 = QLabel(self)  
        self.label4.setText('No Image')
        self.label4.setGeometry(QtCore.QRect(100,450, 180, 32))
        # Create buttons
        self.btn1 = QPushButton('Load image 1',self)  
        self.btn1.setGeometry(QtCore.QRect(100,180, 180, 32)) #(x,y,width,height)
        self.btn1.setStyleSheet(style_btn)
        self.btn1.clicked.connect(self.onButtonClick1) #call func.

        self.btn3 = QPushButton('Save image',self)  
        self.btn3.setGeometry(QtCore.QRect(100,360, 180, 32)) #(x,y,width,height)
        self.btn3.setStyleSheet(style_btn)
        self.btn3.clicked.connect(self.save_quantized_image) #call func.

        self.btn2 = QPushButton('Plot Color Quantization',self)
        self.btn2.setGeometry(QtCore.QRect(100,300, 180, 32)) #(x,y,width,height)
        self.btn2.setStyleSheet(style_btn)
        self.btn2.clicked.connect(self.color_quantization) #call func.

        self.btn4 = QPushButton('Color Inertia',self)  
        self.btn4.setGeometry(QtCore.QRect(100,240, 180, 32)) #(x,y,width,height)
        self.btn4.setStyleSheet(style_btn)
        self.btn4.clicked.connect(self.color_inertia) #call func.

        self.btn5 = QPushButton('Non-coloring',self)  
        self.btn5.setGeometry(QtCore.QRect(100,480, 180, 32)) #(x,y,width,height)
        self.btn5.setStyleSheet(style_btn)
        self.btn5.clicked.connect(self.edge_detection) #call func.

        self.btn1 = QPushButton('Load Quantized image',self)  
        self.btn1.setGeometry(QtCore.QRect(100,420, 180, 32)) #(x,y,width,height)
        self.btn1.setStyleSheet(style_btn)
        self.btn1.clicked.connect(self.onButtonClick2) #call func.

    def onButtonClick1(self):
        # Load image and get filename and filetype
        filename, filetype = QFileDialog.getOpenFileName(self, '開啟檔案', os.getcwd(),'All Files (*);;JPEG Files (*.jpg)')
        if filename:
            print(filename)
            print(filetype)
        self.img1 = cv2.imread(filename)

        # Show image name and change label after loaded another image.
        self.label3.setText('{}'.format(os.path.basename(filename)))

        # Display the loaded image in the original window
        self.hbox1.setStyleSheet(f"background-image: url({filename}); background-repeat: no-repeat; background-position: center;background-size: contain;")

    def onButtonClick2(self):
        # Load image and get filename and filetype
        filename, filetype = QFileDialog.getOpenFileName(self, '開啟檔案', os.getcwd(),'All Files (*);;JPEG Files (*.jpg)')
        if filename:
            print(filename)
            print(filetype)

        # Turn image into grayscale in order to do edge detection
        self.img2 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) 

        # Show image name and change label after loaded another image.
        self.label4.setText('{}'.format(os.path.basename(filename)))

        # Display the loaded image in the original window
        self.hbox1.setStyleSheet(f"background-image: url({filename}); background-repeat: no-repeat; background-position: center;")

    def color_quantization(self):
        self.quantized_image = self.Color_quantization(self.img1)

    def save_quantized_image(self):
        if self.quantized_image is not None:
            filename, _ = QFileDialog.getSaveFileName(self, 'Save Image', os.getcwd(), 'JPEG Files (*.jpg)')
            if filename:
                # Convert quantized image to the appropriate data type and scale it
                quantized_image_uint8 = (self.quantized_image * 255).astype(np.uint8)

                cv2.imwrite(filename, quantized_image_uint8)
                print("Quantized image saved.")
        else:
            print("No quantized image available.")


    def Color_quantization(self, img1):           
        def recreate_image(codebook, labels, w, h):
            #Recreate the (compressed) image from the code book & labels
            return codebook[labels].reshape(w, h, -1)
        
        def quantization(x):
            n_colors = x
            # Convert to floats instead of the default 8 bits integer coding.
            # Dividing by 255 is important so that plt.imshow behaves works well on float data (need to
            # be in the range [0-1])
            img = img1.copy()
            img = np.array(img, dtype=np.float64) / 255

            # Load Image and transform to a 2D numpy array.
            w, h, d = tuple(img.shape)
            assert d == 3
            image_array = np.reshape(img, (w * h, d))
            image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)

            # Do Kmeans which cluster = n_colors
            kmeans = KMeans(n_clusters=n_colors, n_init="auto", random_state=0).fit(image_array_sample)

            # Get labels for all points
            labels = kmeans.predict(image_array)
            final_img = recreate_image(kmeans.cluster_centers_, labels, w, h)
            self.quantized_image = final_img # update img

            # resize window
            ratio = img1.shape[1]/1000
            imgshape_1 = int(img1.shape[1]/ratio)
            imgshape_0 = int(img1.shape[0]/ratio)
            cv2.namedWindow("Color Quantization", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Color Quantization", imgshape_1, imgshape_0)
            cv2.imshow("Color Quantization", final_img)

        # If you choose a new image, destroyWindow before Color Quantization to aviod error 
        try:
            cv2.destroyWindow("Color Quantization")
        except:
            pass

        # resize window
        cv2.namedWindow("Color Quantization",cv2.WINDOW_NORMAL)
        ratio = img1.shape[1]/800
        imgshape_1 = int(img1.shape[1]/ratio)
        imgshape_0 = int(img1.shape[0]/ratio)
        cv2.resizeWindow("Color Quantization", imgshape_1, imgshape_0)
        cv2.createTrackbar('magnitude','Color Quantization',1,100,quantization)
        cv2.setTrackbarPos('magnitude','Color Quantization',0)
        while(True):
            # Press ESC to exit 
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        
            x = cv2.getTrackbarPos('magnitude','Color Quantization') # magnitude
            # origin image
            if x <= 1:
                cv2.imshow('Color Quantization',img1) 

    def color_inertia(self):
        img = self.img1
        img = np.array(img, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array
        w, h, d = tuple(img.shape)
        assert d == 3
        image_array = np.reshape(img, (w * h, d))
        print("Fitting model on a small sub-sample of the data")
        image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)

        # create inertia image
        elbow = []
        for i in range(1, 64):
            kmeans = KMeans(n_clusters=i, n_init="auto", random_state=0).fit(image_array_sample)
            elbow.append(kmeans.inertia_)

        differences = np.diff(elbow)

        # Find the index of the maximum difference
        for i in range(len(differences)-1):
            if abs(differences[i]-differences[i+1]) < 0.01: # tolerance = 0.01
                inflection_index = i
                break

        # Get the corresponding number of clusters
        inflection_clusters = inflection_index + 1

        # Print the inflection point
        inflection_text = f"Inflection point: {inflection_clusters}"
        print(inflection_text)

        # resize figure
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, 64), elbow, marker='o')
        plt.title('Number of Clusters')

        # Set x-axis ticks every 5 units
        x_locator = plt.MultipleLocator(2)
        plt.gca().xaxis.set_major_locator(x_locator)

        # Mark the inflection point
        plt.axvline(inflection_clusters, color='r', linestyle='--', label='Inflection Point')

        # Display the inflection point text on the plot
        plt.text(inflection_clusters, elbow[inflection_index], inflection_text, color='black', fontsize=12, verticalalignment='bottom')

        # Display the plot
        plt.legend()
        plt.show()

    def edge_detection(self):
        self.Edge_detection(self.img2)

    def Edge_detection(self,img2):
        def edge(y):
            edges = cv2.Canny(self.img2,y,y*3)

            # resize window
            ratio = edges.shape[1] / 1000
            imgshape_1 = int(edges.shape[1] / ratio)
            imgshape_0 = int(edges.shape[0] / ratio)
            cv2.namedWindow("Edge Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Edge Detection", imgshape_1, imgshape_0)
            # switch white and black using : 255 - image
            cv2.imshow("Edge Detection", 255 - edges)

        # If you choose a new image, destroyWindow before Color Quantization to aviod error     
        try:
            cv2.destroyWindow("Edge Detection")
        except:
            pass

        # resize
        cv2.namedWindow("Edge Detection",cv2.WINDOW_NORMAL)
        ratio = img2.shape[1]/800
        imgshape_1 = int(img2.shape[1]/ratio)
        imgshape_0 = int(img2.shape[0]/ratio)
        cv2.resizeWindow("Edge Detection", imgshape_1, imgshape_0)
        cv2.createTrackbar('magnitude','Edge Detection',0,400,edge)
        cv2.setTrackbarPos('magnitude','Edge Detection',0)
        while(True):
            # Press ESC to exit 
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            
            y = cv2.getTrackbarPos('magnitude','Edge Detection') # magnitude
            # origin image

            if y == 0:
                cv2.imshow('Edge Detection',self.img2) 
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWidget()
    w.show()
    sys.exit(app.exec_())