import numpy as np  # importing necessary libraries
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import cv2.cv2 as cv2
import imutils

img = None


class Ui_MainWindow(object):  # defining a class to keep all functions together
    def setupUi(self, MainWindow):  # main function that initialize all UI objects
        MainWindow.setObjectName("MainWindow")  # it contains all the labels sliders bars and actions
        MainWindow.resize(792, 730)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 40, 341, 411))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(400, 40, 341, 411))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(60, 480, 31, 16))
        self.label_3.setObjectName("label_3")
        self.horizontalSlider_2 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_2.setGeometry(QtCore.QRect(20, 520, 131, 22))
        self.horizontalSlider_2.setMinimum(-255)
        self.horizontalSlider_2.setMaximum(255)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(60, 550, 47, 13))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(60, 620, 31, 16))
        self.label_5.setObjectName("label_5")
        self.horizontalSlider_3 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_3.setGeometry(QtCore.QRect(20, 580, 131, 22))
        self.horizontalSlider_3.setMinimum(-255)
        self.horizontalSlider_3.setMaximum(255)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.horizontalSlider_4 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_4.setGeometry(QtCore.QRect(20, 640, 131, 22))
        self.horizontalSlider_4.setMinimum(-255)
        self.horizontalSlider_4.setMaximum(255)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.verticalSlider = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider.setGeometry(QtCore.QRect(340, 500, 22, 141))
        self.verticalSlider.setMinimum(0)
        self.verticalSlider.setMaximum(100)
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setObjectName("verticalSlider")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(330, 650, 61, 16))
        self.label_6.setObjectName("label_6")
        self.verticalSlider_2 = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider_2.setGeometry(QtCore.QRect(440, 500, 22, 141))
        self.verticalSlider_2.setMinimum(0)
        self.verticalSlider_2.setValue(10)
        self.verticalSlider_2.setMaximum(20)
        self.verticalSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_2.setObjectName("verticalSlider_2")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(430, 650, 61, 16))
        self.label_7.setObjectName("label_7")
        self.verticalSlider_3 = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider_3.setGeometry(QtCore.QRect(540, 500, 22, 141))
        self.verticalSlider_3.setMinimum(1)
        self.verticalSlider_3.setMaximum(10)
        self.verticalSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_3.setObjectName("verticalSlider_3")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(530, 650, 61, 16))
        self.label_8.setObjectName("label_8")
        self.b1 = QtWidgets.QPushButton(self.centralwidget)
        self.b2 = QtWidgets.QPushButton(self.centralwidget)
        self.b2.setGeometry(QtCore.QRect(190, 560, 110, 30))
        self.b2.setObjectName("b2")
        self.b2.setText("Apply Sliders Values")
        self.b2.clicked.connect(self.Update)
        self.b1.setGeometry(QtCore.QRect(190, 600, 110, 30))
        self.b1.setObjectName("b1")
        self.b1.setText("Reset Button")
        self.b1.clicked.connect(self.Reset_Button)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 792, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 792, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_Image = QtWidgets.QAction(MainWindow)  # connecting actions to relevant functions
        self.actionLoad_Image.setObjectName("actionLoad_Image")
        self.actionLoad_Image.triggered.connect(self.loadImage)

        self.actionSave_Image = QtWidgets.QAction(MainWindow)
        self.actionSave_Image.setObjectName("actionSave_Image")
        self.actionSave_Image.triggered.connect(self.savePhoto)

        self.actionBlur = QtWidgets.QAction(MainWindow)
        self.actionBlur.setObjectName("actionBlur")
        self.actionBlur.triggered.connect(self.Gaussian_blur)

        self.actionDeblur = QtWidgets.QAction(MainWindow)
        self.actionDeblur.setObjectName("actionDeblur")
        self.actionDeblur.triggered.connect(self.Deblur)
        self.actionGrayscale = QtWidgets.QAction(MainWindow)
        self.actionGrayscale.setObjectName("actionGrayscale")
        self.actionGrayscale.triggered.connect(self.GrayScale)

        self.actionCrop = QtWidgets.QAction(MainWindow)
        self.actionCrop.setObjectName("actionCrop")
        self.actionCrop.triggered.connect(self.Crop)

        self.actionFlip = QtWidgets.QAction(MainWindow)
        self.actionFlip.setObjectName("actionFlip")
        self.actionFlip.triggered.connect(self.Flip)

        self.actionMirror = QtWidgets.QAction(MainWindow)
        self.actionMirror.setObjectName("actionMirror")
        self.actionMirror.triggered.connect(self.Mirror)

        self.actionRotate = QtWidgets.QAction(MainWindow)
        self.actionRotate.setObjectName("actionRotate")
        self.actionRotate.triggered.connect(self.Rotation)

        self.actionReserve = QtWidgets.QAction(MainWindow)
        self.actionReserve.setObjectName("actionReserve")
        self.actionReserve.triggered.connect(self.Reverse)

        self.actionAdd_Noise = QtWidgets.QAction(MainWindow)
        self.actionAdd_Noise.setObjectName("actionAdd_Noise")
        self.actionAdd_Noise.triggered.connect(self.AddNoise)

        self.actionEdge_Detection = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection.setObjectName("actionEdge_Detection")
        self.actionEdge_Detection.triggered.connect(self.EdgeDetection)

        self.menuFile.addAction(self.actionLoad_Image)
        self.menuFile.addAction(self.actionSave_Image)
        self.menuEdit.addAction(self.actionBlur)
        self.menuEdit.addAction(self.actionDeblur)
        self.menuEdit.addAction(self.actionGrayscale)
        self.menuEdit.addAction(self.actionCrop)
        self.menuEdit.addAction(self.actionFlip)
        self.menuEdit.addAction(self.actionMirror)
        self.menuEdit.addAction(self.actionRotate)
        self.menuEdit.addAction(self.actionReserve)
        self.menuEdit.addAction(self.actionAdd_Noise)
        self.menuEdit.addAction(self.actionEdge_Detection)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

        self.horizontalSlider_2.valueChanged['int'].connect(self.red_val)  # connecting sliders to relevant functions
        self.horizontalSlider_3.valueChanged['int'].connect(self.green_val)
        self.horizontalSlider_4.valueChanged['int'].connect(self.blue_val)
        self.verticalSlider.valueChanged['int'].connect(self.brightness_val)
        self.verticalSlider_2.valueChanged['int'].connect(self.contrast_val)
        self.verticalSlider_3.valueChanged['int'].connect(self.saturation_val)

        self.filename = None  # initialize first values
        self.output = None
        self.brightness = 0
        self.contrast = 10
        self.saturation = 1
        self.red = 0
        self.green = 0
        self.blue = 0

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def __init__(self):  # defining __init__ function here so that we can use blur_count and isGray in other functions
        self.blur_count = 0
        self.isGray = False
        self.isImageLoaded = False

    def red_val(self, value):
        self.red = value

    def green_val(self, value):
        self.green = value

    def blue_val(self, value):
        self.blue = value

    def brightness_val(self, value):
        self.brightness = value

    def contrast_val(self, value):
        self.contrast = value

    def saturation_val(self, value):
        self.saturation = value

    def Set_red(self):  # red slider function
        if self.red == 0:  # if slider is in original position do nothing
            return
        if self.isGray:  # if image is Gray, we simply can't change colors so we check it
            print("The Red channel could not be changed because the image is gray")
            return
        self.image[:, :, 2] = self.image[:, :, 2] + self.red  # add slider value to red channel and clip for borders
        self.image[:, :, 2] = np.clip(self.image[:, :, 2], 0, 255)
        self.image = np.clip(self.image, 0, 255)
        self.output = self.image
        self.setOutput(self.output)

    def Set_green(self):  # green slider function
        if self.green == 0:  # if slider is in original position do nothing
            return
        if self.isGray:  # if image is Gray, we simply can't change colors so we check it
            print("The Green channel could not be changed because the image is gray")
            return
        self.image[:, :, 1] = self.image[:, :, 1] + self.green  # add slider value to green channel and clip for borders
        self.image[:, :, 1] = np.clip(self.image[:, :, 1], 0, 255)
        self.image = np.clip(self.image, 0, 255)
        self.output = self.image
        self.setOutput(self.output)

    def Set_blue(self):  # blue slider function
        if self.blue == 0:  # if slider is in original position do nothing
            return
        if self.isGray:  # if image is Gray, we simply can't change colors so we check it
            print("The Blue channel could not be changed because the image is gray")
            return
        self.image[:, :, 0] = self.image[:, :, 0] + self.blue  # add slider value to blue channel and clip for borders
        self.image[:, :, 0] = np.clip(self.image[:, :, 0], 0, 255)
        self.image = np.clip(self.image, 0, 255)
        self.output = self.image
        self.setOutput(self.output)

    def Set_Contrast(self):
        if self.contrast == 10:  # if slider is in original position do nothing
            return
        if self.isGray:  # if image is Gray, we simply can't change contrast so we check it
            print("The Contrast could not be changed because the image is gray")
            return
        self.output = cv2.addWeighted(self.output, self.contrast / 10, np.zeros(self.output.shape, self.output.dtype),
                                      0,
                                      self.brightness)  # we divide contrast because we get values ranging 0 to 20
        self.image = self.output  # and we want contrast to range 0 to 2
        self.setOutput(self.output)  # 0-1 for decrease 1-2 for increase

    def Set_Saturation(self):
        if self.saturation == 1:  # if slider is in original position do nothing
            return
        if self.isGray:  # if image is Gray, we simply can't change saturation so we check it
            print("The Saturation could not be changed because the image is gray")
            return
        hsv = cv2.cvtColor(self.output, cv2.COLOR_BGR2HSV)  # first we change color space to hsv because
        h, s, v = cv2.split(hsv)  # we can get saturation easily which is s
        s = s * self.saturation  # then we multiply it with sliders value then clip for borders
        s = np.clip(s, 0, 255)
        hsv = cv2.merge((h, s, v))  # then we merge new s with hv and change color space to bgr
        self.output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.image = self.output
        self.setOutput(self.output)

    def Reset_Button(self):  # function that reset sliders to initial value
        self.verticalSlider.setValue(0)
        self.verticalSlider_2.setValue(10)
        self.verticalSlider_3.setValue(1)
        self.horizontalSlider_2.setValue(0)
        self.horizontalSlider_3.setValue(0)
        self.horizontalSlider_4.setValue(0)

    def Set_Brightness(self):
        if self.brightness == 0:  # if slider is in original position do nothing
            return
        if self.isGray:  # if image is Gray, we simply can't change brightness so we check it
            print("The Brightness could not be changed because the image is gray")
            return
        hsv = cv2.cvtColor(self.output, cv2.COLOR_BGR2HSV)  # first we change color space to hsv because
        h, s, v = cv2.split(hsv)  # we can get brightness easily which is v
        lim = 255 - self.brightness  # then we add slider value and clip
        v[v > lim] = 255  # then merge and change color space to bgr
        v[v <= lim] += self.brightness
        v = np.clip(v, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        self.output = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        self.image = self.output
        self.setOutput(self.output)

    def Update(self):  # we define this function to update all the slider values at once
        if not self.isImageLoaded:  # if there is no image loaded do nothing
            print("No image loaded")
            return
        self.Set_Brightness()
        self.Set_Contrast()
        self.Set_Saturation()
        self.Set_red()
        self.Set_green()
        self.Set_blue()

    def loadImage(self):  # simply load an image
        self.blur_count = 0
        self.isImageLoaded = True
        self.isGray = False
        self.filename, _ = QFileDialog.getOpenFileName(None, 'Open File', '', 'Image Files (*.jpg *.png)')
        self.image = cv2.imread(self.filename)
        self.setPhoto(self.image)

    def setPhoto(self, image):  # set input image
        self.output = image
        image = imutils.resize(image, width=400)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def setOutput(self, image):  # set output image
        image = imutils.resize(image, width=400)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(image))

    def Gaussian_blur(self):  # blur function, we choose gaussian and 9,9 size
        if not self.isImageLoaded:  # if there is no image loaded do nothing
            print("No image loaded")
            return
        gauss = cv2.GaussianBlur(self.image, (9, 9), 10)
        self.output = gauss
        self.image = gauss
        self.setOutput(gauss)
        self.blur_count += 1  # after applying blur we increase blur_count to calculate how many times we can use deblur
        print("Gaussian Blur")
        print("You can use deblur " + str(self.blur_count) + " time(s).")

    """""""""
    def gausskernel(self):  # we were not happy with deblur and tried different things, its one of them
        gausskernel = np.zeros((9, 9), np.float32)  # it calculate gausskernel with any given size and sigma
        for i in range(3):                          # but at the end we did not choose to use it
            for j in range(3):
                norm = math.pow(i - 4, 2) + pow(j - 4, 2)
                gausskernel[i, j] = math.exp(-norm / (2 * math.pow(10, 2))) / 2 * math.pi * pow(10, 2)
        sum = np.sum(gausskernel)
        kernel = gausskernel / sum
        for i in range(3):
            for j in range(3):
                print(kernel[i,j])
        return kernel
    """""""""

    def Deblur(self):
        if not self.isImageLoaded:  # if there is no image loaded do nothing
            print("No image loaded")
            return
        if self.blur_count > 0:  # checking blur count
            sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # define kernel add filter the image
            deblur = cv2.filter2D(self.image, -1, sharpen_kernel)  # with it to deblur

            """""""""
            blurred = cv2.GaussianBlur(self.image,(13, 13), 10)   #working different then current one, we choose
            deblur = 2 * self.image - blurred                     #current one over this  
            deblur = np.maximum(deblur, np.zeros(deblur.shape))
            deblur = np.minimum(deblur, 255 * np.ones(deblur.shape))
            deblur = deblur.round().astype(np.uint8)
            """""""""
            # deblur = cv2.fastNlMeansDenoising(self.image, None, 30, 7, 21)
            self.output = deblur
            self.image = deblur
            self.setOutput(deblur)
            self.blur_count -= 1
            print("Deblur")
            print("You can use deblur " + str(self.blur_count) + " time(s).")

        else:
            print("You can only use deblur function as many times as you have used blur function")
            return

    def GrayScale(self):  # grayscale function
        if not self.isImageLoaded:  # if there is no image loaded do nothing
            print("No image loaded")
            return
        if (self.isGray == False):  # we check if it is already gray
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # we change color space to gray and make it grayscale
            self.output = gray
            self.image = gray
            self.setOutput(gray)
            self.isGray = True  # now image is gray
            print("GrayScale")
        else:
            print("You can't use GrayScale function if the image is already GrayScale")
            return

    def Reverse(self):  # simple reverse function
        if not self.isImageLoaded:  # if there is no image loaded do nothing
            print("No image loaded")
            return
        reverse = cv2.bitwise_not(self.image)  # reverse all the 0 with 1 and vice versa
        self.output = reverse
        self.image = reverse
        self.setOutput(reverse)
        print("Reverse")

    def Flip(self):  # simple flip function
        if not self.isImageLoaded:  # if there is no image loaded do nothing
            print("No image loaded")
            return
        flip = cv2.flip(self.image, 0)  # predefined flip function to flip image
        self.output = flip
        self.image = flip
        self.setOutput(flip)
        print("Flip")

    def Mirror(self):  # simple mirror function
        if not self.isImageLoaded:  # if there is no image loaded do nothing
            print("No image loaded")
            return
        mirror = cv2.flip(self.image, 1)  # predefined flip function to mirror
        self.output = mirror
        self.image = mirror
        self.setOutput(mirror)
        print("Mirror")

    def Rotation(self):  # rotate function
        if not self.isImageLoaded:  # if there is no image loaded do nothing
            print("No image loaded")
            return
        rotation = imutils.rotate_bound(self.image, 90)  # we choose to rotate 90 degree right
        self.output = rotation
        self.image = rotation
        self.setOutput(rotation)
        print("Rotation")

    def AddNoise(self):  # we choose to add gaussian noise, only difference between rgb and gray
        if not self.isImageLoaded:  # if there is no image loaded do nothing
            print("No image loaded")
            return
        # is adding the noise to 3 channel or 1 channel
        if (self.isGray == False):  # if image is rgb
            if not self.isImageLoaded:  # if there is no image loaded do nothing
                print("No image loaded")
                return
            gaussian_noise = self.image.copy()
            gaussian_noise = cv2.randn(gaussian_noise, (0, 0, 0),
                                       (10, 10, 10))  # adding random noise and clip for borders
            gaussian_noise = cv2.add(self.image, gaussian_noise)
            gaussian_noise = np.clip(gaussian_noise, 0, 255)
            self.output = gaussian_noise
            self.image = gaussian_noise
            self.setOutput(gaussian_noise)
            print("Add Noise")
        else:  # if image is gray
            gaussian_noise = self.image.copy()
            gaussian_noise = cv2.randn(gaussian_noise, (0), (10))
            gaussian_noise = cv2.add(self.image, gaussian_noise)
            gaussian_noise = np.clip(gaussian_noise, 0, 255)
            self.output = gaussian_noise
            self.image = gaussian_noise
            self.setOutput(gaussian_noise)

    def EdgeDetection(self):  # predefined edge detection function called canny
        if not self.isImageLoaded:  # if there is no image loaded do nothing
            print("No image loaded")
            return
        edge = cv2.Canny(self.image, 100, 200)  # we choose thresholds 100 and 200
        self.output = edge
        self.image = edge
        self.setOutput(edge)
        self.isGray = True
        print("EdgeDetection")

    def Crop(self):  # crop function, choose 2 point with your mouse and they will be the border of your new image
        posList = []
        if not self.isImageLoaded:  # if there is no image loaded do nothing
            print("No image loaded")
            return

        def onMouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                posList.append((x, y))

        img = self.image.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', onMouse)
        while True:
            cv2.imshow('image', img)  # we show image in new windows to crop
            k = cv2.waitKey(1) & 0xFF
            if k == ord('c'):
                break
            if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
                break
            if len(posList) == 2:  # we look for mouse position and decide if its the start or end position
                if posList[0][0] > posList[1][0]:
                    x_start = posList[1][0]
                    x_end = posList[0][0]
                else:
                    x_start = posList[0][0]
                    x_end = posList[1][0]
                if posList[0][1] > posList[1][1]:
                    y_start = posList[1][1]
                    y_end = posList[0][1]
                else:
                    y_start = posList[0][1]
                    y_end = posList[1][1]
                cropped = img[y_start:y_end, x_start:x_end]
                self.output = cropped
                self.image = cropped
                self.setOutput(cropped)
                break
        cv2.destroyAllWindows()

    def savePhoto(self):  # simple save function
        if not self.isImageLoaded:  # if there is no image loaded do nothing
            print("No image loaded")
            return
        filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
        cv2.imwrite(filename, self.output)
        print('Image saved as:', filename)

    def retranslateUi(self, MainWindow):  # setup_ui function calls it
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ImageTool"))
        self.label.setText(_translate("MainWindow", "Please load an Image"))
        self.label_2.setText(_translate("MainWindow", "Output"))
        self.label_3.setText(_translate("MainWindow", "Red"))
        self.label_4.setText(_translate("MainWindow", "Green"))
        self.label_5.setText(_translate("MainWindow", "Blue"))
        self.label_6.setText(_translate("MainWindow", "Brightness"))
        self.label_7.setText(_translate("MainWindow", "Contrast"))
        self.label_8.setText(_translate("MainWindow", "Saturation"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.actionLoad_Image.setText(_translate("MainWindow", "Load Image"))
        self.actionSave_Image.setText(_translate("MainWindow", "Save Image"))
        self.actionBlur.setText(_translate("MainWindow", "Blur"))
        self.actionDeblur.setText(_translate("MainWindow", "Deblur"))
        self.actionGrayscale.setText(_translate("MainWindow", "Grayscale"))
        self.actionCrop.setText(_translate("MainWindow", "Crop"))
        self.actionFlip.setText(_translate("MainWindow", "Flip"))
        self.actionMirror.setText(_translate("MainWindow", "Mirror"))
        self.actionRotate.setText(_translate("MainWindow", "Rotate"))
        self.actionReserve.setText(_translate("MainWindow", "Reverse Colors"))
        self.actionAdd_Noise.setText(_translate("MainWindow", "Add Noise"))
        self.actionEdge_Detection.setText(_translate("MainWindow", "Edge Detection"))


if __name__ == "__main__":  # main fucntion
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
