import os
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


class PredictionDialog(QDialog):
    def __init__(self, predicted_label):
        super().__init__()
        self.setWindowTitle("Prediction Result")
        self.layout = QVBoxLayout()
        self.label = QLabel(f"Predicted Label: {predicted_label}")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)


class detectObjectScreen(QWidget):
    returnToMainMenu = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.model_path = 'mobilenetv3_grasp_model.h5'
        self.class_labels = ['Palmar_Wrist_Pronated', 'Palmar_wrist_neutral', 'Pinch', 'Tripod']
        self.initUI()
        self.model = load_model(self.model_path)


    def initUI(self):

         # Create the main layout
        main_layout = QVBoxLayout(self)

        # Create the title label and add it to the main layout
        title_label = QLabel("Detect Object: Please capture an image of the object", self)
        title_font = QFont('Arial', 20)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label, alignment=Qt.AlignHCenter)


        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        #main_layout.addLayout(button_layout)

        # Create the back button and add it to the button layout
        back_button = QPushButton('Back', self)
        back_button.setFixedSize(100, 60)
        back_button.clicked.connect(self.backToMainMenu)
        button_layout.addWidget(back_button, alignment=Qt.AlignLeft)

        # Create the capture button and add it to the button layout
        capture_button = QPushButton('Capture', self)
        capture_button.setFixedSize(100, 60)
        capture_button.clicked.connect(self.captureImage)
        button_layout.addWidget(capture_button, alignment=Qt.AlignRight)


        button_layout.addSpacing(60)

########################## fix the reset button #########################################
        reset_button = QPushButton('Reset', self)
        reset_button.setFixedSize(100, 60)
        reset_button.clicked.connect(self.resetCamera)

        # Calculate the coordinates for the reset button
        reset_button_width = reset_button.width()
        reset_button_height = reset_button.height()
        window_width = self.width()
        window_height = self.height()
        reset_button_x = window_width - reset_button_width - 100
        reset_button_y = window_height - reset_button_height - 800

    
        main_layout.addLayout(button_layout)

        # Create a label to display the webcam feed and add it to the main layout
        self.webcam_label = QLabel(self)
        self.webcam_label.setFixedSize(640, 480)
        main_layout.addWidget(self.webcam_label, alignment=Qt.AlignCenter)

        self.setWindowTitle('Object Detection')
        self.setFixedSize(1000, 600)  # Set a fixed window size  #or 600

        # Open the webcam and start capturing frames
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.webcamFeed)
        self.timer.start(30)

    def backToMainMenu(self):
        self.returnToMainMenu.emit()
        self.close()

    def webcamFeed(self):
        # Read a frame from the webcam
        ret, frame = self.capture.read()
        if ret:
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize the frame to fit the label
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            # Convert the frame to QImage
            image = QImage(frame_resized.data, frame_resized.shape[1], frame_resized.shape[0], QImage.Format_RGB888)
            # Set the QImage as the pixmap for the label
            pixmap = QPixmap.fromImage(image)
            self.webcam_label.setPixmap(pixmap)

    def captureImage(self):
        # Read a frame from the webcam
        ret, frame = self.capture.read()
        if ret:
            # Stop the timer to pause the webcam feed
            self.timer.stop()

            # Convert the captured image to RGB format
            captured_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize the captured image to fit the label
            captured_image_resized = cv2.resize(captured_image_rgb, (640, 480))
            # Convert the captured image to QImage
            image = QImage(captured_image_resized.data, captured_image_resized.shape[1], captured_image_resized.shape[0], QImage.Format_RGB888)
            # Set the QImage as the pixmap for the label
            pixmap = QPixmap.fromImage(image)
            self.webcam_label.setPixmap(pixmap)

            # Create a QMessageBox dialog
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Use this Image?")
            msg_box.setText("Do you want to use the captured image?")
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            # Execute the dialog and get the clicked button
            button = msg_box.exec()

            if button == QMessageBox.Yes:
                # Overwrite the captured image file
                image = 'object_image.jpg'
                # Save the captured image in the same directory
                cv2.imwrite(image, frame)

               

                # Perform time-consuming operations here (e.g., image processing, data loading, etc.)
                captured_image = load_img(image, target_size=(224, 224))
                captured_image = img_to_array(captured_image)
                captured_image = captured_image.reshape((1, captured_image.shape[0], captured_image.shape[1], captured_image.shape[2]))
                captured_image = captured_image / 255.0

                # Perform the prediction using the model
                predicted_probabilities = self.model.predict(captured_image)[0]
                predicted_class_index = tf.argmax(predicted_probabilities)
                predicted_class_label = self.class_labels[predicted_class_index]
                print(predicted_class_label)  

                prediction_dialog = PredictionDialog(predicted_class_label)
                prediction_dialog.exec()
                

                 #  NEED TO DISPLAY THE CLASS LABEL ON ANOTHER SCREEN WITH A PICTURE OF THE GRASP
                # Once the operations are done, close the progress dialog
   
            else:
                # Start the timer again to resume the webcam feed
                self.timer.start(30)

    def resetCamera(self):
            # Restart the timer to resume the webcam feed
            self.timer.start(30)

    def closeEvent(self, event):
        if event.spontaneous():
            self.capture.release()
            QApplication.quit()

