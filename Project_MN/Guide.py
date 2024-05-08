import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class GuideScreen(QWidget):
    returnToMainMenu = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create the main layout
        main_layout = QGridLayout(self)
        main_layout.setAlignment(Qt.AlignTop)

        # Create the title label and add it to the main layout
        title_label = QLabel("How to use the Grasp Detector", self)
        title_font = QFont('Arial', 20)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label, 0, 0, 1, 2, alignment=Qt.AlignHCenter)

        # Create the back button and add it to the main layout
        back_button = QPushButton('Back', self)
        back_button.setFixedSize(100, 60)
        back_button.clicked.connect(self.backToMainMenu)
        main_layout.addWidget(back_button, 1, 0, alignment=Qt.AlignLeft)

        self.setWindowTitle('Guide')
        self.setFixedSize(1000, 1000)  # Set a fixed window size

    def backToMainMenu(self):
        self.returnToMainMenu.emit()  # Emit the signal to return to the main menu screen
        self.close()

    def closeEvent(self, event):
        if event.spontaneous():  # Check if the close event was triggered by the window system
            QApplication.quit()  