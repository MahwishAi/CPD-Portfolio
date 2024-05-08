import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Guide import GuideScreen
from Detect_Object import detectObjectScreen


class MenuScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.guide_screen = None

    def initUI(self):
        # Create a vertical layout
        layout = QVBoxLayout()

        # Create a label
        label = QLabel('Grasp Detector', self)
        label.setAlignment(Qt.AlignCenter)
        # Set the font size of the label
        font = QFont('Arial', 18)  # Specify the font family and size
        label.setFont(font)

        # Create three buttons
        detect_button = QPushButton('Detect Object', self)
        guide_button = QPushButton('How to Guide?', self)
        exit_button = QPushButton('Exit', self)

        # Set smaller sizes for the buttons
        detect_button.setFixedSize(180, 60)
        guide_button.setFixedSize(180, 60)
        exit_button.setFixedSize(180, 60)

        # Set the font size of the buttons
        font = QFont()
        font.setPointSize(12)  # Adjust the font size
        detect_button.setFont(font)
        guide_button.setFont(font)
        exit_button.setFont(font)

        # Add label, spacing, and buttons to the layout
        layout.addWidget(label)
        layout.addSpacing(50)
        layout.addWidget(detect_button)
        layout.addSpacing(25)
        layout.addWidget(guide_button)
        layout.addSpacing(25)
        layout.addWidget(exit_button)
        layout.addSpacing(15)  # Adjusted spacing

        exit_button.clicked.connect(QApplication.quit) # Connect the button to quit the application
        guide_button.clicked.connect(self.showGuideScreen)  # Connect the button to show the guide screen
        detect_button.clicked.connect(self.showdetectObjectScreen)  # Connect the button to show the guide screen

        # Set the layout alignment
        layout.setAlignment(Qt.AlignCenter)

        # Set the layout for the window
        self.setLayout(layout)

        # Set the window properties
        self.setGeometry(200, 200, 500, 500)  # Adjusted dimensions
        self.setWindowTitle('Menu')
        self.show()

    def showGuideScreen(self):
        if self.guide_screen is None:
            self.guide_screen = GuideScreen()
            self.guide_screen.returnToMainMenu.connect(self.handleGuideScreenClosed)  # Connect the signal to return to the menu screen
        self.hide()
        self.guide_screen.show()

    def showdetectObjectScreen(self):
        if self.guide_screen is None:
            self.guide_screen = detectObjectScreen()
            self.guide_screen.returnToMainMenu.connect(self.handleGuideScreenClosed)  # Connect the signal to return to the menu screen
        self.hide()
        self.guide_screen.show()

    def handleGuideScreenClosed(self):
        self.show()
        self.guide_screen.deleteLater()
        self.guide_screen = None

    def closeEvent(self, event):
        if self.guide_screen is not None:
            self.guide_screen.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    menu_screen = MenuScreen()
    sys.exit(app.exec_())
        
    
