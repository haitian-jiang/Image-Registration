import sys
from PyQt5.QtWidgets import *
import numpy as np
import window as cv_ui
import button_func as unit1


class MainDialog(QMainWindow):
    """ MainDialog class
    This class includes all the data we get from the UI which will be used in the image registration,
    and also provides functions for UI interaction and picture display.

    Attributes:
    -----------
    ui: Ui_MainWindow class
        Get the UI we created before.
    img: np.ndarray
        Source image for registration. 
    img_tar: np.ndarray
        Target image as the reference in the image registration. (It's size must be the same as img's size)
    img_moved: np.ndarray
        The result of the image registration. 
    img_grey: np.ndarray
        The difference in grayscale between the resulting image and the reference image.
    w/h/c: int
        The width and height of the image and the number of channels.
    lr: float
        Learning rate.
    n_step: int
        Max training iteration.
    method: str
        The method or loss function used for finding the transformation.
    method_map: dict{int: str}
        A dictionary that maps the index in the combo box to methods.
    decay: dict{int: float}
        Specifies when and how much to decay the learning rate. Keys are the steps to do 
        learning rate decay, and the values are the decay scale on these steps.
        Example: {30: 0.1, 60: 0.2} means at step 30 and 60, the learning rate is 
        shrunk by 0.1 and 0.2 respectively.

    """

    def __init__(self, parent=None):
        super(MainDialog, self).__init__(parent)
        self.ui = cv_ui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('图像配准')
        self.img = None
        self.img_tar = None
        self.img_grey = np.ndarray(())
        self.img_moved = np.ndarray(())
        self.w = 0
        self.h = 0
        self.c = 1
        self.lr = 0
        self.n_step = 1
        self.method = 'mse'
        self.method_map = {0: 'mse', 1: 'kl', 2: 'ncc', 3: 'cc', 4: 'nmi', 5: 'optical'}
        self.decay = {}
        # If the pushButton is clicked, then run 'select_button_clicked' function to choose the source image.
        self.ui.pushButton.clicked.connect(self.select_button_clicked)
        # If the pushButton_2 is clicked, then run 'select_button_clicked1' function to choose the target image.
        self.ui.pushButton_2.clicked.connect(self.select_button_clicked1)
        # If the pushButton_4 is clicked, then run 'apply' function to start image registration.
        self.ui.pushButton_4.clicked.connect(self.apply)
        # If the pushButton_5 is clicked, then run 'clear' function to set the image display to be void.
        self.ui.pushButton_9.clicked.connect(self.clear)
        # If comboBox‘s index is changed, then run 'select' function to get the current method.
        self.ui.comboBox.currentIndexChanged.connect(self.select)

    ###################################################
    #    functions linked to function.py
    #    We will explain how they work in there.
    ###################################################

    def clear(self):
        return unit1.clear(self)

    def select_button_clicked(self):
        return unit1.select_button_clicked(self)

    def select_button_clicked1(self):
        return unit1.select_button_clicked1(self)

    def apply(self):
        return unit1.apply(self)

    def select(self, i):
        self.method = self.method_map[i]


if __name__ == '__main__':
    """
    Show the GUI
    """
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    Dlg = MainDialog()
    Dlg.show()
    sys.exit(app.exec_())
