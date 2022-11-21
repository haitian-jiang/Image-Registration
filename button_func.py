from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import os
from PIL import Image, ImageQt
from ImageRegistrant import *


def apply(self):
    """
    Confirm we at least already got the learning rate and max training iterations.
    Then we start the image registration process.
    Finally show the result image and the difference in grayscale 
    between the resulting image and the reference image.
    """

    # check the input image
    if self.img is None or self.img_tar is None:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择输入图像')
        msg_box.exec_()
        return

    # detect the existence of learning rate and max training iterations
    self.lr = self.ui.textEdit.toPlainText()
    self.n_step = self.ui.textEdit_3.toPlainText()

    if self.lr:
        try:
            self.lr = float(self.lr)
        except ValueError:
            msg_box = QMessageBox(QMessageBox.Warning, '提示', '请输入合法的浮点数作为学习率')
            msg_box.exec_()
            return
    elif self.method in ('optical', ):
        # optical flow does not learning rate
        pass
    else:
        # force user to input learning rate
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '学习率不能为空')
        msg_box.exec_()
        return

    if self.n_step:
        try:
            self.n_step = int(self.n_step)
        except ValueError:
            msg_box = QMessageBox(QMessageBox.Warning, '提示', '请输入合法的整数作为步长')
            msg_box.exec_()
            return
    elif self.method in ('optical', ):
        # optical flow does not require step
        pass
    else:
        # force user to input max step
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '最大训练步数不能为空')
        msg_box.exec_()
        return

    # try to get the input of textEdit_2
    lr_decay_text = self.ui.textEdit_2.toPlainText()
    if lr_decay_text:
        # get decay, do not execute if the format is wrong
        if not read_lr_decay(self):
            return
    # start image registration
    reg = ImageRegistrant(self.img, self.img_tar)
    reg.fit(self.lr, self.n_step, decay=self.decay, method=self.method)
    self.img_moved = reg.moved_img * 255
    # calculate the difference in grayscale between the resulting image and the reference image
    self.img_grey = reg.fixed_img - reg.moved_img
    self.img_grey = (
                (self.img_grey - self.img_grey.min()) * (1 / (self.img_grey.max() - self.img_grey.min()) * 255)).astype(
        'uint8')
    refreshShow2(self)
    refreshShow3(self)


def read_lr_decay(self):
    """
    Read the decay in format 'step1,step2;ratio1,ratio2'
    and translate it into a dictionary.
    return: whether the parsing is successful
    """
    # read the input text and split it by ; and ,
    text = self.ui.textEdit_2.toPlainText()
    try:
        step, ratio = text.split(';')
        stepList = step.split(',')
        ratioList = ratio.split(',')
        assert len(stepList) == len(ratioList)
        for i in range(len(stepList)):
            self.decay[int(stepList[i])] = float(ratioList[i])
        return True
    except:
        # give out the warning
        msg_box = QMessageBox(QMessageBox.Warning, '警告', '请检查学习率衰减的输入格式')
        msg_box.exec_()
        return False



def clear(self):
    """
    Reset images and the display table 
    """
    self.img = None
    self.img_tar = None
    self.imgOrg = np.ndarray(())
    self.imgShow = np.ndarray(())
    self.w = 0
    self.h = 0
    self.c = 1
    self.ui.label_66.setPixmap(QtGui.QPixmap(""))
    self.ui.label_67.setPixmap(QtGui.QPixmap(""))
    self.ui.label_68.setPixmap(QtGui.QPixmap(""))
    self.ui.label_69.setPixmap(QtGui.QPixmap(""))


###################################################
#    Functions to read images in particular
#    and display them on the GUI
###################################################

def select_button_clicked(self):
    fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
    print(fileName)
    if fileName == '':
        return
    # Separate the file name from the path
    root_dir, file_name = os.path.split(fileName)
    # get the current working directory
    pwd = os.getcwd()
    if root_dir:
        # changes the current working directory to the specified path
        os.chdir(root_dir)
    self.img = np.array(Image.open(file_name).convert('L'))
    os.chdir(pwd)
    if self.img.size <= 1:
        return
    refreshShow(self)


def select_button_clicked1(self):
    fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
    print(fileName)
    if fileName == '':
        return
    # Separate the file name from the path
    root_dir, file_name = os.path.split(fileName)
    # get the current working directory
    pwd = os.getcwd()
    if root_dir:
        # changes the current working directory to the specified path
        os.chdir(root_dir)
    self.img_tar = np.array(Image.open(file_name).convert('L'))
    os.chdir(pwd)
    if self.img_tar.size <= 1:
        return
    refreshShow1(self)


###################################################
#    Functions to show the image in particular
#    by translating images from np.array to Image
#    and then to Pixmap.
###################################################

def refreshShow(self):
    label_width = self.ui.label_69.width()
    label_height = self.ui.label_69.height()
    imgShow = self.img
    im = Image.fromarray(imgShow)
    im = im.convert('L')
    image = ImageQt.toqpixmap(im)
    image = image.scaled(label_width, label_height)
    self.ui.label_69.setPixmap(image)


def refreshShow1(self):
    label_width = self.ui.label_68.width()
    label_height = self.ui.label_68.height()
    img_tarShow = self.img_tar
    im = Image.fromarray(img_tarShow)
    im = im.convert('L')
    image = ImageQt.toqpixmap(im)
    image = image.scaled(label_width, label_height)
    self.ui.label_68.setPixmap(image)


def refreshShow2(self):
    label_width = self.ui.label_67.width()
    label_height = self.ui.label_67.height()
    img_movedShow = self.img_moved
    im = Image.fromarray(img_movedShow)
    im = im.convert('L')
    image = ImageQt.toqpixmap(im)
    image = image.scaled(label_width, label_height)
    self.ui.label_67.setPixmap(image)


def refreshShow3(self):
    label_width = self.ui.label_66.width()
    label_height = self.ui.label_66.height()
    img_greyShow = self.img_grey
    im = Image.fromarray(img_greyShow)
    im = im.convert('L')
    image = ImageQt.toqpixmap(im)
    image = image.scaled(label_width, label_height)
    self.ui.label_66.setPixmap(image)
