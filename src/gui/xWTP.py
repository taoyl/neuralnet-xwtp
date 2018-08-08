# Application
import sys
sys.path.append("..")

from PyQt5.QtWidgets import QApplication
from mainwindow import AppWindow


qApp = QApplication(sys.argv)
aw = AppWindow()
sys.exit(qApp.exec_())
