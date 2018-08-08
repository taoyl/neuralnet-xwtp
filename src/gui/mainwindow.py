"""
Application Main Window.
Author: Yuliang Tao
Email : nerotao@foxmail.com
"""
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QFrame,
                             QPlainTextEdit, QAction, QMenu, QWidget,
                             QSplitter, QFileDialog, QVBoxLayout, QHBoxLayout)
from PyQt5.QtCore import pyqtSlot, Qt, QSize
from PyQt5.QtGui import QIcon

from dialogs import SettingDialog, UsageDialog
from mplotcanvas import StaticMplotCanvas
from core import ShapeError, NeuralNetwork
from core import DataHandler

progname = "供热能源站二次侧供水温度预测软件 - xWTP"
progversion = "0.1"

class AppWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(progname)
        self.init_ui()

        self.network = None
        self.isDataLoaded = False
        self.training_data = None
        self.test_data = None
        self.predict_y = None
        self.networkSettings = None

    def init_ui(self):
        self.init_menus()
        self.init_toolbar(QSize(40, 40))

        mainWidget = QWidget(self)
        mainbox = QHBoxLayout(mainWidget)
        # self.setLayout(mainbox)

        topFrame = QFrame(mainWidget)
        topFrame.setFrameShape(QFrame.StyledPanel)
        btmFrame = QFrame(mainWidget)
        btmFrame.setFrameShape(QFrame.StyledPanel)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(topFrame)
        splitter.addWidget(btmFrame)
        # logText 30%, Plot 70%
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        mainbox.addWidget(splitter)

        self.init_plot_area(topFrame)

        vboxLog = QVBoxLayout(btmFrame)
        self.logTextEdit = QPlainTextEdit("")
        self.logTextEdit.appendHtml("""<font size='4'>欢迎使用{}</font><p>""".format(progname))
        self.logTextEdit.setReadOnly(True)
        vboxLog.addWidget(self.logTextEdit)

        mainWidget.setFocus()
        self.setCentralWidget(mainWidget)
        self.statusBar().showMessage("Ready")
        self.setWindowIcon(QIcon('res/load_network.png'))
        self.show()

    def init_plot_area(self, parent):
        hboxPlot = QHBoxLayout(parent)
        # errplot_labels = {'t': u'供水温度预测误差', 'x': u'时间', 'y': u'误差百分比(%)'}
        # predplot_labels = {'t': u'供水温度预测值', 'x': u'时间', 'y': u'供水温度(℃)'}
        errplot_labels = {'t': 'Prediction Errors', 'x': 'Time', 'y': 'Error Percent(%)'}
        predplot_labels = {'t': 'Predicted Temperature', 'x': 'Time', 'y': 'Temperature(℃)'}
        self.errPlot = StaticMplotCanvas(parent, labels=errplot_labels)
        self.predPlot = StaticMplotCanvas(parent, labels=predplot_labels)
        hboxPlot.addWidget(self.errPlot)
        hboxPlot.addWidget(self.predPlot)

    def init_toolbar(self, iconSize):
        # data file
        self.loadDataAct = QAction(QIcon('res/load_data.png'), 'Import Training Data', self)
        self.loadDataAct.setShortcut('Ctrl+L')
        self.loadDataAct.triggered.connect(self.loadTrainingDataFile)
        self.saveDataAct = QAction(QIcon('res/save_data.png'), 'Export Predicted Data', self)
        self.saveDataAct.setShortcut('Ctrl+E')
        self.saveDataAct.triggered.connect(self.savePredictDataToFile)
        self.saveDataAct.setEnabled(False)
        # network
        self.loadNetworkAct = QAction(QIcon('res/load_network.png'), 'Load Trained Network', self)
        self.loadNetworkAct.setShortcut('Ctrl+N')
        self.loadNetworkAct.triggered.connect(self.restoreNeuralNetwork)
        self.loadNetworkAct.setEnabled(False)
        self.saveNetworkAct = QAction(QIcon('res/save_network.png'), 'Save Trained Network', self)
        self.saveNetworkAct.setShortcut('Ctrl+S')
        self.saveNetworkAct.triggered.connect(self.saveNeuralNetwork)
        self.saveNetworkAct.setEnabled(False)
        # run & predict
        self.runTrainingAct = QAction(QIcon('res/train_network.png'), 'Train Network', self)
        self.runTrainingAct.setShortcut('Ctrl+R')
        self.runTrainingAct.triggered.connect(self.runNetworkTraining)
        self.runTrainingAct.setEnabled(False)
        self.predictDatakAct = QAction(QIcon('res/predict.png'), 'Predict Data', self)
        self.predictDatakAct.setShortcut('Ctrl+P')
        self.predictDatakAct.triggered.connect(self.predictData)
        self.predictDatakAct.setEnabled(False)
        # clear
        self.resetAct = QAction(QIcon('res/clear.png'), 'Clear data and network', self)
        self.resetAct.setEnabled(False)
        self.resetAct.triggered.connect(self.clearDataAndNetwork)

        dataToolbar = self.addToolBar('Data ToolBar')
        dataToolbar.addAction(self.loadDataAct)
        dataToolbar.addAction(self.saveDataAct)
        dataToolbar.setIconSize(iconSize)

        networkToolbar = self.addToolBar('Network ToolBar')
        networkToolbar.addAction(self.loadNetworkAct)
        networkToolbar.addAction(self.runTrainingAct)
        networkToolbar.addAction(self.predictDatakAct)
        networkToolbar.addAction(self.saveNetworkAct)
        networkToolbar.setIconSize(iconSize)

        resetToolbar = self.addToolBar('Reset ToolBar')
        resetToolbar.addAction(self.resetAct)
        resetToolbar.setIconSize(iconSize)

    def init_menus(self):
        # File
        settingAct = QAction(QIcon('res/settings.png'), '设置', self)
        settingAct.triggered.connect(self.showSettingDialog)

        exitAct = QAction(QIcon('exit.png'), '退出', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.triggered.connect(self.fileQuit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(settingAct)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAct)

        # Help
        helpMenu = QMenu('&Help', self)
        self.menuBar().addMenu(helpMenu)
        helpMenu.addAction('使用说明', self.usage)
        helpMenu.addAction('关于', self.about)

    @pyqtSlot()
    def loadTrainingDataFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Training Data', '.', 'Data File(*.csv)')
        if not fname:
            # self.logStatus("加载数据文件{}失败!".format(fname), 'red', 'E')
            return

        dl = DataHandler(fname)
        self.training_data, self.test_data = dl.load(cb=self.logStatus)
        self.isDataLoaded = True
        self.runTrainingAct.setEnabled(True)
        self.loadNetworkAct.setEnabled(True)
        self.resetAct.setEnabled(True)
        self.logStatus("加载数据文件{}成功".format(fname))
        self.logStatus('请训练神经网络或者加载已经训练的神经网络模型', '#FF8C00', 'T')

    @pyqtSlot()
    def savePredictDataToFile(self):
        if self.predict_y is None:
            # self.logStatus('没有未保存的预测数据, 请先进行数据预测!', 'red', 'E')
            return

        fname, _ = QFileDialog.getSaveFileName(self, 'Save Predicted Data', '.', 'Data File(*.csv)')
        if not fname:
            self.logStatus('保存预测数据文件{}失败!'.format(fname), 'red', 'E')
            return

        test_x, _ = DataHandler.split_xy(self.test_data)
        status = DataHandler.save(np.concatenate((test_x, self.predict_y), axis=0), fname)
        if status:
            self.logStatus('保存预测数据文件{}成功'.format(fname))

    @pyqtSlot()
    def restoreNeuralNetwork(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Network File', '.', 'Network File(*.nf)')
        if not fname:
            # self.logStatus('打开神经网络文件{}失败!'.format(fname), 'red', 'E')
            return

        # clear previous plots
        self.clearPlots()

        training_x, training_y = DataHandler.split_xy(self.training_data)
        self.network = NeuralNetwork(training_x, training_y)
        try:
            self.network.load(fname)
        except ShapeError as e:
            self.logStatus('加载神经网络文件{}失败!'.format(fname), 'red', 'E')
            QMessageBox.warning(self, '警告', '加载神经网络文件失败, 请检查文件格式是否正确!')
            return

        self.logStatus('神经网络文件{}加载成功'.format(fname))
        self.predictDatakAct.setEnabled(True)
        self.logStatus('请执行数据预测', '#FF8C00', 'T')

    @pyqtSlot()
    def saveNeuralNetwork(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save Network File', '.', 'Network File(*.nf)')
        if not fname:
            # self.logStatus('保存神经网络文件{}失败!'.format(fname), 'red', 'E')
            return

        self.network.dump(fname)
        self.logStatus('保存神经网络文件{}成功'.format(fname))

    @pyqtSlot()
    def runNetworkTraining(self):
        if self.network is not None:
            ans = QMessageBox.question(self, '警告',
                                 '系统中已存在训练好的神经网络，请问您需要重新训练神经网络吗?')
            if ans == QMessageBox.No:
                return

        # clear previous plots
        self.clearPlots()

        self.logStatus("正在初始化神经网络结构...", 'blue', 'I')
        training_x, training_y = DataHandler.split_xy(self.training_data)
        # retrieve settings
        epoch0 = 2000
        tol0 = 0.1
        retry_num = 3
        h0size = 4
        h1size = 4
        if self.networkSettings:
            epoch0 = self.networkSettings['epoch']
            tol0 = self.networkSettings['tol']
            retry_num = self.networkSettings['retry']
            h0size = self.networkSettings['h0size']
            h1size = self.networkSettings['h1size']

        self.logStatus("神经网络信息：layer1={}, layer2={}, epoch0={}, retry={}, tol0={}"
                       .format(h0size, h1size, epoch0, retry_num, tol0), 'blue', 'I')

        net = [(training_x.shape[0], ''), (h0size, 'sigmoid'), (h1size, 'sigmoid'), (1, 'feedthrough')]
        try:
            self.network = NeuralNetwork(training_x, training_y, sizes=net)
        except ShapeError as e:
            self.logStatus('初始化神经网络结构失败!')
            QMessageBox.warning(self, '警告', '初始化神经网络结构失败, 请重试!')
            return

        # training
        mu0 = 0.1
        beta = 10
        retry = 0
        self.logStatus("使用LM算法开始训练神经网络...", 'blue', 'I')
        while retry < retry_num:
            residual, mu, citer, msg = \
                self.network.train(retry=retry, epoch=epoch0, mu0=mu0,
                                   beta=beta, tol=tol0, cb=self.logStatus)
            if residual is None:
                if retry == (retry_num - 1):
                    self.logStatus("训练失败!".format(msg), 'red', 'E')
                    return
                else:
                    self.logStatus("训练失败:{}, 重试中...".format(msg), '#FFA07A', 'I')
                    self.network.randomize_wb()
                    # continue
            elif residual > tol0:
                if retry == (retry_num - 1):
                    self.logStatus("训练失败!".format(msg), 'red', 'E')
                    return
                else:
                    self.logStatus("训练失败: 运算未能收敛, 重试中...", '#FFA07A', 'I')
                    self.network.randomize_wb()
                    # continue
            else:
                self.logStatus("神经网络训练完成, 迭代次数={1}, 最终残差={0}"
                               .format(residual, citer+retry*epoch0), 'blue', 'I')
                break
            retry += 1
        self.predictDatakAct.setEnabled(True)
        self.saveNetworkAct.setEnabled(True)
        self.logStatus('请执行数据预测', '#FF8C00', 'T')

    @pyqtSlot()
    def predictData(self):
        # don't forget to clear previous plots
        self.clearPlots()

        self.logStatus("开始进行数据预测...", 'blue', 'I')
        test_x, test_y = DataHandler.split_xy(self.test_data, False)
        self.predict_y = self.network.predict(test_x)
        self.logStatus("开始计算误差...", 'blue', 'I')
        self.predPlot.update_plot(x=np.arange(len(self.predict_y[0])),
                                  y=self.predict_y[0])
        # error plot
        err_percent = (self.predict_y - test_y) * 100.0 / test_y
        self.errPlot.error_plot(x=np.arange(len(err_percent[0])),
                                y=err_percent[0])
        abs_err = np.abs(err_percent)
        self.logStatus("数据预测完成, 最大绝对值误差={}%, 平均绝对值误差={}%"
                       .format(abs_err.max(), abs_err.mean()), 'blue', 'I')
        self.saveDataAct.setEnabled(True)

    @pyqtSlot()
    def clearDataAndNetwork(self):
        ans = QMessageBox.question(self, '警告',
                                   '您希望删除所有的数据和已经训练好的神经网络吗?')
        if ans == QMessageBox.No:
            return

        # reset
        self.network = None
        self.isDataLoaded = False
        self.training_data = None
        self.test_data = None
        # update UI
        self.loadNetworkAct.setEnabled(False)
        self.runTrainingAct.setEnabled(False)
        self.saveDataAct.setEnabled(False)
        self.saveNetworkAct.setEnabled(False)
        self.predictDatakAct.setEnabled(False)
        self.resetAct.setEnabled(False)
        self.logTextEdit.clear()
        # clear plots
        self.clearPlots()

    @pyqtSlot(dict)
    def updateSettings(self, settings):
        self.networkSettings = settings

    @pyqtSlot()
    def showSettingDialog(self):
        dlg = SettingDialog(self, self.networkSettings)
        dlg.show()

    @pyqtSlot()
    def fileQuit(self):
        self.close()

    @pyqtSlot()
    def about(self):
        QMessageBox.about(self, "关于",
                          """<b>{}</b><p>版本号: {}""".format(progname, progversion)
                          )

    @pyqtSlot()
    def usage(self):
        dlg = UsageDialog(self)
        dlg.show()

    def closeEvent(self, ce):
        self.fileQuit()

    def logStatus(self, text, color='green', tag='S'):
        self.logTextEdit.appendHtml("<p><font color='{0}'><b>[{1}]:</b> {2}"
                                "</font></p>".format(color, tag, text))
        # force UI update. An alternative is to use QThread
        QApplication.processEvents()

    def clearPlots(self):
        self.predPlot.clear()
        self.errPlot.clear()
        errplot_labels = {'t': 'Prediction Errors', 'x': 'Time', 'y': 'Error Percent(%)'}
        predplot_labels = {'t': 'Predicted Temperature', 'x': 'Time', 'y': 'Temperature(℃)'}
        self.errPlot.initial_figure(errplot_labels)
        self.predPlot.initial_figure(predplot_labels)
        # force UI update. An alternative is to use QThread
        #QApplication.processEvents()

