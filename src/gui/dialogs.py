from PyQt5.QtWidgets import (QAbstractButton, QGroupBox, QDialog, QListWidget,
                             QDialogButtonBox,QFrame, QStackedWidget,
                             QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
                             QPlainTextEdit, QComboBox)
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt


class SettingDialog(QDialog):
    updateTriggerred = pyqtSignal(dict)

    def __init__(self, parent=None, defSettings=None):
        QDialog.__init__(self, parent=parent)
        self.initUI(parent)
        self.restoreDefaultSettings(defSettings)
        self.updateTriggerred.connect(parent.updateSettings)

    def initUI(self, parent):
        # list views
        listWidget = QListWidget()
        # listWidget.insertItem(0, "图表")
        listWidget.insertItem(0, "网络结构")
        listWidget.insertItem(1, "训练条件")
        listWidget.setFixedWidth(80)

        # pages
        stackPages = self.initSettingPages(parent)
        listWidget.currentRowChanged.connect(stackPages.setCurrentIndex)

        # standard buttons
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok |
                                     QDialogButtonBox.Cancel |
                                     QDialogButtonBox.Apply |
                                     QDialogButtonBox.RestoreDefaults)
        self.buttonBox.clicked.connect(self.updateSettings)

        hbox = QHBoxLayout()
        hbox.addWidget(listWidget)
        hbox.addWidget(stackPages)
        topFrame = QFrame()
        topFrame.setLayout(hbox)

        mainbox = QVBoxLayout()
        mainbox.addWidget(topFrame)
        mainbox.addWidget(self.buttonBox)

        self.setLayout(mainbox)
        self.setWindowTitle("设置")

    def initSettingPages(self, parent):
        # Plot Page
        # label = QLabel('图表字体路径:')
        # self.plotFontLineEdit = QLineEdit('')
        # self.plotFontLineEdit.setMinimumWidth(200)
        # browseBtn = QPushButton('...')
        # browseBtn.clicked.connect(self.browseFonts)
        # browseBtn.setFixedWidth(40)
        #
        # plotGridBox = QGridLayout()
        # plotGridBox.addWidget(label, 0, 0)
        # plotGridBox.addWidget(self.plotFontLineEdit, 0, 1)
        # plotGridBox.addWidget(browseBtn, 0, 2)
        # plotGridBox.setColumnStretch(1, 20)
        # plotGridBox.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        # plotGrp = QGroupBox("图表设置")
        # plotGrp.setLayout(plotGridBox)

        # Network Settings
        self.layer0ComboBox = QComboBox()
        self.layer0ComboBox.addItems(['4', '8', '10'])
        self.layer1ComboBox = QComboBox()
        self.layer1ComboBox.addItems(['4', '8', '10'])
        networkLabels = ['隐藏层1神经元个数:', '隐藏层2神经元个数:']
        networkWidgets = [self.layer0ComboBox, self.layer1ComboBox]

        netFrmBox = QFormLayout()
        for label, wd in zip(networkLabels, networkWidgets):
            netFrmBox.addRow(label, wd)
        netGrp = QGroupBox("神经网络结构设置")
        netGrp.setLayout(netFrmBox)
        netFrmBox.setRowWrapPolicy(QFormLayout.DontWrapRows)
        netFrmBox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        netFrmBox.setFormAlignment(Qt.AlignHCenter | Qt.AlignTop)
        netFrmBox.setLabelAlignment(Qt.AlignRight)

        # Training Settings
        self.epochLineEdit = QLineEdit('')
        self.retryLineEdit = QLineEdit('')
        self.tolLineEdit = QLineEdit('')
        trainLabels = ['单次迭代次数限制:', '重试次数限制:', '训练结束条件(MSE):']
        trainWidgets = [self.epochLineEdit, self.retryLineEdit, self.tolLineEdit]

        trainFrmBox = QFormLayout()
        for label, wd in zip(trainLabels, trainWidgets):
            trainFrmBox.addRow(label, wd)
        trainGrp = QGroupBox("神经网络训练条件设置")
        trainGrp.setLayout(trainFrmBox)
        trainFrmBox.setRowWrapPolicy(QFormLayout.DontWrapRows)
        trainFrmBox.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        trainFrmBox.setFormAlignment(Qt.AlignHCenter | Qt.AlignTop)
        trainFrmBox.setLabelAlignment(Qt.AlignRight)

        stackPages = QStackedWidget()
        # stackPages.addWidget(plotGrp)
        stackPages.addWidget(netGrp)
        stackPages.addWidget(trainGrp)

        return stackPages

    # @pyqtSlot()
    # def browseFonts(self):
    #     font, ok = QFontDialog.getFont()
    #     if ok:
    #         print(font.family(), font.weight(), font.pointSize())
    #     # fname, _ = QFileDialog.getOpenFileName(self, 'Select font file', 'C:/Windows/Fonts/', 'Font file(*.ttc;*.ttf)')
    #     # if not fname:
    #     #     return
    #     # print(fname)

    @pyqtSlot(QAbstractButton)
    def updateSettings(self, btn):
        stdBtn = self.buttonBox.standardButton(btn)
        if stdBtn == QDialogButtonBox.Ok:
            self.emitSettingUpdate()
            self.close()
        elif stdBtn == QDialogButtonBox.Apply:
            self.emitSettingUpdate()
        elif stdBtn == QDialogButtonBox.RestoreDefaults:
            self.restoreDefaultSettings()
        else:
            self.close()

    def emitSettingUpdate(self):
        settings = {
            # 'font': self.plotFontLineEdit.text(),
            'h0size': int(self.layer0ComboBox.currentText()),
            'h1size': int(self.layer1ComboBox.currentText()),
            'epoch': int(self.epochLineEdit.text()),
            'retry': int(self.retryLineEdit.text()),
            'tol': float(self.tolLineEdit.text()),
        }
        self.updateTriggerred.emit(settings)

    def restoreDefaultSettings(self, settings=None):
        if settings:
            # self.plotFontLineEdit.setText(settings['font'])
            id = self.layer0ComboBox.findText(str(settings['h0size']))
            if id != -1:
                self.layer0ComboBox.setCurrentIndex(id)
            else:
                self.layer0ComboBox.setCurrentIndex(0)
            id = self.layer1ComboBox.findText(str(settings['h1size']))
            if id != -1:
                self.layer1ComboBox.setCurrentIndex(id)
            else:
                self.layer1ComboBox.setCurrentIndex(0)
            self.epochLineEdit.setText(str(settings['epoch']))
            self.retryLineEdit.setText(str(settings['retry']))
            self.tolLineEdit.setText(str(settings['tol']))
        else:
            # self.plotFontLineEdit.setText('C:/Windows/Fonts/msyh.ttc')
            self.layer0ComboBox.setCurrentIndex(0)
            self.layer1ComboBox.setCurrentIndex(0)
            self.epochLineEdit.setText('2000')
            self.retryLineEdit.setText('3')
            self.tolLineEdit.setText('0.5')



class UsageDialog(QDialog):

    def __init__(self, parent=None):
        QDialog.__init__(self, parent=parent)
        self.initUI(parent)

    def initUI(self, parent):
        usageTextEdit = QPlainTextEdit("")
        usageTextEdit.appendHtml(
            """
            <ul>
            <li><font size='5'><b>如何训练新的神经网络模型进行预测?</b></font></li>
            <p>&nbsp;&nbsp;&nbsp;&nbsp;1.加载数据文件</p>
            <p>&nbsp;&nbsp;&nbsp;&nbsp;2.文件->设置：配置神经网络及训练参数（可选）</p>
            <p>&nbsp;&nbsp;&nbsp;&nbsp;3.点击运行进行神经网络训练</p>
            <p>&nbsp;&nbsp;&nbsp;&nbsp;4.点击预测按钮进行数据预测</p>
            <p>&nbsp;&nbsp;&nbsp;&nbsp;5.如果预测结果符合预期，点击保存按钮保存训练好的模型</p>
            <p></p>
            <hr />
            <p></p>
            <li><font size='5'><b>如何使用保存的神经网络模型进行预测?</b></font></li>
            <p>&nbsp;&nbsp;&nbsp;&nbsp;1.加载数据文件</p>
            <p>&nbsp;&nbsp;&nbsp;&nbsp;2.点击加载按钮载入已经保存的模型</p>
            <p>&nbsp;&nbsp;&nbsp;&nbsp;3.点击预测按钮进行数据预测</p>
            </ul>
            """
        )
        usageTextEdit.setReadOnly(True)
        usageTextEdit.setMinimumWidth(400)
        usageTextEdit.setMinimumHeight(300)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        buttonBox.clicked.connect(self.clickActions)
        mainbox = QVBoxLayout()
        mainbox.addWidget(usageTextEdit)
        mainbox.addWidget(buttonBox)

        self.setLayout(mainbox)
        self.setWindowTitle("使用说明")

    @pyqtSlot(QAbstractButton)
    def clickActions(self, btn):
        self.close()



