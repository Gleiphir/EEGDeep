


class mainWindow (QMainWindow):
    def __init__(self):
        super().__init__()

        self.setFont(QFont('microsoft Yahei',font_size))
        self.setWindowTitle(conf['Text','windowtitle'])

        self.file_path = ''
        #self.maxLineWidthpx = QApplication.desktop().availableGeometry().width() // 2
        #self.maxLineWidth = self.maxLineWidthpx // (font_size * 4 //3)
        # 1px = 0.75point

        self.maxLineWidth = conf['Intval','LineWidth'] # in char

        self.LHintRT = QLabel(self)
        self.LHintRT.setText(conf['Text', 'hintRT'])

        self.LRTIn = QLineEdit(self)

        self.LRTOut = QLabel(self)
        #self.LRTOut.setMaximumWidth(self.maxLineWidth)
        #self.LRTOut.setWordWrap(True)

        self.LHintFile = QLabel(self)
        self.LHintFile.setText(conf['Text', 'hintFile'])

        self.LFileName = QLabel(self)

        self.markBtn = QPushButton(conf['Text','Bmark'], self)
        self.markBtn.clicked.connect(self.convertRT)

        self.browseBtn = QPushButton(conf['Text','Bbrowse'], self)
        self.browseBtn.clicked.connect(self.browse)

        self.genFileBtn = QPushButton(conf['Text','Bgenerate'], self)
        self.genFileBtn.clicked.connect(self.convertFile)


        self.aboutBtn = QPushButton(conf['Text','Babout'], self)
        self.aboutBtn.clicked.connect(self.about)


        self.QuitBtn = QPushButton(conf['Text','Bquit'], self)
        self.QuitBtn.clicked.connect(self.close)

        self.Cwidget = QFrame(self)

        self.initUI()

    def initUI(self):
        for key in self.__dict__:
            if isinstance(self.__dict__[key],QWidget):
                self.__dict__[key].setObjectName(key)
                #print(key)


        self.setCentralWidget(self.Cwidget)

        grid = QGridLayout()
        grid.setSpacing(5)
        self.Cwidget.setLayout(grid)



        grid.addWidget(self.LHintRT, 0, 0)
        grid.addWidget(self.LRTIn, 1, 0)
        grid.addWidget(self.markBtn, 1, 1)
        grid.addWidget(self.LRTOut, 2, 0)

        grid.addWidget(self.LHintFile, 4, 0)
        grid.addWidget(self.genFileBtn, 4, 1)
        grid.addWidget(self.LFileName, 5, 0)

        grid.addWidget(self.browseBtn, 5, 1)


        grid.addWidget(self.aboutBtn, 6, 1)
        grid.addWidget(self.QuitBtn, 7, 1)

        self.setLayout(grid)
        self.setGeometry(300, 300, 350, 300)
        #print(conf['customFile','stylish'])
        #self.setStyleSheet('QPushButton{color:red;}')
        self.setStyleSheet(conf['customFile','stylish'])