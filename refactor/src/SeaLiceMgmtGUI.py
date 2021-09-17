import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QMenu, QAction
from src.Simulator import Simulator

class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sea Lice Project")
        self.resize(800, 600)
        label = QLabel("Hello sea lice!")
        self.centralWidget = label
        self.centralWidget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setCentralWidget(self.centralWidget)

        self._createActions()
        self._createMenuBar()

    def _createActions(self):
        self.loadDump = QAction("&Load Dump")
        self.helpContentAction = QAction("&Help Content", self)
        self.aboutAction = QAction("&About", self)

    def _createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        fileMenu.addAction(self.loadDump)

        # Help menu
        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addAction(self.aboutAction)

        menuBar.addMenu(fileMenu)
        menuBar.addMenu(helpMenu)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())