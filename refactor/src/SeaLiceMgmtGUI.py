import sys
from pathlib import Path

from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QMenu,
                             QAction, QFileDialog, QMessageBox, QGridLayout, QProgressBar)
from src.Simulator import Simulator

class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SLIM GUI")
        self.setWindowIcon(QtGui.QIcon('res/logo.png'))
        self.resize(800, 600)

        self._createWidgets()

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.progressBar, 3, 0, 1, 2)

        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)

        self.setLayout(mainLayout)

        self._createActions()
        self._connectActions()
        self._createMenuBar()

    def _createWidgets(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 0)
        self.progressBar.setValue(0)

    def _createLoaderWorker(self, filename):
        self.thread = QThread()
        self.worker = SimulatorLoadingWorker(filename)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

        self.loadDumpAction.setEnabled(False)
        self.thread.finished.connect(
            lambda: self.loadDumpAction.setEnabled(True)
        )


    def _createActions(self):
        self.loadDumpAction = QAction("&Load Dump")
        self.helpContentAction = QAction("&Help Content", self)
        self.aboutAction = QAction("&About", self)

    def _connectActions(self):
        self.loadDumpAction.triggered.connect(self.openDump)
        self.aboutAction.triggered.connect(self.about)

    def _createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        fileMenu.addAction(self.loadDumpAction)

        # Help menu
        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addAction(self.aboutAction)

        menuBar.addMenu(fileMenu)
        menuBar.addMenu(helpMenu)

    def openDump(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load a dump", "../", "Pickle file (*.pickle)")

        if filename:
            self._createLoaderWorker(filename)


    def about(self):
        QMessageBox.about(self, "About SLIM",
            """
            GUI for the Sea Lice sIMulator.
            
            For more details see https://github.com/resistance-modelling/slim
            """)

    def helpContent(self):
        pass


class SimulatorLoadingWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, dump_path: str):
        super().__init__()
        self.dump_path = dump_path

    def run(self):
        filename_path = Path(self.dump_path)
        parent_path = filename_path.parent
        sim_name = filename_path.name[len("simulation_name_"):-len(".pickle")]
        Simulator.reload_all_dump(parent_path, sim_name)
        self.finished.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())