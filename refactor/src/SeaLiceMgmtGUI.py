import datetime as dt
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer, QSettings
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QMenu,
                             QAction, QFileDialog, QMessageBox, QGridLayout, QProgressBar)
import pyqtgraph as pg

from src.Simulator import Simulator

class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.states: Optional[List[Simulator]] = None
        self.times: Optional[List[dt.datetime]] = None

        self._createUI()

        # KeyboardInterrupt trick

    def _createUI(self):
        self.setWindowTitle("SLIM GUI")
        self.setWindowIcon(QtGui.QIcon('res/logo.png'))
        self.resize(800, 600)

        pg.setConfigOptions(antialias=True)

        self._createWidgets()

        mainLayout = QGridLayout()
        self.wd.setLayout(mainLayout)


        mainLayout.addWidget(self.plot, 0, 0, 3, 1)
        mainLayout.addWidget(self.progressBar, 2, 0, 1, 2)

        #mainLayout.setRowStretch(1, 1)
        #mainLayout.setRowStretch(2, 1)

        self._createActions()
        self._connectActions()
        self._createMenuBar()
        self._createSettings()


    def _createWidgets(self):
        self.wd = QtWidgets.QWidget(self)
        self.setCentralWidget(self.wd)
        self.plot = pg.PlotWidget()

        # a scientific lorem ipsum
        xs = np.linspace(-100, 100, 1000)
        ys = np.sin(xs) / xs
        self.plot.plot(ys, pen=(255, 255, 255, 200))

        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 1)

    def _createSettings(self):
        self.settings = QSettings("SLIM Project", "Slim")

    def _progressBarLoad(self):
        self.progressBar.setRange(0, 0)
        self.loadDumpAction.setEnabled(False)

    def _progressBarComplete(self):
        self.progressBar.setRange(0, 1)
        self.loadDumpAction.setEnabled(True)

    def _displaySimulatorData(self, states: List[Simulator], times: List[dt.datetime]):
        self.states = states
        self.times = times
        print(f"Successfully loaded {states}")

    def _createLoaderWorker(self, filename):
        # TODO: is there a valid reason why we shouldn't make worker a QThread subclass?
        self.thread = QThread()
        self.worker = SimulatorLoadingWorker(filename)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

        self.worker.finished.connect(self._displaySimulatorData)

        self._progressBarLoad()
        self.thread.finished.connect(
            lambda: self._progressBarComplete()
        )


    def _createActions(self):
        self.loadDumpAction = QAction("&Load Dump")
        self.helpContentAction = QAction("&Help Content", self)
        self.aboutAction = QAction("&About", self)

        self._updateRecentFilesActions()

    def _connectActions(self):
        self.loadDumpAction.triggered.connect(self.openDump)
        self.aboutAction.triggered.connect(self.about)

    def _createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        fileMenu.addAction(self.loadDumpAction)
        fileMenu.addSeparator()
        for action in self.recentFilesActions:
            fileMenu.addAction(action)

        # Help menu
        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addAction(self.aboutAction)

        menuBar.addMenu(fileMenu)
        menuBar.addMenu(helpMenu)

    def _updateRecentFilesActions(self):
        # Recent files
        recent_options = self.recentFilesList
        self.recentFilesActions = []
        if not recent_options:
            return

        new_recent_options = []

        for idx, option in enumerate(recent_options):
            filename_as_path = Path(option)
            if filename_as_path.exists():
                new_action = QAction(f"&{idx}: {option}", self)
                new_action.trigger.connect(lambda filename: self._createLoaderWorker(filename))
                new_recent_options.append(option)
                self.recentFilesActions.append(new_action)

        if new_recent_options != recent_options:
            self.settings.setValue("recentFileList", new_recent_options)

    @property
    def recentFilesList(self):
        return self.settings.value("recentFileList", "str") or []

    @recentFilesList.setter
    def recentFilesList(self, value):
        self.settings.setValue("recentFileList", value)
        self._updateRecentFilesActions()

    def openDump(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load a dump", "", "Pickle file (*.pickle)")

        if filename:
            recentFiles = self.recentFilesList
            if filename not in recentFiles:
                self.recentFilesList = recentFiles + [filename]

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
    finished = pyqtSignal((list, list))

    def __init__(self, dump_path: str):
        super().__init__()
        self.dump_path = dump_path

    def run(self):
        filename_path = Path(self.dump_path)
        parent_path = filename_path.parent
        sim_name = filename_path.name[len("simulation_name_"):-len(".pickle")]
        states, times = Simulator.reload_all_dump(parent_path, sim_name)
        print("Loaded simulation")
        self.finished.emit(states, times)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()

    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    sys.exit(app.exec_())