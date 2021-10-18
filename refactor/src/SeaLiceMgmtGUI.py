import datetime as dt
import sys
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QSettings
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMenu,
                             QAction, QFileDialog, QMessageBox, QProgressBar,
                             QVBoxLayout, QTabWidget)
from src.Simulator import Simulator
from src.gui_utils.configuration import ConfigurationPane
from src.gui_utils.console import ConsoleWidget
from src.gui_utils.plots import PlotPane
from src.gui_utils.model import SimulatorSingleRunState


class Window(QMainWindow):
    newState = pyqtSignal(SimulatorSingleRunState)

    def __init__(self, app: QApplication, parent=None):
        super().__init__(parent)

        self.app = app
        self.states: Optional[List[Simulator]] = None
        self.times: Optional[List[dt.datetime]] = None
        self.states_as_df: Optional[pd.DataFrame] = None

        self._createUI()

    def _createUI(self):
        self.setWindowTitle("SLIM GUI")
        self.setWindowIcon(QtGui.QIcon('res/logo.png'))
        self.resize(800, 600)

        pg.setConfigOptions(antialias=True)

        self._createWidgets()

        mainLayout = QVBoxLayout()
        self.wd.setLayout(mainLayout)
        mainLayout.addWidget(self.plotTabs)
        mainLayout.addWidget(self.progressBar)

        self._createSettings()
        self._createActions()
        self._connectActions()
        self._createMenuBar()

    def _createWidgets(self):
        self.wd = QtWidgets.QWidget(self)
        self.setCentralWidget(self.wd)

        self._createPlotPane()
        self._createConfigurationPane()
        self._createConsole()
        self._createTabs()
        self._createProgressBar()

    def _createPlotPane(self):
        self.plotPane = PlotPane(self)

    def _createConfigurationPane(self):
        self.configurationPane = ConfigurationPane()

    def _createTabs(self):
        self.plotTabs = QTabWidget(self)

        self.plotTabs.addTab(self.plotPane, "Plotter")
        self.plotTabs.addTab(self.configurationPane, "Configuration")
        self.plotTabs.addTab(self.console, "Debugging console")

    def _createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 1)

    def _createConsole(self):
        self.console = ConsoleWidget()

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
        self.states_as_df = Simulator.dump_as_pd(states, times)

        self.configurationPane.newConfig.emit(states[0].cfg)
        self.console.push_vars(vars(self))
        self.newState.emit(SimulatorSingleRunState(states, times, self.states_as_df))

    def _createLoaderWorker(self, filename):
        # TODO: is there a valid reason why we shouldn't make worker a QThread subclass?
        self.worker = SimulatorLoadingWorker(filename)
        print(f"Opening {filename}")

        self.worker.finished.connect(self._displaySimulatorData)
        self.worker.finished.connect(self._progressBarComplete)

        self._progressBarLoad()
        self.worker.start()

    def _createActions(self):
        self.loadDumpAction = QAction("&Load Dump")
        self.paperModeAction = QAction("Set &light mode (paper mode)")
        self.paperModeAction.setCheckable(True)
        self.clearAction = QAction("&Clear plot", self)
        self.aboutAction = QAction("&About", self)
        self.showDetailedGenotypeAction = QAction(self)

        self._updateRecentFilesActions()

    def _connectActions(self):
        self.loadDumpAction.triggered.connect(self.openDump)
        self.paperModeAction.toggled.connect(self.plotPane.setPaperMode)
        self.aboutAction.triggered.connect(self._openAboutMessage)
        self.clearAction.triggered.connect(self.plotPane.cleanPlot)

    def _createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        fileMenu.addAction(self.loadDumpAction)
        fileMenu.addSeparator()
        for action in self.recentFilesActions:
            fileMenu.addAction(action)

        viewMenu = QMenu("&View", self)
        viewMenu.addAction(self.paperModeAction)
        viewMenu.addAction(self.clearAction)

        # Help menu
        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.aboutAction)

        menuBar.addMenu(fileMenu)
        menuBar.addMenu(viewMenu)
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
                # ugly hack as python does not capture "option" but only a lexical reference to it
                new_action.triggered.connect(lambda _, option=option: self._createLoaderWorker(option))
                new_recent_options.append(option)
                self.recentFilesActions.append(new_action)

        if new_recent_options != recent_options:
            self.settings.setValue("recentFileList", new_recent_options)

    @property
    def recentFilesList(self):
        return self.settings.value("recentFileList", type=str) or []

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

    def _openAboutMessage(self):
        QMessageBox.about(self, "About SLIM",
                          """
                          GUI for the Sea Lice sIMulator.
                          
                          For more details see https://github.com/resistance-modelling/slim
                          """)

    def helpContent(self):
        pass


class SimulatorLoadingWorker(QThread):
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
    win = Window(app)
    win.show()

    # KeyboardInterrupt trick
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    sys.exit(app.exec_())