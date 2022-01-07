import sys
from pathlib import Path

import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QSettings
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMenu,
                             QAction, QFileDialog, QMessageBox, QProgressBar,
                             QVBoxLayout, QTabWidget)
from slim.simulation.simulator import Simulator
from slim.gui_utils.configuration import ConfigurationPane
from slim.gui_utils.console import ConsoleWidget
from slim.gui_utils.plots import SingleRunPlotPane, OptimiserPlotPane
from slim.gui_utils.model import SimulatorSingleRunState, SimulatorOptimiserState


class Window(QMainWindow):
    loadedSimulatorState = pyqtSignal(SimulatorSingleRunState)
    loadedOptimiserState = pyqtSignal(SimulatorOptimiserState)

    def __init__(self, parent=None):
        super().__init__(parent)

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
        self.simulationPlotPane = SingleRunPlotPane(self)
        self.optimiserPlotPane = OptimiserPlotPane(self)

    def _createConfigurationPane(self):
        self.configurationPane = ConfigurationPane()

    def _createTabs(self):
        self.plotTabs = QTabWidget(self)

        self.plotTabs.addTab(self.simulationPlotPane, "Simulation plots")
        self.plotTabs.addTab(self.optimiserPlotPane, "Optimiser plots")
        self.plotTabs.addTab(self.configurationPane, "Configuration")
        self.plotTabs.addTab(self.console, "Debugging console")

    def _createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 1)

    def _createConsole(self):
        self.console = ConsoleWidget()
        self.console.push_vars(vars(self))

    def _createSettings(self):
        self.settings = QSettings("SLIM Project", "Slim")

    def _progressBarLoad(self):
        self.progressBar.setRange(0, 0)
        self.loadDumpAction.setEnabled(False)
        self.loadOptimiserDumpAction.setEnabled(False)

    def _progressBarComplete(self):
        self.progressBar.setRange(0, 1)
        self.loadDumpAction.setEnabled(True)
        self.loadOptimiserDumpAction.setEnabled(True)

    def _displaySimulatorData(self, simulator_data: SimulatorSingleRunState):
        self.configurationPane.newConfig.emit(simulator_data.states[0].cfg)
        self.console.push_vars(vars(simulator_data))
        self.loadedSimulatorState.emit(simulator_data)

    def _displayOptimiserData(self, optimiser_data: SimulatorOptimiserState):
        self.console.push_vars(vars(optimiser_data))
        self.loadedOptimiserState.emit(optimiser_data)

    def _createLoaderWorker(self, filename: Path, is_optimiser=False):
        if is_optimiser:
            self.worker = OptimiserLoadingWorker(filename)
            self.worker.finished.connect(self._displayOptimiserData)
        else:
            self.worker = SimulatorLoadingWorker(filename)
            self.worker.finished.connect(self._displaySimulatorData)

        self.worker.finished.connect(self._progressBarComplete)
        self.worker.failed.connect(self._progressBarComplete)

        self._progressBarLoad()
        self.worker.start()

    def _createActions(self):
        self.loadDumpAction = QAction("&Load Dump")
        self.loadOptimiserDumpAction = QAction("&Load Optimiser Dump")
        self.paperModeAction = QAction("Set &light mode (paper mode)")
        self.paperModeAction.setCheckable(True)
        self.clearAction = QAction("&Clear plot", self)
        self.aboutAction = QAction("&About", self)
        self.showDetailedGenotypeAction = QAction(self)

        self._updateRecentFilesActions()

    def _connectActions(self):
        self.loadDumpAction.triggered.connect(self.openSimulatorDump)
        self.loadOptimiserDumpAction.triggered.connect(self.openOptimiserDump)
        self.paperModeAction.toggled.connect(self.simulationPlotPane.setPaperMode)
        self.paperModeAction.toggled.connect(self.optimiserPlotPane.setPaperMode)
        self.aboutAction.triggered.connect(self._openAboutMessage)
        self.clearAction.triggered.connect(self.simulationPlotPane.cleanPlot)

    def _createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        fileMenu.addAction(self.loadDumpAction)
        fileMenu.addAction(self.loadOptimiserDumpAction)
        fileMenu.addSeparator()
        for action in self.recentFilesActions:
            fileMenu.addAction(action)
        # TODO: add recent actions for dumps
        # fileMenu.addSeparator()


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

    def openSimulatorDump(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load a dump", "", "Pickle file (*.pickle)")

        if filename:
            recentFiles = self.recentFilesList
            if filename not in recentFiles:
                self.recentFilesList = recentFiles + [filename]

            self._createLoaderWorker(Path(filename))

    def openOptimiserDump(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load an Optimiser dump", "", "Optimiser artifact (params.json)")

        print(filename)
        if filename:
            dirname = Path(filename).parent
            self._createLoaderWorker(dirname, is_optimiser=True)

    def _openAboutMessage(self):
        QMessageBox.about(self, "About SLIM",
                          """
                          GUI for the Sea Lice sIMulator.
                          
                          For more details see https://github.com/resistance-modelling/slim
                          """)

    def helpContent(self):
        pass


class SimulatorLoadingWorker(QThread):
    finished = pyqtSignal(SimulatorSingleRunState)
    failed = pyqtSignal(Exception)

    def __init__(self, dump_path: Path):
        super().__init__()
        self.dump_path = Path(dump_path)

    def run(self):
        try:
            parent_path = self.dump_path.parent
            sim_name = self.dump_path.name[len("simulation_name_"):-len(".pickle")]
            states, times = Simulator.reload_all_dump(parent_path, sim_name)
            states_as_df = Simulator.dump_as_pd(states, times)
            self.finished.emit(SimulatorSingleRunState(states, times, states_as_df))

        except FileNotFoundError:
            self.failed.emit()


class OptimiserLoadingWorker(QThread):
    finished = pyqtSignal(SimulatorOptimiserState)
    failed = pyqtSignal()

    def __init__(self, dump_path: Path):
        super().__init__()
        self.dir_dump_path = dump_path

    def run(self):
        try:
            states = Simulator.reload_from_optimiser(self.dir_dump_path)
            states_as_df = Simulator.dump_optimiser_as_pd(states)

            print("Loaded optimiser data")
            self.finished.emit(SimulatorOptimiserState(states, states_as_df))

        except NotADirectoryError:
            self.failed.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()

    # KeyboardInterrupt trick
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    sys.exit(app.exec_())