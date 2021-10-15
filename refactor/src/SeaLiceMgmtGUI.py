import datetime as dt
import sys
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer, QSettings
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMenu,
                             QAction, QFileDialog, QMessageBox, QGridLayout, QProgressBar, QCheckBox,
                             QGroupBox, QVBoxLayout, QTabWidget)
from colorcet import glasbey_dark, glasbey_light
from pyqtgraph import LinearRegionItem

from src.LicePopulation import LicePopulation
from src.Simulator import Simulator
from src.gui_utils.configuration import ConfigurationPane
from src.gui_utils.console import ConsoleWidget


class Window(QMainWindow):
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

        self._createSettings()
        self._createActions()
        self._connectActions()
        self._createMenuBar()

    @property
    def _getUniqueFarms(self):
        if self.states:
            return self.states_as_df.reset_index()["farm_name"].unique()
        return []

    @property
    def _colorPalette(self):
        if self.paperModeAction.isChecked():
            return glasbey_light
        return glasbey_dark

    def _createPlots(self):
        num_farms = len(self._getUniqueFarms)

        self.pqgPlotContainer = pg.GraphicsLayoutWidget(self)

        self.licePopulationPlots = [self.pqgPlotContainer.addPlot(title=f"Lice Population of farm {i}", row=0, col=i)
                                    for i in range(num_farms)]
        self.fishPopulationPlots = [self.pqgPlotContainer.addPlot(title=f"Fish Population of farm {i}", row=1, col=i)
                                    for i in range(num_farms)]
        self.payoffPlot = self.pqgPlotContainer.addPlot(title="Cumulated payoff", row=2, col=0)

        self.licePopulationLegend: Optional[pg.LegendItem] = None

        self.geno_to_curve: Dict[str, Dict[str, pg.PlotItem]] = {}
        self.stages_to_curve: Dict[str, Dict[str, pg.PlotItem]] = {}
        self.fish_population: Dict[str, pg.PlotItem] = {}

        # add grid, synchronise Y-range
        # TODO: make this toggleable?
        for plot in self.licePopulationPlots + self.fishPopulationPlots:
            plot.showGrid(x=True, y=True)

        for plotList in [self.licePopulationPlots, self.fishPopulationPlots]:
            for idx in range(1, len(plotList)):
                plotList[idx].setYLink(plotList[idx - 1])
                plotList[idx].setXLink(plotList[idx - 1])

        self.payoffPlot.showGrid(x=True, y=True)

    def _createWidgets(self):
        self.wd = QtWidgets.QWidget(self)
        self.setCentralWidget(self.wd)

        self._createPlotPane()
        self._createConfigurationPane()
        self._createConsole()
        self._createTabs()

    def _createPlotPane(self):
        self.plotPane = QtWidgets.QWidget(self)

        self._createPlots()
        self._createPlotOptionGroup()
        self._createProgressBar()

        mainLayout = QGridLayout()
        self.plotPane.setLayout(mainLayout)

        mainLayout.addWidget(self.pqgPlotContainer, 0, 0, 3, 1)
        mainLayout.addWidget(self.plotButtonGroup, 3, 0)
        mainLayout.addWidget(self.progressBar, 4, 0, 1, 2)

    def _createConfigurationPane(self):
        self.configurationPane = ConfigurationPane()

    def _createTabs(self):
        self.plotTabs = QTabWidget(self)

        self.plotTabs.addTab(self.plotPane, "Plotter")
        self.plotTabs.addTab(self.configurationPane, "Configuration")
        self.plotTabs.addTab(self.console, "Debugging console")

    def _createPlotOptionGroup(self):
        # Panel in the bottom
        self.plotButtonGroup = QGroupBox("Plot options", self)
        self.plotButtonGroupLayout = QVBoxLayout()
        self.showDetailedGenotypeCheckBox = QCheckBox("S&how detailed genotype information", self)
        self.showOffspringDistribution = QCheckBox("Sh&ow offspring distribution", self)

        self.showDetailedGenotypeCheckBox.stateChanged.connect(lambda _: self._updatePlot())
        self.showOffspringDistribution.stateChanged.connect(lambda _: self._updatePlot())

        self.plotButtonGroupLayout.addWidget(self.showDetailedGenotypeCheckBox)
        self.plotButtonGroupLayout.addWidget(self.showOffspringDistribution)

        self.plotButtonGroup.setLayout(self.plotButtonGroupLayout)

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
        # TODO: updatePlot should invoked after a signal
        self._updatePlot()

    def _createLoaderWorker(self, filename):
        # TODO: is there a valid reason why we shouldn't make worker a QThread subclass?
        self.thread = QThread()
        self.worker = SimulatorLoadingWorker(filename)
        print(f"Opening {filename}")
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

    def _cleanPlot(self):
        self.payoffPlot.clear()
        for plot in self.licePopulationPlots + self.fishPopulationPlots:
            plot.clear()

        for curves in self.stages_to_curve.values():
            for curve in curves.values():
                curve.clear()
        for curves in self.geno_to_curve.values():
            for curve in curves.values():
                curve.clear()

        if self.licePopulationLegend:
            self.licePopulationLegend.clear()

    def _remountPlot(self):
        layout: QGridLayout = self.plotPane.layout()
        layout.removeWidget(self.pqgPlotContainer)
        # I need to know the exact number of farms right now
        self._createPlots()
        layout.addWidget(self.pqgPlotContainer, 0, 0, 3, 1)
        self._updatePlot()

    def _updatePlot(self):
        # TODO: not suited for real-time
        # TODO: this function has become an atrocious mess

        self._cleanPlot()
        if self.states is None:
            return

        # if the number of farms has changed
        elif len(self._getUniqueFarms) != len(self.licePopulationPlots):
            self._remountPlot()

        self.licePopulationLegend = self.licePopulationPlots[0].addLegend()

        farm_list = self._getUniqueFarms
        df = self.states_as_df.reset_index()

        if self.showDetailedGenotypeCheckBox.isChecked():
            allele_names = ["a", "A", "Aa"]  # TODO: extract from column names
            colours = dict(zip(allele_names, self._colorPalette[:len(allele_names)]))

            for farm_idx, farm_name in enumerate(farm_list):
                farm_df = df[df["farm_name"] == farm_name]

                allele_data = {allele: farm_df[allele].to_numpy() for allele in allele_names}
                self.geno_to_curve[farm_name] = {
                    allele_name: self.licePopulationPlots[farm_idx].plot(stage_value, name=allele_name,
                                                                         pen=colours[allele_name])
                    for allele_name, stage_value in allele_data.items()
                }
        else:
            for farm_idx, farm_name in enumerate(farm_list):
                farm_df = df[df["farm_name"] == farm_name]
                stages = farm_df[LicePopulation.lice_stages].applymap(lambda geno_data: sum(geno_data.values()))

               # render per stage
                colours = dict(zip(LicePopulation.lice_stages, self._colorPalette[:len(stages)]))

                self.stages_to_curve[farm_name] = {
                    stage: self.licePopulationPlots[farm_idx].plot(
                        series.to_numpy(), name=stage, pen=colours[stage])
                    for stage, series in stages.items()}

                # treatment markers
                treatment_days_df = farm_df[farm_df["is_treating"]]["timestamp"] - self.times[0]
                treatment_days = treatment_days_df.apply(lambda x: x.days).to_numpy()

                # Generate treatment regions by looking for the first non-consecutive treatment blocks.
                # There may be a chance where multiple treatments happen consecutively, on which case
                # we simply consider them as a unique case.
                # TODO: move this in another function
                # TODO: we are not keeping tracks of all the PlotItem's - clear events may not delete them
                treatment_ranges = []
                lo = 0
                for i in range(1, len(treatment_days)):
                    if treatment_days[i] > treatment_days[i - 1] + 1:
                        treatment_ranges.append((treatment_days[lo], treatment_days[i - 1] + 1))
                        lo = i
                treatment_ranges.append((lo, i))

                treatment_lri = [LinearRegionItem(
                    values=range,
                    movable=False) for range in treatment_ranges]
                for lri in treatment_lri:
                    self.licePopulationPlots[farm_idx].addItem(lri)

                if self.showOffspringDistribution.isChecked():
                    # get gross arrivals
                    arrivals_gross = farm_df["arrivals_per_cage"].apply(
                        lambda cages: sum([sum(cage.values()) for cage in cages])).to_numpy()
                    self.licePopulationPlots[farm_idx].plot(arrivals_gross, name="Offspring", pen=self._colorPalette[7])

        # TODO: add this to pandas
        payoffs = [float(state.payoff) for state in self.states]
        self.payoffPlot.plot(payoffs)

        for farm_idx, farm_name in enumerate(farm_list):
            # TODO: condense this loop and the previous one
            num_fish = df[df["farm_name"] == farm_name]["num_fish"].to_numpy()
            self.fish_population[farm_name] = self.fishPopulationPlots[farm_idx].plot(num_fish, pen=self._colorPalette[0])

        # keep ranges consistent
        for plot in self.licePopulationPlots + self.licePopulationPlots:
            plot.setXRange(0, len(payoffs), padding=0)
            plot.vb.setLimits(xMin=0, xMax=len(payoffs))
        self.payoffPlot.setXRange(0, len(payoffs), padding=0)
        self.payoffPlot.vb.setLimits(xMin=0, xMax=len(payoffs))

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
        self.paperModeAction.toggled.connect(self._switchPaperMode)
        self.aboutAction.triggered.connect(self._openAboutMessage)
        self.clearAction.triggered.connect(self._cleanPlot)

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

    def _switchPaperMode(self, lightMode: bool):
        # in light mode prefer a strong white for background
        # in dark mode, prefer a dimmed white for labels
        background = 'w' if lightMode else 'k'
        foreground = 'k' if lightMode else 'd'

        pg.setConfigOption('background', background)
        pg.setConfigOption('foreground', foreground)

        self._remountPlot()

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
    win = Window(app)
    win.show()

    # KeyboardInterrupt trick
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    sys.exit(app.exec_())