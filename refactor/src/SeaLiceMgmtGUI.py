import datetime as dt
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from colorcet import glasbey_dark
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer, QSettings, Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMenu,
                             QAction, QFileDialog, QMessageBox, QGridLayout, QProgressBar, QSplitter, QCheckBox,
                             QGroupBox, QVBoxLayout, QComboBox)

from src.LicePopulation import LicePopulation
from src.Simulator import Simulator


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

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

        mainLayout = QGridLayout()
        self.wd.setLayout(mainLayout)

        mainLayout.addWidget(self.plotPane, 0, 0, 2, 1)
        mainLayout.addWidget(self.plotButtonGroup, 0, 1)
        mainLayout.addWidget(self.progressBar, 2, 0, 1, 2)

        #mainLayout.setRowStretch(1, 1)
        #mainLayout.setRowStretch(2, 1)

        self._createSettings()
        self._createActions()
        self._connectActions()
        self._createMenuBar()

    def _createPlots(self):
        ## PLOTS
        self.plotPane = pg.GraphicsLayoutWidget(self)

        self.licePopulationPlot = self.plotPane.addPlot(title="Lice Population per stage", row=0, col=0)
        self.payoffPlot = self.plotPane.addPlot(title="Cumulated payoff", row=1, col=0)
        self.payoffPlot.setXLink(self.licePopulationPlot)

        self.licePopulationLegend = None

        # We cannot know the genotypes in advance
        self.geno_to_curve = {}
        self.stages_to_curve = {}

        self.licePopulationPlot.showGrid(x=True, y=True)
        self.payoffPlot.showGrid(x=True, y=True)

    def _createWidgets(self):
        self.wd = QtWidgets.QWidget(self)
        self.setCentralWidget(self.wd)

        self._createPlots()

        # Pane on the right
        self.plotButtonGroup = QGroupBox("Plot options", self)
        self.plotButtonGroupLayout = QVBoxLayout()
        self.farmSelector = QComboBox(self)
        self.farmSelector.setEnabled(False)
        self.showDetailedGenotypeCheckBox = QCheckBox("S&how detailed genotype information", self)

        self.showDetailedGenotypeCheckBox.stateChanged.connect(lambda _: self._updatePlot())
        self.plotButtonGroupLayout.addWidget(self.farmSelector)
        self.plotButtonGroupLayout.addWidget(self.showDetailedGenotypeCheckBox)
        self.plotButtonGroup.setLayout(self.plotButtonGroupLayout)

        ## PROGRESS BAR
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
        self.states_as_df = Simulator.dump_as_pd(states, times)

        # how many unique farms are there?
        #farms = self.states[0].organisation.farms

        self._updatePlot()

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

    def _cleanPlot(self):
        self.payoffPlot.clear()
        for curve in self.stages_to_curve.values():
            curve.clear()
        for curve in self.geno_to_curve.values():
            curve.clear()

        if self.licePopulationLegend:
            self.licePopulationLegend.clear()

    def _updatePlot(self):
        # TODO: not suited for real-time
        # TODO: only showing one farm here

        self._cleanPlot()
        if self.states is None:
            return

        self.licePopulationLegend = self.licePopulationPlot.addLegend()

        if self.showDetailedGenotypeCheckBox.isChecked():
            df = self.states_as_df.reset_index()

            first_farm_df = df[df["farm_name"] == "farm_0"]
            allele_names = ["a", "A", "Aa"] # TODO: extract from column names
            colours = dict(zip(allele_names, glasbey_dark[:len(allele_names)]))

            allele_data = {allele: first_farm_df[allele].to_numpy() for allele in allele_names}
            self.geno_to_curve = {allele_name: self.licePopulationPlot.plot(stage_value, name=allele_name, pen=colours[allele_name])
                                  for allele_name, stage_value in allele_data.items()}
        else:
            # TODO: use pandas here
            population_data = []
            for snapshot in self.states:
                population_data.append(snapshot.organisation.farms[0].lice_population)

            # render per stage
            stages = {k: np.array([population.get(k, 0) for population in population_data]) for k in LicePopulation.lice_stages}

            # TODO: use a proper color palette from colorcet or matplotlib
            colours = dict(zip(LicePopulation.lice_stages, glasbey_dark[:len(stages)]))
            self.stages_to_curve = {stage: self.licePopulationPlot.plot(stages[stage], name=stage, pen=colours[stage])
                                    for stage in LicePopulation.lice_stages}

        payoffs = [float(state.payoff) for state in self.states]
        self.payoffPlot.plot(payoffs)

        # keep ranges consistent
        self.licePopulationPlot.setXRange(0, len(payoffs), padding=0)
        self.licePopulationPlot.vb.setLimits(xMin=0, xMax=len(payoffs))
        self.payoffPlot.setXRange(0, len(payoffs), padding=0)
        self.payoffPlot.vb.setLimits(xMin=0, xMax=len(payoffs))


    def _createActions(self):
        self.loadDumpAction = QAction("&Load Dump")
        self.paperModeAction = QAction("Set &light mode (paper mode)")
        self.paperModeAction.setCheckable(True)
        self.aboutAction = QAction("&About", self)
        self.showDetailedGenotypeAction = QAction(self)

        self._updateRecentFilesActions()

    def _connectActions(self):
        self.loadDumpAction.triggered.connect(self.openDump)
        self.paperModeAction.toggled.connect(self._switchPaperMode)
        self.aboutAction.triggered.connect(self._openAboutMessage)

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

        layout: QGridLayout = self.wd.layout()
        layout.removeWidget(self.plotPane)
        self._createPlots()
        layout.addWidget(self.plotPane, 0, 0, 2, 1)
        self._updatePlot()

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
                new_action.triggered.connect(lambda: self._createLoaderWorker(option))
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
    win = Window()
    win.show()

    # KeyboardInterrupt trick
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    sys.exit(app.exec_())