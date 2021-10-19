from __future__ import annotations

from typing import Optional, Dict, TYPE_CHECKING, List

import numpy as np
import pandas as pd
import pyqtgraph as pg
import scipy
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QWidget, QGridLayout, QGroupBox, QVBoxLayout, QCheckBox, QSpinBox, QLabel
from colorcet import glasbey_light, glasbey_dark
from pyqtgraph import LinearRegionItem

from src.LicePopulation import LicePopulation, GenoDistrib
from src.gui_utils.model import SimulatorSingleRunState, SimulatorOptimiserState

if TYPE_CHECKING:
    from src.SeaLiceMgmtGUI import Window


class SingleRunPlotPane(QWidget):
    """
    Main visualisation pane for the plots
    """
    # TODO: Split PlotPane in PlotPane + PlotWidget + PlotWidgetOptions + ...
    def __init__(self, mainPane: Window):
        super().__init__()
        self.mainPane = mainPane
        self.state: Optional[SimulatorSingleRunState] = None

        self._createPlots()
        self._createPlotOptionGroup()

        # Properties used inside the plot pane
        self._paperMode = False
        self._showGenotype = False

        mainLayout = QGridLayout()
        self.setLayout(mainLayout)

        mainLayout.addWidget(self.pqgPlotContainer, 0, 0, 3, 1)
        mainLayout.addWidget(self.plotButtonGroup, 3, 0)

        mainPane.loadedSimulatorState.connect(self._updateModel)

    @property
    def _getUniqueFarms(self):
        if self.state:
            return self.state.states_as_df.reset_index()["farm_name"].unique()
        return []

    @property
    def _colorPalette(self):
        if self._paperMode:
            return glasbey_light
        return glasbey_dark

    @property
    def paperMode(self):
        return self._paperMode

    @property
    def showGenotype(self):
        return self._showGenotype

    def setGenotype(self, value: bool):
        self._showGenotype = value
        self._updatePlot()

    def setPaperMode(self, lightMode: bool):
        # in light mode prefer a strong white for background
        # in dark mode, prefer a dimmed white for labels
        self._paperMode = lightMode
        background = 'w' if lightMode else 'k'
        foreground = 'k' if lightMode else 'd'

        pg.setConfigOption('background', background)
        pg.setConfigOption('foreground', foreground)

        self._remountPlot()


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

    def _createPlotOptionGroup(self):
        # Panel in the bottom
        self.plotButtonGroup = QGroupBox("Plot options", self)
        self.plotButtonGroupLayout = QVBoxLayout()
        self.showDetailedGenotypeCheckBox = QCheckBox("S&how detailed genotype information", self)
        self.showOffspringDistribution = QCheckBox("Sh&ow offspring distribution", self)
        self.convolutionKernelSizeLabel = QLabel("Kernel size")
        self.convolutionKernelSizeBox = QSpinBox(self)
        self.convolutionKernelSizeBox.setRange(1, 10)
        self.convolutionKernelSizeBox.setValue(3)
        self.convolutionKernelSizeBox.valueChanged.connect(lambda _: self._updatePlot())

        self.showDetailedGenotypeCheckBox.stateChanged.connect(lambda _: self._updatePlot())
        self.showOffspringDistribution.stateChanged.connect(lambda _: self._updatePlot())

        self.plotButtonGroupLayout.addWidget(self.showDetailedGenotypeCheckBox)
        self.plotButtonGroupLayout.addWidget(self.showOffspringDistribution)
        self.plotButtonGroupLayout.addWidget(self.convolutionKernelSizeLabel)
        self.plotButtonGroupLayout.addWidget(self.convolutionKernelSizeBox)

        self.plotButtonGroup.setLayout(self.plotButtonGroupLayout)

    def cleanPlot(self):
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
        layout: QGridLayout = self.layout()
        layout.removeWidget(self.pqgPlotContainer)
        # I need to know the exact number of farms right now
        self._createPlots()
        layout.addWidget(self.pqgPlotContainer, 0, 0, 3, 1)
        self._updatePlot()

    def _updateModel(self, model: SimulatorSingleRunState):
        self.state = model
        self._updatePlot()

    def _updatePlot(self):
        # TODO: not suited for real-time
        # TODO: this function has become an atrocious mess

        self.cleanPlot()
        if self.state is None:
            return

        # if the number of farms has changed
        elif len(self._getUniqueFarms) != len(self.licePopulationPlots):
            self._remountPlot()

        self.licePopulationLegend = self.licePopulationPlots[0].addLegend()

        farm_list = self._getUniqueFarms
        df = self.state.states_as_df.reset_index()

        for farm_idx, farm_name in enumerate(farm_list):
            farm_df = df[df["farm_name"] == farm_name]

            # Show fish population
            num_fish = df[df["farm_name"] == farm_name]["num_fish"].to_numpy()
            self.fish_population[farm_name] = self.fishPopulationPlots[farm_idx].plot(num_fish, pen=self._colorPalette[0])

            # Genotype information
            if self.showDetailedGenotypeCheckBox.isChecked():
                allele_names = GenoDistrib.allele_labels
                colours = dict(zip(allele_names, self._colorPalette[:len(allele_names)]))

                allele_data = {allele: self._convolve(farm_df[allele].to_numpy()) for allele in allele_names}
                self.geno_to_curve[farm_name] = {
                    allele_name: self.licePopulationPlots[farm_idx].plot(stage_value, name=allele_name,
                                                                         pen=colours[allele_name])
                    for allele_name, stage_value in allele_data.items()
                }
            # Stage information
            else:
                stages = farm_df[LicePopulation.lice_stages].applymap(
                    lambda geno_data: sum(geno_data.values()))

                # render per stage
                colours = dict(zip(LicePopulation.lice_stages, self._colorPalette[:len(stages)]))

                self.stages_to_curve[farm_name] = {
                    stage: self.licePopulationPlots[farm_idx].plot(
                        self._convolve(series.to_numpy()), name=stage, pen=colours[stage])
                    for stage, series in stages.items()}


                if self.showOffspringDistribution.isChecked():
                    # get gross arrivals
                    arrivals_gross = farm_df["arrivals_per_cage"].apply(
                        lambda cages: sum([sum(cage.values()) for cage in cages])).to_numpy()
                    self.licePopulationPlots[farm_idx].plot(arrivals_gross, name="Offspring", pen=self._colorPalette[7])

            # Plot treatments
            # note: PG does not allow to reuse the same
            treatment_lri = self._getTreatmentRegions(farm_df, 2)
            for lri in treatment_lri:
                self.licePopulationPlots[farm_idx].addItem(lri[0])
                self.fishPopulationPlots[farm_idx].addItem(lri[1])

        # because of the leaky scope, this is going to be something
        payoffs = farm_df["payoff"].to_numpy()
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

    def _convolve(self, signal):
        kernel_size = self.convolutionKernelSizeBox.value()
        # TODO: support multiple rolling averages
        kernel = np.full(kernel_size, 1/kernel_size)
        return scipy.ndimage.convolve(signal, kernel, mode="nearest")


    def _getTreatmentRegions(self, farm_df: pd.DataFrame, shape: int) -> List[List[LinearRegionItem]]:
        # generate treatment regions
        # treatment markers
        treatment_days_df = farm_df[farm_df["is_treating"]]["timestamp"] - self.state.times[0]
        treatment_days = treatment_days_df.apply(lambda x: x.days).to_numpy()

        # Generate treatment regions by looking for the first non-consecutive treatment blocks.
        # There may be a chance where multiple treatments happen consecutively, on which case
        # we simply consider them as a unique case.
        # TODO: move this in another function
        # TODO: we are not keeping tracks of all the PlotItem's - clear events may not delete them
        if len(treatment_days) > 0:
            treatment_ranges = []
            lo = 0
            for i in range(1, len(treatment_days)):
                if treatment_days[i] > treatment_days[i - 1] + 1:
                    treatment_ranges.append((treatment_days[lo], treatment_days[i - 1]))
                    lo = i
            treatment_ranges.append((treatment_days[lo], treatment_days[-1]))

            return [[LinearRegionItem(values=trange, movable=False) for _ in range(shape)]
                    for trange in treatment_ranges]
        return []


class OptimiserPlotPane(QWidget):
    def __init__(self, mainPane: Window):
        super().__init__(mainPane)
        self._addWidgets()

        self.optimiserState: Optional[SimulatorOptimiserState] = None

        mainPane.loadedOptimiserState.connect(self._updateState)

    def _addWidgets(self):
        self.pqgPlotContainer = pg.GraphicsLayoutWidget(self)
        self.payoffPlot = self.pqgPlotContainer.addPlot(title="Hill-climbing payoff walk", row=0, col=0)

        mainLayout = QVBoxLayout()
        self.setLayout(mainLayout)
        mainLayout.addWidget(self.pqgPlotContainer)

    def _updateState(self, state: SimulatorOptimiserState):
        self.optimiserState = state
        self._updatePlot()

    def _updatePlot(self):
        df = self.optimiserState.states_as_df

        print(df)

        self.payoffPlot.plot(df["payoff"])
