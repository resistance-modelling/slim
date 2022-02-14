"""
This module provides plotting widgets and utils.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional, TYPE_CHECKING, List

import numpy as np
import pandas as pd
import pyqtgraph as pg
import scipy
import scipy.ndimage

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QWidget,
    QGridLayout,
    QGroupBox,
    QVBoxLayout,
    QCheckBox,
    QSpinBox,
    QLabel,
    QScrollArea,
    QListWidget,
    QSplitter,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
)
from colorcet import glasbey_light, glasbey_dark
from pyqtgraph import LinearRegionItem, PlotItem, GraphicsLayoutWidget
from pyqtgraph.exporters import ImageExporter, SVGExporter
from pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem

from slim.simulation.lice_population import LicePopulation, GenoDistrib
from slim.gui_utils.model import (
    SimulatorSingleRunState,
    SimulatorOptimiserState,
    CurveListState,
)

if TYPE_CHECKING:
    from slim.SeaLiceMgmtGUI import Window


class SmoothedPlotItemWrap:
    """
    A replacement for PlotItem that also implements convolution and averaging
    """

    color_palette = glasbey_light

    def __init__(
        self,
        plot_item: PlotItem,
        smoothing_size: int,
        average_factor: int,
        method="linear",
    ):
        """
        :param plot_item: the wrapped :class:`PlotItem`
        :param smoothing_size: the amount of smoothing to apply
        :param average_factor: the common divider across lines (used when comparing farms with different cage numbers)
        :param method: if to use linear smoothing. Currently, it is the only one supported.
        """
        self.plot_item = plot_item
        self.kernel_size = smoothing_size
        self.average_factor = average_factor
        self.method = method
        self.original_title = plot_item.titleLabel.text
        self.title_style = plot_item.titleLabel.opts
        self.min_date = None

    def plot(
        self,
        y,
        stage: Optional[str] = None,
        x: Optional[List[dt.datetime]] = None,
        **kwargs,
    ) -> PlotDataItem:
        """
        Plot a list of values

        For a list of options check :meth:`pg.PlotdataItem.__init__`

        :param y: the sequence to plot
        :param stage: if y is a population, pick a selected lifecycle stage
        :param x: provide an optional x-axis
        """
        # compute the signal
        # TODO: support multiple rolling averages
        if self.method == "linear":
            kernel = np.full(self.kernel_size, 1 / self.kernel_size)
        else:
            kernel = np.array([1])

        if stage:
            # determine the strokes to use
            stages_num = len(LicePopulation.lice_stages)
            stages_palette, egg_palette = (
                SmoothedPlotItemWrap.color_palette[:stages_num],
                SmoothedPlotItemWrap.color_palette[stages_num:],
            )
            stages_styles = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]  # enums from Qt.PenStyle

            stages_colours = dict(zip(LicePopulation.lice_stages, stages_palette))
            stages_styles_dict = dict(zip(LicePopulation.lice_stages, stages_styles))

            stage_colour = stages_colours[stage]
            stage_style = stages_styles_dict[stage]
            stage_bio_name = LicePopulation.lice_stages_bio_labels[stage]

            _kwargs = {
                "name": stage_bio_name,
                "pen": {"style": stage_style, "color": stage_colour, "width": 1.5},
                **kwargs,
            }
        else:
            _kwargs = kwargs

        if x is not None:
            x = [(day - self.min_date).days for day in x]

        return self.plot_item.plot(
            x=x,
            y=scipy.ndimage.convolve(y / self.average_factor, kernel, mode="nearest"),
            **_kwargs,
        )

    def setSmoothingFactor(self, kernel_size: int):
        self.kernel_size = kernel_size

    def setAverageFactor(self, average_factor: int):
        self.average_factor = average_factor
        title = self.original_title + (
            " averaged across cages" if average_factor > 1 else ""
        )
        self.plot_item.setTitle(title, **self.title_style)

    def setXAxis(self, timestamps: List[dt.datetime]):
        self.min_date = timestamps[0]

    def heightForWidth(self, width):
        return int(width * 1.5)

    def __getattr__(self, item):
        return getattr(self.plot_item, item)


class ScientificAxisItem(pg.AxisItem):
    """A non-scientific axis. See <https://stackoverflow.com/a/43782129>_"""

    def tickStrings(self, values, scale, spacing):
        if self.logMode:
            return self.logTickStrings(values, scale, spacing)

        return ["%.4g" % (value * scale) for value in values]


class SmoothedGraphicsLayoutWidget(GraphicsLayoutWidget):
    """A wrapper for a GraphicsLayoutWidget that supports smoothing."""

    # TODO these names are terrible
    newKernelSize = pyqtSignal()
    newAverageFactor = pyqtSignal()

    def __init__(self, parent: SingleRunPlotPane):
        super().__init__(parent)
        self.pane = parent
        self.coord_to_plot = {}
        self.timestamps = None

        smoothing_kernel_size_widget = parent.convolutionKernelSizeBox
        averaging_widget = parent.normaliseByCageCheckbox

        smoothing_kernel_size_widget.valueChanged.connect(self._setSmoothingFactor)
        averaging_widget.stateChanged.connect(self._setAverageFactor)

    def _getAverageFactor(self, farm_idx, checkbox_state: int):
        parent = self.pane

        state = parent.state
        if state:
            if state.cfg.nfarms <= farm_idx:
                return 1
            num_cages = state.cfg.farms[farm_idx].n_cages
            average_factor = num_cages if checkbox_state == 2 else 1
            return average_factor
        return 1

    def _setSmoothingFactor(self, value):
        for plot_item in self.coord_to_plot.values():
            plot_item.setSmoothingFactor(value)

        self.newKernelSize.emit()

    def _setAverageFactor(self, value):
        for (farm_idx, _), plot_item in self.coord_to_plot.items():
            average_factor = self._getAverageFactor(farm_idx, value)
            plot_item.setAverageFactor(average_factor)
        self.newAverageFactor.emit()

    def addSmoothedPlot(
        self, exclude_from_averaging=False, force_scientific=False, **kwargs
    ) -> SmoothedPlotItemWrap:
        axis_params = ["left", "right", "top", "bottom"]
        axis_dict = {}
        for axis_param in axis_params:
            if axis_param in kwargs:
                params = kwargs[axis_param]
                if isinstance(params, str):
                    params = {"text": params}
                if force_scientific:
                    axis_item_type = ScientificAxisItem
                else:
                    axis_item_type = pg.AxisItem
                axis_dict[axis_param] = axis = axis_item_type(
                    **{"orientation": axis_param, **params}
                )
                # By default, the axes miss the label and believe they are using SI measures, thus breaking
                # the scaling altogether. In theory, it is not a bug but the behaviour was so horribly visualised
                # it looked like one.
                axis.showLabel(True)
                axis.enableAutoSIPrefix(False)
                del kwargs[axis_param]

        plot: PlotItem = self.addPlot(**kwargs, axisItems=axis_dict)

        # Make titles a bit larger
        if "title" in kwargs:
            plot.setTitle(kwargs["title"], size="12.5pt")

        for axis_item in axis_dict.values():
            axis_item.setParentItem(plot)

        parent = self.pane

        smoothing_kernel_size_widget = parent.convolutionKernelSizeBox
        averaging_widget = parent.normaliseByCageCheckbox

        col = kwargs["col"]
        farm_idx = kwargs["row"]

        smoothing_kernel_size = smoothing_kernel_size_widget.value()
        average = self._getAverageFactor(farm_idx, averaging_widget.isChecked())

        smoothed_plot_item = SmoothedPlotItemWrap(plot, smoothing_kernel_size, average)

        if not exclude_from_averaging:
            self.coord_to_plot[(farm_idx, col)] = smoothed_plot_item

        return smoothed_plot_item

    def disconnectAll(self):
        parent = self.pane

        smoothing_kernel_size_widget = parent.convolutionKernelSizeBox
        averaging_widget = parent.normaliseByCageCheckbox

        smoothing_kernel_size_widget.disconnect()
        averaging_widget.disconnect()

    def deleteLater(self):
        self.disconnectAll()
        super().deleteLater()

    def enforceAspectRatio(self):
        # For some reason size policies are ignored, even when setting the internal GraphicsLayout policy.
        # Maybe there is something else that has to be changed? Need to investigate...
        # This is a hack that works half of the time. Expect flickering.
        size: QSize = self.size()
        num_farms = len(self.pane._farmIDs)
        # 9/60 is not the real aspect ratio, but it takes into account padding
        self.setFixedHeight((num_farms + 1) * size.width() * 9 / 60)

    def setDefaultXAxis(self, timestamps: List[dt.datetime]):
        """
        Set the x-axis of all the plots
        """
        for plot in self.coord_to_plot.values():
            if isinstance(plot, SmoothedPlotItemWrap):
                plot.setXAxis(timestamps)


class LightModeMixin:
    def __init__(self):
        self._paperMode = False

    @property
    def _colorPalette(self):
        if self._paperMode:
            return glasbey_dark
        return glasbey_light

    @property
    def paperMode(self):
        return self._paperMode

    def setPaperMode(self, lightMode: bool):
        # in light mode prefer a strong white for background
        # in dark mode, prefer a dimmed white for labels
        self._paperMode = lightMode
        background = "w" if lightMode else "k"
        foreground = "k" if lightMode else "d"

        pg.setConfigOption("background", background)
        pg.setConfigOption("foreground", foreground)

        # TODO: that's a terrible idea. Why is a class attribute set to an instance's attribute?
        # That's okay because everything is global (and it shouldn't...)
        SmoothedPlotItemWrap.color_palette = self._colorPalette

        self._remountPlot()

    def _remountPlot(self):
        pass


class SingleRunPlotPane(LightModeMixin, QWidget):
    """
    Main visualisation pane for the plots
    """

    # TODO: Split PlotPane in PlotPane + PlotWidget + PlotWidgetOptions + ...
    def __init__(self, mainPane: Window):
        LightModeMixin.__init__(self)
        QWidget.__init__(self, mainPane)

        self.mainPane = mainPane
        self.state: Optional[SimulatorSingleRunState] = None

        # Smoothing depends on the smoothing parameter being set for SmoothedPlotItemWrap
        # Alternatively, one could break the relationship between this and the QSpinBox object, but it may cause
        # synchronisation issues.

        # The overall layout is the following
        # ---------------------------------
        # |                     |         |
        # |  PLOTS              | Options |
        # ...                   | ...     |
        # |                     | ----    |
        # |                     | Curves  |
        # ---------------------------------
        self._createPlotOptionGroup()
        self._createPlots()

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setLayout(QVBoxLayout(self.scrollArea))
        self.scrollArea.setWidget(self.pqgPlotContainer)

        # Properties used inside the plot pane
        self._showGenotype = False

        plotLayout = QGridLayout()

        plotLayout.addWidget(self.scrollArea, 0, 0)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.splitter)
        plotPane = QWidget()
        plotPane.setLayout(plotLayout)
        self.splitter.addWidget(plotPane)
        self.splitter.addWidget(self.plotButtonGroup)
        self.setLayout(mainLayout)

        mainPane.loadedSimulatorState.connect(self._updateModel)

    @property
    def _farmIDs(self):
        """
        if self.state:
            return self.state.states_as_df.reset_index()["farm_id"].unique()
        return []
        """

        if self.state:
            return [f"farm_{i}" for i in range(len(self.state.cfg.farms))]
        return []

    @property
    def _farmNames(self):
        if self.state:
            return [farm.name for farm in self.state.cfg.farms]
        pass

    @property
    def showGenotype(self):
        return self._showGenotype

    def setGenotype(self, value: bool):
        self._showGenotype = value
        self._updatePlot()

    def _createPlots(self):
        num_farms = len(self._farmIDs)

        self.pqgPlotContainer = SmoothedGraphicsLayoutWidget(self)
        self.pqgPlotContainer.newAverageFactor.connect(self._updatePlot)
        self.pqgPlotContainer.newKernelSize.connect(self._updatePlot)

        self.licePopulationPlots = [
            self.pqgPlotContainer.addSmoothedPlot(
                title=f"Lice population of farm {self._farmNames[i]}",
                left="Population",
                bottom="days",
                row=i,
                col=0,
            )
            for i in range(num_farms)
        ]
        self.fishPopulationPlots = [
            self.pqgPlotContainer.addSmoothedPlot(
                title=f"Fish population of farm {self._farmNames[i]}",
                left="Population",
                bottom="days",
                row=i,
                col=1,
            )
            for i in range(num_farms)
        ]
        self.aggregationRatePlot = [
            self.pqgPlotContainer.addSmoothedPlot(
                title=f"Lice aggregation of farm {self._farmNames[i]}",
                left="Population",
                bottom="days",
                row=i,
                col=2,
            )
            for i in range(num_farms)
        ]

        self.payoffPlot = self.pqgPlotContainer.addSmoothedPlot(
            exclude_from_averaging=True,
            title="Cumulated payoff",
            bottom="days",
            left="Payoff",
            force_scientific=True,
            row=num_farms,
            col=0,
        )
        self.extPressureRatios = self.pqgPlotContainer.addSmoothedPlot(
            title="External pressure ratios",
            bottom="days",
            left="Probability",
            row=num_farms,
            col=1,
        )

        self.legends: List[pg.LegendItem] = []

        # add grid, synchronise Y-range
        for plot in self.licePopulationPlots + self.fishPopulationPlots:
            plot.showGrid(x=True, y=False)

        for plotList in [
            self.licePopulationPlots,
            self.fishPopulationPlots,
            self.aggregationRatePlot,
        ]:
            for idx in range(1, len(plotList)):
                plotList[idx].setYLink(plotList[idx - 1])
                plotList[idx].setXLink(plotList[idx - 1])

        self.payoffPlot.showGrid(x=True, y=False)

        self.splitter.splitterMoved.connect(self.pqgPlotContainer.enforceAspectRatio)

    def _createPlotOptionGroup(self):
        # Panel in the bottom
        self.plotButtonGroup = QGroupBox("Plot options", self)
        self.plotButtonGroupLayout = QVBoxLayout()
        self.showDetailedGenotypeCheckBox = QCheckBox(
            "S&how detailed genotype information", self
        )
        self.showOffspringDistribution = QCheckBox("Sh&ow offspring distribution", self)
        self.normaliseByCageCheckbox = QCheckBox("Show average cage population", self)
        self.showFishPopulationDx = QCheckBox(
            "Show fish population mortality rates", self
        )
        self.convolutionKernelSizeLabel = QLabel("Kernel size")
        self.convolutionKernelSizeBox = QSpinBox(self)
        self.convolutionKernelSizeBox.setRange(1, 10)
        self.convolutionKernelSizeBox.setValue(3)

        self._createCurveList()

        self.showDetailedGenotypeCheckBox.stateChanged.connect(
            lambda _: self._updatePlot()
        )
        self.showOffspringDistribution.stateChanged.connect(
            lambda _: self._updatePlot()
        )
        self.showFishPopulationDx.stateChanged.connect(lambda _: self._updatePlot())

        self.plotButtonGroupLayout.addWidget(self.showDetailedGenotypeCheckBox)
        self.plotButtonGroupLayout.addWidget(self.showOffspringDistribution)
        self.plotButtonGroupLayout.addWidget(self.showFishPopulationDx)
        self.plotButtonGroupLayout.addWidget(self.convolutionKernelSizeLabel)
        self.plotButtonGroupLayout.addWidget(self.convolutionKernelSizeBox)
        self.plotButtonGroupLayout.addWidget(self.normaliseByCageCheckbox)
        self.plotButtonGroupLayout.addWidget(self.curveListWidget)

        self.plotButtonGroup.setLayout(self.plotButtonGroupLayout)

    def _createCurveList(self):
        self.splitter = QSplitter(self)
        curveListWidget = self.curveListWidget = QListWidget()

        self.selected_curve = CurveListState()

        curves = {v: k for (k, v) in LicePopulation.lice_stages_bio_long_names.items()}
        curves.update({"External pressure": "ExtP", "Produced eggs": "Eggs"})

        self._curves_to_member = dict(curves)

        for curve_label, curve_id in curves.items():
            item = QListWidgetItem(
                curve_label, curveListWidget, QtCore.Qt.ItemIsUserCheckable
            )
            item.setCheckState(QtCore.Qt.Checked)
            curveListWidget.addItem(item)

        curveListWidget.itemChanged.connect(self._updateItem)

    def _updateItem(self, item: QListWidgetItem):
        setattr(
            self.selected_curve,
            self._curves_to_member[item.text()],
            item.checkState() == QtCore.Qt.Checked,
        )

        self._updatePlot()

    def cleanPlot(self):
        self.payoffPlot.clear()
        self.extPressureRatios.clear()

        for plot in (
            self.licePopulationPlots
            + self.aggregationRatePlot
            + self.fishPopulationPlots
        ):
            plot.clear()

        for legend in self.legends:
            legend.clear()

    def _remountPlot(self):
        self.scrollArea.setWidget(None)
        # I need to know the exact number of farms right now
        self.pqgPlotContainer.deleteLater()
        self._createPlots()
        if self.state:
            self.pqgPlotContainer.setDefaultXAxis(self.state.times)
        self.scrollArea.setWidget(self.pqgPlotContainer)
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
        elif len(self._farmIDs) != len(self.licePopulationPlots):
            self._remountPlot()
            self.pqgPlotContainer.enforceAspectRatio()
            # _remountPlot calls _updatePlot implicitly so avoid recursion
            return

        if len(self._farmIDs) == 0:
            return

        self.legends = [
            plot.addLegend()
            for plot in (
                self.licePopulationPlots
                + self.fishPopulationPlots
                + self.aggregationRatePlot
            )
        ] + [self.extPressureRatios.addLegend()]

        farm_list = self._farmIDs
        df = self.state.states_as_df.reset_index()

        for farm_idx, farm_name in enumerate(farm_list):
            farm_df = df[df["farm_id"] == farm_name]

            # Show fish population
            num_fish = farm_df["num_fish"].to_numpy()

            # Report info
            farm_geo_name = self._farmNames[farm_idx]
            report_df = self.state.report_df
            if report_df is not None:
                report_farm = report_df[report_df["site_name"] == farm_geo_name]

            monocolour_pen = {"colour": self._colorPalette[0], "width": 1.5}

            # TODO: set derivativeMode instead
            if self.showFishPopulationDx.isChecked():
                num_fish_dx = -np.diff(num_fish)
                self.fishPopulationPlots[farm_idx].plot(num_fish_dx, pen=monocolour_pen)
            else:
                self.fishPopulationPlots[farm_idx].plot(
                    num_fish, pen=monocolour_pen, name="Expected"
                )
                if report_df is not None:
                    self.fishPopulationPlots[farm_idx].plot(
                        x=report_farm["date"],
                        y=report_farm["survived_fish"],
                        pen=self._colorPalette[2],
                        name="Ground truth",
                    )

            stages = farm_df[LicePopulation.lice_stages].applymap(
                lambda geno_data: sum(geno_data.values())
            )

            allele_names = GenoDistrib.allele_labels.values()
            allele_colours = dict(
                zip(allele_names, self._colorPalette[: len(allele_names)])
            )

            # Genotype information
            if self.showDetailedGenotypeCheckBox.isChecked():
                allele_data = {
                    allele: farm_df[allele].to_numpy() for allele in allele_names
                }

                for allele_name, stage_value in allele_data.items():
                    pen = {
                        "color": allele_colours[allele_name],
                        "width": monocolour_pen["width"],
                    }
                    self.licePopulationPlots[farm_idx].plot(
                        stage_value, name=allele_name, pen=pen
                    )

            # Stage information
            else:
                # TODO: eggs and ExtP are not stages but graphically speaking they are.
                # render per stage
                stages_num = len(LicePopulation.lice_stages)
                egg_palette = self._colorPalette[stages_num:]

                for stage, series in stages.items():
                    if getattr(self.selected_curve, stage):
                        self.licePopulationPlots[farm_idx].plot(
                            series.to_numpy(), stage=stage
                        )

                if self.selected_curve.Eggs:
                    pen = {"color": egg_palette[0], "width": monocolour_pen["width"]}
                    self.licePopulationPlots[farm_idx].plot(
                        farm_df["eggs"], name="Eggs", pen=pen
                    )
                if self.selected_curve.ExtP:
                    pen = {"color": egg_palette[1], "width": monocolour_pen["width"]}
                    self.licePopulationPlots[farm_idx].plot(
                        farm_df["new_reservoir_lice"], name="External Pressure", pen=pen
                    )

                if self.showOffspringDistribution.isChecked():
                    # get gross arrivals
                    pen = {**monocolour_pen, "color": egg_palette[0]}
                    if self.selected_curve.ExtP:
                        arrivals_gross = (
                            farm_df["arrivals_per_cage"]
                            .apply(
                                lambda cages: sum(
                                    [sum(cage.values()) for cage in cages]
                                )
                            )
                            .to_numpy()
                        )
                        self.licePopulationPlots[farm_idx].plot(
                            arrivals_gross, name="Offspring (L1+L2)", pen=pen
                        )

            aggregation_rate = stages["L5f"].to_numpy() / num_fish
            self.aggregationRatePlot[farm_idx].plot(
                aggregation_rate, pen=monocolour_pen, name="Expected"
            )

            # extract from report
            if report_df is not None:
                self.aggregationRatePlot[farm_idx].plot(
                    x=report_farm["date"],
                    y=report_farm["lice_count"].to_numpy(),
                    pen=self._colorPalette[2],
                    name="Ground truth",
                )

            # Plot treatments
            # note: PG does not allow to reuse the same
            treatment_lri = self._getTreatmentRegions(farm_df, 5)
            if len(treatment_lri) > 0:
                for lri in treatment_lri:
                    self.licePopulationPlots[farm_idx].addItem(lri[0])
                    self.fishPopulationPlots[farm_idx].addItem(lri[1])
                    self.aggregationRatePlot[farm_idx].addItem(lri[2])

                    # Multiple treatment regions may overlap this way, is this okay?
                    self.payoffPlot.addItem(lri[3])
                    self.extPressureRatios.addItem(lri[4])

        # END FARM PLOTTING

        # because of the leaky scope, this is going to be something
        payoffs = farm_df["payoff"].to_numpy()
        self.payoffPlot.plot(payoffs, pen=self._colorPalette[0])

        # External pressure ratios
        # TODO: fix this
        external_pressure_ratios = {
            geno: farm_df["new_reservoir_lice_ratios"].apply(lambda x: x[geno])
            for geno in GenoDistrib.alleles
        }
        for geno, extp_ratios in external_pressure_ratios.items():
            allele_name = GenoDistrib.allele_labels[geno]
            allele_name_bio = GenoDistrib.allele_labels_bio[geno]
            self.extPressureRatios.plot(
                extp_ratios, name=allele_name_bio, pen=allele_colours[allele_name]
            )

        for plot in self.licePopulationPlots:
            plot.setLogMode(False, True)

        # keep ranges consistent
        for plot in (
            self.licePopulationPlots
            + self.fishPopulationPlots
            + self.aggregationRatePlot
        ):
            plot.setXRange(0, len(payoffs), padding=0)
            plot.vb.setLimits(xMin=0, xMax=len(payoffs), yMin=0)

        for singlePlot in [self.payoffPlot, self.extPressureRatios]:
            singlePlot.setXRange(0, len(payoffs), padding=0)

        self.payoffPlot.vb.setLimits(xMin=0, xMax=len(payoffs))
        self.extPressureRatios.vb.setLimits(xMin=0, xMax=len(payoffs), yMin=0, yMax=1)

    def _convolve(self, signal):
        kernel_size = self.convolutionKernelSizeBox.value()
        # TODO: support multiple rolling averages?
        kernel = np.full(kernel_size, 1 / kernel_size)
        return scipy.ndimage.convolve(signal, kernel, mode="nearest")

    def _getTreatmentRegions(
        self, farm_df: pd.DataFrame, shape: int
    ) -> List[List[LinearRegionItem]]:
        # generate treatment regions
        # treatment markers
        treatment_days_df = (
            farm_df[farm_df["is_treating"]]["timestamp"] - self.state.times[0]
        )
        treatment_days = treatment_days_df.apply(lambda x: x.days).to_numpy()

        # Generate treatment regions by looking for the first non-consecutive treatment blocks.
        # There may be a chance where multiple treatments happen consecutively, on which case
        # we simply consider them as a unique case.

        if len(treatment_days) > 0:
            treatment_ranges = []
            lo = 0
            for i in range(1, len(treatment_days)):
                if treatment_days[i] > treatment_days[i - 1] + 1:
                    treatment_ranges.append((treatment_days[lo], treatment_days[i - 1]))
                    lo = i
            treatment_ranges.append((treatment_days[lo], treatment_days[-1]))

            return [
                [LinearRegionItem(values=trange, movable=False) for _ in range(shape)]
                for trange in treatment_ranges
            ]
        return []

    def exportPlots(self):
        """
        Export all the plots. This will open a dialog window asking for a directory where to save the images.

        The images are saved in both PNG and SVG format. The resolution depends on the window resolution.
        (To get greater control on plot resolution I'm afraid we need to export to matplotlib?)

        This will generate the following:

        * An image of the whole scene, called "{sim_id}_all"
        * An image for each plot, called "{sim_id}_{plot_type}_{i}", e.g. "Fyne_150_lice_2"
        """
        if not self.state:
            QMessageBox.information(self, "Export", "You need to load a dump first")
            return

        dir = QFileDialog.getExistingDirectory(
            self, "Export into a folder", ".", QFileDialog.ShowDirsOnly
        )
        if not dir:
            return
        if not Path(dir).is_dir():
            QMessageBox.critical(
                self, "Error", "The given directory does not exist. Halting!"
            )
            return

        def exportPlot(plot, name: str):
            png_exporter = ImageExporter(plot)
            svg_exporter = SVGExporter(plot)
            path = Path(dir) / f"{self.state.sim_name}_{name}"
            # TODO: svg does not work?
            png_exporter.export(str(path) + ".png")
            svg_exporter.export(str(path) + ".svg")

        exportPlot(self.pqgPlotContainer.scene(), "all")
        for idx, lice_plot in enumerate(self.licePopulationPlots):
            exportPlot(lice_plot.plot_item, f"lice_{idx}")
        for idx, fish_plot in enumerate(self.fishPopulationPlots):
            exportPlot(fish_plot.plot_item, f"fish_{idx}")
        for idx, aggregation_plot in enumerate(self.aggregationRatePlot):
            exportPlot(aggregation_plot.plot_item, f"aggregation_{idx}")

        exportPlot(self.payoffPlot.plot_item, "payoff")
        exportPlot(self.extPressureRatios.plot_item, "extpressure")

        QMessageBox.information(
            self, "Success", "The images have been generated with success!"
        )


class OptimiserPlotPane(QWidget, LightModeMixin):
    def __init__(self, mainPane: Window):
        super().__init__(mainPane)
        self.optimiserState: Optional[SimulatorOptimiserState] = None

        self._addWidgets()

        mainPane.loadedOptimiserState.connect(self._updateState)

    def _addWidgets(self):
        mainLayout = QVBoxLayout()
        self._addPlot()
        self.setLayout(mainLayout)
        mainLayout.addWidget(self.pqgPlotContainer)

    def _getUniqueFarms(self):
        if self.optimiserState:
            columns = self.optimiserState.states_as_df.columns
            return [column for column in columns if column.startswith("farm_")]
        return []

    def _addPlot(self):
        farms = self._getUniqueFarms()
        self.pqgPlotContainer = pg.GraphicsLayoutWidget(self)
        self.payoffPlot = self.pqgPlotContainer.addPlot(
            title="Hill-climbing payoff walk", row=0, col=0
        )

        self.individualProbas = []
        for idx, farm in enumerate(farms, 1):
            farm_name = farm[len("farm_") :]
            self.individualProbas.append(
                self.pqgPlotContainer.addPlot(
                    title=f"Defection probability for farm {farm_name}", row=idx, col=0
                )
            )

    def _updateState(self, state: SimulatorOptimiserState):
        self.optimiserState = state
        if len(self.individualProbas) != len(self._getUniqueFarms()):
            self._remountPlot()
        self._updatePlot()

    def _clearPlot(self):
        self.payoffPlot._clear()

        for probaPlot in self.individualProbas:
            probaPlot._clear()

    def _updatePlot(self):
        self._clearPlot()

        if self.optimiserState is None:
            return
        df = self.optimiserState.states_as_df

        penColour = self._colorPalette[0]
        payoff = df["payoff"].map(float).to_numpy()
        self.payoffPlot.plot(payoff, pen=penColour)

        for idx, farm in enumerate(self._getUniqueFarms()):
            self.individualProbas[idx].plot(df[farm], pen=penColour)

    def _remountPlot(self):
        layout = self.layout()
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
        self._addWidgets()
        layout.addWidget(self.pqgPlotContainer)

        self._updatePlot()
