"""
Configuration pane
"""
import json
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableView, QLabel

from slim.Config import Config


class ConfigurationPane(QWidget):
    newConfig = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.newConfig.connect(self._populateConfigurationPanelFromConfig)

        self.mainLayout = QVBoxLayout()
        self.runtimeConfigView = QTableView()
        self.farmConfigView = QTableView()
        self._populateListView(Path("config_data/config.schema.json"), self.runtimeConfigView)
        self._populateListView(Path("config_data/params.schema.json"), self.farmConfigView)

        self.mainLayout.addWidget(QLabel("Runtime Configuration"))
        self.mainLayout.addWidget(self.runtimeConfigView)
        self.mainLayout.addWidget(QLabel("Farm Configuration"))
        self.mainLayout.addWidget(self.farmConfigView)
        self.setLayout(self.mainLayout)

    def _populateListView(self, schema: Path, view: QTableView):
        with schema.open() as f:
            schema_dict = json.load(f)

        properties = schema_dict["properties"]
        model = QStandardItemModel()
        for idx, (property, value) in enumerate(properties.items()):
            item = QStandardItem(property)
            field = QStandardItem("")
            item.setEditable(False)
            field.setEditable(False)
            item.setToolTip(value["description"])
            model.setItem(idx, 0, item)
            model.setItem(idx, 1, field)

        view.setModel(model)

    def _populateConfigurationPanelFromConfig(self, config: Config):
        self._populateViewFromConfig(config, self.runtimeConfigView)
        self._populateViewFromConfig(config, self.farmConfigView)

    def _populateViewFromConfig(self, config: Config, view: QTableView):
        # From the given configuration object, parse the config
        model = view.model()
        positions = {}
        rows = model.rowCount()

        for idx in range(rows):
            item = model.item(idx, 0)
            positions[item.text()] = idx

        print(positions)

        for key, value in vars(config).items():
            if key not in positions:
                continue
            item = model.item(positions[key], 1)
            if isinstance(value, str):
                item.setText(value)
            else:
                item.setText(str(value))