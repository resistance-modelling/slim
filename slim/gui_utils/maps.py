"""
A Wrapper for Qt Location and Maps.
"""
import os
from typing import List

import numpy as np
from PyQt5.QtGui import QColor
from convertbng.util import convert_lonlat
from PyQt5 import QtCore, QtQuickWidgets, QtPositioning
from PyQt5.QtCore import pyqtSignal

from .model import PositionMarker, TransitionEndpoint

__all__ = ["MapWidget"]

from ..simulation.config import Config
import colorcet as cc


class MarkerModel(QtCore.QAbstractListModel):
    """Represent the individual sites"""

    PositionName, PositionRole, SourceRole = range(
        QtCore.Qt.UserRole, QtCore.Qt.UserRole + 3
    )

    def __init__(self, parent=None):
        super(MarkerModel, self).__init__(parent)
        self._markers: List[PositionMarker] = []

    def rowCount(self, parent=QtCore.QModelIndex):
        return len(self._markers)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        # Note: QVariant() maps to an Any
        if 0 <= index.row() < self.rowCount():
            marker = self._markers[index.row()]
            if role == MarkerModel.PositionName:
                return marker.name
            elif role == MarkerModel.PositionRole:
                return marker.position
            elif role == MarkerModel.SourceRole:
                return marker.source
        return QtCore.QVariant()

    def roleNames(self):
        return {
            MarkerModel.PositionName: b"name",
            MarkerModel.PositionRole: b"coords",
            MarkerModel.SourceRole: b"source",
        }

    def appendMarker(self, name, coordinate):
        self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount())
        # Marker icons - TODO change me
        default_marker = QtCore.QUrl(
            "http://maps.gstatic.com/mapfiles/ridefinder-images/mm_20_gray.png"
        )
        marker = PositionMarker(name, coordinate, default_marker)
        self._markers.append(marker)
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self._markers = []
        self.endResetModel()


class NetworkModel(QtCore.QAbstractListModel):
    """Represents pairs between lines"""

    LineWidth, LineColor, Endpoints = range(QtCore.Qt.UserRole, QtCore.Qt.UserRole + 3)

    def __init__(self, parent=None):
        super(NetworkModel, self).__init__(parent)
        self._pairs: List[TransitionEndpoint] = []

    def rowCount(self, parent=QtCore.QModelIndex) -> int:
        return len(self._pairs)

    def roleNames(self):
        return {
            NetworkModel.LineWidth: b"lineWidth",
            NetworkModel.LineColor: b"lineColor",
            NetworkModel.Endpoints: b"endpoints",
        }

    def appendEdge(self, endpoint, intensity, min_intensity, max_intensity):
        self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount())
        width = 3
        t = (intensity - min_intensity) / (max_intensity - min_intensity)
        palette = cc.bgy
        color = QColor(palette[int((len(palette) - 1) * t)])
        color.setAlphaF(0.5)
        edge = TransitionEndpoint(endpoint, color, width)
        self._pairs.append(edge)
        self.endInsertRows()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if 0 <= index.row() < self.rowCount():
            if role == NetworkModel.LineWidth:
                return self._pairs[index.row()].width
            elif role == NetworkModel.LineColor:
                return self._pairs[index.row()].color
            elif role == NetworkModel.Endpoints:
                return list(self._pairs[index.row()].endpoint)
        return QtCore.QVariant()

    def clear(self):
        self.beginResetModel()
        self._pairs = []
        self.endResetModel()


class MapWidget(QtQuickWidgets.QQuickWidget):
    newConfig = pyqtSignal(object)
    setThreshold = pyqtSignal(float)

    def __init__(self, parent=None):
        super(MapWidget, self).__init__(
            parent, resizeMode=QtQuickWidgets.QQuickWidget.SizeRootObjectToView
        )
        self.marker_model = marker_model = MarkerModel(self)
        self.network_model = network_model = NetworkModel(self)

        self.rootContext().setContextProperty("marker_model", marker_model)
        self.rootContext().setContextProperty("network_model", network_model)
        qml_path = os.path.join(os.path.dirname(__file__), "maps.qml")
        self.setSource(QtCore.QUrl.fromLocalFile(qml_path))

        self.newConfig.connect(self._newConfigUpdate)
        self.setThreshold.connect(self._updateMap)

    def _newConfigUpdate(self, cfg: Config):
        self.cfg = cfg
        self.marker_model.clear()
        self.network_model.clear()
        self._updateMap()

    def _updateMap(self):
        cfg = self.cfg
        farms = cfg.farms
        locations = [farm.farm_location for farm in farms]

        northings, easthings = zip(*locations)
        lon_lats = convert_lonlat(northings, easthings)

        # Marker icons - TODO change me
        default_marker = QtCore.QUrl(
            "http://maps.gstatic.com/mapfiles/ridefinder-images/mm_20_gray.png"
        )

        min_intensity = np.min(cfg.interfarm_probs)
        max_intensity = np.max(cfg.interfarm_probs)

        for i, c in enumerate(zip(*lon_lats)):
            name = farms[i].name
            lon, lat = c
            coord = QtPositioning.QGeoCoordinate(lat, lon)
            self.marker_model.appendMarker(name, coord)

            for j, c2 in enumerate(zip(*lon_lats)):
                lon2, lat2 = c2
                coord2 = QtPositioning.QGeoCoordinate(lat2, lon2)
                intensity = cfg.interfarm_probs[i][j]
                self.network_model.appendEdge(
                    (coord, coord2), intensity, min_intensity, max_intensity
                )
