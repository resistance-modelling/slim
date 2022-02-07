"""
A Wrapper for Qt Location and Maps.
"""
import os
from typing import List

from PyQt5 import QtCore, QtQuickWidgets, QtPositioning
from .model import PositionMarker

__all__ = ["MapWidget"]

class MarkerModel(QtCore.QAbstractListModel):
    PositionRole, SourceRole = range(QtCore.Qt.UserRole, QtCore.Qt.UserRole + 2)

    def __init__(self, parent=None):
        super(MarkerModel, self).__init__(parent)
        self._markers: List[PositionMarker] = []

    def rowCount(self, parent=QtCore.QModelIndex):
        return len(self._markers)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        # Note: QVariant() maps to an Any
        if 0 <= index.row() < self.rowCount():
            if role == MarkerModel.SourceRole:
                return self._markers[index.row()].source
            elif role == MarkerModel.PositionRole:
                return self._markers[index.row()].position
        return QtCore.QVariant()

    def roleNames(self):
        return {
            MarkerModel.PositionRole: b"position_marker",
            MarkerModel.SourceRole: b"source_marker",
        }

    def appendMarker(self, marker: PositionMarker):
        self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount())
        self._markers.append(marker)
        self.endInsertRows()


class MapWidget(QtQuickWidgets.QQuickWidget):
    def __init__(self, parent=None):
        super(MapWidget, self).__init__(
            parent, resizeMode=QtQuickWidgets.QQuickWidget.SizeRootObjectToView
        )
        model = MarkerModel(self)
        self.rootContext().setContextProperty("markermodel", model)
        qml_path = os.path.join(os.path.dirname(__file__), "maps.qml")
        self.setSource(QtCore.QUrl.fromLocalFile(qml_path))

        positions = [(56.33278899182579, -4.211768367584389)]

        # Marker icons - TODO change me
        urls = ["http://maps.gstatic.com/mapfiles/ridefinder-images/mm_20_gray.png", 
                "http://maps.gstatic.com/mapfiles/ridefinder-images/mm_20_red.png"]

        for c, u in zip(positions, urls):
            coord = QtPositioning.QGeoCoordinate(*c)
            source = QtCore.QUrl(u)
            model.appendMarker(PositionMarker(coord, source))
