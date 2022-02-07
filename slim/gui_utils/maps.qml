import QtQuick 2.11
import QtPositioning 5.11
import QtLocation 5.11

Rectangle {
    id:mapWidgetRect
    width: 640 /* How to make flexible? */
    height: 480
    Plugin {
        id: osmPlugin
        name: "osm"
    }
    // Somewhere between Perth and Loch Lomond
    property variant locationTC: QtPositioning.coordinate(56.33, -4.21)
    Map {
        id: mapWidgetMap
        anchors.fill: parent
        plugin: osmPlugin
        center: locationTC
        zoomLevel: 8
        MapItemView {
            model: markermodel // a list model-like class that instantiates a set of markers
            delegate: MapQuickItem { // a marker is basically a coordinate with an icon
                coordinate: model.position_marker
                anchorPoint.x: image.width
                anchorPoint.y: image.height
                sourceItem:
                    Image { id: image; source: model.source_marker }
            }
        }
    }
}