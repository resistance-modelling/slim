import QtQuick 2.11
import QtQuick.Controls 2.2
import QtQuick.Window 2.0
import QtPositioning 5.11
import QtLocation 5.11

Rectangle {
    id:mapWidgetRect
    width: 640
    height: 480
    Plugin {
        id: osmPlugin
        name: "osm"
        PluginParameter {
            name: "osm.mapping.host"
            value: "https://cartodb-basemaps-c.global.ssl.fastly.net/light_all/"
        }
        PluginParameter {
            name: "osm.mapping.copyright"
            value: "Map tiles by Carto, under CC BY 3.0. Data by OpenStreetMap, under ODbL."
        }

        /*
        PluginParameter {
            name:"osm.mapping.providersrepository.disable"
            value:true
        }
        */
    }


    // Somewhere between Perth and Loch Lomond
    property var locationTC: QtPositioning.coordinate(56.33, -4.21)
    Map {
        id: mapWidgetMap
        anchors.fill: parent
        activeMapType: supportedMapTypes[supportedMapTypes.length-1]
        plugin: osmPlugin
        center: locationTC
        zoomLevel: 8
        MapItemView {
            id: locations
            model: marker_model
            delegate: MapQuickItem {
                coordinate: model.coords
                anchorPoint.x: image.width
                anchorPoint.y: image.height
                sourceItem: Rectangle {
                    Image { id: image; source: model.source }
                    Text {
                        anchors.left: image.right
                        text: model.name
                    }
                }
                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        ToolTip.timeout = 2000
                        ToolTip.visible = true
                        ToolTip.text = qsTr("Coordinates: %1, %2").arg(model.coords.latitude).arg(model.coords.longitude)
                    }
                }
            }
        }
        MapItemView {
            id: network
            model: network_model
            delegate: MapPolyline {
                line.width: model.lineWidth
                line.color: model.lineColor
                path: model.endpoints
            }
        }
    }
    Slider {
        id: intensityThreshold
        to: 0.05
        value: network_model.threshold
        x: mapWidgetMap.width - 50
        y: 30
        orientation: Qt.Vertical
        onMoved: {
            network_model.threshold = intensityThreshold.value
        }
    }
}