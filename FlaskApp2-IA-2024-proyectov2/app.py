from flask import Flask, render_template
import folium
import matplotlib.colors as mcolors
from folium import Icon
import heapq
import csv
import math
import time
import random
import requests
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
from matplotlib.lines import Line2D
from collections import Counter

app = Flask(__name__)

def inicializar():
    G_lima = ox.load_graphml('../lima_moderna2.graphml')
    return G_lima

def defineCoordsCentrals(uniqueEdges):
    coordinates = []

    for edge in G.edges:
        node1, node2 = edge[0], edge[1]
        uniqueEdges.append((node1, node2))
        x1, y1 = transformNodeToCoordenates(node1)
        #Longitud y Latitud
        x2, y2 = transformNodeToCoordenates(node2) 
        coordinates.append(((y1+y2)/2, (x1+x2)/2))  # Coord Central
    return coordinates




def defineDayHour():
    now = datetime.now()
    dia = now.strftime("%A")  # Día de la semana
    hora = now.strftime("%H:%M")  # Hora en formato HH:MM
    return dia, hora

def congestion_to_color(congestion):
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])
    return mcolors.to_hex(cmap(congestion / 100))

def get_coordinates(location):
    geolocator = Nominatim(user_agent="UPCProject")
    location = geolocator.geocode(location)
    return (location.latitude, location.longitude)

def haversine(lon1,lat1, lon2,lat2):
    R = 6371  # ratio of earth in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def heuristic(node1, node2, G):
    x1, y1 = G.nodes[node1]['x'], G.nodes[node1]['y']
    x2, y2 = G.nodes[node2]['x'], G.nodes[node2]['y']
    return haversine(x1,y1, x2,y2)

def astar(G, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    came_from = {}
    g_costs = {start: 0}
    f_costs = {start: heuristic(start, goal, G)}
    unique = []
    
    path = []
    while open_list:
        # Obtener el nodo con menor costo f (g + h)
        current = heapq.heappop(open_list)[1]
        
        if current == goal:
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
        
        for neighbor in G.neighbors(current):

            
            tentative_g_cost = g_costs[current] + heuristic(current, neighbor, G)
            
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                came_from[neighbor] = current
                g_costs[neighbor] = tentative_g_cost
                f_costs[neighbor] = tentative_g_cost
                heapq.heappush(open_list, (f_costs[neighbor], neighbor))
        
    return path[::-1], g_costs[goal], unique

def transformNodeToCoordenates(node):
    return (G.nodes[node]['y'], G.nodes[node]['x'])

def transformNodesToCoordenates(route):
    coords = []
    for node in route:
        coords.append(transformNodeToCoordenates(node))
    return coords

def congestion_aleatoria():
    return random.randint(0, 100)  # Congestión entre 0% y 100%

@app.route('/')
def mostrar_mapa():
    
    upcSanIsidro_coords = get_coordinates("UPC Monterrico, Lima, Peru")
    upcSanMiguel_coords = get_coordinates("UPC San Miguel, Lima, Peru")
    
    upcSanIsidro_node = ox.distance.nearest_nodes(G, upcSanIsidro_coords[1], upcSanIsidro_coords[0])
    upcSanMiguel_node = ox.distance.nearest_nodes(G, upcSanMiguel_coords[1], upcSanMiguel_coords[0])
    
    route_nodes, _, _ = astar(G, upcSanIsidro_node, upcSanMiguel_node)
    route_coordenates = transformNodesToCoordenates(route_nodes)


    getCongestion_Edge = {
        (0, 1): 20,  
        (1, 2): 60,  
    }

    center_lat, center_lon = route_coordenates[0]
    m = folium.Map(
        location=(center_lat, center_lon),
        zoom_start = 17,  
        tiles='CartoDB.Positron',  
        prefer_canvas=True
    )

    custom_style = '''
        <style>
            .leaflet-tile-pane {
                opacity: 0.9;  /* Hacer el fondo más tenue */
            }
        </style>
    '''
    m.get_root().html.add_child(folium.Element(custom_style))

    for i in range(len(route_coordenates) - 1):
        start_coords = route_coordenates[i]
        end_coords = route_coordenates[i + 1]

        
        congestion_level = getCongestion_Edge.get((i, i + 1), 0)
        congestion_level = congestion_aleatoria()
        color = congestion_to_color(congestion_level)  

        folium.PolyLine(
            [start_coords, end_coords],
            color=color,
            weight=8,  
            opacity=1.0,
            popup=f'Congestión: {congestion_level}%'
        ).add_to(m)

        if i == 0:  
            folium.Marker(
                location=start_coords,
                icon=Icon(icon='cloud', prefix='fa', color='blue', icon_color='white'),
                popup="Punto de inicio"
            ).add_to(m)
        
        elif i == len(route_coordenates) - 2:
            folium.Marker(
                location=end_coords,
                icon=Icon(icon='cloud', prefix='fa', color='red', icon_color='white'),
                popup="Punto final"
            ).add_to(m)

    legend_html = '''
        <div style="position: fixed; 
                    bottom: 20px; right: 20px;
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    z-index: 1000;
                    font-size: 12px;">
            <div style="margin-bottom: 5px;">Nivel de congestión:</div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <span style="background-color: #2ecc71; width: 20px; height: 4px; display: inline-block; margin-right: 5px;"></span>
                <span>Fluido</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <span style="background-color: #f1c40f; width: 20px; height: 4px; display: inline-block; margin-right: 5px;"></span>
                <span>Moderado</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="background-color: #e74c3c; width: 20px; height: 4px; display: inline-block; margin-right: 5px;"></span>
                <span>Congestionado</span>
            </div>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    map_html = m._repr_html_()
    return render_template('index.html', map_html=map_html)

if __name__ == '__main__':
    G = inicializar()
    uniqueEdges = []
    coordinatesCentrals = defineCoordsCentrals(uniqueEdges)
    getCongestion = {}
    getCongestion_Edge = {}
    app.run(debug=True)
