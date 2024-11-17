from flask import Flask, render_template, request
import folium
import matplotlib.colors as mcolors
from folium import Icon
import heapq
import math
import random
import osmnx as ox
import networkx as nx
import torch
import torch.nn as nn
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

app = Flask(__name__)

def inicializar():
    G_lima = ox.load_graphml('../GrafosGuardados_Lima/lima_moderna2.graphml')
    return G_lima

G = inicializar()

class TrafficModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(TrafficModel, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model_path = os.path.abspath("./model/congestion_model_full.pth")
input_size = 4
hidden_sizes = [26] * 14
output_size = 1

traffic_model = TrafficModel(input_size, hidden_sizes, output_size)
traffic_model = torch.load(model_path, map_location=torch.device('cpu'))
traffic_model.eval()

scaler_X = joblib.load("./scalers/scaler_X.pkl")
scaler_y = joblib.load("./scalers/scaler_y.pkl")

calles_principales = [
    'avenida javier prado este', 'avenida javier prado oeste', 'avenida la marina',
    'avenida antonio josé de sucre', 'avenida faustino sanchez carrión', 'avenida del ejército',
    'ovalo josé quiñones', 'circuito de playas', 'avenida del parque norte',
    'avenida general salaverry', 'avenida la paz', 'avenida san borja norte',
    'avenida lima polo', 'avenida panamericana sur', 'avenida primavera',
    'avenida el derby', 'avenida angamos este', 'avenida angamos oeste',
    'avenida alfredo benavides'
]

def predict_congestion(day, time_minutes, lat, lon):
    feature_names = ['Dia', 'Hora', 'Latitud Central', 'Longitud Central']
    
    X = pd.DataFrame([[day, time_minutes, lat, lon]], 
                    columns=feature_names)
    X_norm = scaler_X.transform(X)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    y_pred = traffic_model(X_tensor).detach().numpy()
    y_denorm = scaler_y.inverse_transform(y_pred)[0][0]  

    return max(0, min(100, y_denorm)) 


def get_coordinates(location):
    try:
        geolocator = Nominatim(user_agent="UPCProject")
        location = geolocator.geocode(f"{location}, Lima, Peru", timeout=10)
        if location:
            return (location.latitude, location.longitude)
        return None
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Error getting coordinates: {e}")
        return None

def congestion_to_color(congestion):
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])
    return mcolors.to_hex(cmap(congestion / 100))


def calculate_little_time(distance_km, congestion):
    if congestion is None or congestion == -1:
        return (distance_km / 8.5) * 60
    if congestion <=33:
        congestion = 9 - pow(1.2,(congestion/8.5))
    elif congestion <= 66:
        congestion = 8.5 / pow(1.2,(congestion/8.5))
    elif congestion <= 100:
        congestion = 8.5 / pow(1.3,(congestion/8.5))
    return (distance_km / congestion ) * 60


def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def heuristic(node1, node2, G):
    x1, y1 = G.nodes[node1]['x'], G.nodes[node1]['y']
    x2, y2 = G.nodes[node2]['x'], G.nodes[node2]['y']
    return haversine(x1, y1, x2, y2)

def smastar(G, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    came_from = {}
    g_costs = {start: 0}
    f_costs = {start: heuristic(start, goal, G)}
    
    path = []
    while open_list:
        current = heapq.heappop(open_list)[1]
        
        if current == goal:
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], g_costs[goal]
        
        for neighbor in G.neighbors(current):
            distance = heuristic(current, neighbor, G)

            
            try:
                edge_data = G.get_edge_data(current, neighbor)
                if edge_data is None:
                    continue
                    
                if isinstance(edge_data, dict):
                    if 0 in edge_data:
                        edge_data = edge_data[0]
                
                road_name = ""
                if "name" in edge_data:
                    name_data = edge_data["name"]
                    if isinstance(name_data, list):
                        road_name = " ".join(name_data).lower()
                    elif isinstance(name_data, str):
                        road_name = name_data.lower()

                if any(road in road_name for road in calles_principales):
                    midpoint_lat = (G.nodes[current]["y"] + G.nodes[neighbor]["y"]) / 2
                    midpoint_lon = (G.nodes[current]["x"] + G.nodes[neighbor]["x"]) / 2
                    day, time_minutes = getDayAndTimeMin()
                    congestion = predict_congestion(day, time_minutes, midpoint_lat, midpoint_lon)
                    # estamos aumentando la distancia para que el sma* nos lleve por el camino con menos congestion digamosle
                    distance *= 1 + (congestion / 100) * 0.2
                else: distance *= 1 + (10/100)*0.2 

            except Exception as e:
                print(f"Error processing edge {current}->{neighbor}: {str(e)}")
                continue
            
            tentative_g_cost = g_costs[current] + distance
            
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                came_from[neighbor] = current
                g_costs[neighbor] = tentative_g_cost
                f_costs[neighbor] = tentative_g_cost + heuristic(neighbor, goal, G)
                heapq.heappush(open_list, (f_costs[neighbor], neighbor))
    
    return [], 0

def transformNodeToCoordenates(node):
    return (G.nodes[node]['y'], G.nodes[node]['x'])

def transformNodesToCoordenates(route):
    return [transformNodeToCoordenates(node) for node in route]


def getDayAndTimeMin():
    departure_time = request.form.get('departure_time', '')
    if departure_time:
        departure_datetime = datetime.strptime(departure_time, '%Y-%m-%dT%H:%M')
    else:
        departure_datetime = datetime.now()
        
    day = departure_datetime.weekday()
    time_minutes = departure_datetime.hour * 60 + departure_datetime.minute
    return day, time_minutes

@app.route('/', methods=['GET', 'POST'])
def mostrar_mapa():
    error_message = None
    map_html = None
    route_info = None
    form_data = {
        'start': request.form.get('start', ''),
        'end': request.form.get('end', ''),
        'departure_time': request.form.get('departure_time', '')
    }

    if request.method == 'POST':
        try:
            start_location = form_data['start']
            end_location = form_data['end']
            departure_time = form_data['departure_time']
            
            if not departure_time:
                departure_time = datetime.now().strftime('%Y-%m-%dT%H:%M')
                form_data['departure_time'] = departure_time
            
            if not start_location or not end_location:
                raise ValueError("Por favor ingrese tanto el origen como el destino.")
            
            start_coords = get_coordinates(start_location)
            end_coords = get_coordinates(end_location)
            
            if not start_coords or not end_coords:
                raise ValueError("No se pudieron encontrar una o ambas ubicaciones. Por favor, intente con direcciones más específicas en Lima.")
            
            start_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
            end_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])
            
            if start_node == end_node:
                raise ValueError("El origen y destino están demasiado cerca. Por favor elija ubicaciones más distantes.")
            
            route_nodes, distancia = smastar(G, start_node, end_node)
            
            if not route_nodes:
                raise ValueError("No se pudo encontrar una ruta entre estos puntos. Por favor intente con ubicaciones diferentes.")
            
            route_coordenates = transformNodesToCoordenates(route_nodes)
            
            total_distance = distancia
            total_time = 0
            
            m = folium.Map(
                location=route_coordenates[0],
                zoom_start=13,
                tiles='CartoDB.Positron',
                prefer_canvas=True
            )
            
            custom_style = '''
                <style>
                    .leaflet-tile-pane { opacity: 0.9; }
                </style>
            '''
            m.get_root().html.add_child(folium.Element(custom_style))
            
            for i in range(len(route_coordenates) - 1):
                try:
                    start = route_coordenates[i]
                    end = route_coordenates[i + 1]
                    
                    segment_distance = haversine(start[1], start[0], end[1], end[0])
                    
                    edge_data = G.get_edge_data(route_nodes[i], route_nodes[i + 1])
                    if edge_data is None:
                        continue
                    
                    if isinstance(edge_data, dict):
                        if 0 in edge_data:
                            edge_data = edge_data[0]

                    road_name = ""
                    if "name" in edge_data:
                        name_data = edge_data["name"]
                    if isinstance(name_data, list):
                        road_name = " ".join(name_data).lower()
                    elif isinstance(name_data, str):
                        road_name = name_data.lower()
                    
                    midpoint_lat = (start[0] + end[0]) / 2
                    midpoint_lon = (start[1] + end[1]) / 2
                    
                    if any(road in road_name for road in calles_principales):
                        day, time_minutes = getDayAndTimeMin()
                        congestion = predict_congestion(day, time_minutes, midpoint_lat, midpoint_lon)
                        color = congestion_to_color(congestion)
                    else:
                        congestion = -1
                        color = "gray"
                    
                    segment_time = calculate_little_time(segment_distance, congestion)
                    total_time += segment_time
                    
                    folium.PolyLine(
                        [start, end],
                        color=color,
                        weight=8,
                        opacity=1.0
                    ).add_to(m)
                except Exception as e:
                    print(f"Error processing segment {i}: {str(e)}")
                    continue
            
            departure_datetime = datetime.strptime(departure_time, '%Y-%m-%dT%H:%M')
            arrival_time = departure_datetime + timedelta(minutes=total_time)
            
            route_info = {
                'total_distance': round(total_distance, 2),
                'total_time': round(total_time),
                'departure_time': departure_datetime.strftime('%H:%M'),
                'arrival_time': arrival_time.strftime('%H:%M')
            }
            
            folium.Marker(
                location=route_coordenates[0],
                icon=Icon(icon='play', prefix='fa', color='green', icon_color='white'),
                popup=f"Inicio: {start_location}<br>Salida: {route_info['departure_time']}"
            ).add_to(m)
            
            folium.Marker(
                location=route_coordenates[-1],
                icon=Icon(icon='stop', prefix='fa', color='red', icon_color='white'),
                popup=f"Destino: {end_location}<br>Llegada: {route_info['arrival_time']}"
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
            
        except ValueError as ve:
            error_message = str(ve)
        except Exception as e:
            error_message = f"Error al calcular la ruta: {str(e)}"
            print(f"Detailed error: {e.__class__.__name__}: {str(e)}")
    
    return render_template('index.html', 
                        map_html=map_html, 
                        error_message=error_message, 
                        route_info=route_info, 
                        form_data=form_data)


if __name__ == '__main__':
    print("Feature names used in training:", scaler_X.feature_names_in_)
    app.run(debug=True)