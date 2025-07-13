import streamlit as st
import time
import multiprocessing
from test import start_aegis
import folium
from streamlit_folium import st_folium
import requests
from datetime import datetime

@st.cache_data
def get_route(start, end):
    # OSRM API endpoint for walking directions
    url = f"http://router.project-osrm.org/route/v1/walking/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
    try:
        r = requests.get(url)
        r.raise_for_status()
        route = r.json()['routes'][0]['geometry']['coordinates']
        # OSRM returns lon, lat so we need to swap them for folium
        return [[point[1], point[0]] for point in route]
    except (requests.RequestException, KeyError, IndexError):
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("Aegis")

    # Initialize session state
    if 'shared_state' not in st.session_state:
        print("Initializing State")
        manager = multiprocessing.Manager()
        st.session_state.shared_state = manager.dict()
        st.session_state.shared_state['status'] = 'idle'
        st.session_state.shared_state['threat_level'] = "SAFE"
        st.session_state.shared_state['messages'] = manager.list()
        st.session_state.shared_state['current_location'] = None
        st.session_state.shared_state['historic_locations'] = manager.list()
        st.session_state.shared_state['available_tools'] = manager.list()
        st.session_state.shared_state['current_location_updated'] = False
        st.session_state.aegis_process = None
        st.session_state.home_location = None
        st.session_state.current_location = None

    # --- Sidebar --- 
    with st.sidebar:
        st.header("Controls")
        if st.button("Turn on Aegis", disabled=st.session_state.get('aegis_process') is not None):
            if st.session_state.get('aegis_process') is None:
                print("Starting Aegis")
                st.session_state.aegis_process = multiprocessing.Process(
                    target=start_aegis, args=(st.session_state.shared_state,)
                )
                st.session_state.aegis_process.start()
                st.rerun()

        st.header("Current Threat Level")
        threat_level_value = st.session_state.shared_state.get('threat_level', 'N/A').upper()
        if threat_level_value == "SAFE":
            st.success(f"**{threat_level_value}**")
        elif threat_level_value == "WARNING":
            st.warning(f"**{threat_level_value}**")
        else: 
            st.error(f"**{threat_level_value}**")

        st.header("Model Status")
        is_talking = st.session_state.shared_state.get('status') == 'talking'
        if is_talking:
            st.info("Aegis is Talking...")
        else:
            st.success("Aegis is listening.")

        if 'available_tools' in st.session_state.shared_state and st.session_state.shared_state['available_tools']:
            with st.expander("Available Tools"):
                for tool in st.session_state.shared_state['available_tools']:
                    st.info(tool)

    # --- Main Content ---
    col1, col2 = st.columns([2, 1])

    with col1:
        # Map Display
        st.header("Click Map To Mock Location (Laptop)")
        location_to_place = st.radio("Select location to place on map:", ('Home', 'Current'), horizontal=True)
        m = folium.Map(location=[37.7749, -122.4194], zoom_start=13)

        if st.session_state.home_location:
            folium.Marker(location=st.session_state.home_location, popup="Home", icon=folium.Icon(color="red", icon="home")).add_to(m)
        if st.session_state.current_location:
            folium.Marker(location=st.session_state.current_location, popup="Current Location", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
        if st.session_state.home_location and st.session_state.current_location:
            route = get_route(st.session_state.home_location, st.session_state.current_location)
            if route:
                folium.PolyLine(route, color="black", weight=2.5, opacity=1).add_to(m)

        map_data = st_folium(m, width=700, height=500, key="folium_map")

        if map_data and map_data["last_clicked"]:
            lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
            if location_to_place == 'Home':
                st.session_state.home_location = (lat, lon)
                st.session_state.shared_state['home_location'] = (lat, lon)
            else:
                st.session_state.current_location = (lat, lon)
                st.session_state.shared_state['current_location'] = (lat, lon)
                historic_list = st.session_state.shared_state.get('historic_locations', [])
                historic_list.append({"location": (lat, lon), "timestamp": datetime.now().isoformat()})
                st.session_state.shared_state['historic_locations'] = historic_list
                st.session_state.shared_state['current_location_updated'] = True
            st.rerun()

    with col2:
        st.header("Conversation")
        chat_container = st.container(height=600)
        with chat_container:
            if 'messages' in st.session_state.shared_state:
                for message in st.session_state.shared_state['messages']:
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    avatar = {"user": "🧑‍💻", "model": "🤖", "system": "⚙️"}.get(role)

                    if role == "system":
                        st.info(f"⚙️ {content}")
                    else:
                        # Swap roles for alignment: user->left, model->right
                        alignment_role = "assistant" if role == "user" else "user"
                        with st.chat_message(alignment_role, avatar=avatar):
                            st.markdown(content)

    # Auto-refresh the app to get status updates
    if st.session_state.get('aegis_process') is not None:
        time.sleep(0.5)
        st.rerun()

if __name__ == "__main__":
    main()
