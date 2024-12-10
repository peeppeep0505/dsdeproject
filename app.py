import streamlit as st
import pandas as pd
import pydeck as pdk
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
        page_title="DSDE Project Woohoo~",
        layout="wide"
    )

# Sidebar
st.sidebar.title("Sidebar Here")

# Map type selection
map_layer_type = st.sidebar.radio('Map Type', ["ScatterplotLayer", "HeatmapLayer"])

# Map style selection
map_style = st.sidebar.selectbox(
    'Select Base Map Style',
    options=['Dark', 'Light', 'Road', 'Satellite'],
    index=1
)

# Define map style dictionary
MAP_STYLES = {
    'Dark': 'mapbox://styles/mapbox/dark-v10',
    'Light': 'mapbox://styles/mapbox/light-v10',
    'Road': 'mapbox://styles/mapbox/streets-v11',
    'Satellite': 'mapbox://styles/mapbox/satellite-v9'
}

# DBSCAN clustering parameters
st.sidebar.header('DBSCAN Parameters')
eps_degrees = st.sidebar.slider('eps (degree)', min_value=1, max_value=10, value=3)                       
min_samples = st.sidebar.slider('min_samples', min_value=2, max_value=10, value=5)
num_top_clusters = st.sidebar.slider('Number of Top Clusters to Show', 1, 100, 40)

# Streamlit app title
st.title("DSDE Project Woohoo~")

# Load the data from the CSV file
df = pd.read_csv('City/2018_city_sum_coordinate.csv')  # Replace with your file path
df['sum'] = df['sum'].apply(lambda x: x * 25)

# Display the data as a table
if 'show_data' not in st.session_state:
    st.session_state.show_data = False

if st.button("Show/Hide Data"):
    st.session_state.show_data = not st.session_state.show_data

if st.session_state.show_data:
    st.subheader("DF")
    st.write(df)

try:
    # Perform DBSCAN clustering
    df.dropna(inplace=True)
    coords = df[['latitude', 'longitude']] 
    db = DBSCAN(eps=eps_degrees, min_samples=min_samples).fit(coords)
    
    # Add cluster labels to dataframe
    df['cluster'] = db.labels_
    
    # Analyze clusters
    clusters_count = df['cluster'].value_counts()
    clusters_count = clusters_count[clusters_count.index != -1]  # Exclude noise points
    top_clusters = clusters_count.head(num_top_clusters)
    
    # Generate colors for clusters
    unique_clusters = df[df['cluster'].isin(top_clusters.index)]['cluster'].unique()
    colormap = plt.get_cmap('hsv')
    cluster_colors = {cluster: [int(x*255) for x in colormap(i/len(unique_clusters))[:3]] + [160] 
                     for i, cluster in enumerate(unique_clusters)}
    
    # Create visualization dataframe
    viz_data = df[df['cluster'].isin(top_clusters.index)].copy()
    viz_data['color'] = viz_data['cluster'].map(cluster_colors)

except Exception as e:
    st.error(f"Error in clustering analysis: {e}")

# Create map
def create_map(df):
    if map_layer_type == "ScatterplotLayer":
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[longitude, latitude]',
            get_radius='sum',
            get_fill_color='[0, 0, 255, 140]',
            radius_min_pixels=3,
            radius_max_pixels=80,
            pickable=True,
            auto_highlight=True
        )
    elif map_layer_type == "HeatmapLayer":
        layer = pdk.Layer(
            "HeatmapLayer",
            data=df,
            get_position='[longitude, latitude]',
            get_weight='sum',  # Use the 'sum' column to determine the weight of each point
            opacity=0.6,  # Adjust opacity for better visibility
            radius_pixels=50,  # Adjust the radius of the heatmap points
            threshold=0.03  # Controls the amount of data needed to show heat
        )

    view_state = pdk.ViewState(
        latitude=df['latitude'].mean(),
        longitude=df['longitude'].mean(),
        zoom=2
    )

    deck = pdk.Deck(
        initial_view_state=view_state,
        map_style=MAP_STYLES[map_style],
        layers=[layer]
    )

    return deck

# Generate the map
deck_map = create_map(df)

# Display the map in Streamlit
st.subheader("Map of Cities")
st.pydeck_chart(deck_map)

# Create DBSCAN map
def create_map_dbscan(df):
    if map_layer_type == "ScatterplotLayer":
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=viz_data,
            get_position='[longitude, latitude]',
            get_radius='sum',
            get_fill_color='color',
            radius_min_pixels=3,
            radius_max_pixels=80,
            pickable=True,
            auto_highlight=True
        )
    elif map_layer_type == "HeatmapLayer":
        layer = pdk.Layer(
            "HeatmapLayer",
            data=viz_data,
            get_position='[longitude, latitude]',
            get_weight='sum',  # Use the 'sum' column to determine the weight of each point
            opacity=0.6,  # Adjust opacity for better visibility
            radius_pixels=50,  # Adjust the radius of the heatmap points
            threshold=0.03  # Controls the amount of data needed to show heat
        )

    view_state = pdk.ViewState(
        latitude=df['latitude'].mean(),
        longitude=df['longitude'].mean(),
        zoom=2
    )

    deck = pdk.Deck(
        initial_view_state=view_state,
        map_style=MAP_STYLES[map_style],
        layers=[layer]
    )

    return deck

# Generate the map
deck_map_dbscan = create_map_dbscan(df)

# Display the map in Streamlit
st.subheader("Map of Cities (Clustering")
st.pydeck_chart(deck_map_dbscan)
