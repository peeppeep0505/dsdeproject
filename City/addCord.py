import pandas as pd
from geopy.geocoders import Nominatim

# Initialize geolocator
geolocator = Nominatim(user_agent="city_geocoder")

path = '2019'

# Read CSV data into a DataFrame
df = pd.read_csv('City/' + path + '_city_sum.csv')  # Replace with your CSV file path

# Function to get latitude and longitude
def geocode_city(city):
    location = geolocator.geocode(city, timeout=10000)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None  # In case geocoding fails

# Apply the geocoding function to each city in the DataFrame
df[['latitude', 'longitude']] = df['city'].apply(lambda city: pd.Series(geocode_city(city)))

# Display the updated DataFrame with latitudes and longitudes
print(df)

# Optionally, save the DataFrame with latitudes and longitudes back to CSV
df.to_csv('City/' + path + '_city_sum_coordinate.csv', index=False)
