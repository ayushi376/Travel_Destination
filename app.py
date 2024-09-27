import os
import pandas as pd
from pywebio.input import *
from pywebio.output import *
from flask import Flask
from pywebio.platform.flask import webio_view
import time

# Initialize Flask app
app = Flask(__name__)

# Load the CSV file containing city names and descriptions
# Update the path with the correct path to your CSV file
df = pd.read_csv(r'C:\Users\ayush\OneDrive\Desktop\Travel Desti Recommendation System\travel_destinations.csv')

# List of cities from the 'City' column
print(df.columns)
cities = list(df['City'])

# Function to explore the destinations
def explore():
    put_markdown('## Please wait! Your request is being processed!')

    # Display progress bar
    put_processbar('bar')
    for i in range(1, 11):
        set_processbar('bar', i / 10)
        time.sleep(0.1)
    
    # Display the travel destinations with descriptions and images
    for i in range(len(df)):
        put_html('<hr>')
        put_markdown(f"### *{cities[i]}*")
        
        # Display the description from the DataFrame
        put_text(df.loc[i, 'description'])
        
        # Simulating image display for now (replace with actual image paths later)
        put_markdown(f"![Image of {cities[i]}](https://via.placeholder.com/400?text={cities[i]})")

# Function to handle user choices
def choices():
    put_markdown('# **Travel Destination Recommendation System**')

    # Get user choice
    answer = radio("Choose one", options=['Explore Incredible India!', 'Get Travel Recommendations'])
    
    if answer == 'Explore Incredible India!':
        explore()
    elif answer == 'Get Travel Recommendations':
        put_text("\nRecommendation system coming soon!")

# Map the route to the function using PyWebIO
app.add_url_rule('/tdrs', 'webio_view', webio_view(choices), methods=['GET', 'POST', 'OPTIONS'])

# Run the Flask app
if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
