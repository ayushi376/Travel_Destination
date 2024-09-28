
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from pywebio.input import *
from pywebio.output import *
from IPython.display import Image
from IPython.display import display
import time
from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from flask_cors import CORS
from pywebio import start_server
from flask import Flask, render_template

app = Flask(__name__)
CORS(app)
# Root route ("/")




df = pd.read_csv('travel_destinations.csv')
cities = list(df['City'])
description = list(df['description'])

# Dictionary to map index to travel destination
index_destination_dict = {}
for i in range(len(df)):
    index_destination_dict[i] = df.loc[i]['City']
index_destination_dict

# Dictionary to map travel destination to index
destination_index_dict = {}
for i in range(len(df)):
    destination_index_dict[df.loc[i]['City']] = i
destination_index_dict

df1 = pd.read_csv('destinations_with_processed_text.csv')
corpus = df1['processed_text']


def previously_visited_destination(previously_visited_travel_destination):
    df2 = pd.read_csv('destinations_with_processed_text.csv')
    corpus = df2['processed_text']
    tv = TfidfVectorizer()
    X = tv.fit_transform(corpus)
    vectors = X.toarray()
    correlationMatrix = sigmoid_kernel(vectors, vectors)
    
    idx = destination_index_dict[str(previously_visited_travel_destination)]
    similarity_list = correlationMatrix[idx]
    lst = []
    for i in range(len(similarity_list)):
        lst.append((similarity_list[i], i))
    return sorted(lst, reverse = True)
    
def free_text_based_query():
    free_text = textarea('Enter a free text', rows = 3, placeholder = 'Write anything...\n\'snow winter nature trekking\' ... \'lake boating waterfall tiger\' ... \'market clothes nights history\' ... \'beach cruise camping boats ships\' ... \'temples hills altitude winter line\' ... \'lion safari forests camping nature\' ... ')
    number_of_recommendations = input("Enter the number of recommendations", type = NUMBER)
    
    #Remove Hyperlinks
    processed_query = re.sub(r"http\S+", ' ', str(free_text))   
    #processed_query = re.sub(r'https?:\/\/\S*', '', query, flags=re.MULTILINE)

    #Remove Punctuation Marks and Special Symbols
    processed_query = re.sub('[^a-zA-Z0-9]', ' ', processed_query)

    #Lowercase
    processed_query = processed_query.lower()

    #Create a list of strings using string.split() method
    processed_query = processed_query.split()
    
    wl = WordNetLemmatizer()
    # Prefer Lemmatization over Stemming
    #processed_query = [ps.stem(word) for word in processed_query if not word in stopwords.words('english')]
    processed_query = [wl.lemmatize(word, pos='v') for word in processed_query if not word in stopwords.words('english')]    
    processed_query = ' '.join(processed_query)
    # corpus.append(processed_query)
    # print(i, end = ' ')
    
    new_corpus = []
    for desc in corpus:
        new_corpus.append(desc)
    new_corpus.append(processed_query)
    #new_corpus
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    new_X = cv.fit_transform(new_corpus)
    new_vectors = new_X.toarray()
    
    from sklearn.metrics.pairwise import sigmoid_kernel
    new_correlationMatrix = sigmoid_kernel(new_vectors, new_vectors)
    #print(new_correlationMatrix)
    
    list_of_tuples = []
    for i in range(len(df)):
        list_of_tuples.append((new_correlationMatrix[-1][i], i))
    
    recommendation_list = []
    for element in sorted(list_of_tuples, reverse = True):
        recommendation_list.append(index_destination_dict[element[1]])
    final_rec = recommendation_list[:int(number_of_recommendations)]
    for rec in final_rec:
        put_html('<hr>')
        put_markdown("# *`%s`*" % rec)
        pic = 'DestinationPics/' + str(rec) + '.jpg'
        img = open(pic, 'rb').read()
        put_image(img, width='100%')
        

def select_recommendation_system():
    recommendation_system = select('Which type of recommendation system would you prefer?', ['Recommendation based on free text-based query', 'Recommendations similar to previously visited destination'])
    if(recommendation_system == 'Recommendation based on free text-based query'):
        free_text_based_query()
    if(recommendation_system == 'Recommendations similar to previously visited destination'):
        previously_visited_travel_destination = select('Select the previously visited travel destination', cities)
        recommendations_list = previously_visited_destination(previously_visited_travel_destination)
        number_of_recommendations = input("Enter the number of recommendations", type = NUMBER)
        for element in recommendations_list[:number_of_recommendations]:
            put_markdown("# *`%s`*" % index_destination_dict[element[1]])
            pic = 'DestinationPics/' + str(index_destination_dict[element[1]]) + '.jpg'
            img = open(pic, 'rb').read()
            put_image(img, width='100%')

def explore():
    put_markdown('## Please wait! Your request is being processed!')
    
    #Display Processbar
    put_processbar('bar');
    for i in range(1, 11):
        set_processbar('bar', i / 10)
        time.sleep(0.1)
        
    #Display the Travel Destination along with Description
    for i in range(len(df)):
        put_html('<hr>')
        put_markdown("# *`%s`*" % cities[i])
        pic = 'DestinationPics/' + str(cities[i]) + '.jpg'
        img = open(pic, 'rb').read()
        put_image(img, width='100%')
        #temp = description[i].replace('-', ' ')
        #put_text("     %s" % temp)
    put_markdown("# *In case of copyright issues, please drop an email to `ayushiagg2000@gmail.com`*")
    img = open(r'C:Users/ayush/OneDrive/Desktop/Travel Desti Recommendation System/DestinationPics/India_1.jpg', 'rb').read()
    put_image(img, width='1500px')

def choices():
    img = open(r'C:/Users/ayush/OneDrive/Desktop/Travel Desti Recommendation System/DestinationPics/DesiSafar Logo.jpg', 'rb').read()
    put_image(img, width='900px')
    put_markdown('# **Travel Destination Recommendation System**')
    answer = radio("Choose one", options=['Explore Incredible India!', 'Get Travel Recommendations'])
    if(answer == 'Explore Incredible India!'):
        explore()
    if(answer == 'Get Travel Recommendations'):
        put_text('\nLet\'s get started! ')
        # add recommendation system 
        select_recommendation_system()

if __name__ == '__main__':
    start_server(choices, port=5020)
#app.run()