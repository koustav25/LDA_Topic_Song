from flask import Flask, jsonify, request
from flask_cors import CORS
import pymongo
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import gensim
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.tokenize import word_tokenize

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# MongoDB connection URI
mongo_uri = "mongodb://localhost:27017/"

# Connect to MongoDB
client = pymongo.MongoClient(mongo_uri)

# Access a database
db = client['VisualisationOfSong']

# Access collections
collection = db['lyrics']
collectionInformation = db['information']
collectionGenres = db['genres']
collectionMetadata = db['metadata']

# Download NLTK resources (if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.add('let')
stop_words.add('nigga')
stop_words.add('aaa')

# Initialize global variable for word-to-document ID mappings
topic_to_doc_ids = {}

# Function to preprocess text
def preprocess(text):
    # Clean the text
    text = re.sub(r'\W', ' ', text.lower())
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stopwords
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) >= 3]
    
    return ' '.join(tokens)  # Return as a single string

@app.route('/genres', methods=['GET'])
def get_genres():
    # Fetch genre documents from the collection
    genres_documents = list(collectionGenres.find())
    
    # Initialize a dictionary to store genres and their song counts
    genre_song_count = {}

    # Process each genre document
    for genre_doc in genres_documents:
        genres = genre_doc['genres']
        song_id = genre_doc['id']  # Get the song ID from the document
        
        # Split genres by comma and process each genre
        split_genres = genres.split(',')
        for genre in split_genres:
            genre = genre.strip()  # Remove any leading/trailing whitespace
            if genre in genre_song_count:
                genre_song_count[genre].add(song_id)
            else:
                genre_song_count[genre] = {song_id}

    # Convert sets to counts and sort genres by the number of songs
    genre_song_count = {genre: len(song_ids) for genre, song_ids in genre_song_count.items()}
    sorted_genres = sorted(genre_song_count.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare the response with genres and their song counts
    response = [{'genre': genre, 'song_count': count} for genre, count in sorted_genres]
    # Extract only the genres
    genres_only = [item['genre'] for item in response]
    
    return jsonify(genres_only)

@app.route('/songs_by_genre', methods=['POST'])
def get_songs_by_genre():
    
    genres = request.json.get('genres', [])
    print("Received genres:", genres)  # Debug print
    if not genres:
        return jsonify([])

    # Query to fetch song details by genres
    genre_docs = list(collectionGenres.find({"genres": {"$in": genres}}))
    print("Genre documents found:", genre_docs)  # Debug print
    song_ids = [doc['id'] for doc in genre_docs]
    #print("Song IDs:", song_ids)  # Debug print
    
    if not song_ids:
        return jsonify([])

    # Fetch song popularity from the metadata collection
    popularity_docs = list(collectionMetadata.find({"id": {"$in": song_ids}}, {"id": 1, "popularity": 1,"spotify_id":1,"_id": 0}))
    #print("Popularity documents found:", popularity_docs)  # Debug print

    # Sort the songs by popularity in descending order
    sorted_popularity_docs = sorted(popularity_docs, key=lambda x: x['popularity'], reverse=True)
    
    # Get the first 5 most popular songs
    most_popular_song_ids = [doc['id'] for doc in sorted_popularity_docs[:5]]
    print("Most popular song IDs:", most_popular_song_ids)  # Debug print

    # Get the remaining song IDs
    remaining_song_ids = [doc['id'] for doc in sorted_popularity_docs[5:]]
    #print("Remaining song IDs:", remaining_song_ids)  # Debug print

    # Combine the IDs, ensuring the first 5 are the most popular
    combined_song_ids = most_popular_song_ids + remaining_song_ids
    print("Combined song IDs:", combined_song_ids)  # Debug print

    # Fetch song details from the lyrics collection
    songs = list(collectionInformation.find({"id": {"$in": combined_song_ids}}, {"artist": 1, "song": 1, "album_name": 1,"id": 1, "_id": 0}))
    #print("Songs found:", songs)  # Debug print

    # Sort the songs to match the combined_song_ids order
    songs_sorted = sorted(songs, key=lambda x: combined_song_ids.index(x['id']))
    print("Sorted songs:", songs_sorted)  # Debug print
    
    # Limit the sorted songs to 20 items
    limited_songs_sorted = songs_sorted[:20]
    print("Limited sorted songs:", limited_songs_sorted)  # Debug print
    
    # Append Spotify URLs to the songs
    for song in limited_songs_sorted:
        spotify_id = next((doc['spotify_id'] for doc in sorted_popularity_docs if doc['id'] == song['id']), None)
        song['spotify_url'] = f"https://open.spotify.com/track/{spotify_id}" if spotify_id else None

    return jsonify(limited_songs_sorted)

def get_keywords_for_topic(topic_name):
    # List of tuples containing sets of keywords and their associated topic names
    topic_keywords = [
        ({"night", "day", "waiting", "dark"}, "Songs on hope and despair"),
        ({"never", "thought", "wish", "left"}, "Songs on past grief"),
        ({"somebody", "young", "american", "jungle"}, "German Songs"),
        ({"god", "fight", "devil", "freedom"}, "Songs on war and death"),
        ({"head", "hate", "low", "breathe"}, "Songs on harsh realities of life"),
        ({"around", "talk", "ground", "walking"}, "Songs on easy going life"),
        ({"life", "soul", "world", "rise"}, "Songs on philosophy and hope"),
        ({"got", "work", "want", "back"}, "Songs which contain curse words"),
        ({"love", "need", "touch", "beautiful"}, "Song on romantic songs"),
        ({"let", "boy", "play", "pretty"}, "Song on charm and mischeviousness"),
        ({"know", "way", "thing", "something"}, "Song on cravings"),
        ({"home", "town", "friend", "child"}, "Song on families"),
        ({"like", "watch", "diamond", "wear"}, "Song on chilled life"),
        ({"make", "wait", "happy", "freak"}, "Hip hop songs"),
        ({"dance", "move", "music", "higher"}, "Rock songs"),
        ({"heart", "feel", "fall", "break"}, "Songs on heart break"),
        ({"back", "better", "high", "never"}, "Miscelleneous Songs"),
        ({"one", "world", "alive", "wake"}, "Song on dreams and liveliness"),
        ({"baby", "good", "little", "crazy"}, "Songs dedicated towards crush"),
        ({"song", "sing", "white", "red"}, "Happy songs")
    ]
    
    # Iterate through the list of tuples
    for keywords, name in topic_keywords:
        if name == topic_name:
            return keywords
    
    # Return "Topic not found" if the topic name is not in the list
    return "Topic not found"


def load_lda_model_results():
    global meaningful_topic_words
    with open('lda_results.pkl', 'rb') as file:
        dictionary, meaningful_topic_words, topic_to_doc_ids = pickle.load(file)
    # Print each row of meaningful_topic_words individually
    print("Meaningful Topic Words:")
    for topic_idx, words in enumerate(meaningful_topic_words):
        print(f"Topic {topic_idx + 1}: {words}")
    return dictionary, meaningful_topic_words, topic_to_doc_ids

# Function to initialize the topic-to-doc IDs mapping
def initialize_topic_to_doc_ids():
    global topic_to_doc_ids, meaningful_topic_words
    dictionary, meaningful_topic_words, topic_to_doc_ids = load_lda_model_results()
    

# Initialize global state
initialize_topic_to_doc_ids()

# Route to get word-to-document ID mapping
@app.route('/word_to_doc_ids', methods=['GET'])
def get_word_to_doc_ids():
    global topic_to_doc_ids
    sorted_topic_to_doc_ids = {k: topic_to_doc_ids[k] for k in sorted(topic_to_doc_ids)}
    return jsonify(sorted_topic_to_doc_ids)



@app.route('/songs_by_topic', methods=['POST'])
def get_songs_by_topic():
    global topic_to_doc_ids  # Access the global variable

    topic = request.json.get('topic', '')
    print(f"Received topic: {topic}")  # Debug print

    if not topic:
        print("No topic provided")  # Debug print
        return jsonify([])

    # Ensure that topic exists in the topic-to-doc-IDs mapping
    if topic not in topic_to_doc_ids:
        print(f"Topic '{topic}' not found in topic_to_doc_ids")  # Debug print
        return jsonify([])

    # Get document IDs associated with the topic
    doc_ids = topic_to_doc_ids[topic]
    print(f"Document IDs for topic '{topic}': {doc_ids}")  # Debug print

    if not doc_ids:
        print(f"No document IDs found for topic '{topic}'")  # Debug print
        return jsonify([])

    # Fetch song popularity from the metadata collection
    popularity_docs = list(collectionMetadata.find({"id": {"$in": doc_ids}}, {"id": 1, "popularity": 1,"spotify_id":1, "_id": 0}))

    # Sort the songs by popularity in descending order
    sorted_popularity_docs = sorted(popularity_docs, key=lambda x: x['popularity'], reverse=True)

    # Get the first 5 most popular songs
    most_popular_song_ids = [doc['id'] for doc in sorted_popularity_docs[:5]]
    print(f"Most popular song IDs: {most_popular_song_ids}")  # Debug print

    # Get the remaining song IDs
    remaining_song_ids = [doc['id'] for doc in sorted_popularity_docs[5:]]
    print(f"Remaining song IDs: {remaining_song_ids}")  # Debug print

    # Combine the IDs, ensuring the first 5 are the most popular
    combined_song_ids = most_popular_song_ids + remaining_song_ids
    print(f"Combined song IDs: {combined_song_ids}")  # Debug print

    # Fetch song details from the information collection using the combined song IDs
    songs = list(collectionInformation.find(
        {"id": {"$in": combined_song_ids}},
        {"artist": 1, "song": 1, "album_name": 1, "id": 1, "_id": 0}
    ))
    print(f"Fetched songs from collectionInformation: {songs}")  # Debug print

    # Sort the songs to match the combined_song_ids order
    songs_sorted = sorted(songs, key=lambda x: combined_song_ids.index(x['id']))
    print(f"Sorted songs: {songs_sorted}")  # Debug print

    # Limit the sorted songs to 20 items
    limited_songs_sorted = songs_sorted[:20]
    print(f"Limited sorted songs: {limited_songs_sorted}")  # Debug print

    # Prepare a list of songs to return and add Spotify URLs
    songs_response = []
    for song in limited_songs_sorted:
        spotify_id = next((doc['spotify_id'] for doc in sorted_popularity_docs if doc['id'] == song['id']), None)
        songs_response.append({
            "id": song['id'],
            "artist": song['artist'],
            "song": song['song'],
            "album_name": song.get('album_name', 'Unknown Album'),  # Provide a default value if album_name is not found
            "spotify_url": f"https://open.spotify.com/track/{spotify_id}" if spotify_id else None
        })

    # Return the list of songs in JSON format
    return jsonify(songs_response)


@app.route('/songs_by_genre_and_topic', methods=['POST'])
def get_songs_by_genre_and_topic():
    data = request.json
    genre = data.get('genre', '')
    input_topic = data.get('topic', '')

    # Debug print statements
    print(f"Received genre: {genre}")
    print(f"Received topic: {input_topic}")

    if not genre or not input_topic:
        print("Genre or topic not provided")
        return jsonify([])

    # Split the genres by comma and strip whitespace
    genres = [g.strip() for g in genre.split(',')]
    
    # Identify the topic keywords using the input topic
    #guessed_topic_keywords = get_keywords_for_topic(input_topic)
    #print(f"Guessed topic keywords: {guessed_topic_keywords}")

    #if guessed_topic_keywords == "Topic not found":
    #    print("Topic could not be identified")
    #    return jsonify([])

    # Find songs by any of the genres
    genre_docs = list(collectionGenres.find({"genres": {"$in": genres}}))
    
    if not genre_docs:
        print(f"No documents found for genres '{genres}'")
        return jsonify([])

    genre_song_ids_set = {doc.get('id') for doc in genre_docs if doc.get('id')}
    print(f" genres is {genre_song_ids_set}")

    # Check if the topic exists in the topic_to_doc_ids dictionary
    if input_topic in topic_to_doc_ids:
        # Filter songs by those in topic_to_doc_ids that also match the genres
        topic_song_ids_set = set(topic_to_doc_ids[input_topic])
        print(f" topic is {topic_to_doc_ids[input_topic]}")
        matching_song_ids = genre_song_ids_set.intersection(topic_song_ids_set)
        
        # Limit the number of matches to 20
        matching_song_ids = list(matching_song_ids)[:20]

        print(f"Matching song IDs from topic_to_doc_ids: {matching_song_ids}")
        
        if not matching_song_ids:
            print(f"No matching songs found for the given genre and topic")
            return jsonify([])

        # Fetch song details from the information collection
        songs = list(collectionInformation.find(
            {"id": {"$in": matching_song_ids}},
            {"artist": 1, "song": 1, "album_name": 1, "id": 1, "_id": 0}
        ))

        # Append Spotify URLs to the songs
        for song in songs:
            spotify_doc = collectionMetadata.find_one({"id": song['id']}, {"spotify_id": 1, "_id": 0})
            spotify_id = spotify_doc.get('spotify_id') if spotify_doc else None
            song['spotify_url'] = f"https://open.spotify.com/track/{spotify_id}" if spotify_id else None

        print(f"Fetched songs from collectionInformation: {songs}")

        return jsonify(songs)
    
    else:
        # If the topic is not found in topic_to_doc_ids, consider handling this case.
        # The logic for this part should be added if you need it to perform additional tasks or return an alternative response.
        print(f"Topic '{input_topic}' not found in topic_to_doc_ids")
        return jsonify([])  # Or handle it as needed


@app.route('/topic_by_song_ids', methods=['POST'])
def get_topic_by_song_ids():
    global topic_to_doc_ids  # Access the global variable

    song_ids = request.json.get('song_ids', [])
    print(f"Received song IDs: {song_ids}")  # Debug print

    if not song_ids:
        print("No song IDs provided")  # Debug print
        return jsonify([])

    # Initialize a dictionary to store the topic for each song ID
    song_to_topic = {}

    # Iterate through the topic_to_doc_ids to find the topic for each song ID
    for topic, doc_ids in topic_to_doc_ids.items():
        for song_id in song_ids:
            if song_id in doc_ids:
                song_to_topic[song_id] = topic
                #break  # Stop searching once the topic is found

    print(f"Song to Topic Mapping: {song_to_topic}")  # Debug print

    return jsonify(song_to_topic)



if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        client.close()


