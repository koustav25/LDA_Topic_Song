import gensim
from gensim import corpora
import pickle
import re
import nltk
import pymongo
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
collectionLanguage = db['language']

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')  # Download WordNet for lemmatization

# Load stopwords and initialize lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

def preprocess(text):
    """
    Preprocesses the text by cleaning, tokenizing, lemmatizing, and removing stopwords.
    """
    # Clean the text
    text = re.sub(r'\W', ' ', text.lower())
    tokens = word_tokenize(text)

    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove stopwords and filter out tokens with less than 3 characters
    processed_tokens = [token for token in lemmatized_tokens if token not in stop_words and len(token) >= 3]
    
    return processed_tokens

def guess_topic_name(words):
    # List of tuples containing sets of keywords and their associated topic names
    topic_keywords = [
        ({"night", "day", "well", "young"}, "cheery songs"),
        ({"girl", "woman", "world", "pretty"}, "Songs on relationships"),
        ({"want", "enough", "take", "break"}, "song on ups and downs or day to day struggle"),
        ({"fire", "soul", "death", "pain"}, "Songs on war and violence"),
        ({"free", "higher", "shout", "shut"}, "Chill Songs"),
        ({"god", "lord", "holy", "jesus"}, "Spiritual Songs"),
        ({"time", "every", "make", "people"}, "Miscellaneous Songs"),
        ({"light", "night", "star", "shine"}, "Songs on hope"),
        ({"never", "heart", "time", "find"}, "Songs which has slight pain of loss"),
        ({"away", "around", "running", "bring"}, "Songs on positive vibes"),
        ({"baby", "please", "cry", "babe"}, "Sad love Songs on hope"),
        ({"mouth", "brain", "cut", "human"}, "Dysfunctional relationship"),
        ({"got", "high", "move", "ride"}, "Song on sad love song on families"),
        ({"love", "sweet", "true", "give"}, "Love songs"),
        ({"like", "back", "damn", "never"}, "Songs on curse Words"),
        ({"feel", "little", "touch", "alive"}, "Songs on warmth"),
        ({"home", "call", "phone", "alone"}, "Songs on missing love"),
        ({"rock", "roll", "shake", "mama"}, "Rock Songs"),
        ({"water", "wind", "river", "ocean"}, "Song on nature"),
        ({"know", "good", "cause", "well"}, "Song on Good and Bad")
    ]
    
    for keyword_set, topic_name in topic_keywords:
        if all(word in words for word in keyword_set):
            return topic_name
    
    return "unknown"  # Default topic name if no match is found

def train_and_initialize_lda():
    """
    Trains an LDA model and saves the results including dictionary, topic words, and topic-to-document mapping.
    """
    # Retrieve all documents from the lyrics collection
    all_documents = list(collection.find())

    # Filter documents by language = 'en'
    english_documents = []
    for doc in all_documents:
        song_id = doc['_id']
        language_entry = collectionLanguage.find_one({"id": song_id})
        if language_entry and language_entry.get('lang') == 'en':
            english_documents.append(doc)
            
    print(f"Total documents: {len(all_documents)}, English documents: {len(english_documents)}")
    
    if not english_documents:
        print("No English documents found. Exiting.")
        return
    
    # Collect all document contents for English songs
    texts = [doc['content'] for doc in english_documents]

    # Preprocess the texts
    preprocessed_texts = [preprocess(doc) for doc in texts]

    # Build dictionary and corpus for LDA
    dictionary = corpora.Dictionary(preprocessed_texts)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]

    # Train LDA model
    num_topics = 20
    lda_model = gensim.models.LdaMulticore(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=50,
        iterations=1000,
        random_state=42,
        workers=3
    )

    # Extract meaningful words from the trained LDA model
    def extract_meaningful_words(lda_model, num_topics=20):
        """
        Extracts the most significant words for each topic from the LDA model.
        """
        topic_words_list = []
        for i in range(num_topics):
            topic_terms = lda_model.show_topic(i, topn=20)
            topic_words = [term for term, _ in topic_terms]
            print(f'The topic words are {topic_words}')
            topic_words_list.append(topic_words)
        return topic_words_list

    meaningful_topic_words = extract_meaningful_words(lda_model)

    # Initialize topic to document IDs mapping
    topic_to_doc_ids = {}

    # Initialize lists to store document IDs and texts
    doc_ids = [str(doc['_id']) for doc in english_documents]

    # Map document IDs to topic words based on maximum word matches
    for i, doc in enumerate(preprocessed_texts):
        # Count matches for each topic
        topic_match_counts = {tuple(topic_words): 0 for topic_words in meaningful_topic_words}
        
        for topic_words in meaningful_topic_words:
            matched_words = set(doc) & set(topic_words)
            if matched_words:
                topic_match_counts[tuple(topic_words)] += len(matched_words)
        
        # Determine the topic with the maximum match count
        max_topic_words = max(topic_match_counts, key=topic_match_counts.get)
        
        if topic_match_counts[max_topic_words] > 0:  # Ensure there's at least one match
            topic_name = guess_topic_name(max_topic_words)
            if topic_name != "unknown":
                if topic_name not in topic_to_doc_ids:
                    topic_to_doc_ids[topic_name] = []
                topic_to_doc_ids[topic_name].append(doc_ids[i])

    # Log the results
    print(f'Dictionary: {dictionary}')
    print(f'Meaningful Topic Words: {meaningful_topic_words}')
    print(f'Topic to Document Mapping: {topic_to_doc_ids}')

    # Save the model, topic words, and the mapping to a file
    with open('lda_results.pkl', 'wb') as file:
        pickle.dump((dictionary, meaningful_topic_words, topic_to_doc_ids), file)


# Call the function if the script is run directly
if __name__ == "__main__":
    train_and_initialize_lda()
