import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
movies_data = pd.read_csv(r'C:\Users\Ashut\OneDrive\Desktop\Movie_Recommendation_System\movies.csv')

# Preprocess the data
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + \
                    movies_data['cast'] + ' ' + movies_data['director']

# Convert text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Get similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

# Function to get movie recommendations
def get_movie_recommendations(movie_name):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        st.warning(f"No close match found for '{movie_name}'. Please try a different movie.")
        return []

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    return sorted_similar_movies

# Streamlit app with some styling
st.title("Movie Recommendation System")

# Image strip
image_urls = [
    "https://upload.wikimedia.org/wikipedia/en/2/2d/Vanilla_Sky_poster.png",
    "https://upload.wikimedia.org/wikipedia/en/7/7b/Goodfellas.jpg",
    "https://m.media-amazon.com/images/M/MV5BMDJhMGRjN2QtNDUxYy00NGM3LThjNGQtMmZiZTRhNjM4YzUxL2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BNzQzOTk3OTAtNDQ0Zi00ZTVkLWI0MTEtMDllZjNkYzNjNTc4L2ltYWdlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_UX140_CR0,0,140,209_AL_.jpg",

    "https://m.media-amazon.com/images/M/MV5BNzM3NDFhYTAtYmU5Mi00NGRmLTljYjgtMDkyODQ4MjNkMGY2XkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_UX140_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BYzMzNTJjYmMtZTkxNS00MjI4LWI3YmQtOTQ4MDZjZDJlZjQyXkEyXkFqcGdeQXVyNjc0NzQzNTM@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BMDJlZWZiODItMGE3NC00Yzg3LWFhYTYtZTI2YWNlNjExMDE4XkEyXkFqcGdeQXVyMTA0MjU0Ng@@._V1_UY209_CR2,0,140,209_AL_.jpg",
]

st.image(image_urls, width=100)

# User input form
with st.form("movie_input_form"):
    movie_name = st.text_input("Enter your favorite movie name:")
    st.form_submit_button("Get Recommendations")

# Button to trigger recommendations
if movie_name:
    recommendations = get_movie_recommendations(movie_name)

    # Display recommendations
    st.subheader("Movies suggested for you:")
    if recommendations:
        # Filter out the entered movie from recommendations
        filtered_recommendations = [movie for movie in recommendations if movies_data.iloc[movie[0]]['title'] != movie_name]
        
        for i, movie in enumerate(filtered_recommendations[:10], start=1):
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            st.write(f"{i}. {title_from_index}")
