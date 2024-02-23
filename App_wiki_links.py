import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia

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

    # Exclude the input movie from recommendations
    filtered_recommendations = []

    for movie in sorted_similar_movies:
        movie_index = movie[0]
        movie_title = movies_data.iloc[movie_index]['title']

        # Get Wikipedia link for the movie
        wikipedia_link = get_wikipedia_link(movie_title)

        # Display movie with a working Wikipedia link
        if wikipedia_link:
            filtered_recommendations.append((movie_index, movie_title, wikipedia_link))

    return filtered_recommendations

# Function to get Wikipedia link for a movie title
def get_wikipedia_link(movie_title):
    try:
        # Replace spaces with underscores in the movie title for a more accurate search
        search_title = movie_title.replace(' ', '_')

        # Search for the movie on Wikipedia
        search_results = wikipedia.search(search_title)
        if search_results:
            # Use the first result as the Wikipedia link
            first_option = search_results[0]
            wikipedia_page = wikipedia.page(first_option)
            return wikipedia_page.url
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.HTTPTimeoutError) as e:
        # Simply return None for movies with problematic links
        return None
    except Exception as e:
        # Log the unexpected error for debugging
        st.warning(f"Unexpected error while getting Wikipedia link for '{movie_title}': {e}")
        return None

# Streamlit app with enhanced styling
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

# Sidebar with image strip
st.sidebar.title("My personal favourite movies")
image_urls = [
    "https://upload.wikimedia.org/wikipedia/en/2/2d/Vanilla_Sky_poster.png",
    "https://upload.wikimedia.org/wikipedia/en/7/7b/Goodfellas.jpg",
    "https://m.media-amazon.com/images/M/MV5BMDJhMGRjN2QtNDUxYy00NGM3LThjNGQtMmZiZTRhNjM4YzUxL2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BNzQzOTk3OTAtNDQ0Zi00ZTVkLWI0MTEtMDllZjNkYzNjNTc4L2ltYWdlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_UX140_CR0,0,140,209_AL_.jpg",
    "https://upload.wikimedia.org/wikipedia/en/3/3b/Pulp_Fiction_%281994%29_poster.jpg",
    "https://m.media-amazon.com/images/M/MV5BYzMzNTJjYmMtZTkxNS00MjI4LWI3YmQtOTQ4MDZjZDJlZjQyXkEyXkFqcGdeQXVyNjc0NzQzNTM@._V1_UY209_CR0,0,140,209_AL.jpg",
    "https://m.media-amazon.com/images/M/MV5BMDJlZWZiODItMGE3NC00Yzg3LWFhYTYtZTI2YWNlNjExMDE4XkEyXkFqcGdeQXVyMjkwOTAyMDU@._V1_UY209_CR2,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BNDc2MzNkMjMtZDY5NC00NmQ0LWI1NjctZjRhNWIzZjc4MGRiXkEyXkFqcGdeQXVyMjkwOTAyMDU@._V1_UY209_CR0,0,140,209_AL.jpg",
]
st.sidebar.image(image_urls, width=100)

# Main content
st.title("Explore Movies and Get Recommendations")

# User input form
with st.form("movie_input_form"):
    st.subheader("Enter Your Favorite Movie:")
    movie_name = st.text_input("Movie Name", placeholder="E.g., The Shawshank Redemption")
    st.form_submit_button("Get Recommendations")

# Button to trigger recommendations
if movie_name:
    recommendations = get_movie_recommendations(movie_name)

    # Display recommendations
    st.subheader("Movies Suggested for You:")
    if recommendations:
        for i, (index, title_from_index, wikipedia_link) in enumerate(recommendations[:10], start=1):
            st.write(f"{i}. [{title_from_index}]({wikipedia_link})")
