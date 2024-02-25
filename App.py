import streamlit as st
import pandas as pd
import difflib
from nltk.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies_data = pd.read_csv(r'C:\Users\Ashut\OneDrive\Desktop\Movie_Recommendation_System\movies.csv')


selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + \
                    movies_data['cast'] + ' ' + movies_data['director']


vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)


similarity = cosine_similarity(feature_vectors)


def get_movie_recommendations(movie_name):
    list_of_all_titles = movies_data['title'].tolist()
    
    
    movie_name_lower = movie_name.lower()
    list_of_all_titles_lower = [title.lower() for title in list_of_all_titles]

    find_close_match = difflib.get_close_matches(movie_name_lower, list_of_all_titles_lower, n=1, cutoff=0.6)

    if not find_close_match:
        st.warning(f"No close match found for '{movie_name}'. Please try a different movie.")
        return []

    close_match_lower = find_close_match[0]
    
    
    close_match_index = list_of_all_titles_lower.index(close_match_lower)
    close_match = list_of_all_titles[close_match_index]
    
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    
    filtered_recommendations = [movie for movie in sorted_similar_movies if movies_data.iloc[movie[0]]['title'].lower() != movie_name_lower]

    return filtered_recommendations



st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")


st.sidebar.title("Some movies to choose from")
image_urls = [
    "https://m.media-amazon.com/images/M/MV5BYjZlYmJjYWYtZDM0NS00YmZlLWIyMTAtMDY5ZTNjZTgwMDhjXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BMjc4OTc0ODgwNV5BMl5BanBnXkFtZTcwNjM1ODE0MQ@@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BY2NkZjEzMDgtN2RjYy00YzM1LWI4ZmQtMjIwYjFjNmI3ZGEwXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_UX140_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BMDJhMGRjN2QtNDUxYy00NGM3LThjNGQtMmZiZTRhNjM4YzUxL2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BY2JiYTNmZTctYTQ1OC00YjU4LWEwMjYtZjkwY2Y5MDI0OTU3XkEyXkFqcGdeQXVyNTI4MzE4MDU@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BNzM3NDFhYTAtYmU5Mi00NGRmLTljYjgtMDkyODQ4MjNkMGY2XkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_UX140_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BYzMzNTJjYmMtZTkxNS00MjI4LWI3YmQtOTQ4MDZjZDJlZjQyXkEyXkFqcGdeQXVyNjc0NzQzNTM@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BMDJlZWZiODItMGE3NC00Yzg3LWFhYTYtZTI2YWNlNjExMDE4XkEyXkFqcGdeQXVyMTA0MjU0Ng@@._V1_UY209_CR2,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BNDc2MzNkMjMtZDY5NC00NmQ0LWI1NjctZjRhNWIzZjc4MGRiXkEyXkFqcGdeQXVyMjkwOTAyMDU@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BZWFlZjE5OTYtNWY0ZC00MzgzLTg5MjUtYTFkZjk2NjJkYjM0XkEyXkFqcGdeQXVyNTAyODkwOQ@@._V1_UX140_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BNzQzMzJhZTEtOWM4NS00MTdhLTg0YjgtMjM4MDRkZjUwZDBlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BY2IzNGNiODgtOWYzOS00OTI0LTgxZTUtOTA5OTQ5YmI3NGUzXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_UX140_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BNGNjNjU1YmEtZGM5MC00ODgzLWEyY2MtZmZmNTlhOGU4OWJjXkEyXkFqcGdeQXVyNTAyODkwOQ@@._V1_UY209_CR9,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BODZiMzAxNTctZjdiZC00OGY5LTg2NDAtNWJhNmQwZTcyMWQ2XkEyXkFqcGdeQXVyMjUzOTY1NTc@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BODcxMzY3ODY1NF5BMl5BanBnXkFtZTgwNzg1NDY4MTE@._V1_UX140_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BMWRmYjY1NTUtNjNlMC00MDFjLTk0MTYtZWVlMTFhMjllYjUzXkEyXkFqcGdeQXVyMTUzMDUzNTI3._V1_UY209_CR1,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BNjJlYmNkZGItM2NhYy00MjlmLTk5NmQtNjg1NmM2ODU4OTMwXkEyXkFqcGdeQXVyMjUzOTY1NTc@._V1_UX140_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BYTYxNGMyZTYtMjE3MS00MzNjLWFjNmYtMDk3N2FmM2JiM2M1XkEyXkFqcGdeQXVyNjY5NDU4NzI@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BMTMxNTMwODM0NF5BMl5BanBnXkFtZTcwODAyMTk2Mw@@._V1_UY209_CR0,0,140,209_AL_.jpg",
    "https://m.media-amazon.com/images/M/MV5BMTcwNTE4MTUxMl5BMl5BanBnXkFtZTcwMDIyODM4OA@@._V1_UY209_CR0,0,140,209_AL_.jpg",
]
st.sidebar.image(image_urls, width=100)


st.title("Explore Movies and Get Recommendations🍿")

with st.form("movie_input_form"):
    st.subheader("Enter Your Favorite Movie:")
    movie_name = st.text_input("Movie Name", placeholder="E.g., The Shawshank Redemption")
    st.form_submit_button("Get Recommendations")


if movie_name:
    recommendations = get_movie_recommendations(movie_name)

    
    st.subheader("Movies Suggested for You:")
    if recommendations:
        for i, movie in enumerate(recommendations[:10], start=1):
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            st.write(f"{i}. {title_from_index}")
