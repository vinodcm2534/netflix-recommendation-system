import streamlit as st
import joblib
import pandas as pd

# Load pre-trained models
tfidf = joblib.load(r'D:\Netflix lets recommend\model\TF-IDF-Vectorizer.pkl')
cosine_sim = joblib.load(r'D:\Netflix lets recommend\model\Cosine-Similarity.pkl')

# Load the movie dataset
df = pd.read_csv(r'D:\Netflix lets recommend\data\mymoviedb.csv',lineterminator='\n')

# Generate indices for quick lookup
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

# Get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return "Movie not found. Please check the spelling."
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    return df['Title'].iloc[movie_indices]

# Streamlit UI setup
st.title('Movie Recommendation System')
st.subheader('Enter a movie title to get similar movie recommendations')

movie_title = st.text_input('Movie Title')

if st.button('Get Recommendations'):
    if movie_title:
        recommendations = get_recommendations(movie_title)
        
        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.write("Here are some similar movies:")
            for i, movie in enumerate(recommendations):
                st.write(f"{i+1}. {movie}")
    else:
        st.write("Please enter a movie title.")
