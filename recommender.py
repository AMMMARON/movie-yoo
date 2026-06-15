import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_data(filepath):
    """Load the movie dataset."""
    return pd.read_csv(filepath)

def create_recommendations(df, title, cosine_sim):
    """Recommend movies based on cosine similarity."""
    # Get the index of the movie that matches the title
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    if title not in indices:
        return f"Movie '{title}' not found in the dataset."
    
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
    return df['title'].iloc[movie_indices]

def main():
    # Load dataset
    df = load_data('movies.csv')
    
    # Define TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    df['description'] = df['description'].fillna('')

    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(df['description'])

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Test the recommender
    movie_title = 'The Dark Knight'
    print(f"Recommendations for '{movie_title}':")
    print(create_recommendations(df, movie_title, cosine_sim))

if __name__ == "__main__":
    main()
