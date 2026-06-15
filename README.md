# Movie Yoo

A simple Python-based movie recommendation system that uses Content-Based Filtering.

## How it Works

The system utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** to vectorize movie descriptions and **Cosine Similarity** to calculate the similarity between movies. When a user provides a movie title, the system finds the most similar movies based on their plot descriptions.

## Dataset

The project includes a sample `movies.csv` file with the following columns:
- `id`: Unique identifier for the movie.
- `title`: Title of the movie.
- `description`: A brief plot summary.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AMMMARON/movie-yoo.git
   cd movie-yoo
   ```

2. **Install dependencies:**
   Make sure you have Python installed. Then, install the required libraries:
   ```bash
   pip install pandas scikit-learn
   ```

3. **Run the recommender:**
   ```bash
   python recommender.py
   ```

## Dependencies
- `pandas`: For data manipulation and loading.
- `scikit-learn`: For TF-IDF vectorization and cosine similarity computation.
