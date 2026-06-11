# Movie Recommendation System using KNN Algorithm 🎬

A content-based **Movie Recommendation System** built using **Python** and **Machine Learning**.
This system recommends movies similar to the one entered by the user based on features like genres and keywords.

---

## What it Does

When you run the program with a movie name, it finds and prints **5 similar movies** based on content similarity using the KNN algorithm — similar to how **Netflix** and **Amazon Prime** recommend content.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Loading CSV data and preprocessing |
| Scikit-learn | CountVectorizer + NearestNeighbors |
| KNN Algorithm | Finding similar movies |
| CountVectorizer | Converting text to numerical matrix |
| Cosine Similarity | Measuring similarity between movies |
| TMDB Dataset | 5000+ movies from Kaggle |

---

## How the Code Works

### Step 1 — Load Dataset
```python
movies = pd.read_csv("tmdb_5000_movies.csv")
movies = movies[['title','genres','keywords']]
movies = movies.fillna('')
```
- Reads the TMDB CSV file using Pandas
- Selects only 3 columns — title, genres, keywords
- Fills any missing values with empty string

### Step 2 — Feature Engineering
```python
movies['combined_features'] = movies['genres'] + movies['keywords']
```
- Combines genres and keywords into one single column
- This combined text is used to find similarity between movies

### Step 3 — Vectorization
```python
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(movies['combined_features'])
```
- CountVectorizer converts text into a numerical matrix
- Each word becomes a feature — frequency of each word is counted

### Step 4 — Train KNN Model
```python
model = NearestNeighbors(n_neighbors=6, metric='cosine')
model.fit(feature_matrix)
```
- KNN model is trained with cosine similarity
- n_neighbors=6 because first result is the movie itself — so we get 5 actual recommendations

### Step 5 — Recommend Function
```python
def recommend(movie_name):
    movie_index = movies[movies['title'] == movie_name].index[0]
    distances, indices = model.kneighbors(feature_matrix[movie_index])
    print("Recommended Movies:\n")
    for i in indices[0][1:]:
        print(movies.iloc[i]['title'])
```
- Takes movie name as input
- Finds the movie index in the dataset
- Gets the 6 nearest neighbors using KNN
- Prints top 5 recommended movies (skips index 0 which is the movie itself)

---

## Dataset

**TMDB 5000 Movie Dataset** from Kaggle

Download link: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

File needed: `tmdb_5000_movies.csv`

---

## Project Structure

```
movie-recommendation-knn/
│
├── Movie_recomendation.py    # Main Python file
├── tmdb_5000_movies.csv      # Dataset (download from Kaggle link above)
└── README.md                 # Project documentation
```

---

## How to Run

### Step 1 — Install required libraries
```bash
pip install pandas scikit-learn
```

### Step 2 — Download dataset
Download `tmdb_5000_movies.csv` from Kaggle and place it in the same folder as the Python file.

### Step 3 — Run the program
```bash
python Movie_recomendation.py
```

---

## Sample Output

```
Recommended Movies:

Batman Returns
Batman Forever
Batman & Robin
The Dark Knight
Superman Returns
```

---

## Concepts Used

- **Content-based Filtering** — recommends movies based on content similarity
- **K-Nearest Neighbors (KNN)** — finds K most similar items
- **CountVectorizer** — text to numerical feature conversion
- **Cosine Similarity** — measures angle between feature vectors
- **Data Preprocessing** — handling missing values using fillna()
- **Feature Engineering** — combining multiple text features into one

---

## Academic Context

This project was completed as a **Micro-Project** under:

| Detail | Info |
|---|---|
| Subject | Machine Learning |
| Course Code | 316316 |
| Semester | 6th Semester |
| Course | Diploma in Computer Engineering |
| College | VPM's Polytechnic, Thane |
| Board | MSBTE |

---

## About

**Developer:** Mayur Hanumant Shinde
**Course:** Diploma in Computer Engineering
**College:** VPM's Polytechnic, Thane
**GitHub:** https://github.com/Mayur200330
**LinkedIn:** https://www.linkedin.com/in/mayur-shinde-4514a7379/
