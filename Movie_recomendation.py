import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
movies = pd.read_csv(&quot;tmdb_5000_movies.csv&quot;)
movies = movies[[&#39;title&#39;,&#39;genres&#39;,&#39;keywords&#39;]]
movies = movies.fillna(&#39;&#39;)
movies[&#39;combined_features&#39;] = movies[&#39;genres&#39;] + movies[&#39;keywords&#39;]
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(movies[&#39;combined_features&#39;])
model = NearestNeighbors(n_neighbors=6, metric=&#39;cosine&#39;)
model.fit(feature_matrix)

def recommend(movie_name):
movie_index = movies[movies[&#39;title&#39;] == movie_name].index[0]
distances, indices = model.kneighbors(feature_matrix[movie_index])
print(&quot;Recommended Movies:\n&quot;)
for i in indices[0][1:]:
print(movies.iloc[i][&#39;title&#39;])
recommend(&quot;Batman&quot;)
