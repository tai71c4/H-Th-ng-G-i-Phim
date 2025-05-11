import pandas as pd
import numpy as np
import os  # Thêm import os
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

def load_data():
    movies_path = os.path.join("data", "ml-latest-small", "movies.csv")
    ratings_path = os.path.join("data", "ml-latest-small", "ratings.csv")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"File not found: {movies_path}")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"File not found: {ratings_path}")
    
    # Đọc file vào biến movies và ratings
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    
    # Giả lập metadata
    movies['country'] = np.random.choice(['Mỹ', 'Việt Nam', 'Nhật Bản', 'Pháp', 'Hàn Quốc'], len(movies))
    movies['age_rating'] = np.random.choice(['G', 'PG', 'PG-13', 'R'], len(movies))
    movies['actors'] = np.random.choice(['Diễn viên A', 'Diễn viên B', 'Diễn viên C', 'Diễn viên D'], len(movies))
    movies['year'] = np.random.randint(1980, 2025, len(movies))
    return movies, ratings

def create_user_item_matrix(ratings):
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return user_item_matrix

def compute_similarity(user_item_matrix):
    user_similarity = cosine_similarity(user_item_matrix)
    return user_similarity

def get_recommendations(user_id, user_item_matrix, user_similarity, movies, filters=None, top_n=5):
    user_idx = user_id - 1
    sim_scores = user_similarity[user_idx]
    sim_users = np.argsort(sim_scores)[::-1][1:]
    recommendations = []

    for sim_user in sim_users[:10]:
        sim_user_ratings = user_item_matrix.iloc[sim_user]
        for movie_id, rating in sim_user_ratings.items():
            if rating > 0 and user_item_matrix.iloc[user_idx][movie_id] == 0:
                recommendations.append((movie_id, rating * sim_scores[sim_user]))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    movie_ids = [rec[0] for rec in recommendations]
    rec_movies = movies[movies['movieId'].isin(movie_ids)]

    # Áp dụng bộ lọc
    if filters:
        if 'movie_type' in filters and filters['movie_type']:
            rec_movies = rec_movies[rec_movies['genres'].str.contains(filters['movie_type'], case=False, na=False)]
        if 'country' in filters and filters['country']:
            rec_movies = rec_movies[rec_movies['country'] == filters['country']]
        if 'year' in filters and filters['year']:
            rec_movies = rec_movies[rec_movies['year'] == int(filters['year'])]
        if 'age_rating' in filters and filters['age_rating']:
            rec_movies = rec_movies[rec_movies['age_rating'] == filters['age_rating']]
        if 'genre' in filters and filters['genre']:
            rec_movies = rec_movies[rec_movies['genres'].str.contains(filters['genre'], case=False, na=False)]
        if 'actor' in filters and filters['actor']:
            rec_movies = rec_movies[rec_movies['actors'].str.contains(filters['actor'], case=False, na=False)]

    return rec_movies[['title', 'genres', 'country', 'year', 'age_rating', 'actors']].head(top_n).to_dict('records')

def search_movies(query):
    # Mô phỏng tìm kiếm Google bằng AI (dùng Google Custom Search JSON API cho production)
    url = f"https://www.google.com/search?q={query}+phim"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for item in soup.select('div.tF2Cxc')[:3]:
        title = item.select_one('h3').text if item.select_one('h3') else 'Không có tiêu đề'
        description = item.select_one('.VwiC3b') if item.select_one('.VwiC3b') else 'Không có mô tả'
        results.append({
            'title': title,
            'description': description.text[:100] + '...' if description else 'Không có mô tả'
        })
    return results if results else [{'title': 'Không tìm thấy', 'description': 'Thử từ khóa khác.'}]