from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load data
courses_df = pd.read_csv('courses.csv')
ratings_df = pd.read_csv('ratings.csv') if pd.io.common.file_exists('ratings.csv') else pd.DataFrame()

# Content-Based Filtering
def content_based_recommendations(job_role):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(courses_df['skills_required'] + ' ' + courses_df['sub_skills_required'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = courses_df[courses_df['job_role'] == job_role].index
    if not len(idx):
        return pd.DataFrame()
    
    sim_scores = list(enumerate(cosine_sim[idx[0]]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    course_indices = [i[0] for i in sim_scores]
    
    return courses_df.iloc[course_indices[:10]]

# Collaborative Filtering
def collaborative_filtering_recommendations(user_id):
    if ratings_df.empty:
        return pd.DataFrame()  # Return empty DataFrame if no ratings are available

    # Create a pivot table with users as rows and courses as columns
    user_course_ratings = ratings_df.pivot(index='user_id', columns='course_id', values='rating')
    
    # Fill NaNs with 0
    user_course_ratings = user_course_ratings.fillna(0)
    
    # Compute similarity between users
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_course_ratings)
    
    try:
        distances, indices = model.kneighbors(user_course_ratings.loc[[user_id]], n_neighbors=len(user_course_ratings))
    except KeyError:
        return pd.DataFrame()  # Return empty DataFrame if user_id is not in ratings_df
    
    similar_users = user_course_ratings.index[indices.flatten()]
    similar_users_ratings = user_course_ratings.loc[similar_users].mean(axis=0)
    
    # Predict ratings for courses the user has not rated
    user_ratings = user_course_ratings.loc[user_id]
    recommendations = similar_users_ratings[user_ratings == 0]
    
    # Create a DataFrame of recommendations
    recommendations_df = pd.DataFrame(recommendations, columns=['predicted_rating'])
    recommendations_df = recommendations_df.reset_index()
    recommendations_df.columns = ['course_id', 'predicted_rating']
    
    # Merge with course details
    recommendations_df = recommendations_df.merge(courses_df, on='course_id')
    
    return recommendations_df.sort_values(by='predicted_rating', ascending=False).head(10)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = pd.DataFrame()
    job_roles = courses_df['job_role'].unique()

    if request.method == 'POST':
        job_role = request.form.get('job_role')
        user_id = request.form.get('user_id')

        # Handle cases where user_id might be None
        if user_id:
            try:
                user_id = int(user_id)
            except ValueError:
                return "Error: User ID must be an integer."

        # Initial recommendations for new users
        if user_id is None:
            recommendations = content_based_recommendations(job_role)
        else:
            content_recs = content_based_recommendations(job_role)
            collab_recs = collaborative_filtering_recommendations(user_id)
            recommendations = pd.concat([content_recs, collab_recs]).drop_duplicates()

        return render_template('recommendations.html', recommendations=recommendations.to_dict(orient='records'))

    return render_template('index.html', courses=courses_df, job_roles=job_roles)

@app.route('/get_courses/<job_role>', methods=['GET'])
def get_courses(job_role):
    relevant_courses = courses_df[courses_df['job_role'] == job_role]
    courses_list = relevant_courses[['course_id', 'title']].to_dict(orient='records')
    return jsonify(courses_list)

@app.route('/feedback', methods=['POST'])
def feedback():
    user_id = request.form.get('user_id')
    course_id = request.form.get('course_id')
    rating = request.form.get('rating')
    job_role = request.form.get('job_role')

    # Convert to integers safely
    try:
        user_id = int(user_id)
        course_id = int(course_id)
        rating = int(rating)
    except (ValueError, TypeError):
        return 'Error: Invalid input.'

    relevant_courses = courses_df[courses_df['job_role'] == job_role]['course_id'].astype(int).tolist()
    if course_id not in relevant_courses:
        return 'Error: The course does not match the selected job role.'

    feedback_df = pd.DataFrame({
        'user_id': [user_id],
        'course_id': [course_id],
        'rating': [rating]
    })
    feedback_df.to_csv('ratings.csv', mode='a', header=False, index=False)

    # Reload the ratings data
    global ratings_df
    ratings_df = pd.read_csv('ratings.csv')

    return 'Feedback received'

if __name__ == '__main__':
    app.run(debug=True)
