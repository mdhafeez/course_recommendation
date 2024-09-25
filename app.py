from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import pygwalker as pyg
import os
import plotly.express as px
from recommendation_system import recommend_courses_by_subdomain, domain_df, subdomains_df
import markdown
from markupsafe import Markup
from eda import create_visualization  # Correct import from eda.py

app = Flask(__name__)

# Load data
courses_df = pd.read_csv('courses.csv')
ratings_df = pd.read_csv('ratings.csv') if pd.io.common.file_exists('ratings.csv') else pd.DataFrame()
wdata_df = pd.read_csv('WData2.csv', encoding='latin1')


##################### Content-Based Filtering ################
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


###################### Collaborative Filtering ################
def collaborative_filtering_recommendations(user_id):
    if ratings_df.empty:
        return pd.DataFrame()

    # Sort by timestamp and remove duplicates for user-course combinations
    ratings_df_cleaned = ratings_df.sort_values(by='timestamp').drop_duplicates(subset=['user_id', 'course_id'], keep='last')

    # Create a user-course ratings matrix
    user_course_ratings = ratings_df_cleaned.pivot(index='user_id', columns='course_id', values='rating').fillna(0)

    # Compute similarity between users using NearestNeighbors
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_course_ratings)
    
    try:
        distances, indices = model.kneighbors(user_course_ratings.loc[[user_id]], n_neighbors=len(user_course_ratings))
    except KeyError:
        return pd.DataFrame()

    similar_users = user_course_ratings.index[indices.flatten()]
    similar_users_ratings = user_course_ratings.loc[similar_users].mean(axis=0)
    
    # Predict ratings for unrated courses
    user_ratings = user_course_ratings.loc[user_id]
    recommendations = similar_users_ratings[user_ratings == 0]
    
    # Create a DataFrame of recommendations
    recommendations_df = pd.DataFrame(recommendations, columns=['predicted_rating']).reset_index()
    recommendations_df.columns = ['course_id', 'predicted_rating']
    
    # Merge with course details and return top 10 recommendations
    recommendations_df = recommendations_df.merge(courses_df, on='course_id')
    return recommendations_df.sort_values(by='predicted_rating', ascending=False).head(10)


##################### Index Route ################
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = pd.DataFrame()
    job_roles = courses_df['job_role'].unique()

    if request.method == 'POST':
        job_role = request.form.get('job_role')
        user_id = request.form.get('user_id')

        if user_id:
            try:
                user_id = int(user_id)
            except ValueError:
                return "Error: User ID must be an integer."

        if user_id is None or user_id == "":
            recommendations = content_based_recommendations(job_role)
        else:
            content_recs = content_based_recommendations(job_role)
            collab_recs = collaborative_filtering_recommendations(user_id)
            recommendations = pd.concat([content_recs, collab_recs]).drop_duplicates()

        return render_template('index.html', job_roles=job_roles, recommendations=recommendations.to_dict(orient='records'), user_id=user_id)

    return render_template('index.html', job_roles=job_roles)


##################### Submit Rating Route ################
@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    user_id = request.form.get('user_id')
    course_id = request.form.get('course_id')
    rating = request.form.get('rating')
    timestamp = datetime.now()

    try:
        user_id = int(user_id)
        course_id = int(course_id)
        rating = int(rating)
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid input.'}), 400

    try:
        feedback_df = pd.DataFrame({'user_id': [user_id], 'course_id': [course_id], 'rating': [rating], 'timestamp': [timestamp]})
        feedback_df.to_csv('ratings.csv', mode='a', header=not pd.io.common.file_exists('ratings.csv'), index=False)

        global ratings_df
        ratings_df = pd.read_csv('ratings.csv')

        return jsonify({'status': 'success', 'message': 'Rating submitted successfully'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Failed to save rating.'}), 500


##################### Get Courses by Job Role ################
@app.route('/get_courses/<job_role>', methods=['GET'])
def get_courses(job_role):
    relevant_courses = courses_df[courses_df['job_role'] == job_role]
    courses_list = relevant_courses[['course_id', 'title']].to_dict(orient='records')
    return jsonify(courses_list)


##################### Visualize WData Route ################
@app.route('/visualize_wdata')
def visualize_wdata():
    if not os.path.exists('WData2.csv'):
        return "wdata.csv not found. Please upload the dataset.", 404

    try:
        wdata_df = pd.read_csv('WData2.csv', encoding='latin1')
    except Exception as e:
        return f"Error loading dataset: {e}"

    try:
        walker = pyg.walk(wdata_df)
        pyg_html = walker.to_html()
    except Exception as e:
        return f"Error generating PyGWalker visualization: {e}"

    return render_template('visualize.html', pyg_html=pyg_html)


##################### Domain Route ################
@app.route('/domain')
def domain():
    domains = domain_df.to_dict(orient='records')
    return render_template('domain.html', domains=domains)


##################### Get Subdomains ################
@app.route('/subdomains/<int:domain_id>', methods=['GET'])
def get_subdomains(domain_id):
    subdomains = subdomains_df[subdomains_df['domain_id'] == domain_id].to_dict(orient='records')
    return jsonify(subdomains)


##################### Recommend Courses ################
@app.route('/recommend', methods=['GET'])
def recommend():
    domain_id = int(request.args.get('domain_id'))
    subdomain_id = int(request.args.get('subdomain_id'))
    top_n = int(request.args.get('top_n', 6))

    recommended_courses = recommend_courses_by_subdomain(subdomain_id, top_n)
    recommendations = recommended_courses.to_dict(orient='records')

    no_courses_message = None
    if len(recommendations) == 0:
        no_courses_message = "No courses match the selected domain and subdomain at this time."

    return render_template(
        'domain.html',
        domains=domain_df.to_dict(orient='records'),
        subdomains=subdomains_df[subdomains_df['domain_id'] == domain_id].to_dict(orient='records'),
        recommendations=recommendations,
        selected_domain_id=domain_id,
        selected_subdomain_id=subdomain_id,
        no_courses_message=no_courses_message
    )


##################### Readme Route ################
@app.route('/readme')
def readme():
    with open('README.md', 'r') as readme_file:
        content = readme_file.read()
        md = markdown.markdown(content)
    return render_template('readme.html', content=Markup(md))


##################### Data Visualization 2 ################
# @app.route('/data_visualization_2', methods=['GET', 'POST'])
# def data_visualization_2():
#     selected_visualization = 'year_distribution'
#     if request.method == 'POST':
#         selected_visualization = request.form.get('visualization_selector')

#     fig = create_visualization(selected_visualization)
#     return render_template('data_visualization_2.html', plot=fig.to_html(full_html=False), selected_visualization=selected_visualization)

@app.route('/data_visualization_2', methods=['GET', 'POST'])
def data_visualization_2():
    selected_visualization = 'year_distribution'
    selected_year_range = (2020, 2024)  # Default year range

    if request.method == 'POST':
        selected_visualization = request.form.get('visualization_selector')

        # Capture the selected year range from the sliders
        year_range_min = int(request.form.get('year_range_min', 2020))
        year_range_max = int(request.form.get('year_range_max', 2024))
        selected_year_range = (year_range_min, year_range_max)

    # Pass the year range to the create_visualization function
    fig = create_visualization(selected_visualization, year_range=selected_year_range)

    return render_template(
        'data_visualization_2.html',
        plot=fig.to_html(full_html=False),
        selected_visualization=selected_visualization,
        selected_year_range=selected_year_range
    )




if __name__ == '__main__':
    app.run(debug=True)
