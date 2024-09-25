from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# # Example datasets
# subdomains_df = pd.DataFrame({
#     'subdomain_id': [1, 2, 3],
#     'subdomain_name': ['Cybersecurity', 'Data Science', 'Nursing'],
#     'domain_id': [1, 1, 2],
#     'skills': ['Network Security, Risk Analysis', 'Machine Learning, Data Mining', 'Patient Care, Medication Management']
# })

# courses_df = pd.DataFrame({
#     'course_id': [1, 2, 3, 4],
#     'course_title': ['Intro to Cybersecurity', 'Advanced Network Security', 'Intro to Data Science', 'Fundamentals of Nursing'],
#     'course_description': ['Learn the basics of cybersecurity including network security.', 
#                            'Deep dive into network security and risk management.',
#                            'Explore data science basics including machine learning and data mining.',
#                            'Core principles and practices in nursing.'],
#     'subdomain_id': [1, 1, 2, 3]
# })
domain_df = pd.read_csv('domain_df.csv', encoding='latin1')
subdomains_df = pd.read_csv('subdomain_df.csv', encoding='latin1')
courses_df = pd.read_csv('Updated_Courses_with_Image_URLs.csv', encoding='latin1')



def recommend_courses_by_subdomain(subdomain_id, top_n=10):

    # Retrieve subdomain details
    subdomain = subdomains_df[subdomains_df['subdomain_id'] == subdomain_id]
    
    if subdomain.empty:
        return pd.DataFrame(columns=['course_id', 'course_title', 'course_description', 'subdomain_id'])
    
    # Extract skills related to the subdomain
    subdomain_skills = subdomain['skills'].values[0]
    
    # Filter courses related to the subdomain
    related_courses = courses_df[courses_df['subdomain_id'] == subdomain_id]
    
    if related_courses.empty:
        return pd.DataFrame(columns=['course_id', 'course_title', 'course_description', 'subdomain_id','course_url'])
    
    # Vectorize course descriptions and subdomain skills
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    course_descriptions = related_courses['course_description']
    tfidf_matrix = tfidf_vectorizer.fit_transform(course_descriptions)
    subdomain_vec = tfidf_vectorizer.transform([subdomain_skills])
    
    # Compute cosine similarity between subdomain skills and course descriptions
    cosine_similarities = cosine_similarity(subdomain_vec, tfidf_matrix).flatten()
    
    # Get top N course indices
    top_course_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
    # Get recommended courses
    recommended_courses = related_courses.iloc[top_course_indices]
    
    return recommended_courses

# # Example usage
# subdomain_id = 2  # Example subdomain ID

# recommended_courses = recommend_courses_by_subdomain(subdomain_id)

# # Display the recommended courses
# print(recommended_courses)
