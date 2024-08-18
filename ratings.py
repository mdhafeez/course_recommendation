import pandas as pd
import numpy as np

# Define job roles and courses
courses = {
    'UI/UX Designer': ['1', '5', '21', '42'],
    'Tester': ['2', '10', '12', '15', '31', '35'],
    'Frontend Developer': ['3', '29', '37', '45'],
    'DevOps': ['4', '8', '9', '13', '19', '49'],
    'Web Developer': ['6'],
    'Software Engineer': ['7', '33'],
    'Admin': ['11', '23', '39', '48'],
    'Scrum Master': ['16', '22', '27', '30'],
    'Project Manager': ['24', '34', '50'],
    'Data Analyst': ['14', '17', '20', '28', '38', '40', '43', '47'],
    'Data Engineer': ['18', '26', '36', '44'],
    'Backend Developer': ['46']
}

# Define synthetic data
users = [f'{i}' for i in range(1, 21)]  # 20 users

# Create a DataFrame to store ratings
ratings_data = {
    'user_id': [],
    'course_id': [],
    'rating': []
}

# Generate synthetic ratings
np.random.seed(42)  # For reproducibility

for user in users:
    # Randomly assign a job role to each user
    job_role = np.random.choice(list(courses.keys()))
    # Select courses relevant to the user's job role
    relevant_courses = courses[job_role]
    # Rate a random selection of relevant courses
    rated_courses = np.random.choice(relevant_courses, size=np.random.randint(1, len(relevant_courses)+1), replace=False)
    for course in rated_courses:
        rating = np.random.randint(1, 6)  # Ratings between 1 and 5
        ratings_data['user_id'].append(user)
        ratings_data['course_id'].append(course)
        ratings_data['rating'].append(rating)

# Convert to DataFrame
ratings_df = pd.DataFrame(ratings_data)

# Save to CSV
ratings_df.to_csv('ratings.csv', index=False)

print("Synthetic ratings data generated and saved to 'ratings.csv'.")
