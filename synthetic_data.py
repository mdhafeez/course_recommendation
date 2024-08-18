import pandas as pd
import numpy as np

# Define job roles and their required skills and sub-skills
job_roles = [
    'Data Scientist', 'Data Analyst', 'Data Engineer', 'Web Developer',
    'Backend Developer', 'Frontend Developer', 'Business Analyst',
    'Scrum Master', 'Software Engineer', 'UI/UX Designer', 'Admin',
    'DevOps', 'Project Manager', 'Tester'
]

skills_dict = {
    'Data Scientist': {
        'main': ['Python', 'Statistics', 'Machine Learning'],
        'sub': ['Pandas', 'Scikit-learn', 'TensorFlow', 'Keras']
    },
    'Data Analyst': {
        'main': ['Excel', 'SQL', 'Data Visualization'],
        'sub': ['Tableau', 'Power BI', 'Matplotlib', 'Seaborn']
    },
    'Data Engineer': {
        'main': ['ETL', 'Big Data', 'SQL'],
        'sub': ['Apache Spark', 'Hadoop', 'Kafka', 'Airflow']
    },
    'Web Developer': {
        'main': ['HTML', 'CSS', 'JavaScript'],
        'sub': ['React', 'Angular', 'Vue.js', 'Bootstrap']
    },
    'Backend Developer': {
        'main': ['Node.js', 'API Development', 'Database Management'],
        'sub': ['Express.js', 'MongoDB', 'SQL Server', 'Django']
    },
    'Frontend Developer': {
        'main': ['HTML', 'CSS', 'JavaScript'],
        'sub': ['Sass', 'Webpack', 'JQuery', 'Vue.js']
    },
    'Business Analyst': {
        'main': ['Requirement Gathering', 'Data Analysis', 'Reporting'],
        'sub': ['JIRA', 'Confluence', 'SQL', 'Excel']
    },
    'Scrum Master': {
        'main': ['Agile Methodology', 'Scrum Practices'],
        'sub': ['JIRA', 'Confluence', 'Retrospectives', 'Stand-ups']
    },
    'Software Engineer': {
        'main': ['Programming', 'Software Design', 'Testing'],
        'sub': ['Git', 'JUnit', 'Docker', 'Kubernetes']
    },
    'UI/UX Designer': {
        'main': ['Design Thinking', 'Wireframing', 'Prototyping'],
        'sub': ['Sketch', 'Figma', 'Adobe XD', 'InVision']
    },
    'Admin': {
        'main': ['System Administration', 'Network Management'],
        'sub': ['Linux', 'Windows Server', 'Networking', 'Scripting']
    },
    'DevOps': {
        'main': ['Continuous Integration', 'Continuous Deployment', 'Automation'],
        'sub': ['Jenkins', 'Docker', 'Kubernetes', 'Ansible']
    },
    'Project Manager': {
        'main': ['Project Planning', 'Risk Management', 'Budgeting'],
        'sub': ['MS Project', 'Asana', 'Trello', 'Gantt Charts']
    },
    'Tester': {
        'main': ['Testing Techniques', 'Automation', 'Bug Tracking'],
        'sub': ['Selenium', 'JUnit', 'Bugzilla', 'TestRail']
    }
}

def generate_courses_data(num_courses):
    courses = []
    course_id = 1
    for job_role in np.random.choice(job_roles, num_courses):
        main_skills = skills_dict[job_role]['main']
        sub_skills = skills_dict[job_role]['sub']
        skills_required = ', '.join(np.random.choice(main_skills, size=np.random.randint(1, 3), replace=False))
        sub_skills_required = ', '.join(np.random.choice(sub_skills, size=np.random.randint(1, 3), replace=False))
        course_title = f'{job_role} Course {course_id}'
        course_description = f'This course covers essential skills for the role of {job_role}. Skills required: {skills_required}. Sub-skills required: {sub_skills_required}.'
        courses.append({
            'course_id': course_id,
            'title': course_title,
            'job_role': job_role,
            'skills_required': skills_required,
            'sub_skills_required': sub_skills_required,
            'description': course_description
        })
        course_id += 1
    return pd.DataFrame(courses).drop_duplicates(subset=['job_role', 'skills_required', 'sub_skills_required'])

def generate_synthetic_data():
    num_courses = 50
    courses_df = generate_courses_data(num_courses)
    
    # Save to CSV files
    courses_df.to_csv('courses.csv', index=False)

# Run data generation
generate_synthetic_data()
