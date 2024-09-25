import pandas as pd

# Load the courses and subdomain data
courses_df = pd.read_csv('courses_1.csv', encoding='latin1')
subdomain_df = pd.read_csv('subdomain_df.csv', encoding='latin1')

# Merge the data to insert subdomain names
merged_courses_df = courses_df.merge(subdomain_df[['subdomain_id', 'subdomain_name']], on='subdomain_id', how='left')

# Save the updated DataFrame with subdomain names inserted
merged_courses_df.to_csv('Updated_Courses_with_Subdomain.csv', index=False)

# Print the first few rows to verify the merge
print(merged_courses_df.head())
