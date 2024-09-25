import plotly.express as px
import pandas as pd

# Load the dataset
df = pd.read_csv('WData.csv', encoding='latin1')


# Create a function that generates the desired visualization based on the selected chart and year range
def create_visualization(selection, year_range=None):
    # Filter the dataset by the selected year range, if provided
    if year_range:
        filtered_df = df[(df['Tahun_Kursus'] >= year_range[0]) & (df['Tahun_Kursus'] <= year_range[1])]
    else:
        filtered_df = df

    if selection == 'year_distribution':
        # Group by 'Tahun_Kursus' and 'Kod_Kursus' to ensure unique course count
        grouped_courses = filtered_df.groupby(['Tahun_Kursus', 'Kod_Kursus']).size().reset_index(name='Course Count')
        
        # Histogram: Course Distribution by Year
        fig = px.histogram(
            grouped_courses,
            x='Tahun_Kursus',
            title="Course Distribution by Year",
            labels={'Tahun_Kursus': 'Year', 'count': 'Number of Courses'},
            height=400,
            color_discrete_sequence=['#636EFA']
        )
        
        fig.update_layout(
            bargap=0.095,  # Adds gap between bars
            title={'text': "Course Distribution by Year", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis_title="Year",
            yaxis_title="Number of Courses",
            font=dict(family="Arial, sans-serif", size=14, color="#4A4A4A")
        )
        fig.update_traces(
            hovertemplate='<b>Year: %{x}</b><br>Number of Courses: %{y}<extra></extra>',
            marker_line_color='black',
            marker_line_width=1.5
        )

    elif selection == 'duration_by_field':
        # Group by 'Bidang' and 'Kod_Kursus' to calculate average duration per course
        df_avg_duration = filtered_df.groupby(['Bidang', 'Kod_Kursus'])['Tempoh_Kursus'].mean().reset_index()
        df_avg_duration = df_avg_duration.groupby('Bidang')['Tempoh_Kursus'].mean().reset_index()
        
        # Bar Chart: Average Course Duration by Field
        fig = px.bar(
            df_avg_duration,
            x='Bidang',
            y='Tempoh_Kursus',
            title="Average Course Duration by Field",
            labels={'Tempoh_Kursus': 'Average Duration (Days)', 'Bidang': 'Field'},
            height=800,
            color='Bidang',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            bargap=0.3,
            title={'text': "Average Course Duration by Field", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis_title="Field",
            yaxis_title="Average Duration (Days)",
            font=dict(family="Arial, sans-serif", size=14, color="#4A4A4A")
        )

    elif selection == 'subfield_distribution':
        # Group by 'Bidang', 'Sub_Bidang', and 'Kod_Kursus' to ensure unique course count
        df_filtered = filtered_df.dropna(subset=['Sub_Bidang']).groupby(['Bidang', 'Sub_Bidang', 'Kod_Kursus']).size().reset_index(name='Course Count')
        
        # Now, count unique 'Kod_Kursus' per 'Bidang' and 'Sub_Bidang'
        subfield_counts = df_filtered.groupby(['Bidang', 'Sub_Bidang'])['Kod_Kursus'].nunique().reset_index(name='Course Count')
        
        # Treemap: Courses by Sub-Field
        fig = px.treemap(
            subfield_counts,
            path=['Bidang', 'Sub_Bidang'],
            values='Course Count',
            title="Courses by Sub-Field",
            labels={'Bidang': 'Field', 'Sub_Bidang': 'Sub-Field'},
            color='Course Count',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            title={'text': "Courses by Sub-Field", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            font=dict(family="Arial, sans-serif", size=14, color="#4A4A4A")
        )

    elif selection == 'participant_scheme_distribution':
        # Group by 'Nama_Skim' and 'Kod_Kursus' to ensure unique course count
        df_scheme = filtered_df.groupby(['Nama_Skim', 'Kod_Kursus']).size().reset_index(name='Count')
        scheme_counts = df_scheme.groupby('Nama_Skim')['Kod_Kursus'].nunique().reset_index(name='Number of Participants')
        
        # Pie Chart: Participant Scheme Distribution
        fig = px.pie(
            scheme_counts,
            names='Nama_Skim',
            values='Number of Participants',
            title="Participant Scheme Distribution",
            labels={'Nama_Skim': 'Scheme', 'Number of Participants': 'Number of Participants'},
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.3,
            height=600
        )
        
        fig.update_traces(textinfo='percent+label', textposition='inside')
        fig.update_layout(
            title={'text': "Participant Scheme Distribution", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            legend=dict(orientation="v", yanchor="top", y=0.9, xanchor="left", x=1.0),
            margin=dict(t=50, l=50, r=50, b=50),
            font=dict(family="Arial, sans-serif", size=14, color="#4A4A4A")
        )

    elif selection == 'trends_over_time':
        # Group by 'Tahun_Kursus', 'Bidang', and 'Kod_Kursus' to ensure unique course count
        grouped = filtered_df.groupby(['Tahun_Kursus', 'Bidang', 'Kod_Kursus']).size().reset_index(name='Course Count')
        trends = grouped.groupby(['Tahun_Kursus', 'Bidang'])['Kod_Kursus'].nunique().reset_index(name='Course Count')
        
        # Line Chart: Trends Over Time
        fig = px.line(
            trends,
            x='Tahun_Kursus',
            y='Course Count',
            color='Bidang',
            title="Trends Over Time by Field",
            labels={'Tahun_Kursus': 'Year', 'Course Count': 'Number of Courses'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
    
        fig.update_layout(
            title={'text': "Trends Over Time by Field", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis_title="Year",
            yaxis_title="Number of Courses",
            font=dict(family="Arial, sans-serif", size=14, color="#4A4A4A")
        )

    elif selection == 'trends_schemes_over_time':
        # Group by 'Tahun_Kursus', 'Nama_Skim', and 'Kod_Kursus' to ensure unique course count
        grouped_schemes = filtered_df.groupby(['Tahun_Kursus', 'Nama_Skim', 'Kod_Kursus']).size().reset_index(name='Participant Count')
        trends_schemes = grouped_schemes.groupby(['Tahun_Kursus', 'Nama_Skim'])['Kod_Kursus'].nunique().reset_index(name='Participant Count')
    
        # Bubble chart: Trends in Participant Schemes Over Time
        fig = px.scatter(
            trends_schemes,
            x='Tahun_Kursus',
            y='Nama_Skim',
            size='Participant Count',
            title="Trends in Participant Schemes Over Time",
            labels={'Tahun_Kursus': 'Year', 'Nama_Skim': 'Participant Scheme', 'Participant Count': 'Number of Participants'},
            color='Nama_Skim',
            size_max=60,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
    
        fig.update_layout(
            title={'text': "Trends in Participant Schemes Over Time", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis_title="Year",
            yaxis_title="Participant Scheme",
            font=dict(family="Arial, sans-serif", size=14, color="#4A4A4A")
        )
    
        fig.update_traces(
            marker=dict(sizemode='area', line_width=2),
            hovertemplate='<b>Year: %{x}</b><br>Scheme: %{y}<br>Participants: %{marker.size}<extra></extra>'
        )
    
    return fig
