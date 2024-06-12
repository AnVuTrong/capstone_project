"""
Module name: data_analysis.py
In this module, we will do data analysis on the data we have been issued.
We will look at the data and try to understand it better, try to find some insights from the data.
Process:
- Load the data (Pandas and PySpark)
- Exploratory Data Analysis
- Data Visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalysisCoursera:
    def __init__(self, path_courses, path_reviews, save_path):
        self.df_courses = pd.read_csv(path_courses)
        self.df_reviews = pd.read_csv(path_reviews)
        self.save_path = save_path
        
        
    def load_data(self):
        """Load the data from the specified paths"""
        return self.df_courses, self.df_reviews
    
    def eda(self):
        """Perform Exploratory Data Analysis (EDA)"""
        print("Courses Data Analysis")
        print("=" * 50)
        print("Head:")
        print(self.df_courses.head(), "\n")
        print("-" * 50)
        
        print("Info:")
        self.df_courses.info()
        print("-" * 50)
        
        print("Describe:")
        print(self.df_courses.describe(), "\n")
        print("-" * 50)
        
        print("Null Values:")
        print(self.df_courses.isnull().sum(), "\n")
        print("-" * 50)
        
        print("Duplicate Rows:")
        print(self.df_courses.duplicated().sum(), "duplicate rows found\n")
        print("-" * 50)
        
        print("\nReviews Data Analysis")
        print("=" * 50)
        print("Head:")
        print(self.df_reviews.head(), "\n")
        print("-" * 50)
        
        print("Info:")
        self.df_reviews.info()
        print("-" * 50)
        
        print("Describe:")
        print(self.df_reviews.describe(), "\n")
        print("-" * 50)
        
        print("Null Values:")
        print(self.df_reviews.isnull().sum(), "\n")
        print("-" * 50)
        
        print("Duplicate Rows:")
        print(self.df_reviews.duplicated().sum(), "duplicate rows found\n")
        print("-" * 50)
    
    def data_visualize(self):
        """Visualize the data using various plots"""
        # Create the directory if it doesn't exist
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Distribution of Review Ratings
        plt.figure(figsize=(10, 6))
        sns.countplot(x='RatingStar', data=self.df_reviews)
        plt.title('Distribution of Review Ratings')
        plt.xlabel('Rating Star')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_path, 'distribution_of_review_ratings.png'))
        plt.show()
        plt.close()
        
        # Top 10 Most Reviewed Courses
        review_counts = (
            self.df_courses.groupby('CourseName')['ReviewNumber'].sum().sort_values(ascending=False).head(10)
        )
        plt.figure(figsize=(10, 6))
        review_counts.plot(kind='bar')
        plt.title('Top 10 Most Reviewed Courses')
        plt.xlabel('Course Name')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(save_path, 'top_10_most_reviewed_courses.png'))
        plt.show()
        plt.close()
        
        # Correlation Matrix for Courses
        numeric_df_courses = self.df_courses.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df_courses.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix for Courses')
        plt.savefig(os.path.join(save_path, 'correlation_matrix_for_courses.png'))
        plt.close()
        
        # Review Ratings over Time
        self.df_reviews['DateOfReview'] = pd.to_datetime(self.df_reviews['DateOfReview'])
        plt.figure(figsize=(10, 6))
        self.df_reviews.groupby(self.df_reviews['DateOfReview'].dt.to_period('M'))['RatingStar'].mean().plot()
        plt.title('Average Review Rating Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Rating Star')
        plt.savefig(os.path.join(save_path, 'average_review_rating_over_time.png'))
        plt.show()
        plt.close()
        