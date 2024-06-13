import pandas as pd
import numpy as np


def generate_random_user_data(df_courses, num_courses=5, min_rating=1, max_rating=5):
	"""
	Generate random user data from existing courses.

	Parameters:
	df_courses (pd.DataFrame): DataFrame containing courses data.
	Num_courses (int): Number of courses to be selected randomly. Default is 5.
	Min_rating (int): Minimum rating value. Default is 1.
	Max_rating (int): Maximum rating value. Default is 5.

	Returns:
	pd.DataFrame: DataFrame containing randomly generated user data.
	"""
	num_courses = min(num_courses, df_courses.shape[0])
	selected_courses = df_courses.sample(num_courses)
	selected_courses['RatingStar'] = np.random.randint(min_rating, max_rating + 1, size=num_courses)
	user_data = selected_courses[['CourseID', 'RatingStar']]
	
	return user_data