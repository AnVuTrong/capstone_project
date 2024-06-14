import pandas as pd

from Project_2.modules.data_preprocessing import DataPreprocessing
from Project_2.modules.models import ProcessModels
from Project_2.modules.recommendation import Recommendation


class RecommendationSystem:
	def __init__(self):
		data_preprocessing = DataPreprocessing(
			path_courses='Project_2/data/courses.csv',
			path_reviews='Project_2/data/reviews.csv',
		)
		self.recommendation = Recommendation(
			process_model_module=ProcessModels(data_preprocessing, spark=None, model_dir='Project_2/models/')
		)
	
	def get_gensim_recommendations(self, user_search, num_recommendations=10):
		user_search = user_search.strip()
		df = self.recommendation.gensim_recommender(user_search, num_recommendations)
		df = self._refactor_df(df)
		
		return df
	
	def get_cosine_recommendations(self, user_search, num_recommendations=10):
		user_search = user_search.strip()
		df = self.recommendation.cosine_similarity_recommender(user_search, num_recommendations)
		df = self._refactor_df(df)
		
		return df
	
	def get_collaborative_recommendations(self, user_id=None, user_data=None, n_recommendations=10, preset=True):
		if isinstance(user_data, pd.DataFrame):
			user_data = user_data[['CourseID', 'RatingStar']]
		recommendations_df, user_history = self.recommendation.surprise_recommender(
			current_user_id=user_id,
			user_data=user_data,
			num_recommendations=n_recommendations,
			preset=preset
		)
		
		recommendations_df = self._refactor_df(recommendations_df)
		user_history = self._refactor_df(user_history)
		
		return recommendations_df, user_history
	
	def _refactor_df(self, df):
		# Reset index
		df.reset_index(inplace=True, drop=True)
		
		# Sort by SimilarityScore then by AvgStar if they are present
		columns_to_sort_by = ['SimilarityScore', 'AvgStar']
		if set(columns_to_sort_by).issubset(df.columns):
			df.sort_values(by=columns_to_sort_by, ascending=[False, False], inplace=True)
		
		# List of columns to drop
		columns_to_drop = ['index', 'ReviewNumber', 'SimilarityScore', 'CourseID']
		
		# Attempt to drop columns
		for column in columns_to_drop:
			try:
				df.drop(columns=[column], inplace=True)
			except KeyError:
				continue
		
		# Dictionary of columns to rename
		columns_to_rename = {
			'CourseName': 'Course Name',
			'AvgStar'   : 'Average Rating',
			'Unit'      : 'Provider',
			'Results'   : 'Description',
			'RatingStar': 'User Rating',
		}
		
		# Attempt to rename columns
		for old_name, new_name in columns_to_rename.items():
			try:
				df.rename(columns={old_name: new_name}, inplace=True)
			except KeyError:
				continue
		
		return df
