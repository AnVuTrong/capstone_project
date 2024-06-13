"""
Module name: recommendation.py
In this module, we will implement the recommendation system for our project.
This includes:
- Content-based filtering: Gensim and Cosine Similarity
- Collaborative filtering: Surprise and PySpark ALS.
"""

import pandas as pd


class Recommendation:
	def __init__(
			self,
			process_model_module=None,
	):
		self.df_recommendations = None
		self.process_model_module = process_model_module
	
	def gensim_recommender(self, user_search, num_recommendations) -> pd.DataFrame:
		"""Implement the Gensim model for content-based filtering"""
		try:
			df_recommendations = self.process_model_module.process_gensim(
				user_search,
				num_recommendations=num_recommendations
			)
		except Exception as e:
			raise e
		return df_recommendations
	
	def cosine_similarity_recommender(self, user_search, num_recommendations) -> pd.DataFrame:
		"""Implement the Cosine Similarity model for content-based filtering"""
		try:
			df_recommendations = self.process_model_module.process_cosine_similarity(
				user_search,
				num_recommendations=num_recommendations
			)
		except Exception as e:
			raise e
		return df_recommendations
	
	def surprise_recommender(
			self,
			current_user_id=None,
			user_data=None,
			num_recommendations=10,
			preset=True,
	) -> pd.DataFrame:
		"""Implement the Surprise model for collaborative filtering"""
		try:
			df_recommendations, df_user_reviewed_courses = self.process_model_module.process_surprise(
				current_user_id=current_user_id,
				user_data=user_data,
				num_recommendations=num_recommendations,
				preset=preset,
			)
		except Exception as e:
			raise e
		return df_recommendations, df_user_reviewed_courses
	
	def pyspark_recommender(self, current_user_id, num_recommendations, mode, model_save_path) -> pd.DataFrame:
		try:
			df_recommendations, df_user_reviewed_courses = self.process_model_module.process_pyspark(
				current_user_id,
				num_recommendations,
				mode,
				model_save_path=model_save_path,
			)
		except Exception as e:
			raise e
		return df_recommendations, df_user_reviewed_courses
