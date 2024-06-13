from Project_2.modules.data_preprocessing import DataPreprocessing
from Project_2.modules.models import ProcessModels
from Project_2.modules.recommendation import Recommendation


class RecommendationSystem:
	def __init__(self):
		self.recommendation = Recommendation(
			process_model_module=ProcessModels(DataPreprocessing(), spark=None, model_dir='Project_2/models/')
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
	
	def get_collaborative_recommendations(self, user_id=None, user_data=None, n_recommendations=10):
		recommendations_df, user_history = self.recommendation.surprise_recommender(
			current_user_id=user_id,
			user_data=user_data,
			num_recommendations=n_recommendations,
			preset=False
		)
		
		return recommendations_df, user_history
	
	def _refactor_df(self, df):
		# Reset index, remove unnecessary columns, refactor column names
		df.reset_index(inplace=True)
		df.drop(columns=['index', 'ReviewNumber', 'SimilarityScore'], inplace=True)
		df.rename(columns={
			'CourseName': 'Course Name',
			'AvgStar'   : 'Rating',
			'Unit'      : 'Provider',
			'Results'   : 'Description'
		}, inplace=True)
		
		return df
