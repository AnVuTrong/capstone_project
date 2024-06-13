"""
Module name: models.py
This module will contain all the models' code for our recommendation system.
This includes training and testing the models, as well as making recommendations.
- Content-based filtering: Gensim and Cosine Similarity
- Collaborative filtering: Surprise and PySpark ALS.
"""
import os
import uuid
import pickle
import pandas as pd

from gensim import models, similarities

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Reader, Dataset, SVD
from surprise.model_selection.validation import cross_validate

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


class ProcessModels:
	def __init__(self, data_preprocessing, spark, model_dir='models/'):
		self.data_preprocessing = data_preprocessing
		self.spark = spark
		self.model_dir = model_dir
		os.makedirs(model_dir, exist_ok=True)
	
	def process_gensim(self, query, num_recommendations) -> pd.DataFrame:
		"""Process the data for Gensim model using SparseMatrixSimilarity"""
		model_name = 'gensim_model.pkl'
		model_data = self._load_model(model_name)
		
		if model_data:
			# Load the model
			tfidf, index, dictionary, df_courses = model_data
		else:
			# Train the model
			df_courses, dictionary, corpus = self.data_preprocessing.gensim_preprocessing()
			tfidf = models.TfidfModel(corpus)
			index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))
			self._save_model((tfidf, index, dictionary, df_courses), model_name)
		
		query_tokens = self.data_preprocessing.tokenize(query)
		query_bow = dictionary.doc2bow(query_tokens)
		query_tfidf = tfidf[query_bow]
		similarities_scores = index[query_tfidf]
		similarities_scores = sorted(enumerate(similarities_scores), key=lambda item: -item[1])
		recommended_indices = [idx for idx, score in similarities_scores[:num_recommendations]]
		
		# Include similarity scores in the output
		recommended_courses = df_courses.iloc[recommended_indices].copy()
		recommended_courses['SimilarityScore'] = [score for idx, score in similarities_scores[:num_recommendations]]
		
		return recommended_courses[
			['CourseName', 'ReviewNumber', 'AvgStar', 'Level', 'Unit', 'Results', 'SimilarityScore']]
	
	def process_cosine_similarity(self, query, num_recommendations) -> pd.DataFrame:
		"""Process the data for Cosine Similarity model"""
		model_name = 'cosine_similarity_model.pkl'
		model_data = self._load_model(model_name)
		
		if model_data:
			tfidf_vectorizer, tfidf_matrix, df_courses = model_data
		else:
			df_courses = self.data_preprocessing.cosine_similarity_preprocessing()
			tfidf_vectorizer = TfidfVectorizer()
			tfidf_matrix = tfidf_vectorizer.fit_transform(df_courses['Combined'])
			self._save_model((tfidf_vectorizer, tfidf_matrix, df_courses), model_name)
		
		query_tfidf = tfidf_vectorizer.transform([query])
		cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
		similar_indices = cosine_similarities.argsort()[:-num_recommendations - 1:-1]
		
		recommended_courses = df_courses.iloc[similar_indices].copy()
		recommended_courses['SimilarityScore'] = cosine_similarities[similar_indices]
		
		return recommended_courses[
			['CourseName', 'ReviewNumber', 'AvgStar', 'Level', 'Unit', 'Results', 'SimilarityScore']]
	
	def process_surprise(
			self,
			current_user_id=None,
			user_data=None,
			num_recommendations=10,
			preset=True,
	) -> (pd.DataFrame, pd.DataFrame):
		"""Process the data for Surprise model"""
		df_reviews, df_courses = self.data_preprocessing.surprise_preprocessing()
		new_user_reviews = None
		if preset:
			if current_user_id not in df_reviews['ReviewerID'].unique():
				raise ValueError(f'User: "{current_user_id}" not found in the database. Please try another user.')
			user_id = current_user_id
		else:
			if user_data is None or 'CourseID' not in user_data.columns or 'RatingStar' not in user_data.columns:
				raise ValueError(
					'User data must be provided with "CourseID" and "RatingStar" columns when preset is False.')
			user_id = str(uuid.uuid4())
			new_user_reviews = pd.DataFrame(user_data)
			new_user_reviews['ReviewerID'] = user_id
			df_reviews = pd.concat([df_reviews, new_user_reviews], ignore_index=True)
		
		model_name = 'surprise_model.pkl'
		algo = self._load_model(model_name)
		
		if not algo:
			algo = self._train_surprise(df_reviews, model_name)
		
		if not preset:
			# Create a temporary dataset with the new user data
			reader = Reader(rating_scale=(1, 5))
			data = Dataset.load_from_df(df_reviews[['ReviewerID', 'CourseID', 'RatingStar']], reader)
			trainset = data.build_full_trainset()
			algo.fit(trainset)
		
		all_courses = df_courses['CourseID'].unique()
		predictions = [algo.predict(user_id, course) for course in all_courses]
		recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:num_recommendations]
		recommended_courses = pd.DataFrame([(rec.iid, rec.est) for rec in recommendations],
		                                   columns=['CourseID', 'EstimatedRating'])
		recommended_courses = recommended_courses.merge(df_courses, on='CourseID', suffixes=('', '_course'))
		
		if preset:
			user_reviews = df_reviews[df_reviews['ReviewerID'] == current_user_id]
			user_reviews = user_reviews.merge(df_courses, on='CourseID', suffixes=('', '_course'))
		else:
			user_reviews = new_user_reviews.merge(df_courses, on='CourseID', suffixes=('', '_course'))
		
		return recommended_courses[
			['CourseID', 'CourseName', 'ReviewNumber', 'AvgStar', 'Level', 'Unit', 'Results', 'EstimatedRating']
		], user_reviews[
			['CourseID', 'CourseName', 'ReviewNumber', 'AvgStar', 'Level', 'Unit', 'Results', 'RatingStar']
		]
	
	def process_pyspark(self, current_user_id, num_recommendations, mode, model_save_path):
		df_reviews, df_courses = self.data_preprocessing.pyspark_preprocessing(spark=self.spark)
		
		if mode == 'training':
			# Checkpoint the DataFrame
			df_reviews = df_reviews.checkpoint()
			
			# Ready the data for the ALS model, we only need the ReviewerID, CourseID, and RatingStar columns
			df_fit = df_reviews.select("NumericReviewerID", "NumericCourseID", "RatingStar")
			
			# Split the data into training and testing sets
			(training, test) = df_fit.randomSplit([0.8, 0.2])
			
			# Create the ALS model
			als = ALS(
				userCol="NumericReviewerID",
				itemCol="NumericCourseID",
				ratingCol="RatingStar",
				coldStartStrategy="drop",  # Set to 'drop' to ensure we don't get NaN evaluation metrics
				nonnegative=True,  # Ensures that the model does not output negative ratings
			)
			
			# Tune the model using ParamGridBuilder
			param_grid = (
				ParamGridBuilder()
				.addGrid(als.rank, [12, 14, 16])
				.addGrid(als.maxIter, [18, 20, 22])
				.addGrid(als.regParam, [0.18, 0.20, 0.22])
				.build()
			)
			
			# Define an evaluator
			evaluator = RegressionEvaluator(
				labelCol="RatingStar",
				predictionCol="prediction",
				metricName="rmse",
			)
			
			# Build cross-validation using TrainValidationSplit
			tvs = TrainValidationSplit(
				estimator=als,
				estimatorParamMaps=param_grid,
				evaluator=evaluator,
			)
			
			# Fit the model to the training data
			model = tvs.fit(training)
			
			# Take the best model from the tuning exercise using ParamGridBuilder
			best_model = model.bestModel
			
			# Save the best model
			best_model.write().overwrite().save(model_save_path)
			
			# Generate prediction and evaluate using RMSE
			predictions = best_model.transform(test)
			rmse = evaluator.evaluate(predictions)
			print(f'ALS model training complete.')
			print(f'Best model: {best_model}')
			print(f'Root Mean Squared Error (RMSE) = {str(rmse)}')
			print(f'- Rank: {best_model.rank}')
			print(f'- Max Iterations: {best_model._java_obj.parent().getMaxIter()}')
			print(f'- Regularization Parameter: {best_model._java_obj.parent().getRegParam()}')
			
			return None, None
		
		elif mode == 'predict':
			# Checkpoint the DataFrame
			df_reviews = df_reviews.checkpoint()
			
			# Convert current_user_id to numeric
			numeric_current_user_id = self.data_preprocessing.df_reviews.loc[
				self.data_preprocessing.df_reviews['ReviewerID'] == current_user_id, 'NumericReviewerID'
			].values[0]
			
			if numeric_current_user_id not in df_reviews.select("NumericReviewerID").distinct().rdd.flatMap(
					lambda x: x).collect():
				raise ValueError(f'User: "{current_user_id}" not found in the database. Please try another user.')
			
			# Load the saved model
			best_model = ALSModel.load(model_save_path)
			
			# Get the top N recommendations for the current user
			user_subset = df_reviews.filter(df_reviews.NumericReviewerID == numeric_current_user_id)
			user_recs = best_model.recommendForUserSubset(user_subset, num_recommendations)
			user_recs = user_recs.selectExpr("NumericReviewerID", "explode(recommendations) as rec")
			user_recs = user_recs.select("NumericReviewerID", "rec.NumericCourseID", "rec.rating")
			
			# Convert to pandas DataFrame
			user_recs_df = user_recs.toPandas()
			
			# Map numeric IDs back to original IDs
			id_map = self.data_preprocessing.df_reviews.set_index('NumericCourseID')['CourseID'].to_dict()
			user_recs_df['CourseID'] = user_recs_df['NumericCourseID'].map(id_map)
			
			# Merge with course details
			recommended_courses = user_recs_df.merge(df_courses.toPandas(), on='CourseID', how='left')
			
			# Get user reviews
			user_reviews = df_reviews.filter(df_reviews.NumericReviewerID == numeric_current_user_id).toPandas()
			user_reviews = user_reviews.merge(df_courses.toPandas(), on='CourseID', how='left')
			
			return recommended_courses[
				['CourseID', 'CourseName', 'ReviewNumber', 'AvgStar', 'Level', 'Unit', 'Results', 'rating']
			], user_reviews[
				['CourseID', 'CourseName_x', 'ReviewNumber', 'AvgStar', 'Level', 'Unit', 'Results', 'RatingStar']
			]
		
		else:
			raise ValueError(f'Mode: "{mode}" not recognized. Please use either "training" or "predict".')
		
	def _train_surprise(self, df_reviews, model_name='surprise_model.pkl'):
		reader = Reader(rating_scale=(1, 5))
		data = Dataset.load_from_df(df_reviews[['ReviewerID', 'CourseID', 'RatingStar']], reader)
		algo = SVD()
		cross_val_result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
		print(f'Cross-validation results: {cross_val_result}')
		train_set = data.build_full_trainset()
		algo.fit(train_set)
		self._save_model(algo, model_name)
		
		return algo
	
	def _save_model(self, model, model_name):
		model_path = os.path.join(self.model_dir, model_name)
		with open(model_path, 'wb') as f:
			pickle.dump(model, f)
	
	def _load_model(self, model_name):
		model_path = os.path.join(self.model_dir, model_name)
		if os.path.exists(model_path):
			with open(model_path, 'rb') as f:
				return pickle.load(f)
		return None
