"""
Module name: data_preprocessing.py
In this module, we will pre-process the data for all our operations.
This includes:
- Reading the data
- Data pre-processing for content-based filtering: Gensim and Cosine Similarity
- Data pre-processing for collaborative filtering: Surprise and PySpark ALS.
"""
import uuid
import pandas as pd

from gensim import corpora

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class DataPreprocessing:
	def __init__(
			self,
			path_courses: str = 'data/courses.csv',
			path_reviews: str = 'data/reviews.csv',
	):
		"""Initialize the paths for the data"""
		self.path_courses = path_courses
		self.path_reviews = path_reviews
		
		self.df_courses = None
		self.df_reviews = None
		
		self.download_nltk_resources()
	
	def load_data(self):
		"""Load the data from the specified paths"""
		if self.df_courses is None or self.df_reviews is None:
			self.df_courses = pd.read_csv(self.path_courses)
			self.df_reviews = pd.read_csv(self.path_reviews)
		return self.df_courses, self.df_reviews
	
	def gensim_preprocessing(self):
		"""Pre-process the data for Gensim model"""
		# Reload the data before processing
		self.load_data()
		
		# Tokenize the courses
		self.df_courses['Combined'] = (
				self.df_courses['CourseName'].fillna('') + ' ' +
				self.df_courses['Level'].fillna('') + ' ' +
				self.df_courses['Unit'].fillna('') + ' ' +
				self.df_courses['Results'].fillna('')
		)
		self.df_courses['Tokens'] = self.df_courses['Combined'].apply(self.tokenize)
		
		# Create a Corpus and Dictionary
		dictionary = corpora.Dictionary(self.df_courses['Tokens'])
		corpus = [dictionary.doc2bow(text) for text in self.df_courses['Tokens']]
		
		return self.df_courses, dictionary, corpus
	
	def cosine_similarity_preprocessing(self):
		"""Pre-process the data for Cosine Similarity model"""
		# Reload the data before processing
		self.load_data()
		
		# Combine text fields
		self.df_courses['Combined'] = (
				self.df_courses['CourseName'].fillna('') + ' ' +
				self.df_courses['Level'].fillna('') + ' ' +
				self.df_courses['Unit'].fillna('') + ' ' +
				self.df_courses['Results'].fillna('')
		)
		
		return self.df_courses
	
	def surprise_preprocessing(self):
		"""Pre-process the data for Surprise model"""
		# Ensure that CourseID and ReviewerID columns are present and consistent
		self.create_reviewers_and_courses_id()
		
		return self.df_reviews, self.df_courses
	
	def pyspark_preprocessing(self, spark):
		"""Pre-process the data for PySpark ALS model"""
		self.create_reviewers_and_courses_id()
		"""
		The ALS (Alternating Least Squares) algorithm in PySpark requires the user and item IDs to be numeric
		because it uses matrix factorization techniques, which operate on numerical matrices. The IDs are used
		as indexes in these matrices, hence they need to be integers.
		"""
		# Convert IDs to numeric
		self.df_reviews['NumericReviewerID'] = self.df_reviews['ReviewerID'].astype('category').cat.codes
		self.df_reviews['NumericCourseID'] = self.df_reviews['CourseID'].astype('category').cat.codes
		
		df_reviews = spark.createDataFrame(self.df_reviews)
		df_courses = spark.createDataFrame(self.df_courses)
		
		# Repartition the data
		df_reviews = df_reviews.repartition(100)
		df_courses = df_courses.repartition(100)
		
		# Cache the DataFrames
		df_reviews.cache()
		df_courses.cache()
		
		return df_reviews, df_courses
	
	def tokenize(self, text):
		"""Tokenize the text"""
		stop_words = set(stopwords.words('english'))
		tokens = word_tokenize(text.lower())
		tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
		return tokens
	
	def download_nltk_resources(self):
		"""Download NLTK resources if not already installed"""
		try:
			stopwords.words('english')
		except LookupError:
			nltk.download('stopwords')
			nltk.download('punkt')
		except AttributeError:
			nltk.download('stopwords')
			nltk.download('punkt')
		except Exception as e:
			raise e
	
	def get_all_user_ids(self):
		"""Get all unique ReviewerIDs and corresponding ReviewerNames"""
		self.load_data()
		self.create_reviewers_and_courses_id()
		user_ids = self.df_reviews[['ReviewerName', 'ReviewerID']].drop_duplicates()
		return user_ids
	
	def get_user_history(self, user_id):
		"""Get all courses that a user has reviewed"""
		self.load_data()
		self.create_reviewers_and_courses_id()
		user_courses = self.df_reviews[self.df_reviews['ReviewerID'] == user_id]
		return user_courses
	
	def get_all_courses(self):
		"""Get all available courses"""
		self.load_data()
		self.create_reviewers_and_courses_id()
		courses = self.df_courses.drop_duplicates()
		return courses
	
	def create_reviewers_and_courses_id(self):
		"""Create unique IDs for Reviewers and Courses"""
		self.load_data()
		
		# Ensure that CourseID and ReviewerID columns are present and consistent.
		# This is done using the setdefault method, which returns the value if the key exists, otherwise,
		# it sets the value and returns it.
		reviewer_id_map = {}
		course_id_map = {}
		
		if 'ReviewerID' not in self.df_reviews.columns:
			self.df_reviews['ReviewerID'] = self.df_reviews['ReviewerName'].apply(
				lambda name: reviewer_id_map.setdefault(name, uuid.uuid4().hex)
			)
			self.df_courses['CourseID'] = self.df_courses['CourseName'].apply(
				lambda name: course_id_map.setdefault(name, uuid.uuid4().hex)
			)
			self.df_reviews['CourseID'] = self.df_reviews['CourseName'].map(course_id_map)
			
			self.df_courses.to_csv(self.path_courses, index=False)
			self.df_reviews.to_csv(self.path_reviews, index=False)
	