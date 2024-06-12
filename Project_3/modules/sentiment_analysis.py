"""
In this module, we will implement the SentimentAnalysis class.
This class will be responsible for training and predicting the sentiment of the comments.
The class will have the following methods:
- Training: This method will train the traditional machine learning model, deep learning model,
and transfer learning Bert and pyspark ML, after training, the models will be saved for later use.
- Predict: This method will predict the sentiment of the comment,
the output will the predicted rating of the comment, to judge the sentiment of the user.
- Sentiment Summary: This method will return the average sentiment, number of positive/negative reviews.
"""
import pandas as pd
from tqdm import tqdm

from Project_3.modules.traditional_ml import TraditionalML
from Project_3.modules.pyspark_ml import PySparkML
from Project_3.modules.data_preprocessing import DataPreprocessing
from Project_3.modules.utils import plot_confusion_matrix as plot_cm


class SentimentAnalysis:
	def __init__(self, spark=None,
	             emoji_dict_path='data/files/unicode_all_emojis_vi.txt',
	             teen_dict_path='data/files/teencode.txt',
	             english_dict_path='data/files/english-vnmese.txt',
	             stopwords_path='data/files/vietnamese-stopwords.txt',
	             corrector_model_path='modules/VietnameseOcrCorrection/weights/seq2seq_0.pth'
	             ):
		if spark is not None:
			self.pyspark_ml = PySparkML(spark)
		else:
			self.pyspark_ml = None
		self.traditional_ml = TraditionalML()
		self.data_preprocessing = DataPreprocessing(
			emoji_dict_path=emoji_dict_path,
			teen_dict_path=teen_dict_path,
			english_dict_path=english_dict_path,
			stopwords_path=stopwords_path,
			corrector_model_path=corrector_model_path
		)
	
	def train_traditional_ml(self, df):
		traditional_ml_report = self.traditional_ml.train(df)
		return traditional_ml_report
	
	def train_pyspark_ml(self, df):
		pyspark_ml_report = self.pyspark_ml.train(df)
		return pyspark_ml_report
	
	def predict_traditional_ml(self, comments):
		comments = self.data_preprocessing.preprocess_predictions(comments)
		traditional_ml_predictions = self.traditional_ml.predict(comments)
		predictions = []
		for model_name, labels in tqdm(traditional_ml_predictions.items()):
			for label in labels:
				predictions.append({
					"model"     : model_name,
					"prediction": label
				})
		
		return pd.DataFrame(predictions)
	
	def predict_pyspark_ml(self, comments):
		comments = self.data_preprocessing.preprocess_predictions(comments)
		pyspark_ml_predictions = self.pyspark_ml.predict(comments)
		predictions = []
		for model_name, labels in tqdm(pyspark_ml_predictions.items()):
			for label in labels:
				predictions.append({
					"model"     : model_name,
					"prediction": label
				})
		
		return pd.DataFrame(predictions)
	
	def plot_confusion_matrix(self, cm, model_name):
		plot_cm(cm, model_name)
		return None
