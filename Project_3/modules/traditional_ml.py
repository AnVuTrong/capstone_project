from sklearnex import patch_sklearn
patch_sklearn()

import joblib
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	confusion_matrix,
	classification_report,
)
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

tqdm.pandas()


class TraditionalML:
	def __init__(self):
		self.models = {
			"logistic_regression": LogisticRegression(max_iter=5000, class_weight='balanced', n_jobs=-1),
			"naive_bayes"        : MultinomialNB(),
			"xgboost"            : XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
			"svm"                : SVC(class_weight='balanced'),
			"gradient_boosting"  : GradientBoostingClassifier(random_state=42),
			"ada_boost"          : AdaBoostClassifier(random_state=42),
		}
		self.vectorizer = TfidfVectorizer()
		self.smote = SMOTE(random_state=42)
		self.model_path = "models/traditional_ml/"
		self.vectorizer_path = os.path.join(self.model_path, "vectorizer.pkl")
	
	def train(self, df):
		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)
		
		df = self.feature_engineer(df)
		
		X = self.vectorizer.fit_transform(df['Comment'])
		y = df['Sentiment_ML']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		X_train_res, y_train_res = self.smote.fit_resample(X_train, y_train)
		
		# Save the vectorizer
		joblib.dump(self.vectorizer, self.vectorizer_path)
		
		reports = []
		for model_name, model in tqdm(self.models.items()):
			model_file = os.path.join(self.model_path, f"{model_name}.pkl")
			if os.path.exists(model_file):
				print(f"Model {model_name} already trained. Loading...")
				model = joblib.load(model_file)
			else:
				model.fit(X_train_res, y_train_res)
				joblib.dump(model, model_file)
			
			y_pred = model.predict(X_test)
			report = self._generate_report(y_test, y_pred, model_name)
			reports.append(report)
		
		return pd.concat(reports, ignore_index=True)
	
	def predict(self, texts):
		if not os.path.exists(self.vectorizer_path):
			raise FileNotFoundError("Vectorizer not found. Train the model first.")
		
		# Load the vectorizer
		self.vectorizer = joblib.load(self.vectorizer_path)
		
		X = self.vectorizer.transform(texts)
		results = {}
		for model_name in tqdm(self.models.keys()):
			model_file = os.path.join(self.model_path, f"{model_name}.pkl")
			model = joblib.load(model_file)
			predictions = model.predict(X)
			results[model_name] = [self._convert_class_to_label(pred) for pred in predictions]
		
		return results
	
	def _generate_report(self, y_true, y_pred, model_name):
		accuracy = accuracy_score(y_true, y_pred)
		precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
		recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
		f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
		cm = confusion_matrix(y_true, y_pred)
		classification_rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
		
		report = pd.DataFrame({
			"model"                : [model_name],
			"accuracy"             : [accuracy],
			"precision"            : [precision],
			"recall"               : [recall],
			"f1_score"             : [f1],
			"confusion_matrix"     : [cm],
			"classification_report": [classification_rep]
		})
		
		return report
	
	def feature_engineer(self, df):
		df = self._preprocess(df)
		return df
	
	def _preprocess(self, df):
		df = df.dropna(subset=['Comment'])
		df['Comment'] = df['Comment'].astype(str)
		df['Sentiment_ML'] = df['Rating'].progress_apply(self._convert_rating_to_class)
		return df
	
	def _convert_rating_to_class(self, rating):
		if rating < 5:
			return 0
		elif rating < 8:
			return 1
		else:
			return 2
	
	def _convert_class_to_label(self, cls):
		labels = {
			0: "negative",
			1: "neutral",
			2: "positive",
		}
		return labels.get(cls, "unknown")
	
	def plot_confusion_matrix(self, cm, model_name):
		plt.figure(figsize=(10, 7))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
		            xticklabels=["negative", "neutral", "positive"],
		            yticklabels=["negative", "neutral", "positive"])
		plt.title(f'Confusion Matrix for {model_name}')
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		plt.show()
