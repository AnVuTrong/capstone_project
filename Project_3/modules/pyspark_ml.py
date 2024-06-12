from sklearnex import patch_sklearn

patch_sklearn()

import os
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import LogisticRegression, NaiveBayes, RandomForestClassifier, DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm


class PySparkML:
	def __init__(self, spark=None):
		self.spark = spark
		self.models = {
			"logistic_regression": LogisticRegression(maxIter=1000),
			"naive_bayes"        : NaiveBayes(),
			"random_forest"      : RandomForestClassifier(numTrees=300, maxDepth=10),
			"decision_tree"      : DecisionTreeClassifier(maxDepth=10, maxBins=64),
		}
		self.model_path = "models/pyspark_ml/"
		self.vectorizer_path = "models/pyspark_ml/vectorizer.model"
		self.evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
		self.param_grid_builder = ParamGridBuilder()
		
		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)
	
	def train(self, df):
		df['Sentiment_ML'] = df['Rating'].apply(self._convert_rating_to_class)
		df = self._preprocess(df)
		df = df[['Comment', 'Sentiment_ML']]
		df = df.rename(columns={'Sentiment_ML': 'label'})
		
		df_spark = self.spark.createDataFrame(df)
		df_spark, vectorizer_model = self.feature_engineer(df_spark, fit=True)
		vectorizer_model.write().overwrite().save(self.vectorizer_path)
		df_spark.show()
		# Remove unnecessary columns: Comment, words, raw_features
		df_spark = df_spark.drop("Comment", "words", "c_vector")
		df_spark.show()
		
		# Repartition the dataframe
		df_spark = df_spark.repartition(200)
		
		train, test = df_spark.randomSplit([0.8, 0.2], seed=42)
		# Repartition the split dataframes
		train = train.repartition(300)
		test = test.repartition(300)
		
		reports = []
		for model_name, model in tqdm(self.models.items()):
			model_file = os.path.join(self.model_path, f"{model_name}.model")
			
			if os.path.exists(model_file):
				# Load the model if it already exists
				print(f"Loading model: {model_name}")
				model = PipelineModel.load(model_file)
			else:
				# Train the model if it doesn't exist
				print(f"Training model: {model_name}")
				pipeline = Pipeline(stages=[model])
				model = pipeline.fit(train)
				model.write().overwrite().save(model_file)
			
			# Generate report
			predictions = model.transform(test)
			report = self._generate_report(predictions, model_name)
			reports.append(report)
		
		return pd.concat(reports, ignore_index=True)
	
	def predict(self, texts):
		if not os.path.exists(self.model_path):
			raise FileNotFoundError("Models not found. Train the models first.")
		
		df = pd.DataFrame(texts, columns=['Comment'])
		df['label'] = [0.0 for _ in range(len(df))]
		df_spark = self.spark.createDataFrame(df)
		vectorizer_model = PipelineModel.load(self.vectorizer_path)
		df_spark = self.feature_engineer(df_spark, fit=False, vectorizer_model=vectorizer_model)
		
		results = {}
		for model_name in self.models.keys():
			model_file = os.path.join(self.model_path, f"{model_name}.model")
			model = PipelineModel.load(model_file)
			predictions = model.transform(df_spark)
			labels = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()
			results[model_name] = [self._convert_class_to_label(pred) for pred in labels]
		
		return results
	
	def feature_engineer(self, df, fit=False, vectorizer_model=None):
		tokenizer = Tokenizer(inputCol="Comment", outputCol="words")
		count_vectorizer = CountVectorizer(inputCol="words", outputCol="c_vector")
		idf = IDF(inputCol="c_vector", outputCol="features")
		
		if fit:
			stages = [tokenizer, count_vectorizer, idf]
			pipeline = Pipeline(stages=stages)
			model = pipeline.fit(df)
			return model.transform(df), model
		else:
			if vectorizer_model is None:
				raise ValueError("Vectorizer model must be provided for transformation.")
			return vectorizer_model.transform(df)
	
	def _generate_report(self, predictions, model_name):
		accuracy = self.evaluator.evaluate(predictions)
		y_true = predictions.select("label").rdd.flatMap(lambda x: x).collect()
		y_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()
		
		report = pd.DataFrame({
			"model"           : [model_name],
			"accuracy"        : [accuracy],
			"precision"       : [precision_score(y_true, y_pred, average='weighted', zero_division=0)],
			"recall"          : [recall_score(y_true, y_pred, average='weighted', zero_division=0)],
			"f1_score"        : [f1_score(y_true, y_pred, average='weighted', zero_division=0)],
			"confusion_matrix": [confusion_matrix(y_true, y_pred)]
		})
		
		return report
	
	def _preprocess(self, df):
		df = df.dropna(subset=['Comment'])
		df['Comment'] = df['Comment'].astype(str)
		return df
	
	def _convert_class_to_label(self, cls):
		labels = {
			0: "negative",
			1: "neutral",
			2: "positive",
		}
		return labels.get(cls, "unknown")
	
	def _convert_rating_to_class(self, rating):
		if rating < 5:
			return 0.0
		elif rating < 8:
			return 1.0
		else:
			return 2.0


if __name__ == "__main__":
	from pyspark.sql import SparkSession
	
	spark = SparkSession.builder \
		.appName("SentimentAnalysis") \
		.config("spark.driver.memory", "32g") \
		.config("spark.executor.memory", "32g") \
		.config("spark.sql.autoBroadcastJoinThreshold", -1) \
		.getOrCreate()
	sentiment_analysis = PySparkML(spark)
	
	df = pd.read_csv("../data/ready_data.csv")
	df = df[df['Comment'].notna()]
	
	try:
		reports = sentiment_analysis.train(df)
		print(reports)
		
		comments = ["Mấy món này ngon quá"]
		predictions = sentiment_analysis.predict(comments)
		print(predictions)
	except Exception as e:
		print(e)
	finally:
		print("Closing Spark session...")
		spark.stop()
