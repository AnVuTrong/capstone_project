"""
In this module, we will implement the data preprocessing steps for the project.
It should include three types of data preprocessing scales:
1. Small-scale data preprocessing using Pandas
2. Large-scale data preprocessing using PySpark.
"""

# Import necessary libraries
import uuid
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.window import Window


class PandasDataPreprocessing:
	"""
	How to use:
	preprocessor = PandasDataPreprocessing('Products_with_Prices.csv', 'Transactions.csv')
	df_transaction, df_rfm = preprocessor.run()
	"""
	
	def __init__(self, product_file, transaction_file):
		self.product_file = product_file
		self.transaction_file = transaction_file
		self.df_product = None
		self.df_transaction = None
		self.df_rfm = None
	
	def load_data(self):
		if isinstance(self.product_file, str):
			self.df_product = pd.read_csv(self.product_file)
		else:
			self.df_product = self.product_file
		if isinstance(self.transaction_file, str):
			self.df_transaction = pd.read_csv(self.transaction_file)
		else:
			self.df_transaction = self.transaction_file
	
	def preprocess_data(self):
		# Merge product and transaction data
		self.df_transaction = self.df_transaction.merge(self.df_product[['productId', 'productName', 'price']],
		                                                on='productId', how='left')
		
		# Add uuid to ID the transaction
		self.df_transaction['TransactionID'] = [str(uuid.uuid4()) for _ in range(len(self.df_transaction))]
		
		# Calculate the total payment for each transaction
		self.df_transaction['TotalPayment'] = self.df_transaction['items'] * self.df_transaction['price']
		
		# Convert the date columns to datetime
		if 'Date' in self.df_transaction.columns:
			self.df_transaction['TransactionDate'] = pd.to_datetime(self.df_transaction['Date'], dayfirst=True)
			self.df_transaction.drop('Date', axis=1, inplace=True)
	
	def generate_rfm(self):
		latest_date = self.df_transaction['TransactionDate'].max()
		
		recency = lambda x: (latest_date - x.max()).days
		frequency = lambda x: len(x.unique())
		monetary = lambda x: round(x.sum(), 2)
		
		self.df_rfm = self.df_transaction.groupby('Member_number').agg(
			Recency=('TransactionDate', recency),
			Frequency=('TransactionID', frequency),
			Monetary=('TotalPayment', monetary)
		)
		
		self.df_rfm.columns = ['Recency', 'Frequency', 'Monetary']
		self.df_rfm.sort_values('Monetary', ascending=False, inplace=True)
		
		# Calculate RFM score
		self.df_rfm['RFM_Score'] = self.df_rfm[['Recency', 'Frequency', 'Monetary']].apply(
			lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5]).astype(int)).sum(axis=1)
		
		# Assign customer value
		self.df_rfm['Customer_Value'] = pd.cut(self.df_rfm['RFM_Score'], bins=[0, 5, 10, 15],
		                                       labels=['Low Value', 'Medium Value', 'High Value'])
	
	def run(self):
		self.load_data()
		self.preprocess_data()
		self.generate_rfm()
		return self.df_transaction, self.df_rfm


class PySparkDataPreprocessing:
	"""
	How to use:
	preprocessor = PySparkDataPreprocessing(spark, 'Products_with_Prices.csv', 'Transactions.csv')
	df_transaction, df_rfm = preprocessor.run()
	"""
	
	def __init__(self, spark: SparkSession, product_file: str, transaction_file: str):
		self.spark = spark
		self.product_file = product_file
		self.transaction_file = transaction_file
		self.df_product = None
		self.df_transaction = None
		self.df_rfm = None
	
	def load_data(self):
		self.df_product = self.spark.read.csv(self.product_file, header=True, inferSchema=True)
		self.df_transaction = self.spark.read.csv(self.transaction_file, header=True, inferSchema=True)
		return self.df_product, self.df_transaction
	
	def preprocess_data(self):
		# Merge product and transaction data
		self.df_transaction = self.df_transaction.join(self.df_product.select('productId', 'productName', 'price'),
		                                               on='productId', how='left')
		
		# Add uuid to ID the transaction
		uuid_udf = F.udf(lambda: str(uuid.uuid4()), StringType())
		self.df_transaction = self.df_transaction.withColumn('TransactionID', uuid_udf())
		
		# Calculate the total payment for each transaction
		self.df_transaction = self.df_transaction.withColumn('TotalPayment',
		                                                     self.df_transaction['items'] * self.df_transaction[
			                                                     'price'])
		
		# Rename the date columns to TransactionDate
		if 'Date' in self.df_transaction.columns:
			self.df_transaction = self.df_transaction.withColumn('TransactionDate',
			                                                     F.to_date(F.col('Date'), 'dd-MM-yyyy'))
			self.df_transaction = self.df_transaction.drop('Date')
	
	def generate_rfm(self):
		# Calculate the latest transaction date
		latest_date = self.df_transaction.agg(F.max('TransactionDate')).collect()[0][0]
		
		# Define the recency calculation
		recency_df = self.df_transaction.groupBy('Member_number').agg(
			F.datediff(F.lit(latest_date), F.max('TransactionDate')).alias('Recency')
		)
		
		# Define the frequency calculation
		frequency_df = self.df_transaction.groupBy('Member_number').agg(
			F.countDistinct('TransactionID').alias('Frequency')
		)
		
		# Define the monetary calculation
		monetary_df = self.df_transaction.groupBy('Member_number').agg(
			F.round(F.sum('TotalPayment'), 2).alias('Monetary')
		)
		
		# Combine all three metrics into a single DataFrame
		self.df_rfm = recency_df.join(frequency_df, on='Member_number').join(monetary_df, on='Member_number')
		
		# Calculate the quintiles for each RFM metric
		for metric in ['Recency', 'Frequency', 'Monetary']:
			quintiles = self.df_rfm.approxQuantile(metric, [0.2, 0.4, 0.6, 0.8], 0)
			self.df_rfm = self.df_rfm.withColumn(f'{metric}Score',
			                                     F.when(F.col(metric) <= quintiles[0], 1)
			                                     .when(F.col(metric) <= quintiles[1], 2)
			                                     .when(F.col(metric) <= quintiles[2], 3)
			                                     .when(F.col(metric) <= quintiles[3], 4)
			                                     .otherwise(5))
		
		# Calculate the RFM score
		self.df_rfm = self.df_rfm.withColumn('RFM_Score',
		                                     F.col('RecencyScore') + F.col('FrequencyScore') + F.col('MonetaryScore'))
		
		# Assign customer value
		self.df_rfm = self.df_rfm.withColumn('Customer_Value',
		                                     F.when(F.col('RFM_Score') <= 5, 'Low Value')
		                                     .when(F.col('RFM_Score') <= 10, 'Medium Value')
		                                     .otherwise('High Value'))
	
	def run(self):
		self.load_data()
		self.preprocess_data()
		self.generate_rfm()
		return self.df_transaction, self.df_rfm
