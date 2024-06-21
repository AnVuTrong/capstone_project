import pandas as pd
import polars as pl
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import seaborn as sns


class PandasEDA:
	"""
	A class for performing Exploratory Data Analysis (EDA) on a DataFrame.

	How to use:
	eda = EDA(df_transaction)
	eda.run_all()
	"""
	
	def __init__(self, dataframe: pd.DataFrame):
		self.df = dataframe
	
	def summary_statistics(self):
		"""Generate summary statistics of the DataFrame."""
		try:
			print("Summary Statistics:")
			print(self.df.describe(include='all'))
		except Exception as e:
			print(f"Error in generating summary statistics: {e}")
	
	def check_missing_values(self):
		"""Check for missing values in the DataFrame."""
		try:
			missing_values = self.df.isnull().sum()
			print("Missing Values:")
			print(missing_values[missing_values > 0])
		except Exception as e:
			print(f"Error in checking missing values: {e}")
	
	def plot_distributions(self):
		"""Plot distributions of numerical columns."""
		try:
			numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
			for col in numerical_cols:
				plt.figure(figsize=(10, 5))
				sns.histplot(self.df[col], kde=True)
				plt.title(f'Distribution of {col}')
				plt.show()
		except Exception as e:
			print(f"Error in plotting distributions: {e}")
	
	def plot_correlations(self):
		"""Plot correlation matrix of numerical columns."""
		try:
			numerical_df = self.df.select_dtypes(include=['float64', 'int64'])
			if not numerical_df.empty:
				plt.figure(figsize=(12, 8))
				sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
				plt.title('Correlation Matrix')
				plt.show()
			else:
				print("No numerical columns to plot correlation matrix.")
		except Exception as e:
			print(f"Error in plotting correlations: {e}")
	
	def plot_categorical_counts(self):
		"""Plot count plots for categorical columns."""
		try:
			categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
			for col in categorical_cols:
				plt.figure(figsize=(10, 5))
				sns.countplot(x=col, data=self.df)
				plt.title(f'Count Plot of {col}')
				plt.xticks(rotation=45)
				plt.show()
		except Exception as e:
			print(f"Error in plotting categorical counts: {e}")
	
	def run_all(self):
		"""Run all EDA methods."""
		print("Running summary statistics...")
		self.summary_statistics()
		print("\nChecking missing values...")
		self.check_missing_values()
		print("\nPlotting distributions...")
		self.plot_distributions()
		print("\nPlotting correlations...")
		self.plot_correlations()
		print("\nPlotting categorical counts...")
		self.plot_categorical_counts()
