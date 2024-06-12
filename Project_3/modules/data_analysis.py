"""
In this module, we will implement the data analysis steps for the project.
The dataset is the reviews of restaurants in the Shopee Food dataset.
Data Analysis Steps:
- Load the data.
- Calculate the average rating for the dataset.
- Calculate the average rating for each restaurant.
- Calculate the average rating over time for each restaurant.
- Check for the correlation between the average rating with the price range.
- Check for the correlation between the average rating with the review count.
- Check for the correlation between time of the review with the average rating.
All steps will output the visualizations and insights.
"""

# Import necessary libraries
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from textblob import TextBlob
from wordcloud import WordCloud

class DataAnalysis:
	def __init__(self, df):
		self.df = df
	
	def calculate_dataset_average_rating(self):
		# Calculate the average rating for the dataset and display it as a bell curve plot
		plt.figure(figsize=(10, 6))
		sns.histplot(self.df['Rating'], kde=True)
		plt.title('Average Rating Distribution')
		plt.xlabel('Rating')
		plt.ylabel('Frequency')
		plt.show()
	
	def calculate_average_rating(self):
		# Calculate the average rating for each restaurant
		avg_rating = self.df.groupby('Restaurant')['Rating'].mean().reset_index()
		
		# Plot the average rating and display top 10 restaurants of most and least ratings
		plt.figure(figsize=(10, 6))
		sns.barplot(x='Rating', y='Restaurant', data=avg_rating.sort_values('Rating', ascending=False).head(10))
		plt.title('Top 10 Restaurants with Highest Ratings')
		plt.xlabel('Average Rating')
		plt.ylabel('Restaurant')
		plt.show()
		
		plt.figure(figsize=(10, 6))
		sns.barplot(x='Rating', y='Restaurant', data=avg_rating.sort_values('Rating', ascending=True).head(10))
		plt.title('Top 10 Restaurants with Lowest Ratings')
		plt.xlabel('Average Rating')
		plt.ylabel('Restaurant')
		plt.show()
	
	def calculate_average_rating_over_time(self):
		# Ensure ReviewTime is in datetime format
		self.df['ReviewTime'] = pd.to_datetime(self.df['ReviewTime'], format='%Y-%m-%d %H:%M:%S')
		
		# Create a new column ReviewMonth from ReviewTime
		self.df['ReviewMonth'] = self.df['ReviewTime'].dt.to_period('M')
		
		# Calculate the average rating over time for the entire dataset
		avg_rating_time = self.df.groupby('ReviewMonth')['Rating'].mean().reset_index()
		
		# Convert ReviewMonth back to datetime for plotting
		avg_rating_time['ReviewMonth'] = avg_rating_time['ReviewMonth'].dt.to_timestamp()
		
		# Plot average rating over time for the entire dataset
		plt.figure(figsize=(12, 8))
		sns.lineplot(data=avg_rating_time, x='ReviewMonth', y='Rating')
		plt.xticks(rotation=45)
		plt.title('Average Rating Over Time')
		plt.xlabel('Month')
		plt.ylabel('Average Rating')
		plt.show()
	
	def correlation_with_price_range(self):
		# Calculate the correlation
		correlation = self.df[['Rating', 'MinPrice', 'MaxPrice']].corr().iloc[0, 1:]
		logging.info(f"Correlation between average rating and price range: {correlation}")
		
		# Plot the correlation for Min Price
		plt.figure(figsize=(10, 6))
		sns.scatterplot(x='MinPrice', y='Rating', data=self.df)
		plt.title('Correlation between Min Price and Average Rating')
		plt.xlabel('Min Price (VNĐ)')
		plt.ylabel('Average Rating')
		# Format the x-axis labels as VNĐ
		plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,} VNĐ"))
		plt.xticks(rotation=30)
		plt.show()
		
		# Plot the correlation for Max Price
		plt.figure(figsize=(10, 6))
		sns.scatterplot(x='MaxPrice', y='Rating', data=self.df)
		plt.title('Correlation between Max Price and Average Rating')
		plt.xlabel('Max Price (VNĐ)')
		plt.ylabel('Average Rating')
		# Format the x-axis labels as VNĐ
		plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,} VNĐ"))
		plt.show()
	
	def correlation_with_review_count(self):
		correlation = self.df[['Rating', 'ReviewCount']].corr().iloc[0, 1]
		logging.info(f"Correlation between average rating and review count: {correlation}")
		
		# Plot the correlation
		plt.figure(figsize=(10, 6))
		sns.scatterplot(x='ReviewCount', y='Rating', data=self.df)
		plt.title('Correlation between Review Count and Average Rating')
		plt.xlabel('Review Count')
		plt.ylabel('Average Rating')
		plt.show()
	
	def correlation_time_of_review(self):
		self.df['ReviewHour'] = self.df['ReviewTime'].dt.hour
		correlation = self.df[['Rating', 'ReviewHour']].corr().iloc[0, 1]
		logging.info(f"Correlation between time of review and rating: {correlation}")
		
		# Plot the correlation
		plt.figure(figsize=(10, 6))
		sns.scatterplot(x='ReviewHour', y='Rating', data=self.df)
		plt.title('Correlation between Time of Review and Rating')
		plt.xlabel('Hour of the Day')
		plt.ylabel('Rating')
		plt.show()
	
	def distribution_ratings_by_time_of_day(self):
		# Ensure ReviewTime is in datetime format
		self.df['ReviewTime'] = pd.to_datetime(self.df['ReviewTime'], format='%Y-%m-%d %H:%M:%S')
		self.df['ReviewHour'] = self.df['ReviewTime'].dt.hour
		
		# Plot the distribution of ratings by time of day
		plt.figure(figsize=(14, 8))
		sns.boxplot(x='ReviewHour', y='Rating', data=self.df)
		plt.title('Distribution of Ratings by Time of Day')
		plt.xlabel('Hour of the Day')
		plt.ylabel('Rating')
		plt.show()
	
	def plot_3d_scatter(self):
		fig = plt.figure(figsize=(12, 8))
		ax = fig.add_subplot(111, projection='3d')
		
		x = self.df['MinPrice']
		y = self.df['MaxPrice']
		z = self.df['Rating']
		c = self.df['ReviewCount']
		
		sc = ax.scatter(x, y, z, c=c, cmap='viridis')
		
		ax.set_xlabel('Min Price (VNĐ)', labelpad=20)
		ax.set_ylabel('Max Price (VNĐ)', labelpad=20)
		ax.set_zlabel('Rating')
		plt.colorbar(sc, label='Review Count')
		
		ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,} VNĐ"))
		ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, loc: f"{int(y):,} VNĐ"))
		
		plt.title('3D Scatter Plot of Ratings vs. Price Range vs. Review Count')
		plt.show()
	
	def plot_correlation_heatmap(self):
		plt.figure(figsize=(10, 8))
		correlation_matrix = self.df[['Rating', 'MinPrice', 'MaxPrice', 'ReviewCount']].corr()
		sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
		plt.title('Correlation Matrix Heatmap')
		plt.show()
	
	def simple_sentiment_analysis(self):
		# Ensure comments are strings, replace non-strings with empty strings
		self.df['Comment'] = self.df['Comment'].fillna('').astype(str)
		
		# Apply TextBlob sentiment analysis
		self.df['Sentiment'] = self.df['Comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
		
		# Plot the sentiment polarity distribution
		plt.figure(figsize=(10, 6))
		sns.histplot(self.df['Sentiment'], bins=20, kde=True)
		plt.title('Simple Sentiment Polarity Distribution')
		plt.xlabel('Simple Sentiment Polarity')
		plt.ylabel('Frequency')
		plt.show()
	
	def compare_ratings_by_district(self):
		# Calculate the average rating for each district
		avg_rating_district = self.df.groupby('District')['Rating'].mean().reset_index()
		
		# Sort the data for better visualization
		avg_rating_district = avg_rating_district.sort_values('Rating', ascending=False)
		
		# Plot the average rating for each district
		plt.figure(figsize=(14, 8))
		sns.barplot(x='Rating', y='District', data=avg_rating_district, hue='District', palette='viridis')
		plt.title('Average Rating by District')
		plt.xlabel('Average Rating')
		plt.ylabel('District')
		plt.show()
	
	def average_review_per_district_trailing_days(self, window_size):
		"""
		Calculate and plot the average review per district over a trailing window of specified days.

		Parameters:
		window_size (int): The size of the trailing window in days.
		"""
		# Ensure ReviewTime is in datetime format
		self.df['ReviewTime'] = pd.to_datetime(self.df['ReviewTime'], format='%Y-%m-%d %H:%M:%S')
		
		# Create a new column for the end of the trailing period
		self.df['ReviewMonth'] = self.df['ReviewTime'].dt.to_period('M').dt.to_timestamp()
		
		# Sort the dataframe by ReviewTime to ensure rolling calculations are correct
		self.df = self.df.sort_values('ReviewTime')
		
		# Calculate the trailing average rating for each district based on the specified window size
		self.df['RatingRolling'] = self.df.groupby('District')['Rating'].transform(
			lambda x: x.rolling(window=window_size, min_periods=1).mean())
		
		# Group by District and ReviewMonth to get the average rolling rating
		avg_rating_district_time = self.df.groupby(['District', 'ReviewMonth'])['RatingRolling'].mean().reset_index()
		
		# Plot average rating per district over time
		plt.figure(figsize=(14, 8))
		sns.lineplot(data=avg_rating_district_time, x='ReviewMonth', y='RatingRolling', hue='District')
		plt.xticks(rotation=45)
		plt.title(f'Average Review per District (Trailing {window_size} Days)')
		plt.xlabel('Month')
		plt.ylabel(f'Average Rating (Trailing {window_size} Days)')
		plt.legend(title='District', bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.show()
	
	def generate_word_cloud(self):
		# Combine all comments into one large string
		comment_text = ' '.join(self.df['Comment'].fillna('').astype(str))
		
		# Generate a word cloud image
		wordcloud = WordCloud(background_color='white', max_words=100, contour_color='steelblue').generate(comment_text)
		
		# Display the generated image
		plt.figure(figsize=(10, 6))
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis('off')
		plt.show()
	
	def run(self):
		self.calculate_dataset_average_rating()
		self.calculate_average_rating()
		self.calculate_average_rating_over_time()
		self.correlation_with_price_range()
		self.correlation_with_review_count()
		self.correlation_time_of_review()
		self.distribution_ratings_by_time_of_day()
		self.plot_3d_scatter()
		self.plot_correlation_heatmap()
		self.simple_sentiment_analysis()
		self.compare_ratings_by_district()
		self.average_review_per_district_trailing_days(30*6)
		self.generate_word_cloud()
		