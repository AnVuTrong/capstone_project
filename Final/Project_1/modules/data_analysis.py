import pandas as pd
import polars as pl
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Ensure plots are displayed within Jupyter notebook
pio.renderers.default = 'notebook'


class PandasCustomerSegmentation:
	"""
    A class for performing customer segmentation and visualization.

    How to use:
    segmentation = PandasCustomerSegmentation(df_rfm, df_transaction)
    segmentation.run_all()
    """
	
	def __init__(self, rfm_dataframe: pd.DataFrame, transaction_dataframe: pd.DataFrame):
		self.df_rfm = rfm_dataframe
		self.df_transaction = transaction_dataframe
		self.segments = None
		self.quintile_breakpoints = {}
	
	def create_quintiles(self):
		"""Create quintiles for RFM scores and store their breakpoints."""
		self.df_rfm['Recency_Quintile'], recency_bins = (
			pd.qcut(self.df_rfm['Recency'], 5, labels=range(1, 6), retbins=True)
		)
		self.df_rfm['Frequency_Quintile'], frequency_bins = (
			pd.qcut(self.df_rfm['Frequency'], 5, labels=range(1, 6), retbins=True)
		)
		self.df_rfm['Monetary_Quintile'], monetary_bins = (
			pd.qcut(self.df_rfm['Monetary'], 5, labels=range(1, 6), retbins=True)
		)
		
		self.quintile_breakpoints = {
			'Recency'  : recency_bins,
			'Frequency': frequency_bins,
			'Monetary' : monetary_bins
		}
	
	def segment_customers(self):
		"""Segment customers based on RFM quintiles."""
		self.create_quintiles()
		
		# Remove the 'Segment' column if it already exists
		if 'Segment' in self.df_rfm.columns:
			self.df_rfm.drop('Segment', axis=1, inplace=True)
		
		self.df_rfm['Segment'] = 'Average Customer'
		
		# Define segmentation rules based on quintiles in priority order
		segments_rules = [
			('VIP',
			 (self.df_rfm['Recency_Quintile'] == 1) &
			 (self.df_rfm['Frequency_Quintile'] == 5) &
			 (self.df_rfm['Monetary_Quintile'] == 5)),
			('Loyalist',
			 (self.df_rfm['Recency_Quintile'] <= 2) &
			 (self.df_rfm['Frequency_Quintile'] == 5) &
			 (self.df_rfm['Monetary_Quintile'] >= 3)),
			('Big Spender',
			 (self.df_rfm['Monetary_Quintile'] == 5)),
			('New Customer',
			 (self.df_rfm['Recency_Quintile'] == 1) &
			 (self.df_rfm['Frequency_Quintile'] <= 2)),
			('At Risk',
			 (self.df_rfm['Recency_Quintile'] == 5) &
			 (self.df_rfm['Frequency_Quintile'] >= 3) &
			 (self.df_rfm['Monetary_Quintile'] >= 3)),
			('Hibernating',
			 (self.df_rfm['Recency_Quintile'] == 4) &
			 (self.df_rfm['Frequency_Quintile'] >= 2)),
			('Lost',
			 (self.df_rfm['Recency_Quintile'] == 5) &
			 (self.df_rfm['Frequency_Quintile'] <= 2))
		]
		
		# Assign segments based on rules
		for segment, condition in segments_rules:
			self.df_rfm.loc[condition & (self.df_rfm['Segment'] == 'Average Customer'), 'Segment'] = segment
		
		self.segments = self.df_rfm['Segment'].value_counts().reset_index()
		self.segments.columns = ['Segment', 'Count']
	
	def visualize_segments(self):
		"""Visualize the customer segments."""
		fig = px.bar(
			self.segments, x='Segment', y='Count', title='Customer Segments',
			color='Segment', text='Count', color_discrete_sequence=px.colors.qualitative.Prism
		)
		fig.update_layout(
			template='plotly_white', title_font_size=20, title_x=0.5,
			xaxis_title='', yaxis_title='Number of Customers'
		)
		fig.show()
	
	def visualize_rfm_distribution(self):
		"""Visualize the distribution of RFM scores."""
		fig = go.Figure()
		colors = ['indianred', 'lightseagreen', 'dodgerblue']
		for score, color in zip(['Recency', 'Frequency', 'Monetary'], colors):
			fig.add_trace(go.Box(y=self.df_rfm[score], name=score, marker_color=color))
		
		fig.update_layout(
			template='plotly_white', title='RFM Score Distribution',
			title_font_size=20, title_x=0.5
		)
		fig.show()
	
	def visualize_3d_rfm(self):
		"""Visualize the RFM scores in 3D."""
		fig = px.scatter_3d(
			self.df_rfm, x='Recency', y='Frequency', z='Monetary', color='Segment',
			title='3D Scatter Plot of RFM Scores', symbol='Segment'
		)
		fig.update_layout(template='plotly_white', title_font_size=20, title_x=0.5)
		fig.show()
	
	def visualize_3d_customers_value(self):
		"""Visualize the RFM scores and customer value in 3D."""
		fig = px.scatter_3d(
			self.df_rfm, x='Recency', y='Frequency', z='Monetary', color='Customer_Value',
			title='3D Scatter Plot of RFM Scores and Customer Value', symbol='Segment'
		)
		fig.update_layout(template='plotly_white', title_font_size=20, title_x=0.5)
		fig.show()
	
	def visualize_spending_contribution(self):
		"""Visualize the contribution of each segment to total spending."""
		segment_spending = self.df_rfm.groupby('Segment')['Monetary'].sum().reset_index()
		fig = px.pie(
			segment_spending, names='Segment', values='Monetary',
			title='Contribution to Total Spending by Segment',
			color_discrete_sequence=px.colors.qualitative.Prism
		)
		fig.update_layout(template='plotly_white', title_font_size=20, title_x=0.5)
		fig.show()
	
	def visualize_best_products(self):
		"""Visualize the best products based on total spending in transaction data."""
		# Aggregate total spending by product
		product_sales = (
			self.df_transaction.groupby('productName')['TotalPayment']
			.sum().reset_index().sort_values(by='TotalPayment', ascending=False)
		)
		
		# Visualize the top 10 products based on total spending
		fig = px.bar(
			product_sales.head(10), x='productName', y='TotalPayment',
			title='Top 10 Best-Selling Products by Total Spending',
			color='productName', text='TotalPayment', color_discrete_sequence=px.colors.qualitative.Prism
		)
		fig.update_layout(
			template='plotly_white', title_font_size=20, title_x=0.5,
			xaxis_title='Product', yaxis_title='Total Sales'
		)
		fig.show()
	
	def visualize_segment_preferences(self):
		"""Visualize the products are preferred by each customer segment."""
		# Merge the transaction data with the RFM data to get the segments.
		df_merged = pd.merge(self.df_transaction, self.df_rfm[['Segment']], left_on='Member_number', right_index=True)
		segment_product_sales = (
			df_merged.groupby(['Segment', 'productName'])['TotalPayment']
			.sum().reset_index().sort_values(by='TotalPayment', ascending=False)
		)
		
		# Create a bar plot for each segment
		fig = px.bar(
			segment_product_sales, x='productName', y='TotalPayment', color='Segment', text='TotalPayment',
			title='Product Preferences by Customer Segment', facet_row='Segment',
			color_discrete_sequence=px.colors.qualitative.Prism
		)
		
		fig.update_layout(
			template='plotly_white', title_font_size=20, title_x=0.5,
			height=300 * segment_product_sales['Segment'].nunique(),
			xaxis_title='Product', yaxis_title='Total Spending'
		)
		
		fig.show()
	
	def visualize_trending_product(self):
		"""Visualize the most recently trending product and its revenue contribution."""
		# Convert transaction dates to datetime if not already done
		if not pd.api.types.is_datetime64_any_dtype(self.df_transaction['TransactionDate']):
			self.df_transaction['TransactionDate'] = pd.to_datetime(self.df_transaction['TransactionDate'])
		
		# Determine the latest date in the dataset
		latest_date = self.df_transaction['TransactionDate'].max()
		
		# Define the recent period for trending analysis (e.g., last 30 days from the latest date in the dataset)
		recent_period = latest_date - pd.DateOffset(days=30)
		recent_transactions = self.df_transaction[self.df_transaction['TransactionDate'] >= recent_period]
		
		# Calculate total spending for each product in the recent period
		recent_product_sales = (
			recent_transactions.groupby('productName')['TotalPayment']
			.sum().reset_index().sort_values(by='TotalPayment', ascending=False)
		)
		
		# Identify the most trending product
		trending_product = recent_product_sales.iloc[0]['productName']
		trending_product_revenue = recent_product_sales.iloc[0]['TotalPayment']
		
		# Calculate the total revenue for the shop in the recent period.
		total_revenue_recent_period = recent_transactions['TotalPayment'].sum()
		
		# Calculate the contribution percentage of the trending product
		contribution_percentage = (trending_product_revenue / total_revenue_recent_period) * 100
		
		# Create a bar plot to visualize the trending product revenue contribution.
		fig = go.Figure()
		
		fig.add_trace(go.Bar(
			x=[trending_product],
			y=[trending_product_revenue],
			name='Trending Product Revenue',
			marker_color='indianred'
		))
		
		fig.add_trace(go.Bar(
			x=['Total Revenue in Recent Period'],
			y=[total_revenue_recent_period],
			name='Total Revenue',
			marker_color='lightseagreen'
		))
		
		fig.update_layout(
			template='plotly_white',
			title='Most Recently Trending Product and its Revenue Contribution',
			title_font_size=20,
			title_x=0.5,
			xaxis_title='',
			yaxis_title='Total Spending',
			annotations=[
				dict(
					x=trending_product,
					y=trending_product_revenue,
					text=f"{contribution_percentage:.2f}%",
					showarrow=True,
					arrowhead=2,
					ax=0,
					ay=-40
				)
			]
		)
		
		fig.show()
	
	def visualize_quintile_breakpoints(self):
		"""Visualize the quintile breakpoints for RFM scores."""
		quintiles_df = pd.DataFrame(self.quintile_breakpoints).melt(var_name='RFM Metric', value_name='Breakpoint')
		fig = px.bar(
			quintiles_df, x='RFM Metric', y='Breakpoint', color='RFM Metric', text='Breakpoint',
			title='Quintile Breakpoints for RFM Scores',
			color_discrete_sequence=px.colors.qualitative.Prism
		)
		fig.update_layout(template='plotly_white', title_font_size=20, title_x=0.5)
		fig.show()
	
	def run_all(self):
		"""Run all methods to segment and visualize customer behaviors."""
		self.segment_customers()
		self.visualize_quintile_breakpoints()
		self.visualize_segments()
		self.visualize_rfm_distribution()
		self.visualize_3d_rfm()
		self.visualize_3d_customers_value()
		self.visualize_spending_contribution()
		self.visualize_best_products()
		self.visualize_segment_preferences()
		self.visualize_trending_product()
