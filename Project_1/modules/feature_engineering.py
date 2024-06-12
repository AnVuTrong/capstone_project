import pandas as pd


class PandasFeatureEngineering:
	"""
	Class to perform feature engineering on RFM data for machine learning clustering.
	"""
	
	def __init__(self, rfm_df: pd.DataFrame):
		self.rfm_df = rfm_df
	
	def encode_customer_segment(self):
		# Define the order of segments
		segment_order = ['Lost', 'Hibernating', 'At Risk', 'New Customer', 'Average Customer', 'Loyalist',
		                 'Big Spender', 'VIP']
		# Create a mapping dictionary
		segment_mapping = {segment: idx for idx, segment in enumerate(segment_order)}
		# Encode segments
		self.rfm_df['Segment_Encoded'] = self.rfm_df['Segment'].map(segment_mapping)
	
	def run(self):
		self.encode_customer_segment()
		return self.rfm_df
	