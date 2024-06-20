import streamlit as st
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import squarify
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from GUI.modules.widgets import Widgets
from Project_1.modules.data_preprocessing import PandasDataPreprocessing


class CustomerSegmentation:
	def __init__(self):
		self.header = "Customer Segmentation"
		self.widgets = Widgets()
	
	def gen_page(self):
		# Header
		self._header()
		
		# Data Entry
		products_with_prices, transactions = self._download_sample_csv_button()
		uploaded_products_file, uploaded_transactions_file = self._upload_dataset()
		
		# Check for compatibility
		self._check_for_compatibility(
			uploaded_products_file,
			uploaded_transactions_file,
			products_with_prices,
			transactions,
		)
	
	def _header(self):
		st.header(self.header)
		st.divider()
		st.write(
			"This application is designed for stores, aiming to assist them in clustering customers and aiding in the development of effective business strategies.")
		st.markdown('<h3 style="color:#A4C3A2;">Input Data</h3>', unsafe_allow_html=True)
		st.write(
			"Please download the following two sample CSV files to fill in your store data, so we can assist you with data analysis.")
	
	def _download_sample_csv_button(self):
		# Read the sample csv files
		products_with_prices = pd.read_csv('Project_1/data/Products_with_Prices.csv')
		transactions = pd.read_csv('Project_1/data/Transactions.csv')
		
		# Convert dataframes to CSV
		products_with_prices_csv = products_with_prices.to_csv(index=False)
		transactions_csv = transactions.to_csv(index=False)
		
		# Create download buttons for the sample CSV files
		st.download_button(label="Download products_with_prices.csv", data=products_with_prices_csv,
		                   file_name='products_with_prices.csv', mime='text/csv')
		st.download_button(label="Download transactions.csv", data=transactions_csv, file_name='transactions.csv',
		                   mime='text/csv')
		
		return products_with_prices, transactions
	
	def _upload_dataset(self):
		st.write("Please upload your dataset.")
		uploaded_products_file = st.file_uploader("Upload products_with_prices.csv", type="csv")
		uploaded_transactions_file = st.file_uploader("Upload transactions.csv", type="csv")
		
		return uploaded_products_file, uploaded_transactions_file
	
	def _check_for_compatibility(
			self,
			uploaded_products_file,
			uploaded_transactions_file,
			products_with_prices,
			transactions
	):
		if uploaded_products_file and uploaded_transactions_file:
			products_df = pd.read_csv(uploaded_products_file)
			transactions_df = pd.read_csv(uploaded_transactions_file)
			
			st.write("Uploaded products_with_prices.csv:")
			st.write(products_df)
			
			st.write("Uploaded transactions.csv:")
			st.write(transactions_df)
			
			products_match = set(products_df.columns) == set(products_with_prices.columns)
			transactions_match = set(transactions_df.columns) == set(transactions.columns)
			
			if products_match and transactions_match:
				st.success("Both uploaded files match the sample CSV structures.")
				self._input(products_df, transactions_df)
			else:
				if not products_match:
					st.error(
						"Uploaded products_with_prices.csv does not match the sample CSV structure. Please make sure your file has the correct columns and upload again.")
				if not transactions_match:
					st.error(
						"Uploaded transactions.csv does not match the sample CSV structure. Please make sure your file has the correct columns and upload again.")
				st.write("Please re-upload the correct files.")
	
	def _input(self, products_df, transactions_df):
		st.markdown('<h3 style="color:#A4C3A2;">Data analysis</h3>', unsafe_allow_html=True)
		preprocessor = PandasDataPreprocessing(products_df, transactions_df)
		df_transaction, df_rfm = preprocessor.run()
		
		# Change TotalPayment to Totalsales
		df_transaction.rename(columns={'TotalPayment': 'Totalsales'}, inplace=True)
		
		with st.expander("Please select which analysis you would like to perform"):
			with st.form(key='analysis_form'):
				show_dataframe = st.checkbox("Show Dataframe", value=False, key='show_dataframe')
				show_summary = st.checkbox("Show Summary", value=False, key='show_summary')
				show_graph1 = st.checkbox("Show Graph of insight business", value=False, key='show_graph1')
				show_graph2 = st.checkbox("Show Graph of Recency, Frequency and Monetary", value=False,
				                          key='show_graph2')
				submit = st.form_submit_button("Run Analysis")
			
			if submit:
				if show_dataframe:
					st.write("Dataframe:")
					st.write(df_transaction)
					st.write("RFM Dataframe:")
					st.write(df_rfm)
				if show_summary:
					st.write("Summary Statistics:")
					st.write(df_transaction.describe())
				if show_graph1:
					st.write("Graph of insight business:")
					st.write("Wordcloud product names:")
					self._wordcloud(df_transaction)
					st.write("Top 10 Products by Total Sales:")
					self.plot_top_10_products(df_transaction)
					st.write("Bottom 10 Products by Total Sales:")
					self.plot_bottom_10_products(df_transaction)
					st.write("Top 10 Members by Total Sales:")
					self.plot_top_10_members(df_transaction)
					st.write("Bottom 10 Members by Total Sales:")
					self.plot_bottom_10_members(df_transaction)
				if show_graph2:
					st.write("Graph of Recency, Frequency and Monetary:")
					self.show_rfm_distribution(df_rfm)
		
		if st.button("Continue to Clustering"):
			st.session_state.show_clustering = True
		
		if st.session_state.get('show_clustering', False):
			self.clustering(df_rfm)
	
	def _wordcloud(self, df_transaction):
		product_names = " ".join(df_transaction['productName'].values)
		wordcloud = WordCloud(width=800, height=400, background_color='white').generate(product_names)
		plt.figure(figsize=(10, 5))
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis('off')
		st.pyplot(plt)
	
	def show_rfm_distribution(self, df_rfm):
		plt.figure(figsize=(10, 15))
		
		plt.subplot(3, 1, 1)
		plt.hist(df_rfm['Recency'], bins=20, edgecolor='black')
		plt.title('Distribution of Recency')
		plt.xlabel('Recency')
		
		plt.subplot(3, 1, 2)
		plt.hist(df_rfm['Frequency'], bins=20, edgecolor='black')
		plt.title('Distribution of Frequency')
		plt.xlabel('Frequency')
		
		plt.subplot(3, 1, 3)
		plt.hist(df_rfm['Monetary'], bins=20, edgecolor='black')
		plt.title('Distribution of Monetary')
		plt.xlabel('Monetary')
		
		plt.tight_layout()
		st.pyplot(plt)
	
	def plot_top_10_products(self, df_transaction):
		product_sales = df_transaction.groupby('productName')['Totalsales'].sum().reset_index()
		top_10_products = product_sales.sort_values(by='Totalsales', ascending=False).head(10)
		plt.figure(figsize=(12, 6))
		sns.barplot(data=top_10_products, x='Totalsales', y='productName', palette='viridis')
		plt.title('Top 10 Products by Total Sales')
		plt.xlabel('Total Sales')
		plt.ylabel('Product Name')
		plt.tight_layout()
		st.pyplot(plt)
	
	def plot_bottom_10_products(self, df_transaction):
		product_sales = df_transaction.groupby('productName')['Totalsales'].sum().reset_index()
		bottom_10_products = product_sales.sort_values(by='Totalsales', ascending=True).head(10)
		plt.figure(figsize=(12, 6))
		sns.barplot(data=bottom_10_products, x='Totalsales', y='productName', palette='viridis')
		plt.title('Bottom 10 Products by Total Sales')
		plt.xlabel('Total Sales')
		plt.ylabel('Product Name')
		plt.tight_layout()
		st.pyplot(plt)
	
	def plot_top_10_members(self, df_transaction):
		member_sales = df_transaction.groupby('Member_number')['Totalsales'].sum().reset_index()
		top_10_members = member_sales.sort_values(by='Totalsales', ascending=False).head(10)
		plt.figure(figsize=(12, 6))
		sns.barplot(data=top_10_members, x='Totalsales', y='Member_number', palette='viridis')
		plt.title('Top 10 Members by Total Sales')
		plt.xlabel('Total Sales')
		plt.ylabel('Member Number')
		plt.tight_layout()
		st.pyplot(plt)
	
	def plot_bottom_10_members(self, df_transaction):
		member_sales = df_transaction.groupby('Member_number')['Totalsales'].sum().reset_index()
		bottom_10_members = member_sales.sort_values(by='Totalsales', ascending=True).head(10)
		plt.figure(figsize=(12, 6))
		sns.barplot(data=bottom_10_members, x='Totalsales', y='Member_number', palette='viridis')
		plt.title('Bottom 10 Members by Total Sales')
		plt.xlabel('Total Sales')
		plt.ylabel('Member Number')
		plt.tight_layout()
		st.pyplot(plt)
	
	def clustering(self, df_rfm):
		st.markdown('<h3 style="color:#A4C3A2;">Clustering</h3>', unsafe_allow_html=True)
		n_clusters = st.selectbox("Number of Clusters", [3, 4, 5], index=2, key='num_clusters_clustering')
		clustering_method = st.radio("Choose Clustering Method", ("KMeans", "Hierarchical Clustering"),
		                             key='clustering_method')
		
		if clustering_method == "KMeans":
			self.kmeans_clustering(df_rfm, n_clusters)
		else:
			self.hierarchical_clustering(df_rfm, n_clusters)
	
	def kmeans_clustering(self, df_rfm, n_clusters):
		kmeans = KMeans(n_clusters=n_clusters, random_state=42)
		df_rfm['Cluster'] = kmeans.fit_predict(df_rfm[['Recency', 'Frequency', 'Monetary']])
		
		rfm_agg = df_rfm.groupby('Cluster').agg({
			'Recency'  : 'mean',
			'Frequency': 'mean',
			'Monetary' : ['mean', 'count']}).round(0)
		
		rfm_agg.columns = rfm_agg.columns.droplevel()
		rfm_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
		rfm_agg['Percent'] = round((rfm_agg['Count'] / rfm_agg.Count.sum()) * 100, 2)
		rfm_agg = rfm_agg.reset_index()
		rfm_agg['Cluster'] = 'Cluster ' + rfm_agg['Cluster'].astype('str')
		
		st.write("Cluster Centers:")
		st.write(rfm_agg)
		
		plt.figure(figsize=(12, 8))
		sns.scatterplot(data=df_rfm, x='Recency', y='Monetary', hue='Cluster', palette='viridis', s=100)
		plt.title('KMeans Clustering of Customers')
		plt.xlabel('Recency')
		plt.ylabel('Monetary')
		plt.legend(title='Cluster')
		st.pyplot(plt)
		
		colors = sns.color_palette("husl", n_clusters).as_hex()
		colors_dict2 = {f'Cluster {i}': colors[i] for i in range(n_clusters)}
		
		fig, ax = plt.subplots(figsize=(14, 10))
		squarify.plot(sizes=rfm_agg['Count'],
		              text_kwargs={'fontsize': 12, 'weight': 'bold', 'fontname': "sans serif"},
		              color=[colors_dict2[f'Cluster {i}'] for i in range(n_clusters)],
		              label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(
			              *rfm_agg.iloc[i])
			              for i in range(n_clusters)], alpha=0.5)
		plt.title("Customers Segments", fontsize=26, fontweight="bold")
		plt.axis('off')
		st.pyplot(fig)
		
		if st.button("Continue to Explanation"):
			st.session_state.show_explanation = True
		
		if st.session_state.get('show_explanation', False):
			self.explanation(df_rfm)
	
	def hierarchical_clustering(self, df_rfm, n_clusters):
		hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
		df_rfm['Cluster'] = hierarchical.fit_predict(df_rfm[['Recency', 'Frequency', 'Monetary']])
		
		rfm_agg = df_rfm.groupby('Cluster').agg({
			'Recency'  : 'mean',
			'Frequency': 'mean',
			'Monetary' : ['mean', 'count']}).round(0)
		
		rfm_agg.columns = rfm_agg.columns.droplevel()
		rfm_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
		rfm_agg['Percent'] = round((rfm_agg['Count'] / rfm_agg.Count.sum()) * 100, 2)
		rfm_agg = rfm_agg.reset_index()
		rfm_agg['Cluster'] = 'Cluster ' + rfm_agg['Cluster'].astype('str')
		
		st.write("Cluster Centers:")
		st.write(rfm_agg)
		
		fig, ax = plt.subplots(figsize=(12, 8))
		Z = linkage(df_rfm[['Recency', 'Frequency', 'Monetary']], method='ward')
		dendrogram(Z)
		plt.title('Hierarchical Clustering Dendrogram')
		plt.xlabel('Sample Index')
		plt.ylabel('Distance')
		st.pyplot(fig)
		
		fig, ax = plt.subplots(figsize=(12, 8))
		sns.scatterplot(data=df_rfm, x='Recency', y='Monetary', hue='Cluster', palette='viridis', s=100)
		plt.title('Hierarchical Clustering of Customers')
		plt.xlabel('Recency')
		plt.ylabel('Monetary')
		plt.legend(title='Cluster')
		st.pyplot(fig)
		
		colors = sns.color_palette("husl", n_clusters).as_hex()
		colors_dict2 = {f'Cluster {i}': colors[i] for i in range(n_clusters)}
		
		fig, ax = plt.subplots(figsize=(14, 10))
		squarify.plot(sizes=rfm_agg['Count'],
		              text_kwargs={'fontsize': 12, 'weight': 'bold', 'fontname': "sans serif"},
		              color=[colors_dict2[f'Cluster {i}'] for i in range(n_clusters)],
		              label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(
			              *rfm_agg.iloc[i])
			              for i in range(n_clusters)], alpha=0.5)
		plt.title("Customers Segments", fontsize=26, fontweight="bold")
		plt.axis('off')
		st.pyplot(fig)
		
		if st.button("Continue to Explanation"):
			st.session_state.show_explanation = True
		
		if st.session_state.get('show_explanation', False):
			self.explanation(df_rfm)
	
	def explanation(self, df_rfm):
		st.markdown('<h3 style="color:#A4C3A2;">Explanation of Clustering with K-Means</h3>', unsafe_allow_html=True)
		st.write(
			'The following explanations are based on the sample data provided at the beginning of the page. You can refer to this data to apply it to your store.')
		
		n_clusters = st.selectbox("How many customer clusters do you decide on?", [3, 4, 5], index=2,
		                          key='num_clusters_explanation')
		
		kmeans = KMeans(n_clusters=n_clusters, random_state=42)
		df_rfm['Cluster'] = kmeans.fit_predict(df_rfm[['Recency', 'Frequency', 'Monetary']])
		
		cluster_centers = kmeans.cluster_centers_
		centers_df = pd.DataFrame(cluster_centers, columns=['Recency', 'Frequency', 'Monetary'])
		
		cluster_names = self.get_cluster_names(n_clusters)
		df_rfm['Cluster Name'] = df_rfm['Cluster'].map(cluster_names)
		centers_df['Cluster Name'] = [cluster_names[i] for i in range(n_clusters)]
		
		st.write("Cluster Centers:")
		st.dataframe(centers_df)
		
		st.write("RFM Data with Cluster Labels:")
		st.dataframe(df_rfm)
	
	def get_cluster_names(self, n_clusters):
		if n_clusters == 3:
			return {
				0: 'High Value',
				1: 'Medium Value',
				2: 'Low Value'
			}
		elif n_clusters == 4:
			return {
				0: 'High Value',
				1: 'Medium Value',
				2: 'Low Value',
				3: 'Rising Star'
			}
		elif n_clusters == 5:
			return {
				0: 'Top Tier',
				1: 'Low Engagement',
				2: 'Regulars',
				3: 'Potential Loyalists',
				4: 'New Customers'
			}
