import warnings
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearnex import patch_sklearn
from tqdm import tqdm
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage

patch_sklearn()

from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

class PandasML:
	def __init__(self, data, number_of_clusters):
		self.data = data
		self.number_of_clusters = number_of_clusters
		self.random_state = 42
		self.models = [
			('KMeans', KMeans(n_clusters=number_of_clusters, random_state=self.random_state)),
			('Birch', Birch(n_clusters=number_of_clusters)),
			('AgglomerativeClustering', AgglomerativeClustering(n_clusters=number_of_clusters))
		]
		self.eval_results = None
	
	def evaluate_models(self):
		results = []
		
		for name, model in tqdm(self.models, desc="Processing models", unit="model"):
			try:
				# Fit model
				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", category=UserWarning)
					
					if hasattr(model, 'fit_predict'):
						clusters = model.fit_predict(self.data)
					else:
						model.fit(self.data)
						clusters = model.predict(self.data)
				
				# Calculate evaluation metrics
				silhouette_avg = silhouette_score(self.data, clusters)
				calinski_harabasz_avg = calinski_harabasz_score(self.data, clusters)
				
				results.append({
					'Model'                  : name,
					'Silhouette Score'       : silhouette_avg,
					'Calinski-Harabasz Score': calinski_harabasz_avg
				})
			except Exception as e:
				results.append({
					'Model'                  : name,
					'Silhouette Score'       : None,
					'Calinski-Harabasz Score': None,
					'Error'                  : str(e)
				})
		
		# Convert results to DataFrame
		self.eval_results = pd.DataFrame(results)
		return self.eval_results
	
	def visualize_model_evaluation(self, model_evaluation):
		fig, axs = plt.subplots(2, 1, figsize=(10, 10))
		
		# Model comparison and visualization for Silhouette Score
		sns.barplot(x='Model', y='Silhouette Score', data=model_evaluation, ax=axs[0])
		axs[0].set_title('Silhouette Score Comparison')
		axs[0].tick_params(axis='x', rotation=45)
		
		# Model comparison and visualization for Calinski Hararasz Score
		sns.barplot(x='Model', y='Calinski-Harabasz Score', data=model_evaluation, ax=axs[1])
		axs[1].set_title('Calinski Hararasz Score Comparison')
		axs[1].tick_params(axis='x', rotation=45)
		
		plt.tight_layout()
		plt.show()
	
	def elbow_method_kmeans(self, max_k=15):
		wsse_list = []
		silhouette_scores = {}
		kmeans_models = {}
		K_list = range(2, max_k + 1)
		
		for k in K_list:
			# Create a KMeans instance with k clusters
			kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
			
			# Fit the input data
			kmeans.fit(self.data)
			
			# Append the WSSE to the wsse_list
			wsse_list.append(kmeans.inertia_)
			
			# Append the silhouette score to the silhouette_scores list
			silhouette_scores[k] = silhouette_score(self.data, kmeans.labels_)
			
			# Append the kmeans model to the kmeans_models dictionary
			kmeans_models[k] = kmeans
		
		# Plot the elbow curve
		plt.figure(figsize=(10, 6))
		sns.lineplot(x=list(K_list), y=wsse_list, marker='o')
		plt.xticks(list(K_list))
		plt.xlabel('Number of clusters')
		plt.ylabel('WSSE')
		plt.title('Elbow curve')
		plt.show()
		
		# Plot the silhouette scores
		plt.figure(figsize=(10, 6))
		sns.lineplot(x=list(K_list), y=list(silhouette_scores.values()), marker='o')
		plt.xticks(list(K_list))
		plt.xlabel('Number of clusters')
		plt.ylabel('Silhouette score')
		plt.title('Silhouette score curve')
		plt.show()
		
		return kmeans_models, silhouette_scores
	
	def visualize_kmeans_clusters(self, n_clusters):
		# Fit the KMeans model and get the cluster labels
		model = KMeans(n_clusters=n_clusters, random_state=self.random_state)
		self.data['Cluster'] = model.fit_predict(self.data)
		
		# Calculate the RFM statistics for each cluster
		rfm_stats = self.data.groupby('Cluster').agg({
			'Recency'  : ['mean', 'min', 'max'],
			'Frequency': ['mean', 'min', 'max'],
			'Monetary' : ['mean', 'min', 'max', 'sum']  # sum for total revenue
		}).sort_values(by=('Monetary', 'sum'), ascending=False)
		
		# Create a 3D scatter plot for the clusters
		fig1 = go.Figure(data=go.Scatter3d(
			x=self.data['Recency'],
			y=self.data['Frequency'],
			z=self.data['Monetary'],
			mode='markers',
			marker=dict(
				size=5,
				color=self.data['Cluster'],  # set color to cluster
				colorscale='Viridis',  # choose a colorscale
				opacity=0.8
			)
		))
		fig1.update_layout(
			title='Customer Segments based on RFM (KMeans)',
			scene=dict(
				xaxis_title='Recency',
				yaxis_title='Frequency',
				zaxis_title='Monetary'
			),
			margin=dict(l=0, r=0, b=0, t=40)
		)
		
		# Create a 3D scatter plot for the RFM statistics
		fig2 = go.Figure(data=go.Scatter3d(
			x=rfm_stats[('Recency', 'mean')],
			y=rfm_stats[('Frequency', 'mean')],
			z=rfm_stats[('Monetary', 'mean')],
			mode='markers',
			marker=dict(
				size=10,
				color=rfm_stats.index,  # set color to cluster
				colorscale='Viridis',  # choose a colorscale
				opacity=0.8
			)
		))
		fig2.update_layout(
			title='Mean RFM Values of Clusters (KMeans)',
			scene=dict(
				xaxis_title='Mean Recency',
				yaxis_title='Mean Frequency',
				zaxis_title='Mean Monetary'
			),
			margin=dict(l=0, r=0, b=0, t=40)
		)
		
		# Create a bar plot for the total revenue generated from each cluster
		fig3 = go.Figure(data=go.Bar(
			x=rfm_stats.index,
			y=rfm_stats[('Monetary', 'sum')],
			marker=dict(color=rfm_stats.index, colorscale='Viridis')
		))
		fig3.update_layout(
			title='Total Revenue Generated from Each Cluster (KMeans)',
			xaxis_title='Cluster',
			yaxis_title='Total Revenue',
			margin=dict(l=0, r=0, b=0, t=40)
		)
		
		# Show the plots
		fig1.show()
		fig2.show()
		fig3.show()
	
	def visualize_hierarchical_clusters(self):
		# Perform hierarchical clustering and create a dendrogram
		linked = linkage(self.data[['Recency', 'Frequency', 'Monetary']], method='ward')
		
		plt.figure(figsize=(10, 7))
		dendrogram(linked,
		           orientation='top',
		           distance_sort='descending',
		           show_leaf_counts=True)
		plt.title('Hierarchical Clustering Dendrogram')
		plt.xlabel('Sample index')
		plt.ylabel('Distance')
		plt.show()
		
		# Fit the AgglomerativeClustering model and get the cluster labels
		model = AgglomerativeClustering(n_clusters=self.number_of_clusters)
		self.data['Cluster'] = model.fit_predict(self.data)
		
		# Calculate the RFM statistics for each cluster
		rfm_stats = self.data.groupby('Cluster').agg({
			'Recency'  : ['mean', 'min', 'max'],
			'Frequency': ['mean', 'min', 'max'],
			'Monetary' : ['mean', 'min', 'max', 'sum']  # sum for total revenue
		}).sort_values(by=('Monetary', 'sum'), ascending=False)
		
		# Create a 3D scatter plot for the clusters
		fig1 = go.Figure(data=go.Scatter3d(
			x=self.data['Recency'],
			y=self.data['Frequency'],
			z=self.data['Monetary'],
			mode='markers',
			marker=dict(
				size=5,
				color=self.data['Cluster'],  # set color to cluster
				colorscale='Viridis',  # choose a colorscale
				opacity=0.8
			)
		))
		fig1.update_layout(
			title='Customer Segments based on RFM (Hierarchical)',
			scene=dict(
				xaxis_title='Recency',
				yaxis_title='Frequency',
				zaxis_title='Monetary'
			),
			margin=dict(l=0, r=0, b=0, t=40)
		)
		
		# Create a 3D scatter plot for the RFM statistics
		fig2 = go.Figure(data=go.Scatter3d(
			x=rfm_stats[('Recency', 'mean')],
			y=rfm_stats[('Frequency', 'mean')],
			z=rfm_stats[('Monetary', 'mean')],
			mode='markers',
			marker=dict(
				size=10,
				color=rfm_stats.index,  # set color to cluster
				colorscale='Viridis',  # choose a colorscale
				opacity=0.8
			)
		))
		fig2.update_layout(
			title='Mean RFM Values of Clusters (Hierarchical)',
			scene=dict(
				xaxis_title='Mean Recency',
				yaxis_title='Mean Frequency',
				zaxis_title='Mean Monetary'
			),
			margin=dict(l=0, r=0, b=0, t=40)
		)
		
		# Create a bar plot for the total revenue generated from each cluster
		fig3 = go.Figure(data=go.Bar(
			x=rfm_stats.index,
			y=rfm_stats[('Monetary', 'sum')],
			marker=dict(color=rfm_stats.index, colorscale='Viridis')
		))
		fig3.update_layout(
			title='Total Revenue Generated from Each Cluster (Hierarchical)',
			xaxis_title='Cluster',
			yaxis_title='Total Revenue',
			margin=dict(l=0, r=0, b=0, t=40)
		)
		
		# Show the plots
		fig1.show()
		fig2.show()
		fig3.show()


class PySparkKMeans:
	def __init__(self, spark: SparkSession, data):
		self.spark = spark
		self.data = data.select('Recency', 'Frequency', 'Monetary')
		self.random_state = 42
		self.vector_assembler = VectorAssembler(inputCols=self.data.columns, outputCol='features')
		self.data_vector = self.vector_assembler.transform(self.data).select('features')
	
	def elbow_method_kmeans(self, max_k=15):
		wsse_list = []
		silhouette_scores = {}
		kmeans_models = {}
		K_list = range(2, max_k + 1)
		
		for k in K_list:
			# Create a KMeans instance with k clusters
			kmeans = SparkKMeans(k=k, seed=self.random_state)
			
			# Fit the model
			model = kmeans.fit(self.data_vector)
			
			# Make predictions
			predictions = model.transform(self.data_vector)
			
			# Evaluate clustering by computing WSSE
			wsse = model.summary.trainingCost
			wsse_list.append(wsse)
			
			# Evaluate clustering by computing Silhouette Score
			evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features', metricName='silhouette',
			                                distanceMeasure='squaredEuclidean')
			silhouette_score = evaluator.evaluate(predictions)
			silhouette_scores[k] = silhouette_score
			
			# Store the KMeans model
			kmeans_models[k] = model
		
		# Plot the elbow curve
		plt.figure(figsize=(10, 6))
		sns.lineplot(x=list(K_list), y=wsse_list, marker='o')
		plt.xticks(list(K_list))
		plt.xlabel('Number of clusters')
		plt.ylabel('WSSE')
		plt.title('Elbow curve')
		plt.show()
		
		# Plot the silhouette scores
		plt.figure(figsize=(10, 6))
		sns.lineplot(x=list(K_list), y=list(silhouette_scores.values()), marker='o')
		plt.xticks(list(K_list))
		plt.xlabel('Number of clusters')
		plt.ylabel('Silhouette score')
		plt.title('Silhouette score curve')
		plt.show()
		
		return kmeans_models, silhouette_scores
	
	def visualize_kmeans_clusters(self, n_clusters):
		# Fit the KMeans model and get the cluster labels
		kmeans = SparkKMeans(k=n_clusters, seed=self.random_state)
		model = kmeans.fit(self.data_vector)
		predictions = model.transform(self.data_vector)
		
		# Convert predictions to Pandas DataFrame for plotting
		predictions_pdf = predictions.select('features', 'prediction').toPandas()
		predictions_pdf[['Recency', 'Frequency', 'Monetary']] = pd.DataFrame(predictions_pdf['features'].tolist(),
		                                                                     index=predictions_pdf.index)
		predictions_pdf = predictions_pdf.drop(columns=['features'])
		
		# Calculate the RFM statistics for each cluster
		rfm_stats = predictions_pdf.groupby('prediction').agg({
			'Recency'  : ['mean', 'min', 'max'],
			'Frequency': ['mean', 'min', 'max'],
			'Monetary' : ['mean', 'min', 'max', 'sum']  # sum for total revenue
		}).sort_values(by=('Monetary', 'sum'), ascending=False)
		
		# Create a 3D scatter plot for the clusters
		fig1 = go.Figure(data=go.Scatter3d(
			x=predictions_pdf['Recency'],
			y=predictions_pdf['Frequency'],
			z=predictions_pdf['Monetary'],
			mode='markers',
			marker=dict(
				size=5,
				color=predictions_pdf['prediction'],  # set color to cluster
				colorscale='Viridis',  # choose a colorscale
				opacity=0.8
			)
		))
		fig1.update_layout(
			title='Customer Segments based on RFM (KMeans)',
			scene=dict(
				xaxis_title='Recency',
				yaxis_title='Frequency',
				zaxis_title='Monetary'
			),
			margin=dict(l=0, r=0, b=0, t=40)
		)
		
		# Create a 3D scatter plot for the RFM statistics
		fig2 = go.Figure(data=go.Scatter3d(
			x=rfm_stats[('Recency', 'mean')],
			y=rfm_stats[('Frequency', 'mean')],
			z=rfm_stats[('Monetary', 'mean')],
			mode='markers',
			marker=dict(
				size=10,
				color=rfm_stats.index,  # set color to cluster
				colorscale='Viridis',  # choose a colorscale
				opacity=0.8
			)
		))
		fig2.update_layout(
			title='Mean RFM Values of Clusters (KMeans)',
			scene=dict(
				xaxis_title='Mean Recency',
				yaxis_title='Mean Frequency',
				zaxis_title='Mean Monetary'
			),
			margin=dict(l=0, r=0, b=0, t=40)
		)
		
		# Create a bar plot for the total revenue generated from each cluster
		fig3 = go.Figure(data=go.Bar(
			x=rfm_stats.index,
			y=rfm_stats[('Monetary', 'sum')],
			marker=dict(color=rfm_stats.index, colorscale='Viridis')
		))
		fig3.update_layout(
			title='Total Revenue Generated from Each Cluster (KMeans)',
			xaxis_title='Cluster',
			yaxis_title='Total Revenue',
			margin=dict(l=0, r=0, b=0, t=40)
		)
		
		# Show the plots
		fig1.show()
		fig2.show()
		fig3.show()
		