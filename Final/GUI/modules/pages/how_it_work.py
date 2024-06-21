import streamlit as st


class HowItWork:
	def __init__(self):
		self.title = "How it works"
	
	def gen_page(self):
		st.header(self.title)
		st.divider()
		st.write("The project consists of two main parts: Customer Segmentation and Course Recommendation System.")
		st.markdown("***1. Project 1: Customer Segmentation***")
		st.write(
			"Customer segmentation is done using RFM analysis. "
			"RFM analysis is a marketing technique used to determine quantitatively which customers are "
			"the best ones by examining how recently a customer has purchased (Recency), "
			"how often they purchase (Frequency), and how much the customer spends (Monetary)."
		)
		
		st.markdown("***2. Project 2: Recommendation system***")
		st.write(
			"Recommendations for new users are generated using content-based filtering techniques, using GenSim and Cosine Similarity with TF-IDF.")
		st.image("GUI/img/Picture3.png")
		st.write("Recommendations for existing users are generated using Collaborative filtering method with SVD.")
		st.image("GUI/img/Picture4.png")
		st.image("GUI/img/Picture5.png")
