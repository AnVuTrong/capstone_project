import streamlit as st
from GUI.modules.widgets import Widgets
from GUI.modules.backend.recommendation_system import RecommendationSystem


class ContentBaseFiltering:
	def __init__(self):
		self.title = "Content-Based Filtering Recommendation"
		self.widgets = Widgets()
		self.recommendation_system = RecommendationSystem()
	
	def gen_page(self):
		st.title(self.title)
		st.divider()
		st.info("You can change configurations in the settings.")

		search_query, submit, n_recommendations = self._input()
		
		# Get the recommendations for the course entered by the user.
		if submit:
			with st.spinner(text="In progress"):
				try:
					self._get_recommendations(search_query=search_query, num_recommendations=n_recommendations)
				except Exception as e:
					st.error(f"An error occurred: {e}")
				
	def _input(self):
		# A Searchbar for the user to input the name of the course they want to get recommendations for.
		search_query, submit = self.widgets.search_bar(
			label="Please enter the name of the Data Science course you want recommendations for:",
			button_text="Search",
			result_text="Searching Coursera"
		)
		with st.popover("Setting"):
			n_recommendations = self.widgets.small_selectbox(
				label="Number of Recommendations",
				options=[1, 5, 10, 15, 20, 25],
				index=1,
			)
		return search_query, submit, n_recommendations
	
	def _get_recommendations(self, search_query: str, num_recommendations: int=10):
		if not search_query:
			st.info("Please type something first.")
		else:
			gensim_recommendations_df, cosine_recommendations_df = self._run_model(
				search_query=search_query,
				num_recommendations=num_recommendations
			)

			tab1, tab2 = st.tabs(["Gensim", "Cosine Similarity"])
			
			with tab1:
				self.widgets.progress_bar(100)
				st.write("Gensim Recommendations:")
				st.dataframe(gensim_recommendations_df)
			
			with tab2:
				self.widgets.progress_bar(100)
				st.write("Cosine Similarity Recommendations:")
				st.dataframe(cosine_recommendations_df)
	
	def _run_model(self, search_query: str, num_recommendations: int = 10):
		gensim_recommendations_df = None
		cosine_recommendations_df = None
		self.widgets.progress_bar(0)
		try:
			self.widgets.progress_bar(20)
			gensim_recommendations_df = self.recommendation_system.get_gensim_recommendations(
				user_search=search_query,
				num_recommendations=num_recommendations
			)
			self.widgets.progress_bar(40)
			cosine_recommendations_df = self.recommendation_system.get_cosine_recommendations(
				user_search=search_query,
				num_recommendations=num_recommendations
			)
			self.widgets.progress_bar(90)
		
		except Exception as e:
			st.error(f"An error occurred: {e}")
			gensim_recommendations_df = None
			cosine_recommendations_df = None
		finally:
			return gensim_recommendations_df, cosine_recommendations_df
