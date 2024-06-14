import streamlit as st
from GUI.modules.widgets import Widgets
from GUI.modules.backend.recommendation_system import RecommendationSystem


class ContentBaseFiltering:
	def __init__(self):
		self.header = "Recommendation for new users"
		self.widgets = Widgets()
		self.recommendation_system = RecommendationSystem()
	
	def gen_page(self):
		st.header(self.header)
		st.divider()
		st.write(
			"Recommendations for new users are generated using content-based filtering techniques, specifically Gensim, cosine similarity.")
		st.image("GUI/img/Picture3.png")
		st.write(
			"Please provide some of the following information so we can help recommend suitable Data Science courses for you on Coursera.")
		st.info("You can change configurations in the settings.")
		
		search_query, submit, n_recommendations, show_dataframe = self._input()
		
		# Get the recommendations for the course entered by the user.
		if submit:
			with st.spinner(text="In progress"):
				try:
					self._get_recommendations(
						search_query=search_query,
						num_recommendations=n_recommendations,
						show_dataframe=show_dataframe,
					)
				except Exception as e:
					st.error(f"An error occurred: {e}")
	
	def _input(self):
		# A Searchbar for the user to input the name of the course they want to get recommendations for.
		with st.popover("Setting"):
			n_recommendations = self.widgets.small_selectbox(
				label="Number of Recommendations",
				options=[1, 5, 10, 15, 20, 25],
				index=1,
			)
			
			show_dataframe = st.checkbox("Show Dataframe", value=False)
		
		search_query, submit = self.widgets.search_bar(
			label="What do you want to learn? Please input related keywords.",
			button_text="Click to get recommendations",
			result_text="Searching Coursera"
		)
		
		return search_query, submit, n_recommendations, show_dataframe
	
	def _get_recommendations(self, search_query: str, num_recommendations: int = 10, show_dataframe: bool = False):
		if not search_query:
			st.info("Please type something first.")
		else:
			gensim_recommendations_df, cosine_recommendations_df = self._run_model(
				search_query=search_query,
				num_recommendations=num_recommendations
			)
			self.widgets.progress_bar(100)
			tab1, tab2 = st.tabs(["Gensim", "Cosine Similarity"])

			with tab1:
				if show_dataframe:
					self.widgets.show_raw_dataframe(df=gensim_recommendations_df)
				self.widgets.display_courses_to_columns(gensim_recommendations_df)
			
			with tab2:
				if show_dataframe:
					self.widgets.show_raw_dataframe(df=cosine_recommendations_df)
				self.widgets.display_courses_to_columns(cosine_recommendations_df)
	
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
