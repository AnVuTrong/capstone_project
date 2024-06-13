import streamlit as st
from GUI.modules.widgets import Widgets
from GUI.modules.backend.recommendation_system import RecommendationSystem


class CollaborateFiltering:
	def __init__(self):
		self.title = "Collaborative Filtering Recommendation"
		self.widgets = Widgets()
		self.recommendation_system = RecommendationSystem()
	
	def gen_page(self):
		st.title(self.title)
		st.divider()
		st.info("Demo only, not production ready")
		
		data_type, n_recommendations = self._input()
		
		st.button("Get Recommendations")
		if st.button:
			with st.spinner(text="In progress"):
				self.widgets.progress_bar(0)
				if data_type == "Preset Data":
					user_id = self._preset_data()
				if data_type == "Input Data":
					user_data = self._input_data()
				self._present_recommendations(
					user_id=user_id,
					user_data=user_data,
					n_recommendations=n_recommendations,
				)
	
	def _input(self):
		data_type = st.radio(
			"Choose input type:",
			("Preset Data", "Input Data")
		)
		
		n_recommendations = self.widgets.small_selectbox(
			label="Number of Recommendations",
			options=[10, 15, 20, 25],
			index=0,
		)
		
		return data_type, n_recommendations
	
	def _preset_data(self):
		user_data = "your preset data here"
		return user_data
	
	def _input_data(self):
		user_data = "your input data here"
		return user_data
	
	def _present_recommendations(self, user_id=None, user_data=None, n_recommendations=10):
		recommendations_df, user_history = self.recommendation_system.get_collaborative_recommendations(
			user_id,
			user_data,
			n_recommendations,
		)
	
		recommendations, user_history = st.tabs(["Recommendations", "User History"])
		self.widgets.progress_bar(100)
		st.success("Success")
		with recommendations:
			st.write("Recommendations:")
			st.dataframe(recommendations_df)
		
		with user_history:
			st.write("User History:")
			st.dataframe(user_history)
