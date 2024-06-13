import streamlit as st
from GUI.modules.widgets import Widgets
from GUI.modules.backend.recommendation_system import RecommendationSystem
from Project_2.modules.data_preprocessing import DataPreprocessing


class CollaborateFiltering:
	def __init__(self):
		self.title = "Collaborative Filtering Recommendation"
		self.widgets = Widgets()
		self.recommendation_system = RecommendationSystem()
		self.data_preprocessing = DataPreprocessing(
			path_courses='Project_2/data/courses.csv',
			path_reviews='Project_2/data/reviews.csv',
		)
	
	def gen_page(self):
		st.title(self.title)
		st.divider()
		st.info("Demo only, not production ready")
		
		data_type, n_recommendations = self._input()
		
		press = st.button(f"Get {n_recommendations} recommendations with {data_type}")
		if press:
			with st.spinner(text="In progress"):
				self.widgets.progress_bar(0)
				if data_type == "Preset Data":
					user_id = self._preset_data()
					user_data = None
				if data_type == "Input Data":
					user_data = self._input_data()
					user_id = None
			# self._present_recommendations(
			# 	user_id=user_id,
			# 	user_data=user_data,
			# 	n_recommendations=n_recommendations,
			# )
	
	def _input(self):
		data_type = st.radio(
			"Choose input type:",
			("Preset Data", "Input Data")
		)
		
		with st.popover("Setting"):
			n_recommendations = self.widgets.small_selectbox(
				label="Number of Recommendations",
				options=[10, 15, 20, 25],
				index=0,
			)
		
		return data_type, n_recommendations
	
	def _preset_data(self):
		user_id = self._choose_user()
		# self._show_user_history(user_id)
		return user_id
	
	def _input_data(self):
		st.error("Not implemented yet.")
		user_data = None
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

	def _choose_user(self):
		df_users = self.data_preprocessing.get_all_user_ids()
		user_dict = dict(zip(df_users['ReviewerName'], df_users['ReviewerID']))
		
		selected_user_name = st.selectbox(
			label="Select User",
			options=list(user_dict.keys()),
			index=0,
		)
		
		user_id = user_dict[selected_user_name]
		st.write(f"Selected User {selected_user_name}: {user_id}")
		return user_id
	
	def _show_user_history(self):
		pass
	