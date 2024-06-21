import pandas as pd
import streamlit as st
from GUI.modules.widgets import Widgets
from GUI.modules.backend.recommendation_system import RecommendationSystem
from Project_2.modules.data_preprocessing import DataPreprocessing
from Project_2.modules.helpers import generate_random_user_data


class CollaborateFiltering:
	def __init__(self):
		self.header = "Courses just for you"
		self.subheader = '<h3 style="color:#A4C3A2;">Recommendation for existing users using collaborative filtering</h3>'
		self.widgets = Widgets()
		self.recommendation_system = RecommendationSystem()
		self.data_preprocessing = DataPreprocessing(
			path_courses='Project_2/data/courses.csv',
			path_reviews='Project_2/data/reviews.csv',
		)
	
	def gen_page(self):
		st.header(self.header)
		st.divider()
		st.markdown(self.subheader, unsafe_allow_html=True)
		st.write("Recommendations for existing users are generated using Collaborative filtering method with SVD.")
		st.image("GUI/img/Picture4.png")
		st.image("GUI/img/Picture5.png")
		st.info("You can change configurations in the settings.")
		
		data_type, n_recommendations, user_id, user_data, preset, show_dataframe = self._input()
		
		press = st.button(f"Get recommendations for {data_type}", use_container_width=True)
		if press:
			with st.spinner(text="In progress..."):
				self._present_recommendations(
					user_id=user_id,
					user_data=user_data,
					n_recommendations=n_recommendations,
					preset=preset,
					show_dataframe=show_dataframe,
				)
	
	def _input(self):
		with st.popover("Setting"):
			data_type = st.radio(
				"Choose how to generate user's history:",
				("Preset Data", "Input Data"),
				index=1,
			)
			
			n_recommendations = self.widgets.small_selectbox(
				label="Number of Recommendations",
				options=[1, 5, 10, 15, 20, 25],
				index=1,
			)
			
			show_dataframe = st.checkbox("Show Dataframe", value=False)
		
		if data_type == "Preset Data":
			user_id = self._preset_data()
			user_data = None
			preset = True
		else:
			user_id = None
			user_data = self._input_data()
			preset = False
		
		return data_type, n_recommendations, user_id, user_data, preset, show_dataframe
	
	def _preset_data(self):
		with st.spinner(text="Load users..."):
			user_id = self._choose_user()
			self._show_user_history(user_id)
		return user_id
	
	def _input_data(self):
		method = st.radio(
			"Choose user history method:",
			("Choose courses", "Random courses")
		)
		
		if method == "Choose courses":
			user_data = self._choose_courses()
		else:
			user_data = self._random_courses()
		
		return user_data
	
	def _present_recommendations(self,
	                             user_id=None, user_data=None, n_recommendations=10, preset=True,
	                             show_dataframe=False):
		recommendations_df, user_history_df = self.recommendation_system.get_collaborative_recommendations(
			user_id,
			user_data,
			n_recommendations,
			preset,
		)
		self.widgets.progress_bar(100)
		recommendations, user_history = st.tabs(["Recommendations", "User History"])
		with recommendations:
			if show_dataframe:
				self.widgets.show_raw_dataframe(recommendations_df)
			self.widgets.display_courses_to_columns(recommendations_df)
		with user_history:
			st.write("User History:")
			st.dataframe(user_history_df)
	
	def _choose_user(self):
		df_users = self.data_preprocessing.get_all_user_ids()
		user_dict = dict(zip(df_users['ReviewerName'], df_users['ReviewerID']))
		user_list = list(user_dict.keys())
		
		# Determine the range of options with a slider
		slider_value = st.slider(
			label="Select range of users:",
			min_value=0,
			max_value=len(user_list) - 10,
			value=0,
			step=10,
		)
		
		# Show 10 users at a time in the selectbox
		selected_user_name = st.selectbox(
			label="Select an user",
			options=user_list[slider_value:slider_value + 10],
			index=0,
		)
		
		user_id = user_dict[selected_user_name]
		st.info(f"Selected User {selected_user_name}: :rainbow-background[{user_id}]")
		
		return user_id
	
	def _show_user_history(self, user_id):
		user_history = self.data_preprocessing.get_user_history(user_id)
		st.write("User History:")
		st.dataframe(user_history)
	
	def _choose_courses(self):
		courses_df = self.data_preprocessing.get_all_courses()
		course_names = courses_df['CourseName'].tolist()
		selected_courses = st.multiselect("Select courses", course_names)
		
		if selected_courses:
			selected_courses_df = courses_df[courses_df['CourseName'].isin(selected_courses)]
			ratings = {}
			with st.popover("Edit Ratings"):
				for course in selected_courses:
					rating = st.slider(f"Rate {course}", min_value=1, max_value=5, value=3)
					ratings[course] = rating
				
				user_data = pd.DataFrame({
					'CourseID'  : selected_courses_df['CourseID'],
					'RatingStar': [ratings[course] for course in selected_courses],
				})
				
				return user_data
		else:
			st.warning("Please select at least one course.")
			return pd.DataFrame(columns=['CourseID', 'RatingStar'])
	
	def _random_courses(self):
		courses_df = self.data_preprocessing.get_all_courses()
		num_courses = st.slider("Number of courses to generate", min_value=1, max_value=20, value=5)
		
		user_data = generate_random_user_data(courses_df, num_courses=num_courses)
		st.warning("Generated random user data:")
		selected_courses = courses_df[courses_df['CourseID'].isin(user_data['CourseID'])].copy()
		selected_courses = self.recommendation_system.refactor_df_for_display(selected_courses)
		st.dataframe(selected_courses)
		
		return user_data
	