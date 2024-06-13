import pandas as pd
import streamlit as st
from GUI.modules.widgets import Widgets
from GUI.modules.backend.recommendation_system import RecommendationSystem
from Project_2.modules.data_preprocessing import DataPreprocessing
from Project_2.modules.helpers import generate_random_user_data


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
		
		data_type, n_recommendations, user_id, user_data, preset = self._input()
		
		press = st.button(f"Get recommendations for {data_type}")
		if press:
			with st.spinner(text="In progress..."):
				self._present_recommendations(
					user_id=user_id,
					user_data=user_data,
					n_recommendations=n_recommendations,
					preset=preset,
				)
	
	def _input(self):
		with st.popover("Setting"):
			data_type = st.radio(
				"Choose input type:",
				("Preset Data", "Input Data")
			)
			
			n_recommendations = self.widgets.small_selectbox(
				label="Number of Recommendations",
				options=[10, 15, 20, 25],
				index=0,
			)
		
		if data_type == "Preset Data":
			user_id = self._preset_data()
			user_data = None
			preset = True
		else:
			user_id = None
			user_data = self._input_data()
			preset = False
		
		return data_type, n_recommendations, user_id, user_data, preset
	
	def _preset_data(self):
		with st.spinner(text="Load users..."):
			user_id = self._choose_user()
			self._show_user_history(user_id)
		return user_id
	
	def _input_data(self):
		method = st.radio(
			"Choose method:",
			("Choose courses", "Random courses")
		)
		
		if method == "Choose courses":
			user_data = self._choose_courses()
		else:
			user_data = self._random_courses()
			user_data = self._edit_user_data(user_data)
		
		return user_data
	
	def _present_recommendations(self, user_id=None, user_data=None, n_recommendations=10, preset=True):
		recommendations_df, user_history_df = self.recommendation_system.get_collaborative_recommendations(
			user_id,
			user_data,
			n_recommendations,
			preset,
		)
		self.widgets.progress_bar(100)
		recommendations, user_history = st.tabs(["Recommendations", "User History"])
		with recommendations:
			st.write("Recommendations:")
			st.dataframe(recommendations_df)
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
		st.write(f"Selected User {selected_user_name}: {user_id}")
		
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
			for course in selected_courses:
				rating = st.slider(f"Rate {course}", min_value=1, max_value=5, value=3)
				ratings[course] = rating
			
			user_data = pd.DataFrame({
				'CourseName': selected_courses,
				'RatingStar': [ratings[course] for course in selected_courses],
				'CourseID'  : selected_courses_df['CourseID'],
			})
			
			return user_data
		else:
			st.warning("Please select at least one course.")
			return pd.DataFrame(columns=['CourseID', 'RatingStar'])
	
	def _random_courses(self):
		courses_df = self.data_preprocessing.get_all_courses()
		num_courses = st.slider("Number of courses to generate", min_value=1, max_value=20, value=5)
		
		user_data = generate_random_user_data(courses_df, num_courses=num_courses)
		
		st.write("Generated random user data:")
		st.dataframe(user_data)
		
		return user_data
	
	def _edit_user_data(self, user_data):
		if not user_data.empty:
			st.write("Edit your data:")
			editable_data = user_data.copy()
			
			for index, row in editable_data.iterrows():
				new_rating = st.slider(f"Edit rating for Course ID {row['CourseID']}", min_value=1, max_value=5,
				                       value=row['RatingStar'])
				editable_data.at[index, 'RatingStar'] = new_rating
			
			st.write("Updated user data:")
			st.dataframe(editable_data)
			
			return editable_data
		else:
			st.warning("No user data to edit.")
			return user_data
	