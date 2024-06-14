import streamlit as st
from datetime import datetime
from GUI.modules.page_manager import PageManager


class Sidebar:
	def __init__(self):
		self.header = "Navigation Menu"
		self.options = [
			"**Home**",
			"**About us**",
			"**Searching courses**",
			"**Courses for you**",
			# "UI Test",
		]
		self.captions = [
			"***:grey[Introduction of the project]***",
			"***:grey[Introduce myself and team-member]***",
			"***:grey[Recommendation for new users using content-based filtering]***",
			"***:grey[Recommendation for existing users using collaborative filtering]***",
		]
		
		self.page_mng = PageManager()
		self.image = "GUI/img/Picture2.png"
	
	def draw_sidebar(self):
		with st.sidebar:
			st.header(self.header)
			selected_option = st.sidebar.radio(
				label="Please select a page:",
				options=self.options,
				captions=self.captions,
			)
			st.image(self.image)
			st.divider()
			self._github()
		
		if selected_option == "**Home**":
			self.page_mng.gen_homepage()
		
		elif selected_option == "**About us**":
			self.page_mng.gen_about_us_page()
		
		elif selected_option == "**Searching courses**":
			self.page_mng.gen_content_base_filtering_page()
		
		elif selected_option == "**Courses for you**":
			self.page_mng.gen_collaborative_filtering_page()
	
	# elif selected_option == "UI Test":
	# 	self.page_mng.gen_testing_widgets()
	
	def _github(self):
		""" Redirect to Github """
		date_of_release = datetime(2024, 6, 22)
		if datetime.today() > date_of_release:
			st.link_button("Github", "https://github.com/AnVuTrong/capstone_project")
		else:
			click = st.button("Github")
			st.info(f"Project will be open-sourced on {date_of_release.strftime('%Y-%m-%d')}") if click else None
