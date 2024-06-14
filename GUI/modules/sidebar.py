import streamlit as st
from datetime import datetime
from GUI.modules.page_manager import PageManager


class Sidebar:
	def __init__(self):
		self.title = "Navigation Menu"
		self.options = [
			"Home",
			"Recommendation for new users",
			"Recommendation for existing users",
			# "UI Test",
		]
		
		self.page_mng = PageManager()
		self.image = "GUI/img/Picture2.png"
	
	def draw_sidebar(self):
		with st.sidebar:
			st.header(self.title)
			st.divider()
			selected_option = st.sidebar.selectbox("Please select a below tab:", self.options)
			st.image(self.image)
			self._github()
			st.divider()
			
		
		if selected_option == "Home":
			self.page_mng.gen_homepage()
		
		elif selected_option == "Recommendation for new users":
			self.page_mng.gen_content_base_filtering_page()
		
		elif selected_option == "Recommendation for existing users":
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
			