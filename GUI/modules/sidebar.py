import streamlit as st
from GUI.modules.page_manager import PageManager


class Sidebar:
	def __init__(self):
		self.title = "Sidebar"
		self.options = [
			"Home",
			"Content-based filtering Recommendation",
			"Collaborative filtering Recommendation",
			"UI Test",
		]
		
		self.page_mng = PageManager()
		self.image = "GUI/img/Picture2.png"
	
	def draw_sidebar(self):
		st.sidebar.title(self.title)
		selected_option = st.sidebar.selectbox("Please select below tab", self.options)
		if selected_option == "Home":
			self.page_mng.gen_homepage()
		
		elif selected_option == "Content-based filtering Recommendation":
			self.page_mng.gen_content_base_filtering_page()
		
		elif selected_option == "Collaborative filtering Recommendation":
			self.page_mng.gen_collaborative_filtering_page()
		
		elif selected_option == "UI Test":
			self.page_mng.gen_testing_widgets()
		
		st.sidebar.image(self.image)
