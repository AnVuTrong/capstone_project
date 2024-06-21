import streamlit as st
from datetime import datetime
from GUI.modules.page_manager import PageManager


class Sidebar:
	def __init__(self):
		self.header = "Navigation Menu"
		self.options = [
			"**Homepage**",
			"**About us**",
			"**Customer segmentation**",
			"**Recommend a course**",
			"**How it works**",
			# "UI Test",
		]
		self.captions = [
			'<p style="color:#D7F9FA;">Introduction of the project</p>',
			'<p style="color:#D7F9FA;">Introduce myself and team-member</p>',
			'<p style="color:#D7F9FA;">Recommendation for new users using content-based filtering</p>',
			'<p style="color:#D7F9FA;">Recommendation for existing users using collaborative filtering</p>',
		]
		
		self.page_mng = PageManager()
		# self.image = "GUI/img/Picture2.png"
	
	def draw_sidebar(self):
		if "selected_option" not in st.session_state:
			st.session_state.selected_option = self.options[0]
		
		with st.sidebar:
			st.header(self.header)
			for i, option in enumerate(self.options):
				if st.button(option, type='secondary', use_container_width=True):
					st.session_state.selected_option = option
			# st.markdown(self.captions[i], unsafe_allow_html=True)
			# st.image(self.image)
			st.divider()
			self._github()
		
		selected_option = st.session_state.selected_option
		
		if selected_option == "**Homepage**":
			self.page_mng.gen_homepage()
		
		elif selected_option == "**About us**":
			self.page_mng.gen_about_us_page()
		
		elif selected_option == "**How it works**":
			self.page_mng.gen_how_it_work_page()
		
		elif selected_option == "**Customer segmentation**":
			self.page_mng.gen_customer_segmentation_page()
		
		elif selected_option == "**Recommend a course**":
			self.page_mng.gen_recommendation_system_page()
	
	# elif selected_option == "UI Test":
	#     self.page_mng.gen_testing_widgets()
	
	def _github(self):
		""" Redirect to Github """
		date_of_release = datetime(2024, 6, 22)
		if datetime.today() > date_of_release:
			st.button("Github", on_click=lambda: st.write("Redirecting to Github"))
			st.write("[Github](https://github.com/AnVuTrong/capstone_project)")
		else:
			click = st.button("Github")
			st.info(f"Project will be open-sourced on {date_of_release.strftime('%Y-%m-%d')}") if click else None
