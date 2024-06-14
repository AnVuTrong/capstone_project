import streamlit as st
from GUI.modules import sidebar

class StreamlitUI:
	def __init__(self):
		self.title = ":green[RECOMMENDATION SYSTEM]"
		self.sidebar = sidebar.Sidebar()
		with open("GUI/style.css") as css:
			st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
	
	def run(self):
		st.title(self.title)
		st.divider()
		self.sidebar.draw_sidebar()
		