import streamlit as st
from GUI.modules import sidebar


class StreamlitUI:
	def __init__(self):
		self.title = '<h1 style="color:#5D7B6F;">RECOMMENDATION SYSTEM</h1>'
		self.sidebar = sidebar.Sidebar()
		with open("GUI/style.css") as css:
			st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
	
	def run(self):
		st.markdown(self.title, unsafe_allow_html=True)
		st.divider()
		self.sidebar.draw_sidebar()
