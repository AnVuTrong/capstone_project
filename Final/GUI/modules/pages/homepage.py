import streamlit as st


class Homepage:
	def __init__(self):
		self.title = "Homepage"
		self.subheader = '<h3 style="color:#A4C3A2;">Welcome</h3>'
	
	def gen_page(self):
		st.header(self.title)
		st.divider()
		st.markdown(self.subheader, unsafe_allow_html=True)
		st.write(
			'Welcome to our project. We are working on two projects (Customer Segmentation and Course Recommender System). Please choose one of the projects in the sidebar to explore more.')
		st.markdown("***1. Project 1: Customer Segmentation***")
		st.write(
			'This application is designed for stores, aiming to assist them in clustering customers and aiding in the development of effective business strategies.')
		st.image("GUI/img/Picture6.png")
		
		st.markdown("***2. Project 2: Recommendation system***")
		st.write("This application is designed for users seeking suitable courses on Coursera.")
		st.image("GUI/img/Picture1.png")
		st.write(
			"We hope you find these two applications useful. Wishing you peace and happiness. Thank you for visiting us.")
		