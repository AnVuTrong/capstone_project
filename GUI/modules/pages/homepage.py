import streamlit as st


class Homepage:
	def __init__(self):
		self.title = "Homepage"
		
	def gen_page(self):
		st.title(self.title)
		st.markdown("***1. Introduction to the project***")
		st.write("This project aims to develop a Recommender System to suggest courses to users. The target audience consists of two types of users:")
		st.write("-	New users seeking courses that align with their personal objectives.")
		st.write("-	Returning users who have previously registered for courses on Coursera, to whom suitable next courses will be recommended.")
		st.write("The project utilizes two groups of algorithms:")
		st.write("-	Content-based filtering: Gensim, cosine_similarity.")
		st.write("-	Collaborative filtering: Surprise (SVD).")

		st.markdown("***2. Introduction to Coursera***")
		st.image("GUI/img/Picture1.png")
		
		st.write("Coursera is a global online learning platform that offers anyone, anywhere, access to online courses and degrees from leading universities and companies.")
		st.write("Coursera is an online learning platform featuring many different subjects across an array of learning formats, such as courses, Specializations, Professional Certificates, degrees, and tutorials. Over 300 leading universities and companies provide instruction on Coursera, including Stanford, Duke, Illinois, University of Colorado Boulder, Google, IBM, Microsoft, and Meta.")
		st.write("Whether you're looking to develop a new skill, strengthen an existing one, further your learning with a formal credential like a Professional Certificate or degree, or just learn something for fun, Coursera has many options to choose from.")
		
		st.markdown("***3. Introduction to data collection***")
		st.write("Analytical data is collected from course information from Coursera (879 courses) and course reviews (223,543 reviews).")
		