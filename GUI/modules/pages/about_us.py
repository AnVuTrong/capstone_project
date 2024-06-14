import streamlit as st


class AboutUs:
	def __init__(self):
		self.header = "About Us"
	
	def gen_page(self):
		""" Introduce myself and team-member """
		st.header(self.header)
		
		st.write(
			"""
			This is my capstone project for the Data Science course.
			"""
		)
		
		self._portraits()
	
	def _portraits(self):
		img_path = "GUI/img/"
		vta = f"{img_path}vta.jpg"
		lthb = f"{img_path}lthb.jpg"
		
		# Apply CSS to make the image round
		st.markdown(
			"""
			<style>
			.round-image {
				border-radius: 50%;
				width: 200px;
				height: 200px;
				object-fit: cover;
			}
			</style>
			""",
			unsafe_allow_html=True
		)
		
		# Display images using Streamlit's st.image
		st.markdown(f'<img src="{vta}" class="round-image">', unsafe_allow_html=True)
		st.write(
			"""
			An Vu Trong is a student at the University of Science, Ho Chi Minh City. He is passionate about Data Science,
			especially in the field of recommendation systems. He is the author of this project.
			"""
		)
		
		st.markdown(f'<img src="{lthb}" class="round-image">', unsafe_allow_html=True)
		st.write(
			"""
			Le Thi Hai Binh is a teacher at the University of Science, Ho Chi Minh City. She is the mentor of this project.
			"""
		)
