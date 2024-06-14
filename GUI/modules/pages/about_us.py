import streamlit as st


class AboutUs:
	def __init__(self):
		self.header = "About Us"
		self.img_path = "GUI/img/"
		self.vta = f"{self.img_path}vta.png"
		self.lthb = f"{self.img_path}lthb.png"
	
	def gen_page(self):
		""" Introduce myself and team-member """
		st.header(self.header)
		st.divider()
		col1, col2 = st.columns(2, gap='large')
		with col1:
			self._portraits(
				img=self.vta,
				name="Vu Trong An",
				description="I am a student at the CSC. I am passionate about data science and machine learning."
			)
		with col2:
			self._portraits(
				img=self.lthb,
				name="Le Thi Hai Binh",
				description="I am a teacher at UEH, I'm joining the project to better support my researches."
			)
	
	def _portraits(self, img, name, description):
		# with st.container(border=True, height=600):
			st.image(img)
			st.write("")
			st.subheader(name)
			st.write(
				description
			)
			click = st.button("Contact", key=name)
			if click:
				st.warning("No contact information available.")
				