import cv2
import numpy as np
import pandas as pd
import streamlit as st
#from PIL import Image 
from pathlib import Path
from data_reader import Data
import matplotlib.pyplot as plt
#from dimensional_analysis import DimensionalAnalysis




class Data:
    def __init__(self, file, pandas=False):
        self.file_location = '' if pandas else file
        self.data = file if pandas else self.read_file(self.file_location)
        self.parameters = self.generate_list_of_parameters()

    @staticmethod
    def read_file(file_location):
        data = pd.read_csv(file_location)
        return data

    def generate_list_of_parameters(self):
        # TODO add the ability to convert to standard units (i.e. mm to m) using Convert and ConvertTemperature
        parameters = ListOfParameters([])
        for key in self.data:
            try:
                print(f"Processing key: {key}")  # Debugging statement
                parts = key.split('-')
                if len(parts) > 1 and parts[1]:  # Check if there's a valid second part
                    unit_key = parts[1]
                    unit = getattr(Units, unit_key, None)
                    if unit is not None:
                        parameters.append(Parameter(value=[value for value in self.data[key]],
                                                units=unit,
                                                name=parts[0]))
                    else:
                        print(f"Attribute '{unit_key}' not found in Units class")
                else:
                    print(f"Key '{key}' does not have a valid second part after hyphen, put into form parameter name-base unit")
            except Exception as e:
                print(f"Error processing key '{key}': {e}")
        return parameters



#if __name__ == "__main__":
 #   experiment = Data("C:/Users/truma/Downloads/test - bernoulli_v2.csv")
  #  d = DimensionalAnalysis(experiment.parameters)
 #   d.plot()

  #  # [print(group, '\n', group.repeating_variables) for group in d.pi_group_sets]

   # values = [80, 20, 9.8, 1, 1, 1]
   # test = ListOfParameters([])
    #for i, parameter in enumerate(experiment.parameters[1:]):
     #   # print(Parameter(value=values[i], units=parameter.units, name=parameter.name))
      #  test.append(Parameter(value=values[i], units=parameter.units, name=parameter.name))
    #print('test', test)
    ## test = d.predict(experiment.parameters)


















@st.cache_data
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def generate_plots(dimensional_analysis):
    plt.close('all')
    for h, pi_group_set in enumerate(dimensional_analysis.pi_group_sets):
        text = pi_group_set.repeating_variables[0].name
        for repeating in pi_group_set.repeating_variables[1:]:
            text += ', ' + repeating.name
        with st.expander(text, expanded=True):
            for i, pi_group in enumerate(pi_group_set.pi_groups[1:]):
                plt.figure()
                plt.scatter(pi_group.values, pi_group_set.pi_groups[0].values)
                plt.xlabel(pi_group.formula, fontsize=14)
                plt.ylabel(pi_group_set.pi_groups[0].formula, fontsize=14)
                st.pyplot(plt)
        my_bar.progress((h+1) / len(dimensional_analysis.pi_group_sets))


def find_contours(img, threshold1=100, threshold2=200, blur=3):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (blur, blur), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=threshold1, threshold2=threshold2)
    return cv2.findContours(image=edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE), edges


# st.set_page_config(layout="wide")
st.title("Data Processor")

instructions = 'Upload a CSV file. Make sure the first row contains header information which should have the following formate: Name-units (e.g. Gravity-acceleration). Also avoid values of zero in the data set as this tends to lead to division by zero.'

option = st.sidebar.selectbox('Select the type of data to be processed', ('Select an Option', 'Images', 'CSV File'))
file = None
if option == 'CSV File':
    file = st.sidebar.file_uploader('CSV file', type=['csv'], help=instructions)
    st.subheader('Dimensional Analysis')
    with st.expander('What is Dimensional Analysis?'):
        intro_markdown = read_markdown_file("readme.md")
        st.markdown(intro_markdown)
    with st.expander('Instructions'):
        st.markdown(instructions)

    if file is not None:
        ds = pd.read_csv(file)
        st.sidebar.write("Here is the dataset used in this analysis:")
        st.sidebar.write(ds)

        data = Data(ds, pandas=True)
        d = DimensionalAnalysis(data.parameters)
        # figure, axes = d.pi_group_sets[0].plot()

        st.subheader('Generating Possible Figures')
        my_bar = st.progress(0)
        st.write('Different Sets of Repeating Variables')
        generate_plots(d)
        st.balloons()

elif option == 'Images':
    image_files = st.sidebar.file_uploader('Image Uploader', type=['tif', 'png', 'jpg'], help='Upload .tif files to to test threshold values for Canny edge detection. Note multiple images can be uploaded but there is a 1 GB RAM limit and the application can begin to slow down if more than a couple hundred images are uploaded', accept_multiple_files=True)
    st.subheader('Edges Detection and Contour Tracking')
    with st.expander('Instructions'):
        st.write('Upload images and then use the sliders to select the image and threshold values.')
    if len(image_files) > 0:
        image_number = 1
        if len(image_files) > 1:
            image_number = st.sidebar.slider('Image Number', min_value=1, max_value=len(image_files))
        image = np.array(Image.open(image_files[image_number-1]))
        try:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except cv2.error:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_copy = image.copy()

        threshold1 = st.sidebar.slider('Minimum Threshold', min_value=0, max_value=200, value=100, help='Any pixel below this threshold is eliminated, and any above are consider possible edges')
        threshold2 = st.sidebar.slider('Definite Threshold', min_value=0, max_value=200, value=200, help='Any pixel above this threshold is consider a definite edge. Additionally any pixel above the minimum threshold and connected to a pixel already determined to be an edge will be declared an edge')
        blur = st.sidebar.slider('blur', min_value=1, max_value=10, value=2, help='Filters out noise. Note: blur values must be odd so blur_value = 2 x slider_value + 1')

        (contours, _), edge_img = find_contours(image, threshold1=threshold1, threshold2=threshold2, blur=2*blur-1)
        image_copy = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 100, 55), thickness=1, lineType=cv2.LINE_AA)
        if st.sidebar.checkbox("Show just edges"):
            st.image(edge_img)
        else:
            st.image(image_copy)

else:
    st.subheader('Use the side bar to select the type of data you would like to process.')

