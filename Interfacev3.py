import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import re
import cv2
from PIL import Image

# ---------------------
# IMAGE PROCESSING SECTION (UNCHANGED)
# ---------------------
st.title("Image Processing & Dimensional Analysis App")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_array = np.array(image)

    st.image(image_array, caption='Original Image', use_column_width=True)

    if image_array.ndim == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        st.image(gray_image, caption='Grayscale Image', use_column_width=True)

        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        st.image(blurred_image, caption='Blurred Image', use_column_width=True)

        edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)
        st.image(edges, caption='Edge Detected Image', use_column_width=True)
    else:
        st.warning("Image appears to be grayscale already.")

# ---------------------
# DIMENSIONAL ANALYSIS SECTION
# ---------------------

st.subheader("Upload CSV File for Dimensional Analysis")
uploaded_csv = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.write("Raw Data Preview:")
    st.dataframe(df.head())

    # Parse units from column names like "velocity-L/T"
    def parse_unit(unit_str):
        unit_map = {'L': 1, 'T': 2, 'M': 3}
        exponents = [0, 0, 0]  # [L, T, M]
        matches = re.findall(r'([MTL])(?:\^(-?\d+))?', unit_str)
        for dim, exp in matches:
            idx = unit_map[dim] - 1
            exponents[idx] += int(exp) if exp else 1
        return tuple(exponents)

    class Parameter:
        def __init__(self, name, values, unit_str):
            self.name = name
            self.values = np.array(values, dtype=float)
            self.unit_str = unit_str
            self.exponents = parse_unit(unit_str)  # (L, T, M)

    parameters = []
    for col in df.columns:
        try:
            name, unit_str = col.split("-")
            param = Parameter(name.strip(), df[col].values, unit_str.strip())
            parameters.append(param)
        except ValueError:
            st.warning(f"Column `{col}` ignored. Must be in format: name-unit (e.g., velocity-L/T)")
            continue

    if len(parameters) < 4:
        st.warning("You need at least 4 parameters for meaningful Pi groups.")
    else:
        st.success("Parameters parsed successfully!")

        # Create dimensional matrix
        exponents_matrix = np.array([p.exponents for p in parameters]).T
        rank = np.linalg.matrix_rank(exponents_matrix)
        num_pi_groups = len(parameters) - rank

        st.write(f"Dimensional Matrix Rank: {rank}")
        st.write(f"Number of Pi Groups: {num_pi_groups}")

        # Try different combinations of repeating variables
        st.write("Generated Pi Groups (plotted below):")
        pi_plots = []

        found = False
        for combo in combinations(parameters, rank):
            A = np.array([p.exponents for p in combo]).T
            if np.linalg.matrix_rank(A) < rank:
                continue

            remaining = [p for p in parameters if p not in combo]
            for target in remaining:
                b = np.array(target.exponents)
                try:
                    x = np.linalg.solve(A, b)
                    exponents = np.round(-x, 4)
                    pi_values = target.values.copy()
                    for coeff, base_param in zip(exponents, combo):
                        pi_values *= base_param.values ** coeff

                    fig, ax = plt.subplots()
                    ax.scatter(range(len(pi_values)), pi_values)
                    label = f"{target.name} × " + " × ".join([
                        f"{p.name}^{round(e,2)}" for p, e in zip(combo, exponents)
                    ])
                    ax.set_title(f"Pi Group: {label}")
                    ax.set_ylabel("Dimensionless Value")
                    ax.set_xlabel("Data Point Index")
                    st.pyplot(fig)
                    found = True
                except np.linalg.LinAlgError:
                    continue

        if not found:
            st.error("Failed to compute any Pi groups. Check for linearly independent base units.")
