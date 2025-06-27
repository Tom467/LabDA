import streamlit as st
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path

# --- Utility functions ---
def get_prime():
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def factorization(value):
    factors = []
    for prime in get_prime():
        while value % prime == 0:
            factors.append(prime)
            value //= prime
        if value == 1:
            break
    return factors

def combinations(elements, k):
    from itertools import combinations as it_combinations
    return list(it_combinations(elements, k))

# --- Units ---
class BaseUnits:
    Nondimensional = 1
    Time = 2
    Mass = 3
    Angle = 1
    Length = 5
    Temperature = 7

class Unit:
    def __init__(self, numerator=1, denominator=1):
        self.n = numerator if isinstance(numerator, int) else numerator.n
        self.d = denominator if isinstance(denominator, int) else denominator.d
        self.factorization = [factorization(self.n), factorization(self.d)]
        self.independent_dimensions = list(dict.fromkeys(self.factorization[0] + self.factorization[1]))

    def __mul__(self, other):
        return Unit(self.n * other.n, self.d * other.d)

    def __truediv__(self, other):
        return Unit(self.n * other.d, self.d * other.n)

    def __pow__(self, power):
        return Unit(self.n**power, self.d**power)

    def __eq__(self, other):
        return self.n / self.d == other.n / other.d

    def __str__(self):
        if self.n == self.d:
            return '(non-dimensional)'
        return f"{self.n}/{self.d}"

class Units:
    nondimensional = Unit()
    T = Unit(numerator=BaseUnits.Time)
    M = Unit(numerator=BaseUnits.Mass)
    L = Unit(numerator=BaseUnits.Length)
    velocity = L / T
    length = L
    density = M / L**3
    viscosity_dynamic = M / (L * T)
    pressure = M / (L * T**2)

# --- Parameters ---
class Parameter:
    def __init__(self, value=None, units=Units.nondimensional, formula=None, name=''):
        self.values = np.array(value, dtype=np.float64)
        self.units = units
        self.formula = formula
        self.name = name

class ListOfParameters:
    def __init__(self, parameters):
        self._list = parameters
        self.units = [param.units for param in self._list]
        self.independent_dimensions = list({dim for u in self.units for dim in u.independent_dimensions})

    def __getitem__(self, i): return self._list[i]
    def __len__(self): return len(self._list)
    def __iter__(self): return iter(self._list)
    def append(self, item): self._list.append(item)



class DimensionalMatrix:
    def __init__(self, parameters):  # parameters should be of type ListOfParameters or ListOfUnits
        self.parameters = parameters if isinstance(parameters, ListOfParameters) else ListOfParameters(parameters)
        self.M = DimensionalMatrix.create_dimensional_matrix(self.parameters.units)
        self.rank = np.linalg.matrix_rank(self.M) if len(self.M) > 0 else 0

    def __str__(self):
        return str(self.M)

    @staticmethod
    def create_dimensional_matrix(units_of_parameters):
        matrix = np.zeros([len(units_of_parameters.independent_dimensions), len(units_of_parameters)])
        for i, dimension in enumerate(units_of_parameters.independent_dimensions):
            for j, unit in enumerate(units_of_parameters):
                matrix[i, j] = unit.factorization[0].count(dimension) - unit.factorization[1].count(dimension)
        return matrix



class PiGroup:
    def __init__(self, parameters):  # parameters should be of type ListOfParameters with the first parameter plus repeating variables
        self.parameters = parameters
        self.values = None
        self.exponents = None
        self._define_pi_group()
        self.formula = None
        self.formula_inverse = None
        self._define_formula()
        self.repeating_variables = parameters[1:]
        # TODO add some check to see if the Pi group is something common like the Reynold's Number

    def __str__(self):
        return self.formula  # str(self.values) + ' ' + str(self.formula)

    def __eq__(self, other):
        return self.formula == other.formula or self.formula == other.formula_inverse

    def _define_pi_group(self):
        M = DimensionalMatrix(self.parameters.units).M
        A, B = M[:, 1:], M[:, 0]
        self.exponents = np.round(-(np.linalg.inv(A) @ B), 2)  # TODO rounding might cause problems with small fractions
        self.values = self.calculate_value(self.parameters)
        # TODO add logic to make sure x is a vector of integers if raising units to this power

    def calculate_value(self, parameters):
        value = copy.deepcopy(parameters[0].values)
        for i, parameter in enumerate(parameters[1:]):
            value *= parameter.values**self.exponents[i]
        return value
        # TODO figure out what to return in addition to the total

    def contains(self, other_name):
        for param in self.parameters:
            if param.name == other_name:
                return True
        return False

    def _define_formula(self):
        top = ''
        bottom = ''
        for i, parameter in enumerate(self.parameters):
            if i == 0:
                top += f'({parameter.name})'
            else:
                if self.exponents[i-1] > 0:
                    if self.exponents[i-1] == 1:
                        top += f'({parameter.name})'
                    else:
                        top += f'({parameter.name}^'+'{'+f'{self.exponents[i-1]}'+'})'
                elif self.exponents[i-1] < 0:
                    if self.exponents[i-1] == -1:
                        bottom += f'({parameter.name})'
                    else:
                        bottom += f'({parameter.name}^'+'{'+f'{-self.exponents[i-1]}'+'})'

        self.formula = r'$\frac{t}{b}$'.replace('t', top).replace('b', bottom) if bottom else top
        self.formula_inverse = f'{bottom} / {top}' if top else bottom

class PiGroupSet:
    def __init__(self, parameters, repeating_variables):
        self.pi_groups = []
        self.parameters = parameters
        self.repeating_variables = repeating_variables
        self.create_pi_groups()

    def __len__(self):
        return len(self.pi_groups)

    def __str__(self):
        return str([pi_group.formula for pi_group in self.pi_groups])

    def __getitem__(self, index):
        return self.pi_groups[index]

    def __iter__(self):
        for elem in self.pi_groups:
            yield elem

    def create_pi_groups(self):
        # Clear pi_groups list before creating new groups
        self.pi_groups = []

        # We want to create a Pi group for each parameter that is NOT
        # one of the repeating variables.
        non_repeating = [p for p in self.parameters if p not in self.repeating_variables]

       # For each non-repeating parameter, create a PiGroup with it + all repeating variables
for param in non_repeating:
    # The Pi group consists of this param + all repeating variables
    # Put param first, then repeating variables to match your PiGroup init
    group_params = [param] + self.repeating_variables._list
    pi_group = PiGroup(group_params)
    self.pi_groups.append(pi_group)
     
    def plot(self):
        figure, axis = plt.subplots(1, len(self.pi_groups))
        for i, pi_group in enumerate(self.pi_groups[1:]):
            axis[i].scatter(pi_group.values, self.pi_groups[0].values)
            axis[i].set_ylabel(self.pi_groups[0].formula)
        return figure, axis


def plot(pi_group_x, pi_group_y):
    fig, ax = plt.subplots()
    ax.scatter(pi_group_x.values, pi_group_y.values)
    ax.set_xlabel(pi_group_x.formula)
    ax.set_ylabel(pi_group_y.formula)
    ax.grid(True)
    return fig




class DimensionalAnalysis:
    def __init__(self, parameters):
        self.parameters = ListOfParameters(parameters)
        self.repeating_variables_list = self.find_repeating_variables()

        # Now build PiGroupSet objects for each repeating variable set
        self.pi_group_sets = []
        for repeating_vars in self.repeating_variables_list:
            pi_set = PiGroupSet(self.parameters, repeating_vars)
            self.pi_group_sets.append(pi_set)

    def find_repeating_variables(self):
        combos = combinations(self.parameters._list, 3)
        return [ListOfParameters(list(c)) for c in combos[:3]]


# --- Data Reader ---
class Data:
    def __init__(self, file, pandas=False):
        self.data = file if pandas else pd.read_csv(file)
        self.parameters = self.generate_list_of_parameters()

    def generate_list_of_parameters(self):
        params = ListOfParameters([])
        for col in self.data:
            try:
                name, unit_str = col.split('-')
                unit = getattr(Units, unit_str, Units.nondimensional)
                param = Parameter(value=self.data[col].values, units=unit, name=name)
                params.append(param)
            except:
                pass
        return params

@st.cache_data
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def generate_plots(dimensional_analysis):
    st.write("Generated plots for Pi group sets")
    for i, pi_set in enumerate(dimensional_analysis.pi_group_sets):
        with st.expander(f"Group {i+1}"):
            st.write(f"Repeating variables: {[p.name for p in pi_set.repeating_variables]}")

            st.write("Sample values for each repeating variable:")
            for p in pi_set.repeating_variables:
                st.write(f"{p.name}: {p.values[:5]} ({p.units})")

            if len(pi_set) >= 2:
                st.pyplot(plot(pi_set[0], pi_set[1]))
            else:
                st.info("Not enough Pi groups to plot.")




def find_contours(img, threshold1=100, threshold2=200, blur=3):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (blur, blur), 0)
    edges = cv2.Canny(img_blur, threshold1, threshold2)
    return cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE), edges

# --- Streamlit UI ---
st.title("LabDA: User-friendly Dimensional Analysis and Edge Detection")

# CSV Upload and Processing
st.sidebar.header("Upload CSV")
csv_file = st.sidebar.file_uploader("Upload CSV File", type="csv", key="csv_upload")

if csv_file:
    df = pd.read_csv(csv_file)
    st.subheader("CSV Data Preview")
    st.write(df.head())
    data = Data(df, pandas=True)
    da = DimensionalAnalysis(data.parameters)
    generate_plots(da)

# Image Upload and Processing
st.sidebar.header("Upload Images")
image_files = st.sidebar.file_uploader(
    "Upload Image Files", 
    type=["png", "jpg"], 
    accept_multiple_files=True, 
    key="image_upload"
)

if image_files:
    t1 = st.sidebar.slider("Min Threshold", 0, 255, 100, key="image_min_thresh")
    t2 = st.sidebar.slider("Max Threshold", 0, 255, 200, key="image_max_thresh")
    blur = st.sidebar.slider("Blur (odd)", 1, 9, 3, key="image_blur")
    show_original = st.sidebar.checkbox("Show Original Images", value=False, key="image_show_orig")

    st.subheader(f"Processed Images ({len(image_files)} uploaded)")
    for i, file in enumerate(image_files):
        img = np.array(Image.open(file))
        (contours, _), edge_img = find_contours(img, t1, t2, blur)

        st.markdown(f"**Image {i + 1}**")
        if show_original:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original", use_container_width=True)
            with col2:
                st.image(edge_img, caption="Edge Map", use_container_width=True)
        else:
            st.image(edge_img, caption=f"Edge Map {i + 1}", use_container_width=True)

else:
    st.sidebar.info("Upload one or more images to perform edge detection.")
