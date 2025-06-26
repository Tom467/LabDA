import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import copy




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



    
class DimensionalAnalysis:
    def __init__(self, parameters, dependent_parameter=None):  # parameters should be of type ListOfParameter
        # TODO add logic to accept just a list of units list of parameters without values
        self.parameters = ListOfParameters(parameters)
        self.dependent_parameter = dependent_parameter
        self.units_of_parameters = self.parameters.units
        self.independent_dimensions = self.units_of_parameters.independent_dimensions
        self.number_of_pi_groups = self.calculate_number_of_pi_groups()
        self.dimensional_matrix = DimensionalMatrix(self.units_of_parameters)
        self.repeating_variables = self.find_repeating_variables()
        self.pi_groups = []
        self.pi_group_sets = [PiGroupSet(self.parameters, repeating_group) for repeating_group in self.repeating_variables]
        # self.create_pi_groups()
        self.regression_model = self.generate_model(self.pi_group_sets[0])
        # self.generate_model()

    def predict(self,):
        values = np.array([copy.deepcopy(pi_group.values) for pi_group in self.pi_groups[1:]])
        return self.regression_model.predict(values.T)

    @staticmethod
    def generate_model(pi_group_set):
        x = np.array([copy.deepcopy(pi_group.values) for pi_group in pi_group_set[1:]])
        y = copy.deepcopy(pi_group_set[0].values)
        return GradientDescent(np.transpose(x), y)

    def calculate_number_of_pi_groups(self):
        n = len(self.units_of_parameters)
        r = len(self.independent_dimensions)
        return n - r

    def find_repeating_variables(self):
        repeating_variables = []
        combinations = Util.combinations(self.parameters, self.dimensional_matrix.rank)
        for group in combinations:
            M = DimensionalMatrix(group)
            if M.rank == self.dimensional_matrix.rank:
                repeating_variables.append(ListOfParameters(group))
        return repeating_variables

    # def create_pi_groups(self):
    #     group = self.parameters - self.repeating_variables[0]
    #     for variable in group:
    #         pi_group = PiGroup(ListOfParameters([variable]) + self.repeating_variables[0])
    #         self.pi_groups.append(pi_group)
    #     test = True
    #     if test:
    #         pass
    #     else:
    #         The following loop can find All the possible pi groups from all the different combinations of repeating variables
    #         for repeating_variables in self.repeating_variables:
    #             group = self.parameters - repeating_variables
    #             for variable in group:
    #                 pi_group = PiGroup(ListOfParameters([variable]) + repeating_variables)
    #                 self.pi_groups.append(pi_group)
    #                 # TODO the following if statement should not be needed
    #                 # if pi_group not in self.pi_groups:
    #                 #     self.pi_groups.append(pi_group)

    def plot(self):
        for i, pi_group_set in enumerate(self.pi_group_sets):
            axis = pi_group_set.plot()
            # y1, y = self.generate_model(pi_group_set).plot_data()
            # axis[-1].plot(y1, y1, c='r')  # , label='predicted')
            # axis[-1].scatter(y1, y, s=10, c='b', marker="s")  # , label='measured')
            # axis[2, h].legend(loc='upper left')
        return axis  # plt.show()


if __name__ == '__main__':

    # U, y, h, rho, mu, dpdx
    u = Parameter(value=np.array([1,1,1]), units=Units.velocity, name='u')
    U = Parameter(value=np.array([1,1,1]), units=Units.velocity, name='U')
    y = Parameter(value=np.array([1,1,1]), units=Units.length, name='y')
    h = Parameter(value=np.array([1,1,1]), units=Units.length, name='h')
    rho = Parameter(value=np.array([1,1,1]), units=Units.density, name='rho')
    mu = Parameter(value=np.array([1,1,1]), units=Units.viscosity_dynamic, name='mu')
    p = Parameter(value=np.array([1,1,1]), units=Units.pressure, name='p')
    x = Parameter(value=np.array([1,1,1]), units=Units.length, name='x')
    problem = ListOfParameters([U, h, rho, mu, u, y, p, x])
    solution = DimensionalAnalysis(problem)
    print(solution.repeating_variables)
    # for group in solution.pi_groups:
    #     print('pi group', group.formula)

    # TODO test a list of parameters that are dimensionless then with groups of ranks 1-5
    # TODO test a dimensionless group that has fractional exponents



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

    def __str__(self):
        return str([pi_group.formula for pi_group in self.pi_groups])

    def __getitem__(self, index):
        return self.pi_groups[index]

    def __iter__(self):
        for elem in self.pi_groups:
            yield elem  # TODO what does yield do?

    def create_pi_groups(self):
        non_repeating = self.parameters - self.repeating_variables
        for variable in non_repeating:
            pi_group = PiGroup(ListOfParameters([variable]) + self.repeating_variables)
            self.pi_groups.append(pi_group)

    def plot(self):
        figure, axis = plt.subplots(1, len(self.pi_groups))
        for i, pi_group in enumerate(self.pi_groups[1:]):
            axis[i].scatter(pi_group.values, self.pi_groups[0].values)
            axis[i].set_ylabel(self.pi_groups[0].formula)
        return figure, axis

        
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



if __name__ == "__main__":
    experiment = Data("C:/Users/truma/Downloads/test - bernoulli_v2.csv")
    d = DimensionalAnalysis(experiment.parameters)
    d.plot()

    # [print(group, '\n', group.repeating_variables) for group in d.pi_group_sets]

    values = [80, 20, 9.8, 1, 1, 1]
    test = ListOfParameters([])
    for i, parameter in enumerate(experiment.parameters[1:]):
        # print(Parameter(value=values[i], units=parameter.units, name=parameter.name))
        test.append(Parameter(value=values[i], units=parameter.units, name=parameter.name))
    print('test', test)
    # test = d.predict(experiment.parameters)

    

class Parameter:
    def __init__(self, value=None, units=Units.nondimensional, formula=None, name=''):
        # TODO use units to convert the value (example: if value=10 & units=cm, convert to value=0.01 & units=m)
        self.values = np.array(value, dtype=np.float64)
        self.units = units
        self.formula = formula
        self.name = name

    def __str__(self):
        return self.name + (f' {self.values}' if self.values is not None else '') + f' {self.units}' if self.units else ''

    def __add__(self, other):
        if isinstance(other, ListOfParameters):
            return ListOfParameters(ListOfParameters([self]) + other)
        return Parameter(value=(self.values + other.values), units=self.units)

    def __sub__(self, other):
        return Parameter(value=(self.values - other.values), units=self.units)

    def __mul__(self, other):
        if isinstance(other, Parameter):
            return Parameter(value=(self.values * other.values), units=(self.units * other.units))
        return Parameter(value=(self.values * other), units=self.units)

    def __truediv__(self, other):
        if isinstance(other, Parameter):
            return Parameter(value=(self.values / other.values), units=(self.units / other.units))
        return Parameter(value=(self.values / other), units=self.units)

    def __pow__(self, power, modulo=None):
        if self.values is not None:
            return Parameter(value=(self.values ** power), units=(self.units ** power))
        else:
            return Parameter(units=(self.units ** power))

    def __eq__(self, other):
        return (self.values == other.values).all() and self.units == other.units


class Values:
    def __init__(self, values):
        if isinstance(values, Values):
            self._list = [value for value in values]
        else:
            self._list = values if isinstance(values, list) else [values]

    def __getitem__(self, index):
        return self._list[index]

    def __str__(self):
        return str([str(item) for item in self._list])

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        for elem in self._list:
            yield elem

    def __eq__(self, other):
        if not isinstance(other, Values):
            return False
        if len(self) == len(other):
            for item in self:
                if item not in other:
                    return False
            return True
        return False

    def __mul__(self, other):
        if isinstance(other, Values):
            total = Values([])
            for i, _ in enumerate(self):
                value = self[i] * other[i]
                total.append(value)
            return total
        return Values([value * other for value in self])

    def __truediv__(self, other):
        return [value / other for value in self]

    def __pow__(self, power, modulo=None):
        return Values([value**power if value else None for value in self])

    def append(self, item):
        self._list.append(item)


class ListOfParameters:
    # TODO make a generic class for custom lists and have ListOfParameters and ListOfUnits inherit from it
    def __init__(self, list_of_parameters):  # items in list should be of type Parameters or Units
        # TODO make this a dictionary instead of a list
        self._list = [parameter if isinstance(parameter, Parameter) else Parameter(units=parameter) for parameter in list_of_parameters]
        self.names = [parameter.name for parameter in self]
        self.units = None
        self.independent_dimensions = None
        self.calculate_units()
        self.length = len(self)

    def __getitem__(self, index):
        return self._list[index]

    def __str__(self):
        return str([parameter.name for parameter in self])

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        for elem in self._list:
            yield elem  # TODO what does yield do?

    def __eq__(self, other):
        if len(self) == len(other):
            for item in self:
                if item not in other:
                    return False
            return True
        return False

    def __add__(self, other):
        if isinstance(other, Parameter):
            return ListOfParameters(other + ListOfParameters([self]))
        # TODO find better naming
        stuff = [parameter for parameter in self]
        for elem in other:
            stuff.append(elem)
        return ListOfParameters(stuff)

    def __sub__(self, other):
        parameters = ListOfParameters(self)
        for elem in other:
            parameters.delete(elem)
        return ListOfParameters(parameters)

    def calculate_units(self):
        self.units = ListOfUnits([parameter.units for parameter in self._list])
        self.calculate_independent_dimensions()

    def calculate_independent_dimensions(self):
        units = []
        for unit in self.units:
            units += unit.independent_dimensions
        self.independent_dimensions = list(dict.fromkeys(units))

    def delete_duplicate_named_parameters(self):
        # TODO fix this function (it deletes all the parameters in the list that are the same, instead of leaving one)
        for parameter in self:
            if self.name_count(parameter) > 1:
                self.delete(parameter)

    def append(self, item):
        self._list.append(item)
        self.calculate_units()

    def delete(self, element):
        for parameter in self:
            if parameter == element:
                self._list.remove(parameter)
                self.calculate_units()
                return 0

    def count(self, element):
        counter = 0
        for item in self:
            if item == element:
                counter += 1
        return counter

    def name_count(self, element):
        counter = 0
        for item in self:
            if item.name == element.name:
                counter += 1
        return counter
        
    def dot(self, other):
        # TODO test this method, it is completely untested
        return ListOfParameters([item * other[i] for i, item in enumerate(self._list)])

    def included_within(self, other):
        for group in other:
            if self == group:
                return True
        return False


class Common:
    def __init__(self):
        self.stuff = None
    constants = []


if __name__ == "__main__":
    print('Parameter() main script')
    test = ListOfParameters([Units.velocity, Units.length])
    test2 = [1, 2, 3]
    test1 = [1, 2, 3.0]
    print(test2 == test1)

    print(test.length)

    
def plot(x_parameter, y_parameter):
    plt.plot(x_parameter.value, y_parameter.value)
    plt.xlabel(f'{x_parameter.name} ({x_parameter.units})')
    plt.ylabel(f'{y_parameter.name} ({y_parameter.units})')

    
class BaseUnits:
    Nondimensional = 1
    Time = 2
    Mass = 3
    Angle = 1
    Length = 5
    Temperature = 7

    @staticmethod
    def numbers_to_names(value, number_to_name):
        factorization = Util.factorization(value)
        units = ''
        for number in number_to_name:
            count = factorization.count(number)
            if count == 0:
                pass
            else:
                units += number_to_name[number] + (f'^{count}' if count > 1 else '') + ' * '
        return units.rstrip(" *")


class GaloisField:
    def __init__(self, members):
        self.conversion = {Util.get_prime()[i]: variable for i, variable in enumerate(members)}


# SIUnits = GaloisField(['kg', 's', 'rad', 'm', 'K'])


class UnitSystem:
    def __init__(self, mass='Mass', time='Time', angle='Angle', length='Length', temperature='Temperature'):
        self.number_to_name = {BaseUnits.Mass: mass,
                               BaseUnits.Time: time,
                               BaseUnits.Angle: angle,
                               BaseUnits.Length: length,
                               BaseUnits.Temperature: temperature}

    def factorization_to_names(self, factorization):
        # TODO print '1' instead of '' when there is no unit in the numerator
        top, bottom = '', ''
        for number in self.number_to_name:
            count_top = factorization[0].count(number)
            count_bottom = factorization[1].count(number)
            if count_top == 0:
                pass
            else:
                top += self.number_to_name[number] + (f'^{count_top}' if count_top > 1 else '') + ' * '
            if count_bottom == 0:
                pass
            else:
                bottom += self.number_to_name[number] + (f'^{count_bottom}' if count_bottom > 1 else '') + ' * '
        return '(' + top.rstrip(" *") + (') / (' + bottom.rstrip(" *") + ')' if bottom.rstrip(" *") else ')')


FundamentalUnits = UnitSystem(mass='Mass', time='Time', angle='Angle', length='m', temperature='Temp')
SIUnits = UnitSystem(mass='kg', time='s', angle='rad', length='m', temperature='K')


class Unit:
    def __init__(self, numerator=1, denominator=1, unit_system=SIUnits):
        if isinstance(numerator, Unit):
            self.n = numerator.n
            self.d = numerator.d
        else:
            self.n = numerator
        if isinstance(denominator, Unit):
            self.n = self.n * denominator.d
            self.d = self.d * denominator.n
        else:
            self.d = denominator
        self.unit_system = unit_system
        self.factorization = [Util.factorization(self.n), Util.factorization(self.d)]
        self.independent_dimensions = list(dict.fromkeys(self.factorization[0] + self.factorization[1]))
        self.simplify_common_factors()

    def __mul__(self, other):
        # TODO Check if unit system of self and other are the same
        numerator = self.n * other.n
        denominator = self.d * other.d
        return Unit(numerator=numerator, denominator=denominator, unit_system=self.unit_system)

    def __pow__(self, power, modulo=None):  # power should be of type int
        # TODO Figure out how to handle franctional powers
        if not isinstance(power, int):
            raise Warning('Units can only only be raised to the power of integers')
        result = self
        if power > 0:
            for _ in range(1, power):
                result *= self
            return Unit(numerator=result.n, denominator=result.d, unit_system=self.unit_system)
        elif power < 0:
            for _ in range(1, -power):
                result *= self
            return result.inv()
        return Unit(unit_system=self.unit_system)

    def __truediv__(self, other):
        # TODO Check if unit system of self and other are the same
        numerator = self.n * other.d
        denominator = self.d * other.n
        return Unit(numerator=numerator, denominator=denominator, unit_system=self.unit_system)

    def __eq__(self, other):
        # TODO Check if unit system of self and other are the same
        return self.n / self.d == other.n / other.d

    def __ne__(self, other):
        # TODO Check if unit system of self and other are the same
        return self.n / self.d != other.n / other.d

    def __str__(self):
        # TODO
        if self.n == self.d:
            return '(non-dimensional)'
        if self.unit_system is not None:
            return self.unit_system.factorization_to_names(self.factorization)
        else:
            return str(self.n) + ' / ' + str(self.d)

    def inv(self):
        return Unit(numerator=self.d, denominator=self.n, unit_system=self.unit_system)

    def simplify_common_factors(self):
        b = False
        # if self.factorization[0] == [2, 3, 11, 11]:
        #     b = True
        #     print(self.factorization)
        n_factors = self.factorization[0]
        d_factors = self.factorization[1]
        # if b:
        #     for test in n_factors:
        #         print('test', test)
        for number in n_factors:
            # if b:
            #     print('before', number, n_factors, d_factors)
            if number in d_factors:
                n_factors.remove(number)
                d_factors.remove(number)
            # if b:
            #     print('after', number, n_factors, d_factors)
        self.n, self.d = 1, 1
        for x in n_factors:
            self.n *= x
        for x in d_factors:
            self.d *= x


class Units:
    nondimensional = Unit()
    T = Unit(numerator=BaseUnits.Time)
    M = Unit(numerator=BaseUnits.Mass)
    L = Unit(numerator=BaseUnits.Length)
    theta = Unit(numerator=BaseUnits.Angle)
    Temp = Unit(numerator=BaseUnits.Temperature)

    acceleration = L / T ** 2  # 2.75
    angle = theta  # 5
    angular_acceleration = theta / T ** 2  # 1.25
    angular_velocity = theta / T  # 2.5
    area = L ** 2  # 121
    charge = L ** 2 * T
    density = M / L ** 3  # 0.002253944402704733
    energy = M * L ** 2 / T ** 2  # 90.75
    entropy = energy / Temp  # ###########################################
    force = M * L / T ** 2  # 8.25
    frequency = Unit(numerator=1, denominator=BaseUnits.Time)  # 0.5
    heat = M * L ** 2 / T ** 2  # 90.75
    length = L  # 11
    mass = M  # 3
    modulus_of_elasticity = M / L / T ** 2  # 0.0681818181818
    moment_of_force = M * L ** 2 / T ** 2  # 90.75
    moment_of_inertia_area = L ** 4  # 14641
    moment_of_inertia_mass = M * L ** 2  # 363
    momentum = M * L / T  # 16.5
    power = M * L ** 2 / T ** 3  # 45.375
    pressure = M / L / T ** 2  # 0.06818181818
    specific_heat = L ** 2 / T ** 2 / Temp  # 2.326923076923077
    specific_weight = M / L ** 2 / T ** 2  # 0.006198347107438017
    strain = L/L  # 1
    stress = M / L / T ** 2  # 0.0681818181818
    surface_tension = M / T ** 2  # 0.75
    temperature = Temp  # 13
    time = T  # 2
    torque = M * L ** 2 / T ** 2  # 90.75
    velocity = L / T  # 5.5
    viscosity_dynamic = M / L / T  # 0.136363636
    viscosity_kinematic = L ** 2 / T  # 60.5
    voltage = M * L ** 2 / T ** 3 / L ** 2
    volume = L ** 3  # 1331
    work = M * L ** 2 / T ** 2  # 90.75
    g = L / T ** 2  # 2.75
    Q = volume / T  # 665.5
    A = area  # 121

    # Constants
    boltzmanns_constant = force * L / Temp
    plancks_constant = L**2 * M / T

    def get_units(self):
        return [name for name in dir(self) if '__' not in name]


class ListOfUnits:
    def __init__(self, list_of_units):
        self._list_of_units = list_of_units
        units = []
        for unit in self._list_of_units:
            units += unit.independent_dimensions
        self.independent_dimensions = list(dict.fromkeys(units))

    def __getitem__(self, index):
        return self._list_of_units[index]

    def __str__(self):
        return str([str(unit) for unit in self._list_of_units])

    def __len__(self):
        return len(self._list_of_units)

    def __eq__(self, other):
        if len(self) == len(other):
            for param in self:
                if param not in other:
                    return False
            return True
        return False

    def __iter__(self):
        for elem in self._list_of_units:
            yield elem

    def __add__(self, other):
        for elem in other:
            self.append(elem)

    def append(self, item):
        self._list_of_units.append(item)


if __name__ == "__main__":
    c = Units.area * Units.viscosity_dynamic / (Units.mass * Units.length)
    test = ListOfUnits([Units.mass, Units.density])
    print(len(test))
    print(Units().get_units())
    print(Units.force)
    # TODO why is the following code returning (kg * m) / (kg) instead of (m)?? "print(Units.mass * Units.velocity * Units.length / (Units.area * Units.viscosity_dynamic))"




# class Units:
#     def __init__(self):
#         base = BaseUnits()
#         T = Unit(numerator=base.Time)
#         M = Unit(numerator=base.Mass)
#         L = Unit(numerator=base.Length)
#         theta = Unit(numerator=base.Angle)
#         Temp = Unit(numerator=base.Temperature)
#
#         self.acceleration = L/T**2  # 2.75
#         self.angle = theta  # 5
#         self.angular_acceleration = theta/T**2  # 1.25
#         self.angular_velocity = theta/T  # 2.5
#         self.area = L**2  # 121
#
#         self.density = M/L**3  # 0.002253944402704733
#         self.energy = M*L**2/T**2  # 90.75
#         self.force = M*L/T**2  # 8.25
#         self.frequency = Unit(numerator=1, denominator=base.Time)  # 0.5
#         self.heat = M*L**2/T**2  # 90.75
#
#         self.length = L  # 11
#         self.mass = M  # 3
#         self.modulus_of_elasticity = M/L/T**2  # 0.0681818181818
#         self.moment_of_force = M*L**2/T**2  # 90.75
#         self.moment_of_inertia_area = L**4  # 14641
#
#         self.moment_of_inertia_mass = M*L**2  # 363
#         self.momentum = M*L/T  # 16.5
#         self.power = M*L**2/T**3  # 45.375
#         self.pressure = M/L/T**2  # 0.06818181818
#         self.specific_heat = L**2/T**2/Temp  # 2.326923076923077
#
#         self.specific_weight = M/L**2/T**2  # 0.006198347107438017
#         self.strain = L/L  # 1
#         self.stress = M/L/T**2  # 0.0681818181818
#         self.surface_tension = M/T**2  # 0.75
#         self.temperature = Temp  # 13
#
#         self.time = T  # 2
#         self.torque = M*L**2/T**2  # 90.75
#         self.velocity = L/T  # 5.5
#         self.viscosity_dynamic = M/L/T  # 0.136363636
#         self.viscosity_kinematic = L**2/T  # 60.5
#
#         self.volume = L**3  # 1331
#         self.work = M*L**2/T**2  # 90.75
#
#         self.g = L/T**2  # 2.75
#         self.Q = self.volume/T  # 665.5
#         self.A = self.area  # 121

class Util:
    @staticmethod
    def factorization(value):
        prime_numbers = Util.get_prime()
        prime_factors = []
        i = 0
        while value > 1:
            remainder = value % prime_numbers[i]
            if remainder == 0:
                prime_factors.append(prime_numbers[i])
                value /= prime_numbers[i]
                i -= 1
            i += 1
            try:
                prime = prime_numbers[i]
            except IndexError:
                # print('Warning: factorization includes prime numbers greater than 1009')
                return []
        return prime_factors

    @staticmethod
    def primatize(array):
        prime = Util.get_prime()
        group = dict()
        for i, element in enumerate(array):
            group[prime[i]] = element
        return group

    @staticmethod
    def combinations(array, r):  # r is the length of the desired combination arrays
        # TODO this algorithm for finding unique combinations can be improved
        primatized = Util.primatize(array)
        groups = [0] * int(Util.factorial(len(array)) / (Util.factorial(r) * Util.factorial(len(array) - r)))
        value = 2
        for i, _ in enumerate(groups):
            combination = []
            while len(combination) != r or not Util.combination_is_unique(combination) or not Util.test(combination, primatized):
                combination = Util.factorization(value)
                value += 1
            groups[i] = [primatized[j] for j in combination]
        return groups

    @staticmethod
    def test(combination, primification):
        try:
            [primification[j] for j in combination]
        except KeyError:
            return False
        return True

    @staticmethod
    def combination_is_unique(combination):
        for element in combination:
            if combination.count(element) != 1:
                return False
        return True

    @staticmethod
    def simplify_common_factors(numerator, denominator):
        n_factors = Util.factorization(numerator)
        d_factors = Util.factorization(denominator)
        for number in n_factors:
            if number in d_factors:
                n_factors.remove(number)
                d_factors.remove(number)
        numerator, denominator = 1, 1
        for x in n_factors:
            numerator *= x
        for x in d_factors:
            denominator *= x
        return numerator, denominator

    @staticmethod
    def multiply_list_values(list):
        value = list[0] / list[0]
        for item in list:
            value *= item
        return value

    @staticmethod
    def factorial(integer):
        factorial = 1
        for i in range(1, integer+1):
            factorial *= i
        return factorial


########################################################################################################################

    @staticmethod
    def read_prime_numbers():
        f = open('prime_numbers.txt', 'r')
        c = []
        for a in f:
            for b in a.split():
                c.append(int(b))
        return c
        f.close()

    @staticmethod
    def get_prime():
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
                211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331,
                337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457,
                461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
                601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
                739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877,
                881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009]


if __name__ == '__main__':
    primatization = Util.primatize(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o'])
    comb = Util.combinations(primatization, 3)
    print('length', len(comb))

    
class Convert:
    def __init__(self):
        # common constants
        self.g = 9.81  # on earth
        self.sg = 1000  # specific gravity

        # Length
        self.m = 1
        self.km = 1000 * self.m
        self.cm = 1 / 100 * self.m
        self.mm = 1 / 1000 * self.m
        self.inch = 2.54 * self.cm
        self.ft = 12 * self.inch
        self.yard = 3 * self.ft
        self.mi = 5280 * self.ft

        # Time
        self.s = 1
        self.min = 60 * self.s
        self.hr = 60 * self.min
        self.day = 24 * self.hr
        self.week = 7 * self.day
        self.year = 365 * self.day

        # Mass
        self.kg = 1
        self.gram = 1 / 1000 * self.kg
        self.lbm = 1 / 2.2046 * self.kg
        self.slug = 14.594 * self.kg

        # Angle
        self.rad = 1
        self.deg = 3.141592653589793238462643383279/180 * self.rad

        # Force
        self.N = self.kg * self.m / self.s ** 2
        self.lbs = 4.448 * self.N

        # Energy
        self.J = self.N * self.m

        # Pressure
        self.Pa = 1
        self.psi = 6.895e3 * self.Pa

        self.prefix = {
            'peta':  10 ** 15,
            'tera':  10 ** 12,
            'giga':  10 ** 9,
            'mega':  10 ** 6,
            'kilo':  10 ** 3,
            'hecto': 10 ** 2,
            'deka':  10,
            'deci':  10 ** -1,
            'centi': 10 ** -2,
            'milli': 10 ** -3,
            'micro': 10 ** -6,
            'nano':  10 ** -9,
            'pico':  10 ** -12,
            'femto': 10 ** -15,
            'atto':  10 ** -18
        }


class ConvertTemperature:
    def __init__(self, temp):
        
        # Temperature
        self.K = 1
        self.C = (temp + 273.15) / temp
        self.F = ((temp - 32) * 5 / 9 + 273.15) / temp
        self.Rankine = 0.556
