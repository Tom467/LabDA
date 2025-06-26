import copy
import numpy as np
import matplotlib.pyplot as plt
from parameter import ListOfParameters
from dimensional_matrix import DimensionalMatrix


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