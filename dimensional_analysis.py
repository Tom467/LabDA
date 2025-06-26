import copy
import numpy as np
import matplotlib.pyplot as plt

from util import Util
from units import Units
from parameter import Parameter, ListOfParameters
from pi_group import PiGroup, PiGroupSet
from dimensional_matrix import DimensionalMatrix


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
