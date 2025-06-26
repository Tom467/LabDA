import numpy as np
from parameter import ListOfParameters


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