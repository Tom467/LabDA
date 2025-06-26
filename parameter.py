import numpy as np
from units import Units, ListOfUnits


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