from util import Util


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