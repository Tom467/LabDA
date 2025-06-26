
class Fluid:
    def __init__(self, ideal_gas=False, temp=None, density=None, specific_weight=None, dynamic_viscosity=None, kinematic_viscosity=None, R=None, specific_heat_ratio=None):
        self.ideal_gas = ideal_gas
        self.temp = temp
        self.density = density
        self.specific_weight = specific_weight
        self.dynamic_viscosity = dynamic_viscosity
        self.kinematic_viscosity = kinematic_viscosity
        self.R = R
        self.specific_heat_ratio = specific_heat_ratio


class Flow:
    def __init__(self, fluid, length, velocity):
        self.fluid = fluid
        self.length = length
        self.velocity = velocity
        self.Re = None
        self.Ma = None


air = Fluid(ideal_gas=True, temp=15, density=1.23e1, specific_weight=1.2e1, dynamic_viscosity=1.79e-5, kinematic_viscosity=1.46e-5, R=2.869e2, specific_heat_ratio=1.4)
water = Fluid(ideal_gas=False, temp=20, density=998)

if __name__ == "__main__":
    print(0)