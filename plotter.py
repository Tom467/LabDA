import matplotlib.pyplot as plt


def plot(x_parameter, y_parameter):
    plt.plot(x_parameter.value, y_parameter.value)
    plt.xlabel(f'{x_parameter.name} ({x_parameter.units})')
    plt.ylabel(f'{y_parameter.name} ({y_parameter.units})')