import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

st.set_page_config(layout="wide")
st.title("🔬 Buckingham Pi Theorem – 3D Pi Group Visualizer")

st.markdown("""
Upload a `.csv` file with numeric columns only.  
Define the physical dimensions for each variable using any of the 7 SI base units:

- `M`: Mass  
- `L`: Length  
- `T`: Time  
- `I`: Electric Current  
- `Θ`: Temperature  
- `N`: Amount of Substance  
- `J`: Luminous Intensity
""")

uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

SI_UNITS = ["M", "L", "T", "I", "Θ", "N", "J"]

def parse_dimensions(dim_str):
    dims = {u: 0 for u in SI_UNITS}
    for item in dim_str.split():
        if "^" in item:
            symbol, power = item.split("^")
            dims[symbol] = float(power)
        else:
            dims[item] = 1.0
    return [dims[u] for u in SI_UNITS]

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Data Preview")
    st.dataframe(df.head())

    variables = list(df.columns)
    st.write("### 🛠 Define Physical Dimensions")

    dimensions = []
    valid_inputs = True
    for var in variables:
        dim_input = st.text_input(
            f"Dimensions for `{var}` (e.g., 'M^1 L^2 T^-2')", key=var
        )
        if dim_input:
            try:
                dims = parse_dimensions(dim_input)
                dimensions.append(dims)
            except Exception as e:
                st.error(f"Error parsing dimensions for {var}: {e}")
                valid_inputs = False
        else:
            valid_inputs = False

    if valid_inputs and len(dimensions) == len(variables):
        st.success("✅ All dimensions parsed.")
        dim_matrix = np.array(dimensions).T
        rank = np.linalg.matrix_rank(dim_matrix)
        num_pi_groups = len(variables) - rank
        st.write(f"### 📐 Number of Dimensionless Groups: {num_pi_groups}")

        exponents = sp.Matrix(dim_matrix).nullspace()
        if len(exponents) < 3:
            st.warning("⚠️ At least 3 Pi groups are needed for 3D plots.")
        else:
            var_symbols = sp.symbols(variables)
            pi_data = []
            pi_expressions = []
            pi_names = []

            st.write("### 🧮 Dimensionless Groups (π₁, π₂, ...)")
            for idx, vec in enumerate(exponents):
                exps = np.array([float(e) for e in vec])
                pi_vals = np.prod(
                    [df[var] ** power for var, power in zip(variables, exps)],
                    axis=0
                )
                pi_data.append(pi_vals)
                pi_names.append(f"π{idx + 1}")

                # Display formula
                symbolic_expr = sp.Mul(*[sym**p for sym, p in zip(var_symbols, vec) if not sp.Eq(p, 0)])
                pi_expressions.append(symbolic_expr)
                st.latex(f"\\pi_{{{idx + 1}}} = {sp.latex(symbolic_expr)}")

            st.write("### 📈 3D Scatter Plots of π Groups")

            for (i, j, k) in combinations(range(len(pi_data)), 3):
                x = np.array(pi_data[i], dtype=np.float64)
                y = np.array(pi_data[j], dtype=np.float64)
                z = np.array(pi_data[k], dtype=np.float64)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, alpha=0.7)

                ax.set_xlabel(pi_names[i])
                ax.set_ylabel(pi_names[j])
                ax.set_zlabel(pi_names[k])
                ax.set_title(f"{pi_names[k]} vs {pi_names[j]} vs {pi_names[i]}")
                st.pyplot(fig)
