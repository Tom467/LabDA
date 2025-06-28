import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔬 Buckingham Pi Theorem Analyzer (with Reciprocal Plots and FL Basis Support)")

st.markdown("""
Upload a `.csv` file with only numeric columns.  
Then choose a **dimension basis** and assign dimensions for each variable.

### Available Bases:
- `SI`: Mass (`M`), Length (`L`), Time (`T`), Electric Current (`I`), Temperature (`Θ`), Amount of Substance (`N`), Luminous Intensity (`J`)
- `FL`: Force (`F`), Length (`L`), Time (`T`)
""")

basis_choice = st.selectbox("Choose Dimension Basis", options=["SI (M L T I Θ N J)", "FL"])

if "SI" in basis_choice:
    DIM_UNITS = ["M", "L", "T", "I", "Θ", "N", "J"]
else:
    DIM_UNITS = ["F", "L", "T"]

def parse_dimensions(dim_str):
    dims = {u: 0 for u in DIM_UNITS}
    for item in dim_str.split():
        if "^" in item:
            symbol, power = item.split("^")
            dims[symbol] = float(power)
        else:
            dims[item] = 1.0
    return [dims[u] for u in DIM_UNITS]

uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

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
            f"Dimensions for `{var}` (e.g., 'M^1 L^2 T^-2' or 'F L^-2')", key=var
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
        if not exponents:
            st.error("❌ No nullspace found — variables are dimensionally dependent.")
        else:
            var_symbols = sp.symbols(variables)
            pi_data = []
            pi_labels = []

            st.write("### 🧮 Dimensionless Groups")
            for idx, vec in enumerate(exponents):
                exps = np.array([float(e) for e in vec])
                pi_vals = np.prod(
                    [df[var] ** power for var, power in zip(variables, exps)],
                    axis=0
                )
                pi_data.append(pi_vals)

                # Label
                terms = []
                for var, power in zip(variables, exps):
                    if abs(power) > 1e-8:
                        if abs(power - 1.0) < 1e-8:
                            terms.append(f"{var}")
                        else:
                            terms.append(f"{var}^{power:.2f}")
                label = " * ".join(terms) if terms else "1"
                pi_labels.append(label)

                latex_expr = sp.Mul(*[sym**p for sym, p in zip(var_symbols, vec) if not sp.Eq(p, 0)])
                st.latex(f"\\pi_{{{idx+1}}} = {sp.latex(latex_expr)}")

            st.write("### 📈 Plots of Pi Groups and Their Reciprocals")

            for i in range(len(pi_data)):
                for j in range(i + 1, len(pi_data)):
                    # Regular plot with smaller size and padding
                    fig1, ax1 = plt.subplots(figsize=(5, 4))
                    ax1.scatter(pi_data[i], pi_data[j], alpha=0.7)
                    ax1.set_xlabel(f"π{i+1}: {pi_labels[i]}")
                    ax1.set_ylabel(f"π{j+1}: {pi_labels[j]}")
                    ax1.set_title(f"π{j+1} vs π{i+1}")

                    # Axis limits with padding
                    x_min, x_max = np.nanmin(pi_data[i]), np.nanmax(pi_data[i])
                    y_min, y_max = np.nanmin(pi_data[j]), np.nanmax(pi_data[j])
                    x_pad = (x_max - x_min) * 0.05 if x_max != x_min else 1
                    y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 1
                    ax1.set_xlim(x_min - x_pad, x_max + x_pad)
                    ax1.set_ylim(y_min - y_pad, y_max + y_pad)
                    ax1.grid(True)

                    st.pyplot(fig1)

                    # Reciprocal plot
                    with np.errstate(divide='ignore', invalid='ignore'):
                        reciprocal_x = np.where(pi_data[i] != 0, 1 / pi_data[i], np.nan)
                        reciprocal_y = np.where(pi_data[j] != 0, 1 / pi_data[j], np.nan)

                    valid = ~np.isnan(reciprocal_x) & ~np.isnan(reciprocal_y)
                    if np.any(valid):
                        fig2, ax2 = plt.subplots(figsize=(5, 4))
                        ax2.scatter(reciprocal_x[valid], reciprocal_y[valid], alpha=0.7, color='orange')
                        ax2.set_xlabel(f"1 / π{i+1}")
                        ax2.set_ylabel(f"1 / π{j+1}")
                        ax2.set_title(f"Reciprocal: 1/π{j+1} vs 1/π{i+1}")

                        # Axis limits with padding for reciprocal plots
                        x_min, x_max = np.nanmin(reciprocal_x[valid]), np.nanmax(reciprocal_x[valid])
                        y_min, y_max = np.nanmin(reciprocal_y[valid]), np.nanmax(reciprocal_y[valid])
                        x_pad = (x_max - x_min) * 0.05 if x_max != x_min else 1
                        y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 1
                        ax2.set_xlim(x_min - x_pad, x_max + x_pad)
                        ax2.set_ylim(y_min - y_pad, y_max + y_pad)
                        ax2.grid(True)

                        st.pyplot(fig2)
                    else:
                        st.warning(f"⚠️ Reciprocal plot for π{i+1} vs π{j+1} skipped (all values zero or invalid).")
