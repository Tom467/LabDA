import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔬 Buckingham Pi Theorem Analyzer (with Reciprocal Plots)")

st.markdown("""
Upload a `.csv` file with only numeric columns.  
Then assign dimensions using any combination of the 7 SI base units:

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
        if not exponents:
            st.error("❌ No nullspace found — variables are dimensionally dependent.")
        else:
            var_symbols = sp.symbols(variables)
            pi_data = []
            pi_labels = []

            st.write("### 🧮 Dimensionless Groups")
            for vec in exponents:
                exps = np.array([float(e) for e in vec])
                pi_vals = np.prod(
                    [df[var] ** power for var, power in zip(variables, exps)],
                    axis=0
                )
                pi_data.append(pi_vals)

                # Human-readable label
                terms = []
                for var, power in zip(variables, exps):
                    if abs(power) > 1e-8:
                        if abs(power - 1.0) < 1e-8:
                            terms.append(f"{var}")
                        else:
                            terms.append(f"{var}^{power:.2f}")
                label = " * ".join(terms)
                pi_labels.append(label)

                # Display as LaTeX
                st.latex(
                    f"\\text{{Dimensionless Group}}: {sp.latex(sp.Mul(*[sym**p for sym, p in zip(var_symbols, vec) if not sp.Eq(p, 0)]))}"
                )

            st.write("### 📈 Plots of Pi Groups and Their Reciprocals")

            # Loop through all unique pairs of Pi groups
            for i in range(len(pi_data)):
                for j in range(i + 1, len(pi_data)):

                    # Plot original
                    fig1, ax1 = plt.subplots()
                    ax1.scatter(pi_data[i], pi_data[j], alpha=0.7, color='blue')
                    ax1.set_xlabel(pi_labels[i])
                    ax1.set_ylabel(pi_labels[j])
                    ax1.set_title(f"{pi_labels[j]} vs {pi_labels[i]}")
                    st.pyplot(fig1)

                    # Compute reciprocals safely
                    reciprocal_x = np.array(pi_data[i], dtype=np.float64)
                    reciprocal_y = np.array(pi_data[j], dtype=np.float64)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        reciprocal_x = np.where(reciprocal_x != 0, 1 / reciprocal_x, np.nan)
                        reciprocal_y = np.where(reciprocal_y != 0, 1 / reciprocal_y, np.nan)

                    # Filter valid entries
                    valid = ~np.isnan(reciprocal_x) & ~np.isnan(reciprocal_y)
                    if np.any(valid):
                        fig2, ax2 = plt.subplots()
                        ax2.scatter(reciprocal_x[valid], reciprocal_y[valid], alpha=0.7, color='orange')
                        ax2.set_xlabel(f"1 / ({pi_labels[i]})")
                        ax2.set_ylabel(f"1 / ({pi_labels[j]})")
                        ax2.set_title(f"Reciprocal: 1/({pi_labels[j]}) vs 1/({pi_labels[i]})")
                        st.pyplot(fig2)
                    else:
                        st.warning(f"⚠️ Reciprocal plot for {pi_labels[i]} vs {pi_labels[j]} skipped (all values zero or invalid).")
Upload a `.csv` file with only numeric columns.  
Then assign dimensions using any combination of the 7 SI base units:

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
        if not exponents:
            st.error("❌ No nullspace found — variables are dimensionally dependent.")
        else:
            var_symbols = sp.symbols(variables)
            pi_data = []
            pi_labels = []

            st.write("### 🧮 Dimensionless Groups")
            for vec in exponents:
                exps = np.array([float(e) for e in vec])
                pi_vals = np.prod(
                    [df[var] ** power for var, power in zip(variables, exps)],
                    axis=0
                )
                pi_data.append(pi_vals)

                # Human-readable label
                terms = []
                for var, power in zip(variables, exps):
                    if abs(power) > 1e-8:
                        if abs(power - 1.0) < 1e-8:
                            terms.append(f"{var}")
                        else:
                            terms.append(f"{var}^{power:.2f}")
                label = " * ".join(terms)
                pi_labels.append(label)

                # Display as LaTeX
                st.latex(
                    f"\\text{{Dimensionless Group}}: {sp.latex(sp.Mul(*[sym**p for sym, p in zip(var_symbols, vec) if not sp.Eq(p, 0)]))}"
                )

            st.write("### 📈 Plots of Pi Groups and Their Reciprocals")

            # Loop through all unique pairs of Pi groups
            for i in range(len(pi_data)):
                for j in range(i + 1, len(pi_data)):

                    # Normal plot
                    fig, ax = plt.subplots()
                    ax.scatter(pi_data[i], pi_data[j], alpha=0.7)
                    ax.set_xlabel(pi_labels[i])
                    ax.set_ylabel(pi_labels[j])
                    ax.set_title(f"{pi_labels[j]} vs {pi_labels[i]}")
                    st.pyplot(fig)

                    # Reciprocal plot
                    with np.errstate(divide='ignore', invalid='ignore'):
                        reciprocal_x = np.where(pi_data[i] != 0, 1 / pi_data[i], np.nan)
                        reciprocal_y = np.where(pi_data[j] != 0, 1 / pi_data[j], np.nan)

                    fig_r, ax_r = plt.subplots()
                    ax_r.scatter(reciprocal_x, reciprocal_y, alpha=0.7, color='orange')
                    ax_r.set_xlabel(f"1 / ({pi_labels[i]})")
                    ax_r.set_ylabel(f"1 / ({pi_labels[j]})")
                    ax_r.set_title(f"Reciprocal: 1/({pi_labels[j]}) vs 1/({pi_labels[i]})")
                    st.pyplot(fig_r)
