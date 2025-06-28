import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

st.title("🔬 Buckingham Pi Theorem Analyzer (Full SI Base Support)")

st.markdown("""
Upload a `.csv` file with **only numeric columns** (each column is a variable).  
Then assign the **dimensions using any combination of the 7 SI base units**:
- M: Mass
- L: Length
- T: Time
- I: Electric Current
- Θ: Temperature
- N: Amount of Substance
- J: Luminous Intensity
""")

uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

# Order of SI base dimensions
SI_UNITS = ["M", "L", "T", "I", "Θ", "N", "J"]

def parse_dimensions(dim_str):
    """Parse a user input like 'M^1 L^2 T^-2' into a list of exponents."""
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
                st.error(f"Error in dimensions for {var}: {e}")
                valid_inputs = False
        else:
            valid_inputs = False

    if valid_inputs and len(dimensions) == len(variables):
        st.success("✅ All dimensions parsed.")

        dim_matrix = np.array(dimensions).T
        rank = np.linalg.matrix_rank(dim_matrix)
        num_pi_groups = len(variables) - rank
        st.write(f"### 📐 Number of Pi Groups: {num_pi_groups}")

        # Null space gives exponents for Pi groups
        exponents = sp.Matrix(dim_matrix).nullspace()
        if not exponents:
            st.error("❌ No nullspace found — all variables dimensionally dependent.")
        else:
            var_symbols = sp.symbols(variables)
            pi_data = []
            pi_labels = []

            for vec in exponents:
                exps = np.array([float(e) for e in vec])
                pi_vals = np.prod(
                    [df[var] ** power for var, power in zip(variables, exps)],
                    axis=0
                )
                pi_data.append(pi_vals)

                # Create label like "F^1 * D^2 / mu^2"
                terms = []
                for var, power in zip(variables, exps):
                    if abs(power) > 1e-8:
                        if abs(power - 1.0) < 1e-8:
                            terms.append(f"{var}")
                        else:
                            terms.append(f"{var}^{power:.2f}")
                label = " * ".join(terms)
                pi_labels.append(label)

            st.write("### 📈 Pi Group Scatter Plots")

            # Plot all unique combinations of Pi groups
            for i in range(len(pi_data)):
                for j in range(i + 1, len(pi_data)):
                    fig, ax = plt.subplots()
                    ax.scatter(pi_data[i], pi_data[j], alpha=0.7)
                    ax.set_xlabel(pi_labels[i])
                    ax.set_ylabel(pi_labels[j])
                    ax.set_title(f"{pi_labels[j]} vs {pi_labels[i]}")
                    st.pyplot(fig)
