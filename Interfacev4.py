import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

st.title("🔍 Buckingham Pi Theorem Analyzer")

st.markdown("""
Upload a `.csv` file where each column is a variable.  
Then specify the dimensions for each variable to perform dimensional analysis.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

def parse_dimensions(dim_str):
    # Simple parser for dimensions like "M L T^-2"
    dims = {"M": 0, "L": 0, "T": 0}
    for item in dim_str.split():
        if "^" in item:
            symbol, power = item.split("^")
            dims[symbol] = int(power)
        else:
            dims[item] = 1
    return [dims["M"], dims["L"], dims["T"]]

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Preview of Data")
    st.write(df.head())

    variables = list(df.columns)
    st.write("### ⚙️ Assign Physical Dimensions")

    dimensions = []
    for var in variables:
        dim_input = st.text_input(f"Dimensions for {var} (e.g., 'M L T^-2')", key=var)
        if dim_input:
            dims = parse_dimensions(dim_input)
            dimensions.append(dims)

    if len(dimensions) == len(variables):
        st.success("✅ All dimensions provided!")

        # Perform dimensional matrix analysis
        dim_matrix = np.array(dimensions).T
        rank = np.linalg.matrix_rank(dim_matrix)
        num_pi_groups = len(variables) - rank
        st.write(f"### 🔢 Number of Pi Groups: {num_pi_groups}")

        # Use sympy for symbolic Pi group construction
        exponents = sp.Matrix(dim_matrix).nullspace()
        if not exponents:
            st.error("❌ No nullspace found – dimensionally dependent variables.")
        else:
            st.write("### 🧮 Dimensionless Pi Groups")
            var_symbols = sp.symbols(variables)
            pi_exprs = []

            for i, vec in enumerate(exponents):
                pi = 1
                for sym, power in zip(var_symbols, vec):
                    pi *= sym**power
                pi_exprs.append(pi)
                st.latex(f"\\pi_{{{i+1}}} = {sp.latex(pi)}")

            # Calculate Pi values from data
            st.write("### 📈 Plots of Dimensionless Groups")
            pi_data = []
            for vec in exponents:
                exps = np.array([float(e) for e in vec])
                pi_vals = np.prod([df[var]**power for var, power in zip(variables, exps)], axis=0)
                pi_data.append(pi_vals)

            # Plot Pi groups
            for i in range(len(pi_data)):
                for j in range(i+1, len(pi_data)):
                    fig, ax = plt.subplots()
                    ax.scatter(pi_data[i], pi_data[j], alpha=0.7)
                    ax.set_xlabel(f"$\\pi_{i+1}$")
                    ax.set_ylabel(f"$\\pi_{j+1}$")
                    ax.set_title(f"$\\pi_{i+1}$ vs $\\pi_{j+1}$")
                    st.pyplot(fig)
