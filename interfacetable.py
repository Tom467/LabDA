import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import plotly.express as px
from fractions import Fraction
import math

st.set_page_config(layout="wide")
st.title("User-friendly Dimensional Analysis")

st.markdown("""
Upload a `.csv` file with numeric data where **each column header** contains:
- Variable name
- A hyphen (`-`)
- The physical dimensions for that variable

Example header:
rho-M^1 L^-3,mu-M^1 L^-1 T^-1,D-L^1,v-L^1 T^-1,F-M^1 L^1 T^-2



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
            dims[symbol.strip()] = float(power)
        else:
            dims[item.strip()] = 1.0
    return [dims[u] for u in DIM_UNITS]

def scale_to_integers(vec):
    """Convert vector to smallest possible integer ratio form."""
    fracs = [Fraction(float(v)).limit_denominator(10) for v in vec]
    denoms = [f.denominator for f in fracs]
    lcm_den = np.lcm.reduce(denoms)
    ints = [int(f * lcm_den) for f in fracs]
    # Reduce by GCD
    gcd_val = math.gcd.reduce(ints) if hasattr(math, "gcd") else np.gcd.reduce(ints)
    if gcd_val != 0:
        ints = [i // gcd_val for i in ints]
    return ints

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    # Parse headers
    variables = []
    dimensions = []
    for col in df_raw.columns:
        try:
            name, dim_str = col.split("-", 1)
            variables.append(name.strip())
            dimensions.append(parse_dimensions(dim_str.strip()))
        except ValueError:
            st.error(f"Column '{col}' must contain a hyphen separating name and dimensions.")
            st.stop()

    # Rename DataFrame columns to just variable names
    df = df_raw.copy()
    df.columns = variables

    st.write("### Data Preview")
    st.dataframe(df.head())

    # Dimension matrix
    dim_matrix = np.array(dimensions).T
    rank = np.linalg.matrix_rank(dim_matrix)
    num_pi_groups = len(variables) - rank
    st.write(f"### Number of Dimensionless Groups: {num_pi_groups}")

    # Nullspace
    exponents = sp.Matrix(dim_matrix).nullspace()
    if not exponents:
        st.error("No nullspace found — variables are dimensionally dependent.")
    else:
        var_symbols = sp.symbols(variables)
        pi_data = []
        pi_labels = []

        st.write("### Dimensionless Groups")
        for idx, vec in enumerate(exponents):
            # Convert to scaled integers
            exps_int = scale_to_integers(vec)
            exps_float = np.array([float(e) for e in exps_int])

            pi_vals = np.prod(
                [df[var] ** power for var, power in zip(variables, exps_float)],
                axis=0
            )
            pi_data.append(pi_vals)

            # Label for plots
            terms = []
            for var, power in zip(variables, exps_int):
                if power != 0:
                    if power == 1:
                        terms.append(f"{var}")
                    else:
                        terms.append(f"{var}^{power}")
            label = " * ".join(terms) if terms else "1"
            pi_labels.append(label)

            # LaTeX formula
            latex_expr = sp.Mul(*[sym**p for sym, p in zip(var_symbols, exps_int) if p != 0])
            st.latex(f"\\pi_{{{idx+1}}} = {sp.latex(latex_expr)}")

        st.write("### Interactive Plots of Pi Groups and Their Reciprocals")

        for i in range(len(pi_data)):
            for j in range(i + 1, len(pi_data)):
                plot_df = pd.DataFrame({
                    f"π{i+1}": pi_data[i],
                    f"π{j+1}": pi_data[j]
                })

                # Regular plot
                fig1 = px.scatter(
                    plot_df,
                    x=f"π{i+1}",
                    y=f"π{j+1}",
                    title=f"π{j+1} vs π{i+1}",
                    labels={f"π{i+1}": pi_labels[i], f"π{j+1}": pi_labels[j]},
                    width=600,
                    height=450
                )
                fig1.update_traces(marker=dict(size=6, opacity=0.7))
                fig1.update_layout(margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig1, use_container_width=True)

                # Reciprocal plot
                with np.errstate(divide='ignore', invalid='ignore'):
                    reciprocal_x = np.where(pi_data[i] != 0, 1 / pi_data[i], np.nan)
                    reciprocal_y = np.where(pi_data[j] != 0, 1 / pi_data[j], np.nan)

                valid = ~np.isnan(reciprocal_x) & ~np.isnan(reciprocal_y)
                if np.any(valid):
                    reciprocal_df = pd.DataFrame({
                        f"1/π{i+1}": reciprocal_x[valid],
                        f"1/π{j+1}": reciprocal_y[valid]
                    })

                    fig2 = px.scatter(
                        reciprocal_df,
                        x=f"1/π{i+1}",
                        y=f"1/π{j+1}",
                        title=f"Reciprocal: 1/π{j+1} vs 1/π{i+1}",
                        width=600,
                        height=450
                    )
                    fig2.update_traces(marker=dict(size=6, opacity=0.7, color='orange'))
                    fig2.update_layout(margin=dict(l=40, r=40, t=40, b=40))
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning(f"Reciprocal plot for π{i+1} vs π{j+1} skipped — no valid data.")
