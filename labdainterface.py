import streamlit as st 
import pandas as pd
import numpy as np
import sympy as sp
import plotly.express as px

st.set_page_config(layout="wide")
st.title(" User-friendly Dimensional Analysis")

st.markdown("""
Upload a `.csv` file with only numeric columns, where **each column header** contains the variable name, a hyphen `-`, and then the dimension.  
Example header: `rho-M^1 L^-3, mu-M^1 L^-1 T^-1, D-L^1, v-L^1 T^-1, F-M^1 L^1 T^-2`
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

uploaded_file = st.file_uploader(" Upload CSV File", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    # Parse variable names and dimensions from headers
    variables = []
    dimensions = []
    valid_inputs = True
    for col in df_raw.columns:
        if '-' not in col:
            st.error(f"Column '{col}' header must include a hyphen separating variable name and dimensions.")
            valid_inputs = False
            break
        var_name, dim_str = col.split('-', 1)
        variables.append(var_name.strip())
        try:
            dims = parse_dimensions(dim_str.strip())
            dimensions.append(dims)
        except Exception as e:
            st.error(f"Error parsing dimensions for {var_name}: {e}")
            valid_inputs = False
            break

    if valid_inputs:
        # Rename dataframe columns to variable names only
        df = df_raw.copy()
        df.columns = variables

        st.write("###  Data Preview")
        st.dataframe(df.head())

        dim_matrix = np.array(dimensions).T
        rank = np.linalg.matrix_rank(dim_matrix)
        num_pi_groups = len(variables) - rank
        st.write(f"###  Number of Dimensionless Groups: {num_pi_groups}")

        exponents = sp.Matrix(dim_matrix).nullspace()
        if not exponents:
            st.error(" No nullspace found — variables are dimensionally dependent.")
        else:
            var_symbols = sp.symbols(variables)
            pi_data = []
            pi_labels = []

            st.write("###  Dimensionless Groups")
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

            st.write("###  Interactive Plots of Pi Groups and Their Reciprocals")

            for i in range(len(pi_data)):
                for j in range(i + 1, len(pi_data)):
                    # Create DataFrame for plotting
                    plot_df = pd.DataFrame({
                        f"π{i+1}": pi_data[i],
                        f"π{j+1}": pi_data[j]
                    })

                    # Regular Plot - adjusted for clarity
                    fig1 = px.scatter(
                        plot_df,
                        x=f"π{i+1}",
                        y=f"π{j+1}",
                        title=f"π{j+1} vs π{i+1}",
                        labels={f"π{i+1}": pi_labels[i], f"π{j+1}": pi_labels[j]},
                        width=450,
                        height=450
                    )
                    fig1.update_traces(marker=dict(size=8, opacity=1.0, color='blue'))
                    fig1.update_layout(
                        margin=dict(l=40, r=40, t=40, b=40),
                        xaxis=dict(showgrid=True, zeroline=True, title_font=dict(size=14, color='black')),
                        yaxis=dict(showgrid=True, zeroline=True, title_font=dict(size=14, color='black')),
                        plot_bgcolor='white',
                        title_font=dict(size=16, color='black')
                    )
                    fig1.update_xaxes(title_text=f"π<sub>{i+1}</sub>")
                    fig1.update_yaxes(title_text=f"π<sub>{j+1}</sub>")
                    st.plotly_chart(fig1, use_container_width=False)

                    # Reciprocal plot - adjusted for clarity
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
                            width=450,
                            height=450
                        )
                        fig2.update_traces(marker=dict(size=8, opacity=1.0, color='orange'))
                        fig2.update_layout(
                            margin=dict(l=40, r=40, t=40, b=40),
                            xaxis=dict(showgrid=True, zeroline=True, title_font=dict(size=14, color='black')),
                            yaxis=dict(showgrid=True, zeroline=True, title_font=dict(size=14, color='black')),
                            plot_bgcolor='white',
                            title_font=dict(size=16, color='black')
                        )
                        fig2.update_xaxes(title_text=f"1/π<sub>{i+1}</sub>")
                        fig2.update_yaxes(title_text=f"1/π<sub>{j+1}</sub>")
                        st.plotly_chart(fig2, use_container_width=False)
                    else:
                        st.warning(f" Reciprocal plot for π{i+1} vs π{j+1} skipped — no valid data.")


