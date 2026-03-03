import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import tempfile
import os
import io
import warnings
from io import BytesIO

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Pre-Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEPS = [
    "Upload Data",
    "Data Preview & Filters",
    "Data Cleaning",
    "Weighted Stats",
    "Endorsements",
    "Correlation",
    "Factor Analysis",
    "Regression",
    "Export",
]


def init_session_state():
    defaults = {
        "current_step": 0,
        "data_view": None,
        "data_view_renamed": None,
        "variable_view": None,
        "mapping_dict": None,
        "filtered_df": None,
        "sorted_df": None,
        "meta": None,
        "filters": [],
        "endorsement_result": None,
        "correlation_results": {},
        "factor_results": None,
        "regression_results": {},
        "export_sheets": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .step-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .step-header h2 { color: white; margin: 0; }
    .step-header p  { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; }

    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 { margin: 0; color: #667eea; font-size: 2rem; }
    .metric-card p  { margin: 0.3rem 0 0 0; color: #6c757d; font-size: 0.9rem; }

    .upload-zone {
        border: 3px dashed #667eea;
        border-radius: 16px;
        padding: 4rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f5f7ff 0%, #ede7f6 100%);
        transition: all 0.3s ease;
    }
    .upload-zone:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #ede7f6 0%, #e8eaf6 100%);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Sidebar Navigation ──────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 Pre-Analysis")
        st.markdown("---")

        for i, step_name in enumerate(STEPS):
            disabled = i > 0 and st.session_state.data_view is None
            icon = "✅" if i < st.session_state.current_step else ("▶️" if i == st.session_state.current_step else "⬜")
            if st.button(
                f"{icon}  {step_name}",
                key=f"nav_{i}",
                disabled=disabled,
                use_container_width=True,
            ):
                st.session_state.current_step = i
                st.rerun()

        st.markdown("---")
        if st.session_state.data_view is not None:
            shape = st.session_state.data_view.shape
            st.markdown(f"**Rows:** {shape[0]:,}")
            st.markdown(f"**Columns:** {shape[1]:,}")
            if st.session_state.filtered_df is not None:
                st.markdown(f"**Filtered Rows:** {st.session_state.filtered_df.shape[0]:,}")


render_sidebar()


# ─── Helper: step header ─────────────────────────────────────────────────────
def step_header(title, subtitle=""):
    st.markdown(
        f'<div class="step-header"><h2>{title}</h2><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# ─── Helper: next / prev buttons ─────────────────────────────────────────────
def nav_buttons(show_prev=True, show_next=True, next_label="Next Step →", next_disabled=False):
    cols = st.columns([1, 6, 1])
    if show_prev and st.session_state.current_step > 0:
        with cols[0]:
            if st.button("← Back"):
                st.session_state.current_step -= 1
                st.rerun()
    if show_next and st.session_state.current_step < len(STEPS) - 1:
        with cols[2]:
            if st.button(next_label, disabled=next_disabled):
                st.session_state.current_step += 1
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Upload Data
# ══════════════════════════════════════════════════════════════════════════════
def page_upload():
    step_header("Upload Your SPSS Data", "Drop your .sav file below to get started")

    st.markdown(
        '<div class="upload-zone">'
        "<h3>📁 Drag & Drop your <code>.sav</code> file here</h3>"
        "<p>or click <b>Browse files</b> below</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    uploaded = st.file_uploader(
        "Choose a .sav file",
        type=["sav"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        with st.spinner("Reading SPSS file..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            df, meta = pyreadstat.read_sav(tmp_path)
            os.unlink(tmp_path)

            def apply_value_labels(data, meta):
                for col in data.columns:
                    if col in meta.variable_value_labels:
                        data[col] = data[col].map(meta.variable_value_labels[col])
                return data

            data_view = apply_value_labels(df.copy(), meta)
            variable_view = pd.DataFrame(
                {"Variable Name": meta.column_names, "Variable Label": meta.column_labels}
            )
            mapping_dict = variable_view.set_index("Variable Name")["Variable Label"].to_dict()
            data_view_renamed = data_view.rename(columns=mapping_dict)

            st.session_state.data_view = data_view
            st.session_state.data_view_renamed = data_view_renamed
            st.session_state.variable_view = variable_view
            st.session_state.mapping_dict = mapping_dict
            st.session_state.meta = meta
            st.session_state.filtered_df = data_view_renamed.copy()
            st.session_state.sorted_df = None
            st.session_state.filters = []

        st.success(f"File loaded successfully! {data_view.shape[0]:,} rows × {data_view.shape[1]:,} columns")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric-card"><h3>{data_view.shape[0]:,}</h3><p>Total Rows</p></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><h3>{data_view.shape[1]:,}</h3><p>Total Columns</p></div>',
                unsafe_allow_html=True,
            )
        with c3:
            num_cols = data_view.select_dtypes(include="number").shape[1]
            st.markdown(
                f'<div class="metric-card"><h3>{num_cols}</h3><p>Numeric Columns</p></div>',
                unsafe_allow_html=True,
            )

        nav_buttons(show_prev=False)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data Preview & Filters
# ══════════════════════════════════════════════════════════════════════════════
def page_preview():
    step_header("Data Preview & Filters", "Explore and filter your dataset")
    df_renamed = st.session_state.data_view_renamed

    tab_data, tab_vars, tab_filter = st.tabs(["📋 Data View", "📝 Variable View", "🔍 Apply Filters"])

    with tab_data:
        st.dataframe(df_renamed.head(100), use_container_width=True, height=400)

    with tab_vars:
        st.dataframe(st.session_state.variable_view, use_container_width=True, height=400)

    with tab_filter:
        st.markdown("#### Add Column Filters")
        st.info("Select a column and values to filter. You can add multiple filters and apply them all at once.")

        filter_col = st.selectbox("Select Column", options=df_renamed.columns.tolist(), key="filter_col")

        if filter_col:
            unique_vals = df_renamed[filter_col].dropna().unique().tolist()
            selected_vals = st.multiselect(
                f"Select values for **{filter_col}**",
                options=unique_vals,
                key="filter_vals",
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("➕ Add Filter", use_container_width=True):
                    if selected_vals:
                        st.session_state.filters.append((filter_col, selected_vals))
                        st.success(f"Filter added: {filter_col} → {selected_vals}")
                        st.rerun()
            with c2:
                if st.button("🗑️ Clear All Filters", use_container_width=True):
                    st.session_state.filters = []
                    st.session_state.filtered_df = df_renamed.copy()
                    st.rerun()

        if st.session_state.filters:
            st.markdown("#### Active Filters")
            for i, (col, vals) in enumerate(st.session_state.filters):
                st.markdown(f"**{i+1}.** `{col}` in `{vals}`")

            if st.button("✅ Apply All Filters", type="primary", use_container_width=True):
                filtered = df_renamed.copy()
                for col, vals in st.session_state.filters:
                    filtered = filtered[filtered[col].isin(vals)]
                st.session_state.filtered_df = filtered
                st.success(f"Filters applied! {filtered.shape[0]:,} rows remaining.")
                st.rerun()

        if st.session_state.filtered_df is not None:
            st.markdown(f"**Current filtered data:** {st.session_state.filtered_df.shape[0]:,} rows × {st.session_state.filtered_df.shape[1]:,} columns")
            st.dataframe(st.session_state.filtered_df.head(50), use_container_width=True, height=300)

    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Data Cleaning (Yes/No conversion, drop nulls, sort)
# ══════════════════════════════════════════════════════════════════════════════
def page_cleaning():
    step_header("Data Cleaning", "Convert Yes/No columns, remove nulls, and sort your data")
    fdf = st.session_state.filtered_df

    tab_convert, tab_clean, tab_sort = st.tabs(["🔄 Yes/No Conversion", "🧹 Remove Nulls", "📊 Sort Data"])

    with tab_convert:
        st.markdown("#### Convert Yes/No columns to 1/0")
        st.info("Select columns where 'Yes' should become 1 and 'No' should become 0.")

        all_cols = fdf.columns.tolist()
        yes_no_candidates = [
            c for c in all_cols if fdf[c].dropna().isin(["Yes", "No"]).any()
        ]

        selected_cols = st.multiselect(
            "Select columns to convert",
            options=yes_no_candidates if yes_no_candidates else all_cols,
            key="yes_no_cols",
        )

        if st.button("🔄 Convert Selected Columns", type="primary"):
            if selected_cols:
                for col in selected_cols:
                    fdf[col] = fdf[col].apply(lambda x: 1 if x == "Yes" else (0 if x == "No" else x))
                st.session_state.filtered_df = fdf
                st.success(f"Converted {len(selected_cols)} columns!")
                st.rerun()
            else:
                st.warning("Please select at least one column.")

    with tab_clean:
        st.markdown("#### Remove Null Values")
        null_counts = fdf.isnull().sum()
        null_cols = null_counts[null_counts > 0]

        if len(null_cols) > 0:
            st.dataframe(
                pd.DataFrame({"Column": null_cols.index, "Null Count": null_cols.values}),
                use_container_width=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🧹 Drop All Rows with Nulls"):
                    cleaned = fdf.dropna()
                    st.session_state.filtered_df = cleaned
                    st.success(f"Dropped {fdf.shape[0] - cleaned.shape[0]:,} rows. {cleaned.shape[0]:,} remaining.")
                    st.rerun()
            with c2:
                drop_cols = st.multiselect("Or select specific columns", options=null_cols.index.tolist(), key="drop_null_cols")
                if st.button("Drop nulls in selected columns only"):
                    if drop_cols:
                        cleaned = fdf.dropna(subset=drop_cols)
                        st.session_state.filtered_df = cleaned
                        st.success(f"Dropped rows with nulls in selected columns. {cleaned.shape[0]:,} remaining.")
                        st.rerun()
        else:
            st.success("No null values found in the data!")

    with tab_sort:
        st.markdown("#### Sort Data")
        sort_cols = st.multiselect(
            "Select columns to sort by",
            options=fdf.columns.tolist(),
            key="sort_cols",
        )
        sort_asc = st.checkbox("Sort Ascending", value=True, key="sort_asc")

        if st.button("📊 Sort DataFrame", type="primary"):
            if sort_cols:
                sorted_df = fdf.sort_values(by=sort_cols, ascending=sort_asc)
                st.session_state.filtered_df = sorted_df
                st.session_state.sorted_df = sorted_df
                st.success("Data sorted!")
                st.dataframe(sorted_df.head(50), use_container_width=True, height=300)
            else:
                st.warning("Please select at least one column to sort by.")

        if st.button("✅ Finalize Cleaned Data", type="primary", use_container_width=True):
            st.session_state.sorted_df = st.session_state.filtered_df.copy()
            st.success("Cleaned data finalized and ready for analysis!")

    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Weighted Mean & Frequency
# ══════════════════════════════════════════════════════════════════════════════
def page_weighted_stats():
    step_header("Weighted Statistics", "Calculate weighted mean and weighted frequency")
    working_df = st.session_state.sorted_df if st.session_state.sorted_df is not None else st.session_state.filtered_df

    tab_mean, tab_freq = st.tabs(["📈 Weighted Mean", "📊 Weighted Frequency"])

    numeric_cols = working_df.select_dtypes(include="number").columns.tolist()

    with tab_mean:
        st.markdown("#### Calculate Weighted Mean")

        weight_col = st.selectbox("Select Weight Column", options=numeric_cols, key="wm_weight")
        value_cols = st.multiselect("Select Value Columns", options=numeric_cols, key="wm_values")

        if st.button("📈 Calculate Weighted Mean", type="primary"):
            if weight_col and value_cols:
                results = []
                for vc in value_cols:
                    w_mean = np.sum(working_df[vc] * working_df[weight_col]) / np.sum(working_df[weight_col])
                    results.append({"Measure": vc, "Weighted Mean": round(w_mean, 4)})
                result_df = pd.DataFrame(results)
                st.dataframe(result_df, use_container_width=True)
                st.session_state.export_sheets["Weighted Mean"] = result_df
                st.success("Weighted mean calculated! Results will be included in export.")
            else:
                st.warning("Please select weight and value columns.")

    with tab_freq:
        st.markdown("#### Calculate Weighted Frequency")

        weight_col_f = st.selectbox("Select Weight Column", options=numeric_cols, key="wf_weight")
        cat_cols = st.multiselect("Select Category Columns", options=working_df.columns.tolist(), key="wf_cats")

        if st.button("📊 Calculate Weighted Frequency", type="primary"):
            if weight_col_f and cat_cols:
                all_freq_dfs = []
                for cat_col in cat_cols:
                    wf = (
                        working_df.groupby(cat_col)
                        .apply(lambda x: np.sum(x[weight_col_f]))
                        .reset_index(name="Weighted Frequency")
                    )
                    wf.columns = [cat_col, "Weighted Frequency"]
                    st.markdown(f"**{cat_col}**")
                    st.dataframe(wf, use_container_width=True)
                    all_freq_dfs.append(wf)
                st.session_state.export_sheets["Weighted Frequency"] = pd.concat(all_freq_dfs, ignore_index=True)
                st.success("Weighted frequency calculated!")
            else:
                st.warning("Please select weight and category columns.")

    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Endorsements
# ══════════════════════════════════════════════════════════════════════════════
def page_endorsements():
    step_header("Endorsements", "Calculate weighted endorsements split by Brand")
    working_df = st.session_state.sorted_df if st.session_state.sorted_df is not None else st.session_state.filtered_df

    if "Brand" not in working_df.columns:
        st.warning("No 'Brand' column found. Please ensure your data has a Brand column (check variable labels).")
        brand_col = st.selectbox("Select the column to use as Brand", options=working_df.columns.tolist(), key="brand_col_endorse")
    else:
        brand_col = "Brand"

    numeric_cols = working_df.select_dtypes(include="number").columns.tolist()
    all_cols = working_df.columns.tolist()

    weight_col = st.selectbox(
        "Select Weight Column (optional — leave as None for unweighted)",
        options=[None] + numeric_cols,
        key="endorse_weight",
    )

    selected_cols = st.multiselect(
        "Select columns for endorsement calculation",
        options=numeric_cols,
        key="endorse_cols",
    )

    if st.button("📊 Calculate Endorsements", type="primary", use_container_width=True):
        if selected_cols:
            numeric_selected = working_df[selected_cols].select_dtypes(include="number")

            if weight_col:
                mean_by_brand = working_df.groupby(brand_col).apply(
                    lambda x: (
                        numeric_selected.loc[x.index].multiply(x[weight_col], axis=0).sum()
                        / x[weight_col].sum()
                    )
                )
            else:
                mean_by_brand = working_df.groupby(brand_col)[numeric_selected.columns.tolist()].mean()

            mean_pct = mean_by_brand * 100
            mean_transposed = mean_pct.T
            mean_transposed.columns.name = None
            mean_transposed.index.name = "ENDORSEMENTS"

            base_counts = working_df.groupby(brand_col).size()
            base_row = pd.DataFrame(base_counts).T
            base_row.index = ["BASE"]

            result_df = pd.concat([base_row, mean_transposed])
            result_df.iloc[1:] = result_df.iloc[1:].applymap(
                lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
            )

            st.session_state.endorsement_result = result_df
            st.session_state.export_sheets["Endorsement"] = result_df

            st.success("Endorsements calculated!")
            st.dataframe(result_df, use_container_width=True, height=400)
        else:
            st.warning("Please select at least one column.")

    if st.session_state.endorsement_result is not None:
        st.markdown("#### Previous Result")
        st.dataframe(st.session_state.endorsement_result, use_container_width=True, height=300)

    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Correlation
# ══════════════════════════════════════════════════════════════════════════════
def page_correlation():
    step_header("Correlation Analysis", "Calculate correlation matrices by brand with optional weights")
    working_df = st.session_state.sorted_df if st.session_state.sorted_df is not None else st.session_state.filtered_df

    if "Brand" not in working_df.columns:
        brand_col = st.selectbox("Select brand column", options=working_df.columns.tolist(), key="brand_col_corr")
    else:
        brand_col = "Brand"

    brands = working_df[brand_col].unique().tolist()
    numeric_cols = working_df.select_dtypes(include="number").columns.tolist()

    selected_brands = st.multiselect("Select Brands", options=brands, key="corr_brands")

    weight_col = st.selectbox(
        "Select Weight Column (optional)",
        options=[None] + numeric_cols,
        key="corr_weight",
    )

    selected_cols = st.multiselect(
        "Select columns for correlation",
        options=numeric_cols,
        key="corr_cols",
    )

    if st.button("📊 Calculate Correlation", type="primary", use_container_width=True):
        if selected_brands and selected_cols:
            filtered = working_df[working_df[brand_col].isin(selected_brands)]
            numeric_data = filtered[selected_cols].select_dtypes(include="number")

            if weight_col:
                weighted = numeric_data.multiply(filtered[weight_col], axis=0)
                corr_matrix = weighted.corr().round(3)
                suffix = " (Weighted)"
            else:
                corr_matrix = numeric_data.corr().round(3)
                suffix = ""

            brand_label = ", ".join(str(b) for b in selected_brands)
            sheet_name = f"Corr_{brand_label[:20]}{suffix}"

            st.session_state.correlation_results[sheet_name] = corr_matrix
            st.session_state.export_sheets[sheet_name] = corr_matrix

            st.success(f"Correlation calculated for: {brand_label}")

            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title=f"Correlation Matrix — {brand_label}{suffix}",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(corr_matrix, use_container_width=True)

            high_corr = []
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) > 0.5:
                        high_corr.append(
                            {
                                "Variable 1": corr_matrix.index[i],
                                "Variable 2": corr_matrix.columns[j],
                                "Correlation": val,
                            }
                        )
            if high_corr:
                st.markdown("#### High Correlations (|r| > 0.5)")
                st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
        else:
            st.warning("Please select at least one brand and one column.")

    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Factor Analysis
# ══════════════════════════════════════════════════════════════════════════════
def page_factor_analysis():
    step_header("Factor Analysis", "Principal component analysis with varimax rotation")
    working_df = st.session_state.sorted_df if st.session_state.sorted_df is not None else st.session_state.filtered_df

    if "Brand" not in working_df.columns:
        brand_col = st.selectbox("Select brand column", options=working_df.columns.tolist(), key="brand_col_fa")
    else:
        brand_col = "Brand"

    brands = working_df[brand_col].unique().tolist()
    numeric_cols = working_df.select_dtypes(include="number").columns.tolist()

    selected_brands = st.multiselect("Select Brands", options=brands, key="fa_brands")
    selected_cols = st.multiselect("Select columns for factor analysis", options=numeric_cols, key="fa_cols")
    num_factors = st.number_input("Number of Factors", min_value=1, max_value=20, value=3, key="fa_num")

    if st.button("🔬 Run Factor Analysis", type="primary", use_container_width=True):
        if selected_brands and selected_cols and num_factors:
            try:
                from factor_analyzer import FactorAnalyzer

                filtered = working_df[working_df[brand_col].isin(selected_brands)]
                analysis_df = filtered[selected_cols].select_dtypes(include="number")

                if num_factors > len(selected_cols):
                    st.error(f"Number of factors ({num_factors}) cannot exceed selected columns ({len(selected_cols)}).")
                    return

                cov_matrix = analysis_df.cov()
                fa = FactorAnalyzer(n_factors=num_factors, method="principal", rotation="varimax")
                fa.fit(cov_matrix)

                ev, v = fa.get_eigenvalues()

                fig_scree = go.Figure()
                fig_scree.add_trace(
                    go.Scatter(
                        x=list(range(1, len(ev) + 1)),
                        y=ev,
                        mode="lines+markers",
                        name="Eigenvalues",
                        marker=dict(size=10, color="#667eea"),
                        line=dict(color="#667eea", width=2),
                    )
                )
                fig_scree.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Eigenvalue = 1")
                fig_scree.update_layout(
                    title="Scree Plot",
                    xaxis_title="Factors",
                    yaxis_title="Eigenvalue",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig_scree, use_container_width=True)

                loadings = fa.loadings_
                loadings_df = pd.DataFrame(
                    loadings,
                    index=selected_cols,
                    columns=[f"Factor {i+1}" for i in range(num_factors)],
                )

                st.markdown("#### Factor Loadings")
                st.dataframe(loadings_df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), use_container_width=True)

                st.session_state.factor_results = loadings_df
                st.session_state.export_sheets["Factor Analysis"] = loadings_df
                st.success("Factor analysis completed!")

            except Exception as e:
                st.error(f"Error during factor analysis: {e}")
        else:
            st.warning("Please select brands, columns, and number of factors.")

    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Regression
# ══════════════════════════════════════════════════════════════════════════════
def page_regression():
    step_header("Regression Analysis", "Enter and Stepwise regression methods")
    working_df = st.session_state.sorted_df if st.session_state.sorted_df is not None else st.session_state.filtered_df

    if "Brand" not in working_df.columns:
        brand_col = st.selectbox("Select brand column", options=working_df.columns.tolist(), key="brand_col_reg")
    else:
        brand_col = "Brand"

    brands = working_df[brand_col].unique().tolist()
    numeric_cols = working_df.select_dtypes(include="number").columns.tolist()

    selected_brands = st.multiselect("Select Brands", options=brands, key="reg_brands")

    dependent_var = st.selectbox("Dependent Variable", options=numeric_cols, key="reg_dep")
    independent_vars = st.multiselect(
        "Independent Variables",
        options=[c for c in numeric_cols if c != dependent_var],
        key="reg_indep",
    )
    method = st.radio("Regression Method", options=["Enter", "Stepwise"], horizontal=True, key="reg_method")

    if st.button("🚀 Run Regression", type="primary", use_container_width=True):
        if selected_brands and dependent_var and independent_vars:
            import statsmodels.api as sm
            from sklearn.preprocessing import StandardScaler

            filtered = working_df[working_df[brand_col].isin(selected_brands)]
            X = filtered[independent_vars].dropna()
            y = filtered.loc[X.index, dependent_var].dropna()
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]

            if method == "Enter":
                X_const = sm.add_constant(X)
                model = sm.OLS(y, X_const).fit()

                coefficients = model.params
                std_coefficients = coefficients * (X_const.std() / y.std())

                result_df = pd.DataFrame(
                    {
                        "Unstandardized B": model.params,
                        "Standardized Beta": std_coefficients,
                        "Std Error": model.bse,
                        "t-value": model.tvalues,
                        "p-value": model.pvalues,
                    }
                )

                st.markdown("#### Enter Regression Results")
                st.dataframe(
                    result_df.style.applymap(
                        lambda v: "color: green" if isinstance(v, float) and v < 0.05 else "",
                        subset=["p-value"],
                    ),
                    use_container_width=True,
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("R-squared", f"{model.rsquared:.4f}")
                with c2:
                    st.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
                with c3:
                    st.metric("F-statistic", f"{model.fvalue:.4f}")

                st.session_state.regression_results["Enter"] = result_df
                st.session_state.export_sheets["Enter Regression"] = result_df
                st.success("Enter regression completed!")

            elif method == "Stepwise":
                X = X.reset_index(drop=True)
                y = y.reset_index(drop=True)

                included = []
                stepwise_output = []
                step = 1

                threshold_in = 0.05
                threshold_out = 0.10

                max_iterations = len(independent_vars) * 2
                iteration = 0

                while iteration < max_iterations:
                    iteration += 1
                    excluded = list(set(X.columns) - set(included))
                    new_pval = pd.Series(index=excluded, dtype=float)

                    for col in excluded:
                        X_temp = X[included + [col]]
                        scaler = StandardScaler()
                        X_scaled = pd.DataFrame(scaler.fit_transform(X_temp), columns=X_temp.columns)
                        model = sm.OLS(y, sm.add_constant(X_scaled)).fit()
                        new_pval[col] = model.pvalues.get(col, 1.0)

                    best_pval = new_pval.min() if len(new_pval) > 0 else 1.0

                    if best_pval < threshold_in:
                        best_feature = new_pval.idxmin()
                        included.append(best_feature)

                        X_inc = X[included]
                        scaler = StandardScaler()
                        X_scaled = pd.DataFrame(scaler.fit_transform(X_inc), columns=included)
                        model = sm.OLS(y, sm.add_constant(X_scaled)).fit()

                        step_df = pd.DataFrame(
                            {
                                "Standardized Beta": model.params[1:],
                                "p-value": model.pvalues[1:],
                            }
                        )
                        stepwise_output.append((f"Step {step}: Add {best_feature}", step_df))
                        step += 1

                    if included:
                        X_inc = X[included]
                        scaler = StandardScaler()
                        X_scaled = pd.DataFrame(scaler.fit_transform(X_inc), columns=included)
                        model = sm.OLS(y, sm.add_constant(X_scaled)).fit()
                        pvalues = model.pvalues[1:]
                        worst_pval = pvalues.max()

                        if worst_pval > threshold_out:
                            worst_feature = pvalues.idxmax()
                            included.remove(worst_feature)

                            if included:
                                X_inc = X[included]
                                scaler = StandardScaler()
                                X_scaled = pd.DataFrame(scaler.fit_transform(X_inc), columns=included)
                                model = sm.OLS(y, sm.add_constant(X_scaled)).fit()

                                step_df = pd.DataFrame(
                                    {
                                        "Standardized Beta": model.params[1:],
                                        "p-value": model.pvalues[1:],
                                    }
                                )
                                stepwise_output.append((f"Step {step}: Remove {worst_feature}", step_df))
                                step += 1
                        else:
                            if best_pval >= threshold_in:
                                break
                    else:
                        if best_pval >= threshold_in:
                            break

                if stepwise_output:
                    st.markdown("#### Stepwise Regression Results")
                    for step_name, step_result in stepwise_output:
                        st.markdown(f"**{step_name}**")
                        st.dataframe(step_result, use_container_width=True)

                    final_step_name, final_df = stepwise_output[-1]
                    st.session_state.regression_results["Stepwise"] = final_df
                    st.session_state.export_sheets["Stepwise Regression"] = final_df

                    if included:
                        X_final = X[included]
                        scaler = StandardScaler()
                        X_scaled = pd.DataFrame(scaler.fit_transform(X_final), columns=included)
                        final_model = sm.OLS(y, sm.add_constant(X_scaled)).fit()
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("R-squared", f"{final_model.rsquared:.4f}")
                        with c2:
                            st.metric("Adj. R-squared", f"{final_model.rsquared_adj:.4f}")
                        with c3:
                            st.metric("Final Variables", len(included))

                    st.success("Stepwise regression completed!")
                else:
                    st.warning("No variables met the threshold criteria for inclusion.")
        else:
            st.warning("Please select brands, dependent variable, and independent variables.")

    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Export
# ══════════════════════════════════════════════════════════════════════════════
def page_export():
    step_header("Export Results", "Download all analysis results as an Excel file")

    sheets = st.session_state.export_sheets

    if not sheets:
        st.info("No analysis results to export yet. Complete the analysis steps first.")
    else:
        st.markdown("#### Available Results for Export")
        for name in sheets:
            st.markdown(f"- ✅ **{name}**")

        if st.button("📥 Generate Excel File", type="primary", use_container_width=True):
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                for sheet_name, df in sheets.items():
                    safe_name = sheet_name[:31]
                    df.to_excel(writer, sheet_name=safe_name, index=True, startrow=2)

            st.download_button(
                label="⬇️ Download Excel Report",
                data=buffer.getvalue(),
                file_name="Pre_Analysis_Output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.success("Excel file ready for download!")

    st.markdown("---")
    st.markdown("#### Quick Data Export")
    working_df = st.session_state.sorted_df if st.session_state.sorted_df is not None else st.session_state.filtered_df
    if working_df is not None:
        buf2 = BytesIO()
        working_df.to_excel(buf2, index=False, engine="openpyxl")
        st.download_button(
            label="⬇️ Download Cleaned/Filtered Data as Excel",
            data=buf2.getvalue(),
            file_name="Cleaned_Data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    nav_buttons(show_next=False)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    step = st.session_state.current_step

    if step == 0:
        page_upload()
    elif st.session_state.data_view is None:
        step_header("No Data Loaded", "Please go back to Step 1 and upload a .sav file")
        st.warning("You need to upload a data file first.")
        if st.button("← Go to Upload"):
            st.session_state.current_step = 0
            st.rerun()
    elif step == 1:
        page_preview()
    elif step == 2:
        page_cleaning()
    elif step == 3:
        page_weighted_stats()
    elif step == 4:
        page_endorsements()
    elif step == 5:
        page_correlation()
    elif step == 6:
        page_factor_analysis()
    elif step == 7:
        page_regression()
    elif step == 8:
        page_export()


if __name__ == "__main__":
    main()
