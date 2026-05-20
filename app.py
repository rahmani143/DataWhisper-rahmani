import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st

st.cache_data.clear()
st.cache_resource.clear()



# Import custom modules
from src.data_loader import load_data, get_dataframe_info, get_data_preview
from src.eda import (
    generate_summary_stats, 
    plot_missing_values, 
    plot_correlation_matrix, 
    plot_distributions, 
    plot_count_plots
)
from src.llm_insights import generate_insights, explain_chart, generate_auto_summary
from src.recommendations import generate_recommendations # Keeping it for now
from src.auth import authenticate_user
from src.ui_components import render_header, render_insight_card, render_step_indicator, render_info_box, add_custom_css

# Global error handling for the entire app
def main_with_error_handling():
    try:
        main()
    except Exception as e:
        st.error(f"⚠️ A critical error occurred: {str(e)}")
        st.info("Try refreshing the page or clearing the cache in the sidebar.")
        if st.button("Restart App"):
            st.session_state.clear()
            st.rerun()

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="DataWhisper | AI-Powered EDA", 
    layout="wide", 
    page_icon="📊", 
    initial_sidebar_state="expanded"
)
st.title("DataWhisper")


def load_css():
    """Loads external CSS for SaaS styling."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    add_custom_css()

def init_session_state():
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0 # 0: Upload, 1: Analyze, 2: EDA, 3: Insights, 4: Export
    if 'insights' not in st.session_state:
        st.session_state.insights = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'auto_summary' not in st.session_state:
        st.session_state.auto_summary = None
    if 'ai_task' not in st.session_state:
        st.session_state.ai_task = "General Analysis"

def main():
    load_css()
    init_session_state()
    
    # 1. Enforce Authentication
    is_authenticated, authenticator = authenticate_user()
    if not is_authenticated:
        return
        
    st.sidebar.divider()
    st.sidebar.markdown("## DataWhisper")
    st.sidebar.markdown("Transform your data into actionable insights.")
    
    # Check for API Key
    if not os.getenv("GROQ_API_KEY"):
        st.sidebar.error("⚠️ GROQ_API_KEY not found in .env")
        st.stop()
    else:
        st.sidebar.success("✅ AI Engine Ready (Groq)")

    if st.sidebar.button("🧹 Clear Cache & Restart"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

    st.sidebar.divider()
    
    # Navigation Sidebar
    st.sidebar.markdown("### Navigation")
    steps = ["1. Upload Data", "2. Dataset Overview", "3. Visual EDA", "4. AI Insights", "5. Export Report"]
    
    # Map steps to st.session_state.current_step
    app_mode = st.sidebar.radio("Go to:", steps, index=min(st.session_state.current_step, len(steps)-1))
    st.session_state.current_step = steps.index(app_mode)
    
    render_step_indicator(st.session_state.current_step)

    # 1. Upload Data
    if st.session_state.current_step == 0:
        render_header("Upload Dataset", "Start by uploading your CSV file.", "📤")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
            if uploaded_file:
                st.session_state.uploaded_file = uploaded_file
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.df = df
                    st.success(f"Successfully loaded {uploaded_file.name}!")
                    if st.button("Proceed to Overview →", type="primary"):
                        # NEW FEATURE: Auto-generate AI Insights Summary for speedy start
                        with st.spinner("✨ Generating AI Insights Summary..."):
                            df_info = str(get_dataframe_info(df))
                            st.session_state.auto_summary = generate_auto_summary(df_info)
                        
                        st.session_state.current_step = 1
                        st.rerun()
        
        with col2:
            render_info_box("Instructions", "Upload a CSV file with headers. Ensure numeric columns are properly formatted for correlation analysis.")
            if st.button("Load Sample Data (Titanic)", use_container_width=True):
                st.session_state.uploaded_file = "sample_data/titanic.csv"
                df = load_data("sample_data/titanic.csv")
                st.session_state.df = df
                st.session_state.current_step = 1
                st.rerun()

    # If no data is loaded, stop here
    if st.session_state.df is None and st.session_state.current_step > 0:
        st.warning("Please upload a dataset first.")
        st.session_state.current_step = 0
        st.rerun()
        return

    df = st.session_state.df
    info = None
    if df is not None:
        info = get_dataframe_info(df)

    # NEW FEATURE: Display Auto Summary at the top if available
    if st.session_state.auto_summary and st.session_state.current_step > 0:
        with st.expander("✨ AI Executive Summary", expanded=True):
            from src.ui_components import render_insight_card
            render_insight_card("Auto Summary", st.session_state.auto_summary, icon="✨")
        st.markdown("---")

    # 2. Dataset Overview
    if st.session_state.current_step == 1:
        render_header("Dataset Overview", "A quick look at your data structure.", "📋")
        st.subheader("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", info['shape'][0])
        col2.metric("Columns", info['shape'][1])
        col3.metric("Missing Values", sum(info['missing_values'].values()))
        col4.metric("Duplicates", info['duplicates'])
        
        st.subheader("Data Preview (Top 10)")
        st.dataframe(get_data_preview(df), use_container_width=True)
        
        with st.expander("Show Column Details"):
            dtypes_df = pd.DataFrame({
                "Column": info['columns'],
                "Type": [info['dtypes'][c] for c in info['columns']],
                "Missing": [info['missing_values'][c] for c in info['columns']]
            })
            st.table(dtypes_df)
        
        if st.button("Proceed to EDA →", type="primary"):
            st.session_state.current_step = 2
            st.rerun()

    # 3. Visual EDA
    elif st.session_state.current_step == 2:
        render_header("Visual Exploratory Data Analysis", "Comprehensive visualizations powered by Seaborn.", "📈")
        st.subheader("📈 Visual Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Summary & Missing", "Correlations", "Distributions"])
        
        with tab1:
            st.subheader("Summary Statistics")
            st.dataframe(generate_summary_stats(df), use_container_width=True)
            
            fig_missing = plot_missing_values(df)
            if fig_missing:
                st.pyplot(fig_missing)
                if st.button("Ask AI to explain this chart"):
                    with st.spinner("🧠 AI is analyzing the missing values..."):
                        explanation = explain_chart({"type": "Heatmap", "columns": "Missing values across all columns"}, data_context=f"Missing count: {sum(info['missing_values'].values())}")
                        st.info(explanation)
            else:
                st.success("No missing values found!")

        with tab2:
            st.subheader("Correlation Analysis")
            fig_corr = plot_correlation_matrix(df)
            if fig_corr:
                st.pyplot(fig_corr)
                if st.button("Explain Correlations"):
                    with st.spinner("🧠 AI is analyzing correlations..."):
                        explanation = explain_chart({"type": "Correlation Matrix", "columns": "Numerical column relationships"})
                        st.info(explanation)
            else:
                st.info("Not enough numeric columns for correlation matrix.")

        with tab3:
            st.subheader("Numerical & Categorical Distributions")
            figs_dist = plot_distributions(df)
            figs_count = plot_count_plots(df)
            
            if figs_dist or figs_count:
                all_figs = {**figs_dist, **figs_count}
                cols = st.columns(2)
                for i, (col_name, fig) in enumerate(all_figs.items()):
                    with cols[i % 2]:
                        st.pyplot(fig)
                        if st.button(f"Explain {col_name} chart", key=f"btn_{col_name}"):
                            with st.spinner(f"🧠 AI is analyzing {col_name}..."):
                                # Smarter context
                                context = f"Column: {col_name}, Type: {info['dtypes'].get(col_name)}"
                                explanation = explain_chart({"type": "Distribution/Count Plot", "columns": col_name}, data_context=context)
                                from src.ui_components import render_insight_card
                                render_insight_card(f"Analysis: {col_name}", explanation, icon="🧠")
            else:
                st.info("No visualizations could be generated.")

        st.markdown("### 🛠️ Suggested AI Actions")
        c1, c2, c3 = st.columns(3)
        if c1.button("🔍 Find Anomalies"):
            st.session_state.ai_task = "Identify potential anomalies, outliers, and data quality issues."
            st.session_state.insights = None # Force regeneration
            st.session_state.current_step = 3
            st.rerun()
        if c2.button("📊 Deep Correlations"):
            st.session_state.ai_task = "Analyze deep correlations and find hidden relationships between variables."
            st.session_state.insights = None # Force regeneration
            st.session_state.current_step = 3
            st.rerun()
        if c3.button("🧠 Full Dataset Summary"):
            st.session_state.ai_task = "Provide a comprehensive high-level summary of the entire dataset."
            st.session_state.insights = None # Force regeneration
            st.session_state.current_step = 3
            st.rerun()

        if st.button("Generate Full AI Insights →", type="primary"):
            st.session_state.ai_task = "General Analysis"
            st.session_state.current_step = 3
            st.rerun()

    # 4. AI Insights
    elif st.session_state.current_step == 3:
        render_header("AI Powered Insights", icon="🧠")
        st.subheader("🧠 AI Insights")
        
        if st.session_state.insights is None or st.button("Regenerate Insights"):
            with st.spinner("🤖 Analyzing dataset and thinking..."):
                summary = generate_summary_stats(df).to_string()
                insights = generate_insights(
                    df_summary=summary,
                    column_names=str(info['columns']),
                    dtypes=str(info['dtypes']),
                    missing_values=str(info['missing_values']),
                    correlations="Calculated from numeric columns",
                    task=st.session_state.get('ai_task', 'General Analysis')
                )
                st.session_state.insights = insights
        
        st.markdown(st.session_state.insights)
        
        st.divider()
        
        if st.checkbox("Show Data Preprocessing Recommendations"):
            if 'recommendations' not in st.session_state or st.button("Regenerate Recommendations"):
                with st.spinner("🔍 Analyzing for recommendations..."):
                    info_str = str(get_dataframe_info(df))
                    st.session_state.recommendations = generate_recommendations(info_str)
            st.markdown(st.session_state.recommendations)
        
        if st.button("Proceed to Export Report →", type="primary"):
            st.session_state.current_step = 4
            st.rerun()

    # 5. Export Report
    elif st.session_state.current_step == 4:
        render_header("Export Report", "Download your analysis as a document.", "📄")
        st.info("Download a comprehensive report containing your dataset info, summary, and AI insights.")
        
        if st.button("Generate HTML Report", type="primary"):
            with st.spinner("Brewing your report..."):
                from src.report_generator import generate_html_report
                stats_df = generate_summary_stats(df)
                info = get_dataframe_info(df)
                # Note: Currently passing None for figs because generating them for report needs care 
                # (Plotly figs to HTML works differently than Matplotlib)
                # For now, keeping it simple as per original report_generator's expectation of figs
                # I might need to update report_generator later to support Plotly.
                
                # Generate figures for the report
                fig_missing = plot_missing_values(df)
                fig_corr = plot_correlation_matrix(df)
                
                html_report = generate_html_report(
                    info, 
                    stats_df, 
                    fig_missing,
                    fig_corr,
                    st.session_state.get('insights', 'No insights generated. Please visit the AI Insights tab first.'), 
                    st.session_state.get('recommendations', 'No recommendations generated. Please visit the AI Insights tab first.'),
                    st.session_state.get('auto_summary')
                )
                
                st.download_button(
                    label="⬇️ Download Report (HTML)",
                    data=html_report,
                    file_name="datawhisper_report.html",
                    mime="text/html"
                )

if __name__ == "__main__":
    main_with_error_handling()
