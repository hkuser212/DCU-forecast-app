import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
model = joblib.load('model_base_2021_24.pkl')
features = joblib.load('features_base_2021_24.pkl')
features = [f.replace("FeedFlow _Pass3", "FeedFlow_Pass3") for f in features]

# Streamlit page setup
st.set_page_config(page_title="DCU SAD Forecast", layout="wide")
st.title("Forecast DCU Shutdown (SAD) proactively using furnace telemetry data — powered by Machine Learning")
st.markdown("Predict **Days Until Next SAD** using real-time or uploaded furnace parameters")

# Upload CSV
st.header("📁 Upload Furnace CSV Data")
uploaded_file = st.file_uploader("Upload a .csv file with furnace readings", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=4)

    st.subheader("📄 Raw Data Preview")
    st.write(df.head())

    # Rename columns
    column_mapping = {
        'Timestamp': "timestamp",
        'KG/CM2': "Coil_Inlet_Pressure_Pass1",
        'KG/CM2.1': "Coil_Inlet_Pressure_Pass2",
        'KG/CM2.2': "Coil_Inlet_Pressure_Pass3",
        'KG/CM2.3': "Coil_Inlet_Pressure_Pass4",
        'Unnamed: 7': 'FeedFlow_Pass1',
        'Unnamed: 8': 'FeedFlow_Pass2',
        'Unnamed: 9': 'FeedFlow_Pass3',
        'Unnamed: 10': 'FeedFlow_Pass4',
        'DEGC': 'COT_Pass1',
        'DEGC.1': 'COT_Pass2',
        'DEGC.2': 'COT_Pass3',
        'DEGC.3': 'COT_Pass4',
        'KG/CM2.4': 'COIL_dP_Pass1',
        'KG/CM2.5': 'COIL_dP_Pass2',
        'KG/CM2.6': 'COIL_dP_Pass3',
        'KG/CM2.7': 'COIL_dP_Pass4',
        'KG/HR': 'BFW_Injection_Pass1',
        'KG/HR.1': 'BFW_Injection_Pass2',
        'KG/HR.2': 'BFW_Injection_Pass3',
        'KG/HR.3': 'BFW_Injection_Pass4',
        'DEGC.4': 'MAX_SkinTemp_Pass1',
        'DEGC.5': 'Max_SkinTemp_Pass2',
        'DEGC.6': 'Max_SkinTemp_Pass3',
        'DEGC.7': 'Max_SkinTemp_Pass4',
    }
    df.rename(columns=column_mapping, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)

    # Check missing columns
    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

    # Clean data
    for f in features:
        df[f] = pd.to_numeric(df[f], errors='coerce')

    df.dropna(subset=features, inplace=True)
    if df.empty:
        st.warning("⚠️ No valid rows left after cleaning.")
        st.stop()

    df.set_index('timestamp', inplace=True)

    # Prediction
    df['Predicted_Days_to_SAD'] = model.predict(df[features])
    df['Rolling_SAD'] = df['Predicted_Days_to_SAD'].rolling(window=12, min_periods=1).mean()

    # Get key values
    min_pred = df['Rolling_SAD'].min()
    min_ts = df['Rolling_SAD'].idxmin()
    est_sad_from_min = min_ts + pd.Timedelta(days=min_pred) if pd.notna(min_ts) else pd.NaT

    min_pred_raw = df['Predicted_Days_to_SAD'].min()
    min_ts_raw = df['Predicted_Days_to_SAD'].idxmin()
    est_sad_raw = min_ts_raw + pd.Timedelta(days=min_pred_raw) if pd.notna(min_ts_raw) else pd.NaT

    # Format for display
    min_ts_str = min_ts.strftime('%d-%b-%Y') if pd.notna(min_ts) else "Unknown"
    est_sad_from_min_str = est_sad_from_min.strftime('%d-%b-%Y') if pd.notna(est_sad_from_min) else "Unknown"
    min_ts_raw_str = min_ts_raw.strftime('%d-%b-%Y') if pd.notna(min_ts_raw) else "Unknown"
    est_sad_raw_str = est_sad_raw.strftime('%d-%b-%Y') if pd.notna(est_sad_raw) else "Unknown"

    # Display alerts
    if min_pred > 250:
        st.success("✅ DCU is operating in a healthy range. No SAD required soon.")
    elif min_pred > 90:
        st.warning("🟡 Conditions are stable, but monitor for trends. SAD may approach within 2–3 months.")
    else:
        st.error("🔴 Alert: SAD likely required soon! Take action.")

    # Display info
    st.info(f"📉 Minimum (smoothed) Days to SAD: **{min_pred:.1f}** (on {min_ts_str})")
    st.success(f"📍 Estimated SAD Date (Rolling Avg): **{est_sad_from_min_str}**")
    st.subheader("🔴 Critical Risk Estimate (Raw Minimum)")
    st.info(f"📉 Most Critical Prediction: **{min_pred_raw:.1f} days to SAD** (from {min_ts_raw_str})")
    st.warning(f"📍 Estimated SAD Date (Raw): **{est_sad_raw_str}**")

    # Ensure chronological ordering
    # Ensure dates are in correct order
    start_date, end_date = sorted([est_sad_from_min_str, est_sad_raw_str])

    # Determine which estimate comes first and construct detailed descriptions
    if est_sad_raw_str == start_date:
        planning_msg = (
            f"→ **Raw Minimum ({est_sad_raw_str})**: This is the **earliest predicted SAD date**, triggered by a sudden drop in model outputs. "
            "It signals a possible abrupt onset of fouling or performance degradation. Treat it as a **conservative trigger** for emergency preparedness, "
            "closer surveillance, and operational flexibility."
        )
        strategy_msg = (
            f"→ **Rolling Average ({est_sad_from_min_str})**: This reflects a **sustained declining trend** based on smoothed model behavior. "
            "It’s recommended for **maintenance planning, scheduling of shutdown resources, and coordination with production timelines**."
        )
    else:
        planning_msg = (
            f"→ **Rolling Average ({est_sad_from_min_str})**: This represents the **expected planning window** based on consistent model patterns. "
            "Use it for **routine shutdown preparation and logistics coordination**."
        )
        strategy_msg = (
            f"→ **Raw Minimum ({est_sad_raw_str})**: This is a **later critical point** based on a localized prediction dip. "
            "It serves as a **safety margin** — a backup alert window if trend-based planning is deferred or delayed."
        )

    # Display final info
    st.info(
        f"🔧 Based on current model outputs, the next SAD (Shutdown and Decoking) event is expected between **{start_date} and {end_date}**.\n\n"
        f"{planning_msg}\n\n"
        f"{strategy_msg}\n\n"
        "Monitor pressure drop, temp gradients, and feed stability weekly."
    )



    # Plot
    st.subheader("📈 Predicted Days to SAD vs Time")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df.index, df['Predicted_Days_to_SAD'], label='Predicted Days to SAD', color='blue')
    ax.plot(df.index, df['Rolling_SAD'], label='Rolling Avg (Window=12)', color='green', linestyle='--')
    ax.axhline(30, color='orange', linestyle='--', label='Warning Threshold')
    ax.axhline(10, color='red', linestyle='--', label='Critical Threshold')
    ax.set_ylabel('Days to SAD')
    ax.set_title('SAD Prediction Timeline')
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate(rotation=45)
    st.pyplot(fig)

    # Dual plot
    selected_param = st.selectbox("📌 Select a parameter to compare with SAD Prediction:", features)
    fig2, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(df.index, df[selected_param], color='green', label=selected_param)
    ax1.set_ylabel(selected_param, color='green')
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['Predicted_Days_to_SAD'], color='blue', alpha=0.6, label='Predicted Days to SAD')
    ax2.set_ylabel('Predicted Days to SAD', color='blue')
    fig2.legend(loc='upper right')
    st.pyplot(fig2)

    # Feature Importance
    # ✅ Feature Importance Block
    st.subheader("📊 Feature Importance Analysis")
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

    # ✅ Feature Importance Plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis', ax=ax3)
    ax3.set_title("Top Influential Parameters for SAD Prediction")
    ax3.set_xlabel("Importance Score")
    ax3.set_ylabel("Feature")
    ax3.grid(True)
    st.pyplot(fig3)

    # ✅ Dynamic Recommendation Message
    top_n = 3
    top_features = feat_imp_df['Feature'].head(top_n)

    feature_explanations = {
        "Coil_Inlet_Pressure_Pass1": "coil inlet pressure (Pass 1)",
        "Coil_Inlet_Pressure_Pass2": "coil inlet pressure (Pass 2)",
        "Coil_Inlet_Pressure_Pass3": "coil inlet pressure (Pass 3)",
        "Coil_Inlet_Pressure_Pass4": "coil inlet pressure (Pass 4)",
        "FeedFlow_Pass1": "feed flow rate (Pass 1)",
        "FeedFlow_Pass2": "feed flow rate (Pass 2)",
        "FeedFlow_Pass3": "feed flow rate (Pass 3)",
        "FeedFlow_Pass4": "feed flow rate (Pass 4)",
        "COT_Pass1": "coil outlet temperature (Pass 1)",
        "COT_Pass2": "coil outlet temperature (Pass 2)",
        "COT_Pass3": "coil outlet temperature (Pass 3)",
        "COT_Pass4": "coil outlet temperature (Pass 4)",
        "COIL_dP_Pass1": "pressure drop across coil (Pass 1)",
        "COIL_dP_Pass2": "pressure drop across coil (Pass 2)",
        "COIL_dP_Pass3": "pressure drop across coil (Pass 3)",
        "COIL_dP_Pass4": "pressure drop across coil (Pass 4)",
        "BFW_Injection_Pass1": "BFW injection rate (Pass 1)",
        "BFW_Injection_Pass2": "BFW injection rate (Pass 2)",
        "BFW_Injection_Pass3": "BFW injection rate (Pass 3)",
        "BFW_Injection_Pass4": "BFW injection rate (Pass 4)",
        "MAX_SkinTemp_Pass1": "maximum skin temperature (Pass 1)",
        "Max_SkinTemp_Pass2": "maximum skin temperature (Pass 2)",
        "Max_SkinTemp_Pass3": "maximum skin temperature (Pass 3)",
        "Max_SkinTemp_Pass4": "maximum skin temperature (Pass 4)",
    }

    # Convert to friendly explanation
    explanations = [f"**{feature_explanations.get(f, f)}**" for f in top_features]

    # Join in readable format
    if len(explanations) == 1:
        rec_line = explanations[0]
    elif len(explanations) == 2:
        rec_line = f"{explanations[0]} and {explanations[1]}"
    else:
        rec_line = ", ".join(explanations[:-1]) + f", and {explanations[-1]}"

    # Show recommended action
    st.markdown(
        f"📌 *Recommended Action:* Closely monitor {rec_line}. "
        "Significant deviations may shift the predicted SAD window and require earlier intervention."
    )
