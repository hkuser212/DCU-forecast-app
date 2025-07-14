# # -------------------- Load Regression Model --------------------
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from streamlit_navigation_bar import st_navbar


# -------------------- Load Regression Model --------------------



#2021-2024 model and features..
model = joblib.load('model_base_2021_24.pkl')
features = joblib.load('features_base_2021_24.pkl')
#2021-2024 model and features retrianed with 2025
# model = joblib.load('model_retrained_2025.pkl')
# features = joblib.load('features_base_2021_24.pkl')
# Fix column name typo if any
features = [f.replace("FeedFlow _Pass3", "FeedFlow_Pass3") for f in features]

st.set_page_config(page_title="DCU SAD Forecast", layout="wide")
st.title("Forecast DCU Shutdown (SAD) proactively using furnace telemetry data — powered by Machine Learning")
st.markdown("Predict **Days Until Next SAD** using real-time or uploaded furnace parameters")


st.header("📁 Upload Furnace CSV Data")
uploaded_file = st.file_uploader("Upload a .csv file with furnace readings", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=4)


    st.subheader("📄 Raw Data Preview")
    st.write(df.head())


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
    print("✅ Available columns after rename:")
    print(df.columns.tolist())
    # Rename columns
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=False, errors='coerce')

    # STEP 4: Display parsed results

    # Ensure required features are present
    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
    else:
        # Clean and predict
        for f in features:
            df[f] = pd.to_numeric(df[f], errors='coerce')

        # STEP 2: Normalize column names

        # STEP 3: Parse timestamp carefully


        if df.empty:
            st.warning("⚠️ No valid rows after cleaning. All rows had missing values.")
        else:
            # Make prediction
            df['Predicted_Days_to_SAD'] = model.predict(df[features])



            # 1. Clean and process all rows
            df['Predicted_Days_to_SAD'] = model.predict(df[features])


            # 2️⃣ Get earliest warning row
            # Apply rolling average with a window of 12
            # RAW CRITICAL METHOD
            min_pred_raw = df['Predicted_Days_to_SAD'].min()
            min_row_raw = df[df['Predicted_Days_to_SAD'] == min_pred_raw].iloc[0]

            if 'timestamp' in df.columns:
                min_ts_raw = pd.to_datetime(min_row_raw['timestamp'], errors='coerce')
            else:
                min_ts_raw = pd.to_datetime(min_row_raw.name, errors='coerce')

            est_sad_raw = min_ts_raw + pd.Timedelta(days=min_pred_raw)

            df['Rolling_SAD'] = df['Predicted_Days_to_SAD'].rolling(window=12, min_periods=1).mean()

            # ✅ Visualization: Predicted Days to SAD over time
            st.subheader("📈 Predicted Days to SAD vs Time")
            df.set_index('timestamp', inplace=True)
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

            # ✅ Allow user to pick a parameter for dual-plot
            selected_param = st.selectbox("📌 Select a parameter to compare with SAD Prediction:", features)
            fig2, ax1 = plt.subplots(figsize=(15, 5))
            ax1.plot(df.index, df[selected_param], color='green', label=selected_param)
            ax1.set_ylabel(selected_param, color='green')

            ax2 = ax1.twinx()
            ax2.plot(df.index, df['Predicted_Days_to_SAD'], color='blue', alpha=0.6, label='Predicted Days to SAD')
            ax2.set_ylabel('Predicted Days to SAD', color='blue')
            fig2.legend(loc='upper right')
            st.pyplot(fig2)

            # ✅ Feature Importance Visualization
            import seaborn as sns
            import numpy as np

            st.subheader("📊 Feature Importance Analysis")
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis', ax=ax3)
            ax3.set_title("Top Influential Parameters for SAD Prediction")
            ax3.set_xlabel("Importance Score")
            ax3.set_ylabel("Feature")
            ax3.grid(True)
            st.pyplot(fig3)

            st.info("✅ Higher importance score = more influence on SAD timing. Monitor these parameters closely.")

            min_pred = df['Rolling_SAD'].min()
            min_index = df['Rolling_SAD'].idxmin()
            min_row = df.loc[min_index]
            min_index_dt = pd.to_datetime(min_index)

            # Extract timestamp from column or index
            if 'timestamp' in df.columns:
                min_ts = pd.to_datetime(min_row['timestamp'], errors='coerce')
            else:
                min_ts = pd.to_datetime(min_row.name, errors='coerce')

            # Validate earliest timestamp
            if pd.isna(min_ts):
                st.error("❌ Earliest timestamp (for SAD) is not valid.")
            else:
                est_sad_from_min = min_ts + pd.Timedelta(days=min_pred)

                # Display alert
                if min_pred > 250:
                    st.success("✅ DCU is operating in a healthy range. No SAD required soon.")
                elif min_pred > 90:
                    st.warning("🟡 Conditions are stable, but monitor for trends. SAD may approach within 2–3 months.")
                else:
                    st.error("🔴 Alert: SAD likely required soon! Take action.")
                st.info(f"📉 Minimum (smoothed) Days to SAD: **{min_pred:.1f}** (on {min_index_dt:%d-%b-%Y})")
                st.success(f"📍 Estimated SAD Date (Rolling Avg): **{est_sad_from_min.strftime('%d-%b-%Y')}**")
                st.subheader("🔴 Critical Risk Estimate (Raw Minimum)")
                st.info(f"📉 Most Critical Prediction: **{min_pred_raw:.1f} days to SAD** (from {min_ts_raw:%d-%b-%Y})")
                st.warning(f"📍 Estimated SAD Date (Raw): **{est_sad_raw:%d-%b-%Y}**")

                st.info(
                    f"🔧 Based on current patterns, the model predicts next SAD between **{est_sad_from_min.strftime('%d-%b-%Y')} and {est_sad_raw:%d-%b-%Y}**.\n\n"
                    f"→ **Use Rolling Avg {est_sad_from_min.strftime('%d-%b-%Y')}** for planning.\n"
                    f"→ **Use Raw Min {est_sad_raw:%d-%b-%Y}** as a conservative bound for emergency readiness.\n\n"
                    "Monitor pressure drop, temp gradients, and feed stability weekly."
                )

#                 # column_mapping = {


