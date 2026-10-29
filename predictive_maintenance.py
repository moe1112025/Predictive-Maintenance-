"""Industrial sensor and synthetic maintenance-pattern classifier.

Author: Moe Htet Ar Kar (Phoe Cho)

This learning example uses generated sensor data and rule-created labels.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title = "Predictive Maintenance", page_icon = "⚙️", layout = "wide")
st.markdown("<style>" + """.stApp{background:linear-gradient(145deg,#17120c 0%,#262016 55%,#16130f 100%);color:#fff7e8;} 
[data-testid='stSidebar']{background:#252016;} [data-testid='stMetric']{background:#30271a;border:1px solid #77532a;padding:14px;
border-radius:12px;} .stButton>button{background:#f0a43c;color:#281708;border:0;border-radius:9px;font-weight:700;} h1,h2,h3{color:#fff3df;}
""" + "</style>", unsafe_allow_html = True)

st.title("⚙️ Predictive Maintenance Monitor")
st.caption("Sensor inputs · Gradient Boosting classifier")
st.warning("Synthetic-data learning project. This model does not estimate real-world machine failure probability.")

FEATURES = ["Air Temp (K)", "Process Temp (K)", "Rotational Speed (rpm)", "Torque (Nm)", "Tool Wear (min)"]


@st.cache_resource
def train_maintenance_model():

    """Train on generated sensor readings with an explicit target rule."""
    rng = np.random.default_rng(42)
    sample_count = 1000
    training = pd.DataFrame(
        {
            FEATURES[0]: rng.uniform(295, 305, sample_count),
            FEATURES[1]: rng.uniform(305, 315, sample_count),
            FEATURES[2]: rng.integers(1100, 2800, sample_count),
            FEATURES[3]: rng.uniform(10, 80, sample_count),
            FEATURES[4]: rng.integers(0, 250, sample_count),
        }
    )

    # A synthetic class label, not an observed machine-failure event.
    labels = ((training[FEATURES[3]] > 60) & (training[FEATURES[4]] > 180)).astype(int)
    classifier = GradientBoostingClassifier(n_estimators = 100, random_state = 42)
    classifier.fit(training[FEATURES], labels)
    return classifier


model = train_maintenance_model()
st.sidebar.header("Machine sensor inputs")
air_temp = st.sidebar.slider("Air temperature (K)", 295.0, 305.0, 298.0)
process_temp = st.sidebar.slider("Process temperature (K)", 305.0, 315.0, 308.0)
rotational_speed = st.sidebar.slider("Rotational speed (rpm)", 1100, 2800, 1500)
torque = st.sidebar.slider("Torque (Nm)", 10.0, 80.0, 40.0)
tool_wear = st.sidebar.slider("Tool wear (min)", 0, 250, 50)

sensor_row = pd.DataFrame([[air_temp, process_temp, rotational_speed, torque, tool_wear]], columns = FEATURES)
left, right = st.columns([1.35, 1])

with left:
    st.subheader("Current sensor readings")
    st.dataframe(sensor_row, use_container_width = True, hide_index = True)
    st.caption("Inputs are compared with a synthetic dataset that uses the same five features.")

with right:
    st.subheader("Maintenance-pattern result")
    if st.button("Analyze sensor pattern", use_container_width = True):
        predicted_class = int(model.predict(sensor_row[FEATURES])[0])
        pattern_score = float(model.predict_proba(sensor_row[FEATURES])[0, 1])
        st.metric("Model score for synthetic risk class", f"{pattern_score * 100:.2f}%")

        if predicted_class == 1:
            st.error("SYNTHETIC RISK PATTERN FLAGGED")
            st.write("The entered values resemble the rule-defined class used for training. This is not a real failure forecast.")
        else:
            st.success("SYNTHETIC RISK PATTERN NOT FLAGGED")
            st.write("The model did not classify these inputs as the rule-defined class. This is not proof that equipment is healthy.")
        with st.expander("Training label definition"):
            st.code("risk_class = (torque > 60) and (tool_wear > 180)")
            st.write("The target is rule-generated; no actual machine-failure records are included.")
    else:
        st.info("Adjust the sensor values and click **Analyze sensor pattern**.")

st.divider()
st.caption("Educational industrial ML prototype · Synthetic data · Not for safety-critical maintenance decisions")
