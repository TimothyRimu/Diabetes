"""Pima diabetes risk screen: logistic regression, honest pipeline (median imputation fitted on training data only)."""
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

st.set_page_config(page_title="Diabetes risk screen", page_icon="🩺", layout="centered")


@st.cache_resource
def load_model():
    d = pd.read_csv(ROOT / "data" / "pima_diabetes.csv")
    y = d.pop("Outcome")
    d[ZERO_AS_MISSING] = d[ZERO_AS_MISSING].replace(0, np.nan)
    pipe = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("m", LogisticRegression(max_iter=2000))]).fit(d, y)
    return pipe, d


pipe, X = load_model()
med = X.median()

st.title("Diabetes risk screen")
st.warning("**Educational demo, not medical advice.** Trained on 768 Pima Indian women aged 21+. "
           "It does not generalise to other populations and cannot diagnose anything. "
           "Cross-validated ROC AUC is about 0.84; see the notebook for the honest evaluation.")

st.caption("Leave a field at 0 if you do not know it: it is treated as missing and filled with the training median.")
c1, c2 = st.columns(2)
with c1:
    preg = st.number_input("Pregnancies", 0, 20, 2)
    glu = st.number_input("Glucose (mg/dL, 2h OGTT)", 0, 250, int(med.Glucose))
    bp = st.number_input("Blood pressure (diastolic, mm Hg)", 0, 130, int(med.BloodPressure))
    skin = st.number_input("Skin thickness (mm)", 0, 100, 0)
with c2:
    ins = st.number_input("Insulin (mu U/ml)", 0, 900, 0)
    bmi = st.number_input("BMI", 0.0, 70.0, float(round(med.BMI, 1)), step=0.1)
    dpf = st.number_input("Diabetes pedigree function", 0.0, 2.5, float(round(X.DiabetesPedigreeFunction.median(), 2)), step=0.01)
    age = st.number_input("Age", 21, 90, 33)

cut = st.select_slider("Flagging cut-off (lower = catches more, flags more)", [0.5, 0.4, 0.35, 0.3, 0.25], value=0.35)

row = pd.DataFrame([[preg, glu, bp, skin, ins, bmi, dpf, age]], columns=X.columns)
row[ZERO_AS_MISSING] = row[ZERO_AS_MISSING].replace(0, np.nan)
p = float(pipe.predict_proba(row)[0, 1])

st.metric("Estimated probability", f"{p:.0%}")
st.progress(min(p, 1.0))
st.write("**Flagged for follow-up**" if p >= cut else "Not flagged at this cut-off")

tt = pd.read_csv(ROOT / "results" / "threshold_tradeoff.csv").set_index("cut-off")
st.subheader("What the cut-off trades off")
st.caption("Out-of-fold results on the full dataset.")
st.dataframe(tt.style.format("{:.0%}"), width="stretch")

coef = pd.Series(pipe.named_steps["m"].coef_[0], index=X.columns).sort_values()
with st.expander("What drives the estimate (standardised coefficients)"):
    st.bar_chart(coef)
