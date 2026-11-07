import streamlit as str_lit
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_auc_score)

# Page Configuration in StreamLit
str_lit.set_page_config(page_title="Mobile Price Classification", layout="wide")

# Load Model and data
model = joblib.load("model/logistic_regression.pkl")
scaler = joblib.load("model/scaler.pkl")

# Set Application Title
str_lit.title("Mobile Price Classification")

# Navigation Menu
page = str_lit.sidebar.selectbox("Select Page", ["Home", "Test Dataset Evaluation","ML Classifier Performance Comparison","Prediction"])

str_lit.sidebar.markdown("---")
str_lit.sidebar.markdown("## Mobile Price Classification")

if page == "Home": #Home Page

    str_lit.header("Problem Statement")
    str_lit.info("""
The objective of this project is to predict the price category of a mobile phone
based on its hardware and technical specifications. Twenty features such as battery
power, RAM, internal memory, processor cores, camera resolution, display
characteristics, and connectivity options are used as input variables.

In order to predict the price, five machine learning classification algorithms have been
implemented and compared: Logistic Regression, Decision Tree, K-Nearest Neighbors
(KNN), Naive Bayes, and Support Vector Machine (SVM). Each classifier has been evaluated
using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation
Coefficient (MCC).

Among the evaluated classifiers, Logistic Regression achieved the best overall
performance with an accuracy of 95.75%. The trained models can be used to classify
mobile devices into one of four categories: Low Cost, Medium Cost, High Cost,
and Very High Cost.
""")

    train_df = pd.read_csv("data/train.csv")

    str_lit.header("Dataset Information")
    str_lit.write("Dataset Shape:", train_df.shape)
    str_lit.write("Number of Records:", train_df.shape[0])
    str_lit.write("Number of Features:", train_df.shape[1] - 1)

    str_lit.subheader("Price Category Distribution")

    price_distribution = (train_df["price_range"].value_counts().sort_index())

    price_distribution.index = ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]

    str_lit.bar_chart(price_distribution)

    str_lit.subheader("Dataset Preview")
    with str_lit.expander("View Dataset Preview"):
        str_lit.dataframe(train_df.head(), hide_index=True, use_container_width=True)

elif page == "Test Dataset Evaluation":

    str_lit.header("Model Evaluation Using Uploaded Test Dataset")

    file_uploaded_by_user = str_lit.file_uploader("Upload Test Dataset CSV",type=["csv"])

    selected_model = str_lit.selectbox("Select Classifier Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Support Vector Machine"
        ]
    )

    if file_uploaded_by_user is not None:

        user_test_df = pd.read_csv(file_uploaded_by_user)

        str_lit.subheader("Uploaded Dataset Preview")
        str_lit.dataframe(user_test_df.head(), hide_index=True, use_container_width=True)

        if "price_range" not in user_test_df.columns:

            str_lit.error("The uploaded CSV must contain the target column 'price_range'.")

        else:

            X_test = user_test_df.drop("price_range", axis=1)

            y_test = user_test_df["price_range"]

            model_paths = {
                "Logistic Regression":
                    "model/logistic_regression.pkl",

                "Decision Tree":
                    "model/decision_tree_classifier.pkl",

                "KNN":
                    "model/knn_classifier.pkl",

                "Naive Bayes":
                    "model/naive_bayes_classifier.pkl",

                "Support Vector Machine":
                    "model/svm_classifier.pkl"
            }

            ld_model = joblib.load(model_paths[selected_model])

            # Models trained on scaled features

            X_test_processed = scaler.transform(X_test)

            ld_mdl_predict = ld_model.predict(X_test_processed)

            ld_mdl_prob = ld_model.predict_proba(X_test_processed)

            acc_score = accuracy_score(y_test, ld_mdl_predict)

            auc_score = roc_auc_score(y_test, ld_mdl_prob, multi_class="ovr")

            prec_score = precision_score(y_test, ld_mdl_predict, average="weighted")

            recall_score = recall_score(y_test, ld_mdl_predict, average="weighted")

            f1 = f1_score(y_test, ld_mdl_predict, average="weighted")

            mcc_score = matthews_corrcoef(y_test, ld_mdl_predict)

            str_lit.subheader("Evaluation Metrics")

            # split 6 metrics into 3 columns and 2 rows for display
            col1, col2, col3 = str_lit.columns(3)

            col1.metric("Accuracy", f"{acc_score:.4f}")

            col2.metric("AUC",f"{auc_score:.4f}")

            col3.metric("Precision",f"{prec_score:.4f}")

            col1.metric("Recall", f"{recall_score:.4f}")

            col2.metric("F1 Score", f"{f1:.4f}")

            col3.metric("MCC", f"{mcc_score:.4f}")

            str_lit.subheader("Confusion Matrix")
            class_names = ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]

            conf_mtx = confusion_matrix(y_test, ld_mdl_predict)
            conf_mtx_df = pd.DataFrame(conf_mtx, index=class_names, columns=class_names)

            str_lit.dataframe(conf_mtx_df)            

            str_lit.subheader("Classification Report")

            class_rpt = classification_report(y_test,ld_mdl_predict,target_names=class_names, output_dict=True)
            class_rpt_df = pd.DataFrame(class_rpt).transpose()
            class_rpt_df = class_rpt_df.loc[class_names]
            str_lit.dataframe(class_rpt_df.round(4), use_container_width=True)

elif page == "ML Classifier Performance Comparison": # Model Comparison Page
    
    str_lit.header("Machine Learning Classifier Performance")
    perf_metrics_df = pd.read_csv("data/metrics.csv")
    str_lit.subheader("Classifier Evaluation Metrics")

    disp_metrics_df = perf_metrics_df.copy()

    disp_metrics_df.insert(0, "Rank", range(1, len(disp_metrics_df) + 1))

    disp_metrics_df = disp_metrics_df.set_index("Rank")

    str_lit.dataframe(disp_metrics_df, use_container_width=True)

    best_perf_model = perf_metrics_df.iloc[0]["Model"]

    str_lit.success(f"Best Performing Classifier Model: {best_perf_model}")

    selected_evaluation_metric = str_lit.selectbox("Select Performance Metric",
        [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC"
        ]
    )

    chart_metric_df = perf_metrics_df.sort_values(by=selected_evaluation_metric, ascending=False)

    str_lit.subheader(f"{selected_evaluation_metric} Comparison")

    str_lit.bar_chart(chart_metric_df.set_index("Model")[selected_evaluation_metric])

elif page == "Prediction": # Prediction Page

    str_lit.header("Mobile Price Prediction")

    fst_col, snd_col = str_lit.columns(2)

    with fst_col:

        battery_power = str_lit.number_input("Battery Power", min_value=500, max_value=2000, value=1000)

        blue = str_lit.selectbox("Bluetooth", [0, 1])

        clock_speed = str_lit.number_input("Clock Speed", min_value=0.5, max_value=3.0, value=2.0)

        dual_sim = str_lit.selectbox("Dual SIM", [0, 1])

        fc = str_lit.number_input("Front Camera (MP)", min_value=0, max_value=20, value=5)

        four_g = str_lit.selectbox("4G Support",[0, 1])

        int_memory = str_lit.number_input("Internal Memory (GB)", min_value=2, max_value=128, value=32)

        m_dep = str_lit.number_input("Mobile Depth", min_value=0.1, max_value=1.0, value=0.5)

        mobile_wt = str_lit.number_input("Mobile Weight", min_value=80, max_value=250, value=150)

        n_cores = str_lit.number_input("Number of CPU Cores", min_value=1, max_value=8, value=4)

    with snd_col:

        pc = str_lit.number_input("Primary Camera (MP)", min_value=0, max_value=25, value=10)

        px_height = str_lit.number_input("Pixel Height", min_value=0, value=1000)

        px_width = str_lit.number_input("Pixel Width", min_value=0, value=1500)

        ram = str_lit.number_input("RAM", min_value=256, max_value=4000, value=2000)

        sc_h = str_lit.number_input("Screen Height", min_value=5, max_value=25, value=15)

        sc_w = str_lit.number_input("Screen Width", min_value=0, max_value=20, value=8)

        talk_time = str_lit.number_input("Talk Time", min_value=2, max_value=24, value=12)

        three_g = str_lit.selectbox("3G Support", [0, 1])

        touch_screen = str_lit.selectbox("Touch Screen", [0, 1])

        wifi = str_lit.selectbox("WiFi", [0, 1])

    if str_lit.button("Predict Price Category"):

        user_input = pd.DataFrame({
            "battery_power": [battery_power],
            "blue": [blue],
            "clock_speed": [clock_speed],
            "dual_sim": [dual_sim],
            "fc": [fc],
            "four_g": [four_g],
            "int_memory": [int_memory],
            "m_dep": [m_dep],
            "mobile_wt": [mobile_wt],
            "n_cores": [n_cores],
            "pc": [pc],
            "px_height": [px_height],
            "px_width": [px_width],
            "ram": [ram],
            "sc_h": [sc_h],
            "sc_w": [sc_w],
            "talk_time": [talk_time],
            "three_g": [three_g],
            "touch_screen": [touch_screen],
            "wifi": [wifi]
        })

        user_input_scaled = scaler.transform(user_input)

        user_inp_pred = model.predict(user_input_scaled)[0]

        user_inp_pred_prob = model.predict_proba(user_input_scaled)

        confidence = (user_inp_pred_prob.max() * 100)

        labels = {
            0: "Low Cost",
            1: "Medium Cost",
            2: "High Cost",
            3: "Very High Cost"
        }

        str_lit.success(f"Predicted Price Category: {labels[user_inp_pred]}")

        str_lit.info(f"Prediction Confidence: {confidence:.2f}%")
