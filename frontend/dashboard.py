import streamlit as st
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Medical Imaging System",
    page_icon="🩻",
    layout="wide"
)

st.title("🩻 AI-Powered Medical Imaging System")
st.caption("Chest X-ray Classification | MRI Segmentation | CT Segmentation")

st.warning(
    "Educational and research prototype only. "
    "Predictions are not medical diagnoses and must not guide clinical decisions."
)

# API connection
try:
    health_response = requests.get(f"{API_URL}/health", timeout=10)
    health_response.raise_for_status()
    st.success("FastAPI backend connected.")
except requests.RequestException as exc:
    st.error(f"Cannot connect to API: {exc}")
    st.stop()

# Model information
try:
    info_response = requests.get(f"{API_URL}/model-info", timeout=20)
    info_response.raise_for_status()
    model_info = info_response.json()

    with st.expander("Model Information", expanded=False):
        for modality in ["xray", "mri", "ct"]:
            details = model_info.get(modality)
            if details:
                st.subheader(modality.upper())
                st.json(details)

except requests.RequestException as exc:
    model_info = {}
    st.error(f"Could not retrieve model information: {exc}")

st.divider()

modality = st.selectbox(
    "Select Imaging Modality",
    ["Chest X-ray", "MRI Brain", "CT Lung"]
)

# ---------------- CHEST X-RAY ----------------

if modality == "Chest X-ray":
    st.subheader("Chest X-ray Classification")

    uploaded_file = st.file_uploader(
        "Upload chest X-ray",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(
            BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Chest X-ray",
                use_container_width=True
            )

        with col2:
            st.write("Filename:", uploaded_file.name)
            st.write("Dimensions:", f"{image.width} × {image.height}")
            st.write("Format:", uploaded_file.type)

        if st.button("Analyze X-ray", type="primary"):
            try:
                with st.spinner("Running DenseNet121 inference..."):
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type
                            )
                        },
                        timeout=300
                    )

                response.raise_for_status()
                result = response.json()

                predictions = result["predictions"]
                df = pd.DataFrame(predictions)

                st.session_state["xray_result"] = result
                st.session_state["xray_df"] = df

            except requests.RequestException as exc:
                st.error(f"X-ray inference failed: {exc}")
            except (KeyError, ValueError) as exc:
                st.error(f"Unexpected API response: {exc}")

    if "xray_result" in st.session_state:
        result = st.session_state["xray_result"]
        df = st.session_state["xray_df"]

        st.divider()
        st.subheader("Classification Results")

        display_df = df.rename(columns={
            "condition": "Condition",
            "probability": "Probability",
            "percentage": "Score (%)"
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        if "condition" in df.columns and "percentage" in df.columns:
            st.subheader("Prediction Score Distribution")
            st.bar_chart(
                df.set_index("condition")[["percentage"]],
                horizontal=True
            )

        st.download_button(
            "Download X-ray Results CSV",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="xray_predictions.csv",
            mime="text/csv"
        )

        st.caption(
            f"Model: {result.get('model')} | "
            f"Checkpoint epoch: {result.get('checkpoint_epoch')}"
        )

        st.warning(result.get("disclaimer", "Research use only."))

# ---------------- MRI ----------------

elif modality == "MRI Brain":
    st.subheader("MRI Brain Tumour Segmentation")

    st.info(
        "Uses an existing preprocessed MRI case from "
        "data/processed/mri."
    )

    case_id = st.text_input(
        "MRI Case ID",
        value="BRATS_001",
        help="Example: BRATS_001"
    ).strip()

    if st.button("Analyze MRI", type="primary"):
        if not case_id:
            st.error("Enter an MRI case ID.")
        else:
            try:
                with st.spinner("Running MRI 3D U-Net inference..."):
                    response = requests.post(
                        f"{API_URL}/predict/mri/{case_id}",
                        timeout=600
                    )

                response.raise_for_status()
                result = response.json()
                st.session_state["mri_result"] = result

            except requests.RequestException as exc:
                st.error(f"MRI inference failed: {exc}")

    if "mri_result" in st.session_state:
        result = st.session_state["mri_result"]

        st.divider()
        st.subheader("MRI Segmentation Results")

        st.write("Case ID:", result.get("case_id"))
        st.write("Model:", result.get("model"))
        st.write("Checkpoint epoch:", result.get("checkpoint_epoch"))
        st.write("Validation patch Dice:", result.get("validation_patch_dice"))

        segmentation = result.get("segmentation", {})
        st.write("Volume shape:", segmentation.get("volume_shape"))
        st.metric("Total voxels", segmentation.get("total_voxels", 0))

        distribution = segmentation.get("class_distribution", {})
        if distribution:
            df = pd.DataFrame([
                {
                    "Class": name,
                    "Voxels": values["voxels"],
                    "Percentage": values["percentage"]
                }
                for name, values in distribution.items()
            ])

            st.dataframe(df, use_container_width=True, hide_index=True)
            st.bar_chart(df.set_index("Class")[["Percentage"]])

            st.download_button(
                "Download MRI Segmentation Summary CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{result.get('case_id')}_mri_summary.csv",
                mime="text/csv"
            )

        st.warning(result.get("disclaimer", "Research use only."))

# ---------------- CT ----------------

elif modality == "CT Lung":
    st.subheader("CT Lung Lesion Segmentation")

    st.info(
        "Uses an existing preprocessed CT case from "
        "data/processed/ct."
    )

    case_id = st.text_input(
        "CT Case ID",
        value="lung_001",
        help="Example: lung_001"
    ).strip()

    if st.button("Analyze CT", type="primary"):
        if not case_id:
            st.error("Enter a CT case ID.")
        else:
            try:
                with st.spinner("Running CT 3D U-Net inference..."):
                    response = requests.post(
                        f"{API_URL}/predict/ct/{case_id}",
                        timeout=600
                    )

                response.raise_for_status()
                result = response.json()
                st.session_state["ct_result"] = result

            except requests.RequestException as exc:
                st.error(f"CT inference failed: {exc}")

    if "ct_result" in st.session_state:
        result = st.session_state["ct_result"]

        st.divider()
        st.subheader("CT Segmentation Results")

        st.write("Case ID:", result.get("case_id"))
        st.write("Model:", result.get("model"))
        st.write("Checkpoint epoch:", result.get("checkpoint_epoch"))
        st.write("Validation patch Dice:", result.get("validation_patch_dice"))

        segmentation = result.get("segmentation", {})
        st.write("Volume shape:", segmentation.get("volume_shape"))
        st.metric("Total voxels", segmentation.get("total_voxels", 0))

        distribution = segmentation.get("class_distribution", {})
        if distribution:
            df = pd.DataFrame([
                {
                    "Class": name,
                    "Voxels": values["voxels"],
                    "Percentage": values["percentage"]
                }
                for name, values in distribution.items()
            ])

            st.dataframe(df, use_container_width=True, hide_index=True)
            st.bar_chart(df.set_index("Class")[["Percentage"]])

            st.download_button(
                "Download CT Segmentation Summary CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{result.get('case_id')}_ct_summary.csv",
                mime="text/csv"
            )

        st.warning(result.get("disclaimer", "Research use only."))

st.divider()
st.caption("AIML-02 | AI Medical Imaging System | Educational Research Prototype")
