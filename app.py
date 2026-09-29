import hashlib
import os
import joblib
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(page_title="Multi-Class Digit Classifier", layout="centered")

# Custom CSS for compact buttons
st.markdown(
    """
    <style>
    div.stButton > button {
        padding: 2px 10px !important;
        font-size: 13px !important;
        min-height: 0px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown("### Digit Classifier")

# ==============================================================================
# 2. STATE MANAGEMENT & MODEL LOADING
# ==============================================================================
if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0

STANDARD_MODEL_PATH = "models/svc_mnist_digit_classifier.joblib"
HIGH_PREC_MODEL_PATH = "models/svc_mnist_digit_classifier_58k.joblib"


@st.cache_resource
def _load_classifier(model_path, model_signature):
    """Cache each model revision, not just its filename."""
    return joblib.load(model_path)


def load_classifier(model_path):
    # A deployment can replace a model without changing its path or this code.
    with open(model_path, "rb") as model_file:
        model_signature = hashlib.sha256(model_file.read()).hexdigest()
    return _load_classifier(model_path, model_signature)


# Helper function to find a model file in 'models/' or root directory
def resolve_model_path(path):
    if os.path.exists(path):
        return path
    # Check if file exists in root instead of models/ directory
    base_name = os.path.basename(path)
    if os.path.exists(base_name):
        return base_name
    return None


# ==============================================================================
# 3. SIDEBAR CONTROLS & MODEL SELECTION
# ==============================================================================
st.sidebar.header("Controls & Settings")
stroke_width = st.sidebar.slider("Brush Size", 10, 40, 25)
enable_centering = st.sidebar.checkbox("Enable MNIST Auto-Centering", value=True)

# Model selection checkbox (Default: Unchecked)
use_high_prec = st.sidebar.checkbox("Use High-Precision Model (58k)", value=False)

# Determine which model file to load
selected_path = None
if use_high_prec:
    resolved_high_prec = resolve_model_path(HIGH_PREC_MODEL_PATH)
    if resolved_high_prec:
        selected_path = resolved_high_prec
    else:
        st.sidebar.info(
            "High-precision model (`58k`) file not found. Falling back to standard model."
        )
        selected_path = resolve_model_path(STANDARD_MODEL_PATH)
else:
    selected_path = resolve_model_path(STANDARD_MODEL_PATH)

# Load the active model
if selected_path and os.path.exists(selected_path):
    estimator = load_classifier(selected_path)
else:
    st.error("No valid model file found! Please verify your model path.")
    st.stop()

labels = estimator.classes_.tolist()
supported_digits = ", ".join(str(digit) for digit in labels)
st.sidebar.caption(f"Recognizes digits: {supported_digits}")


# ==============================================================================
# 4. HELPER FUNCTION: TRUE MNIST CENTERING & RESIZING
# ==============================================================================
def center_digit_image(img_28x28):
    """Preprocesses drawn images to match original 1990s MNIST specs:

    1. Scales stroke into a 20x20 box (preserving aspect ratio).
    2. Centers the digit using Center of Mass (intensity centroid).
    """
    rows, cols = np.where(img_28x28 > 0.05)
    if len(rows) == 0 or len(cols) == 0:
        return img_28x28

    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    crop = img_28x28[row_min : row_max + 1, col_min : col_max + 1]
    crop_h, crop_w = crop.shape

    # Scale to fit inside 20x20 box
    if crop_h > crop_w:
        new_h = 20
        new_w = max(1, int(round((crop_w / crop_h) * 20)))
    else:
        new_w = 20
        new_h = max(1, int(round((crop_h / crop_w) * 20)))

    crop_img = Image.fromarray((crop * 255).astype(np.uint8))
    resized_crop = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_arr = np.array(resized_crop) / 255.0

    canvas_28x28 = np.zeros((28, 28), dtype=np.float32)
    start_r = (28 - new_h) // 2
    start_c = (28 - new_w) // 2
    canvas_28x28[
        start_r : start_r + new_h, start_c : start_c + new_w
    ] = resized_arr

    # Center of mass alignment
    cy, cx = center_of_mass(canvas_28x28)
    if not np.isnan(cy) and not np.isnan(cx):
        shift_y = int(round(13.5 - cy))
        shift_x = int(round(13.5 - cx))
        canvas_28x28 = np.roll(canvas_28x28, shift_y, axis=0)
        canvas_28x28 = np.roll(canvas_28x28, shift_x, axis=1)

    return canvas_28x28


# ==============================================================================
# 5. USER INTERFACE (CANVAS & BUTTONS)
# ==============================================================================
st.write(f"Draw a single digit ({supported_digits}) below:")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"digit_canvas_{st.session_state['canvas_key']}",
    update_streamlit=True,
    return_image_data=True,
)

col1, col2, _ = st.columns([0.2, 0.25, 0.55])

with col1:
    predict_clicked = st.button("Predict", type="primary")

with col2:
    if st.button("Clear Canvas"):
        st.session_state["canvas_key"] += 1
        st.rerun()

# ==============================================================================
# 6. PREPROCESSING & INFERENCE LOGIC
# ==============================================================================
if predict_clicked:
    if canvas_result.image_data is not None:
        rgba_array = canvas_result.image_data.astype("uint8")
        img = Image.fromarray(rgba_array)
        img_gray = img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
        raw_28x28 = np.array(img_gray) / 255.0

        if enable_centering:
            final_28x28 = center_digit_image(raw_28x28)
        else:
            final_28x28 = raw_28x28

        df_test_features = final_28x28.reshape(1, -1)

        predicted_class = estimator.predict(df_test_features)[0]
        decision_scores = estimator.decision_function(df_test_features)[0]
        # SVC score columns follow classes_, which need not be digits 0-9.
        if len(labels) == 2 and np.ndim(decision_scores) == 0:
            decision_scores = [-decision_scores, decision_scores]
        scores_by_digit = dict(zip(labels, decision_scores))

        st.success(f"### Predicted Digit: **{predicted_class}**")

        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.write("**Model Input (28x28 Grid):**")
            st.image(final_28x28, width=140, clamp=True)

        with preview_col2:
            st.write("**Decision Scores:**")
            best_digit = int(predicted_class)
            st.write(
                f"Top Digit ({best_digit}) Score: `{scores_by_digit[best_digit]:.3f}`"
            )

        with st.expander("See All Class Decision Scores"):
            for digit, score in scores_by_digit.items():
                st.write(f"**Digit {digit}:** Score = `{score:.3f}`")
    else:
        st.warning("Please draw a digit on the canvas before predicting.")
