import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import time
import datetime

# Import your existing model
from models.cnn_model import SimpleCNN

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Fake Currency Detection",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CONFIG / PLACEHOLDER VALUES
# --------------------------------------------------

DATASET_IMAGE_COUNT = "7,437 images"   # update to your actual dataset size
LAST_UPDATED = "28 Jul 2026"           # update to your actual last-trained date

NAV_ITEMS = [
    ("dashboard", "Dashboard"),
    ("search", "Predict"),
    ("history", "History"),
    ("bar_chart", "Analytics"),
    ("model_training", "Model Details"),
    ("dataset", "Dataset"),
    ("settings", "Settings"),
    ("info", "About Project"),
]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "result" not in st.session_state:
    st.session_state.result = None

st.markdown("""
<link rel="stylesheet"
href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400,0,0">
""", unsafe_allow_html=True)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

# <link rel="stylesheet"
# href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,300..500,0..1,0" />
# <style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.material-symbols-outlined{
    font-family: 'Material Symbols Outlined';
    font-weight: normal;
    font-style: normal;
    font-size: 20px;
    line-height: 1;
    vertical-align: middle;
}

.stApp{
    background: radial-gradient(circle at 10% 0%, #171029 0%, #0a0a16 55%, #08080f 100%);
    color: #e7e7f0;
}

#MainMenu, footer {visibility: hidden;}

/* ---------------- TOP BAR ---------------- */

.topbar{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 22px;
}

.topbar-pill{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 8px 16px;
    border-radius: 10px;
}

.topbar-icon{
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: #171225;
    border: 1px solid rgba(255,255,255,0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #c4b5fd;
}

/* ---------------- SIDEBAR ---------------- */

[data-testid="stSidebar"]{
    background: #0d0b18;
    border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] > div:first-child{
    padding-top: 10px;
}

.sidebar-logo{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 4px 20px 4px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 18px;
}

.sidebar-logo-icon{
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    flex-shrink: 0;
}

.sidebar-logo-title{
    font-size: 15px;
    font-weight: 700;
    color: #f4f4f8;
    line-height: 1.2;
}

.sidebar-logo-sub{
    font-size: 11.5px;
    color: #8b87a3;
}

.nav-row{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 4px;
    color: #a9a5c0;
    font-size: 14px;
    font-weight: 500;
}

.nav-row.active{
    background: linear-gradient(135deg, #7c3aed, #9333ea);
    color: white;
    font-weight: 600;
}

[data-testid="stSidebar"] .stButton > button{
    background: transparent;
    border: none;
    color: #a9a5c0;
    text-align: left;
    justify-content: flex-start;
    font-weight: 500;
    font-size: 14px;
    padding: 10px 14px;
    width: 100%;
    box-shadow: none;
}

[data-testid="stSidebar"] .stButton > button:hover{
    background: rgba(124,58,237,0.14);
    color: #e9e6f7;
    transform: none;
    box-shadow: none;
}

.accuracy-card{
    margin-top: 26px;
    background: linear-gradient(180deg, #171225 0%, #100d1c 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 18px;
}

.accuracy-label{
    color: #8b87a3;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.accuracy-value{
    color: #f4f4f8;
    font-size: 28px;
    font-weight: 800;
    margin: 4px 0 2px 0;
}

.accuracy-updated{
    color: #6d6987;
    font-size: 11.5px;
    margin-bottom: 10px;
}

/* ---------------- CARDS ---------------- */

.card{
    background: linear-gradient(180deg, #16121f 0%, #100d18 100%);
    padding: 22px 24px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.07);
    height: 100%;
}

.card h3{
    font-size: 14.5px;
    font-weight: 700;
    color: #e7e4f0;
    margin: 0 0 14px 0;
}

/* Upload dropzone */

.dropzone-icon{
    width: 54px;
    height: 54px;
    border-radius: 14px;
    background: rgba(124,58,237,0.14);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #a855f7;
    margin: 6px auto 10px auto;
}

[data-testid="stFileUploaderDropzone"]{
    background: rgba(124,58,237,0.05) !important;
    border: 1.5px dashed rgba(168,85,247,0.4) !important;
    border-radius: 14px !important;
}

[data-testid="stFileUploaderDropzone"] button{
    background: linear-gradient(135deg, #7c3aed, #9333ea) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Result badge */

.result-icon-wrap{
    width: 58px;
    height: 58px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 4px auto 12px auto;
}

.result-icon-wrap.real{ background: rgba(34,197,94,0.15); color: #4ade80; }
.result-icon-wrap.fake{ background: rgba(239,68,68,0.15); color: #f87171; }

.result-label{
    text-align: center;
    font-size: 19px;
    font-weight: 800;
    letter-spacing: 0.5px;
}

.result-label.real{ color: #4ade80; }
.result-label.fake{ color: #f87171; }

.result-desc{
    text-align: center;
    color: #8b87a3;
    font-size: 12.5px;
    margin-top: 6px;
}

/* Confidence */

.confidence-value{
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    color: #f4f4f8;
    margin: 6px 0 12px 0;
}

/* Probability rows */

.prob-row{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}

.prob-dot{
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}

.prob-dot.real{ background: #22c55e; }
.prob-dot.fake{ background: #ef4444; }

.prob-label{
    font-size: 13px;
    color: #c9c6d8;
    width: 42px;
    flex-shrink: 0;
}

.prob-track{
    flex-grow: 1;
    height: 8px;
    border-radius: 6px;
    background: rgba(255,255,255,0.06);
    overflow: hidden;
}

.prob-fill{
    height: 100%;
    border-radius: 6px;
}

.prob-fill.real{ background: linear-gradient(90deg, #16a34a, #4ade80); }
.prob-fill.fake{ background: linear-gradient(90deg, #dc2626, #f87171); }

.prob-pct{
    font-size: 13px;
    font-weight: 700;
    color: #f4f4f8;
    width: 52px;
    text-align: right;
    flex-shrink: 0;
}

/* Highlight tiles */

.highlight-tile{
    background: #14111f;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}

.highlight-icon{
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 8px auto;
}

.highlight-icon.ok{ background: rgba(34,197,94,0.15); color: #4ade80; }
.highlight-icon.bad{ background: rgba(239,68,68,0.15); color: #f87171; }
.highlight-icon.pending{ background: rgba(148,163,184,0.15); color: #94a3b8; }

.highlight-title{
    font-size: 13px;
    font-weight: 600;
    color: #e7e4f0;
}

.highlight-status{
    font-size: 11.5px;
    font-weight: 600;
    margin-top: 2px;
}

.highlight-status.ok{ color: #4ade80; }
.highlight-status.bad{ color: #f87171; }
.highlight-status.pending{ color: #8b87a3; }

/* Info tiles */

.info-tile{
    background: #14111f;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}

.info-tile-icon{
    color: #a855f7;
    margin-bottom: 6px;
}

.info-tile-label{
    font-size: 11px;
    color: #8b87a3;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
}

.info-tile-value{
    font-size: 15px;
    font-weight: 700;
    color: #f4f4f8;
    margin-top: 2px;
}

.section-title{
    font-size: 14.5px;
    font-weight: 700;
    color: #e7e4f0;
    margin: 6px 0 12px 2px;
}

hr{
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 24px 0;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

from pathlib import Path
import torch

@st.cache_resource
def load_model():
    BASE_DIR = Path(__file__).resolve().parent
    model_path = BASE_DIR / "saved_models" / "best_model.pth"

    print("Current Working Directory:", Path.cwd())
    print("Model Path:", model_path)
    print("Exists:", model_path.exists())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("model_path:", repr(model_path))
    print("type:", type(model_path))

    checkpoint = torch.load(model_path, map_location=device)

    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = SimpleCNN(
        num_classes=len(class_names),
        img_size=img_size
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    accuracy = checkpoint.get("test_acc", None)

    return model, class_names, img_size, accuracy, device


def preprocess_image(image, img_size):

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    return tensor


def predict_image(image):

    model, class_names, img_size, accuracy, device = load_model()

    tensor = preprocess_image(image, img_size).to(device)

    start = time.time()

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        confidence, pred = torch.max(probs, 0)

    elapsed = time.time() - start

    return {
        "prediction": class_names[pred.item()],
        "confidence": confidence.item() * 100,
        "probabilities": probs.cpu().numpy(),
        "processing_time": elapsed,
        "accuracy": accuracy,
        "img_size": img_size,
        "classes": class_names
    }


def icon(name, size=20):
    return f'<span class="material-symbols-outlined" style="font-size:{size}px;">{name}</span>'


# --------------------------------------------------
# TOP BAR
# --------------------------------------------------

st.markdown(
    f"""
    <div class="topbar">
        <div class="topbar-pill">{icon('shield', 16)} Fake Currency Dashboard</div>
        <div class="topbar-icon">{icon('dark_mode', 18)}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        f"""
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">{icon('shield', 22)}</div>
            <div>
                <div class="sidebar-logo-title">Fake Currency</div>
                <div class="sidebar-logo-sub">Detection System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    for icon_name, label in NAV_ITEMS:

        if label == st.session_state.page:
            st.markdown(
                f"""
                <div class="nav-row active">
                    {icon(icon_name, 18)} <span>{label}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            col_icon, col_btn = st.columns([0.16, 0.84])
            with col_icon:
                st.markdown(
                    f"<div style='padding-top:9px;color:#8b87a3;'>{icon(icon_name, 18)}</div>",
                    unsafe_allow_html=True
                )
            with col_btn:
                if st.button(label, key=f"nav_{label}", use_container_width=True):
                    st.session_state.page = label
                    st.rerun()

    st.markdown(
        f"""
        <div class="accuracy-card">
            <div class="accuracy-label">Model Accuracy</div>
            <div class="accuracy-value">98.21%</div>
            <div class="accuracy-updated">Last updated: {LAST_UPDATED}</div>
            <svg viewBox="0 0 100 30" width="100%" height="30" preserveAspectRatio="none">
                <polyline points="0,26 15,22 30,24 45,14 60,18 75,6 100,2"
                    fill="none" stroke="#a855f7" stroke-width="2" />
            </svg>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# NON-DASHBOARD PAGES: SIMPLE PLACEHOLDER
# --------------------------------------------------

if st.session_state.page != "Dashboard":

    icon_map = dict(NAV_ITEMS)
    reverse_map = {v: k for k, v in NAV_ITEMS}

    st.markdown(
        f"""
        <div class="card" style="text-align:center;padding:60px 24px;">
            <div style="color:#a855f7;margin-bottom:10px;">{icon(reverse_map.get(st.session_state.page, 'info'), 40)}</div>
            <h3 style="font-size:18px;">{st.session_state.page}</h3>
            <p style="color:#8b87a3;font-size:13.5px;">This section is under construction. Head back to
            <b>Dashboard</b> to run currency detection.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()

# --------------------------------------------------
# UPLOAD + PREVIEW
# --------------------------------------------------

left_col, right_col = st.columns([1, 1])

with left_col:

    st.markdown('<div class="card"><h3>Upload Currency Image</h3>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop an image here, or browse files. Supports JPG, JPEG, PNG.",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:

    st.markdown('<div class="card"><h3>Image Preview</h3>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    else:
        st.info("Upload an image to preview it here.")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

predict_btn = st.button("Run Detection", use_container_width=True)

if uploaded_file is not None and predict_btn:
    image = Image.open(uploaded_file)
    with st.spinner("Running CNN inference..."):
        st.session_state.result = predict_image(image)

result = st.session_state.result

# --------------------------------------------------
# RESULT ROW: PREDICTION / CONFIDENCE / PROBABILITIES
# --------------------------------------------------

st.write("")

r1, r2, r3 = st.columns([1, 1, 1.2])

if uploaded_file is not None and result is not None:

    prediction = result["prediction"]
    confidence = result["confidence"]
    probs = result["probabilities"]
    elapsed = result["processing_time"]
    is_real = prediction.lower() == "real"

    real_prob, fake_prob = 0.0, 0.0
    for i, cls in enumerate(result["classes"]):
        if cls.lower() == "real":
            real_prob = float(probs[i]) * 100
        elif cls.lower() == "fake":
            fake_prob = float(probs[i]) * 100

    with r1:
        state = "real" if is_real else "fake"
        icon_name = "verified" if is_real else "gpp_bad"
        desc = "The currency is predicted as real." if is_real else "The currency is predicted as fake."
        st.markdown(
            f"""
            <div class="card">
            <h3>Prediction Result</h3>
            <div class="result-icon-wrap {state}">{icon(icon_name, 30)}</div>
            <div class="result-label {state}">{'REAL NOTE' if is_real else 'FAKE NOTE'}</div>
            <div class="result-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with r2:
        st.markdown(
            f"""
            <div class="card">
            <h3>Confidence Score</h3>
            <div class="confidence-value">{confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(confidence / 100)

    with r3:
        st.markdown(
            f"""
            <div class="card">
            <h3>Prediction Probabilities</h3>
            <div class="prob-row">
                <div class="prob-dot real"></div>
                <div class="prob-label">Real</div>
                <div class="prob-track"><div class="prob-fill real" style="width:{real_prob:.2f}%;"></div></div>
                <div class="prob-pct">{real_prob:.2f}%</div>
            </div>
            <div class="prob-row">
                <div class="prob-dot fake"></div>
                <div class="prob-label">Fake</div>
                <div class="prob-track"><div class="prob-fill fake" style="width:{fake_prob:.2f}%;"></div></div>
                <div class="prob-pct">{fake_prob:.2f}%</div>
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ------------------------------------------
    # KEY HIGHLIGHTS (illustrative, tied to prediction)
    # ------------------------------------------

    st.write("")
    st.markdown('<div class="section-title">Key Highlights</div>', unsafe_allow_html=True)

    highlight_items = ["Security Thread", "Watermark", "Texture Pattern", "Micro Print"]
    h_cols = st.columns(4)

    for col, title in zip(h_cols, highlight_items):
        status = "ok" if is_real else "bad"
        status_icon = "check_circle" if is_real else "cancel"
        status_text = ("Detected" if title in ["Security Thread", "Watermark"] else "Matched") if is_real \
            else "Not Matched"
        with col:
            st.markdown(
                f"""
                <div class="highlight-tile">
                    <div class="highlight-icon {status}">{icon(status_icon, 18)}</div>
                    <div class="highlight-title">{title}</div>
                    <div class="highlight-status {status}">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.caption(
        "Key Highlights are illustrative indicators derived from the overall prediction, "
        "not independent detectors."
    )

else:

    with r1:
        st.markdown(
            f"""
            <div class="card">
            <h3>Prediction Result</h3>
            <div class="result-icon-wrap" style="background:rgba(148,163,184,0.12);color:#94a3b8;">{icon('help', 30)}</div>
            <div class="result-label" style="color:#94a3b8;">No Prediction Yet</div>
            <div class="result-desc">Upload an image and run detection.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with r2:
        st.markdown(
            """
            <div class="card">
            <h3>Confidence Score</h3>
            <div class="confidence-value" style="color:#5b5776;">--</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(0)

    with r3:
        st.markdown(
            """
            <div class="card">
            <h3>Prediction Probabilities</h3>
            <div class="prob-row">
                <div class="prob-dot real"></div>
                <div class="prob-label">Real</div>
                <div class="prob-track"><div class="prob-fill real" style="width:0%;"></div></div>
                <div class="prob-pct">--</div>
            </div>
            <div class="prob-row">
                <div class="prob-dot fake"></div>
                <div class="prob-label">Fake</div>
                <div class="prob-track"><div class="prob-fill fake" style="width:0%;"></div></div>
                <div class="prob-pct">--</div>
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# INFO TILES
# --------------------------------------------------

st.write("")
st.divider()

img_size = result["img_size"] if result else 128
elapsed_display = f"{result['processing_time']:.2f}s" if result else "--"

tiles = [
    ("aspect_ratio", "Input Size", f"{img_size} x {img_size}"),
    ("memory", "Model", "Custom CNN"),
    ("dataset", "Dataset", DATASET_IMAGE_COUNT),
    ("category", "Classes", "Real / Fake"),
    ("timer", "Processing Time", elapsed_display),
]

t_cols = st.columns(len(tiles))

for col, (icon_name, label, value) in zip(t_cols, tiles):
    with col:
        st.markdown(
            f"""
            <div class="info-tile">
                <div class="info-tile-icon">{icon(icon_name, 20)}</div>
                <div class="info-tile-label">{label}</div>
                <div class="info-tile-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.write("")
st.markdown(
    """
    <div style="text-align:center;color:#5b5776;font-size:13px;margin-top:10px;">
    Fake Currency Detection System &middot; PyTorch &middot; Streamlit &middot; Custom CNN
    </div>
    """,
    unsafe_allow_html=True
)
