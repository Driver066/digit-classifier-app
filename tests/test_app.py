"""Regression tests for model selection, score labels, and model replacement.

Run with: python -m pytest -q
The drawing component is replaced with pixel data; the app and SVCs run normally.
"""

import os
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.svm import SVC
import streamlit as st
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
STANDARD = "svc_mnist_digit_classifier.joblib"
HIGH_PRECISION = "svc_mnist_digit_classifier_58k.joblib"


def sample_image(digit):
    image = np.zeros((28, 28), dtype=np.uint8)
    image[5:23, 2 + 2 * digit : 4 + 2 * digit] = 255
    return image


def train_model(digits):
    features = np.array([sample_image(d).ravel() / 255.0 for d in digits])
    return SVC(kernel="linear").fit(features, digits)


@pytest.fixture
def app_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    st.cache_resource.clear()
    yield tmp_path
    st.cache_resource.clear()


def save_model(workspace, model, name=STANDARD):
    path = workspace / "models" / name
    joblib.dump(model, path)
    return path


def open_app(monkeypatch, image):
    if image is None:
        rgba = None
    else:
        rgba = np.dstack([image, image, image, np.full_like(image, 255)])
    # Components v2 registers browser assets during import, so isolate the
    # drawing frontend before importing the app in the headless test runner.
    canvas_module = ModuleType("streamlit_drawable_canvas")
    canvas_module.st_canvas = lambda **kwargs: SimpleNamespace(image_data=rgba)
    monkeypatch.setitem(sys.modules, "streamlit_drawable_canvas", canvas_module)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    return app


def predict(app, high_precision=False, centering=False):
    app.sidebar.checkbox[0].set_value(centering)
    app.sidebar.checkbox[1].set_value(high_precision)
    app.button[0].click().run()
    assert not app.exception
    return "\n".join(item.value for item in app.markdown)


def assert_class_scores(app, text, model, image):
    features = image.reshape(1, -1) / 255.0
    prediction = model.predict(features)[0]
    raw_scores = model.decision_function(features)[0]
    if len(model.classes_) == 2:
        expected_scores = [-raw_scores, raw_scores]
    else:
        expected_scores = raw_scores
    assert app.success[0].value == f"### Predicted Digit: **{prediction}**"
    displayed = [line for line in text.splitlines() if line.startswith("**Digit ")]
    expected = [
        f"**Digit {label}:** Score = `{expected_scores[column]:.3f}`"
        for column, label in enumerate(model.classes_)
    ]
    assert displayed == expected
    prediction_column = model.classes_.tolist().index(prediction)
    assert (
        f"Top Digit ({prediction}) Score: `{expected_scores[prediction_column]:.3f}`"
        in text
    )


@pytest.mark.parametrize("digit", [3, 5, 6, 8])
def test_nonconsecutive_digit_labels(app_workspace, monkeypatch, digit):
    model = train_model([3, 5, 6, 8])
    save_model(app_workspace, model)
    image = sample_image(digit)
    app = open_app(monkeypatch, image)
    text = predict(app)
    assert_class_scores(app, text, model, image)
    assert app.success[0].value == f"### Predicted Digit: **{digit}**"
    assert app.sidebar.caption[0].value == "Recognizes digits: 3, 5, 6, 8"
    assert "Draw a single digit (3, 5, 6, 8) below:" in text


@pytest.mark.parametrize("digit", [3, 8])
def test_binary_model_scores(app_workspace, monkeypatch, digit):
    model = train_model([3, 8])
    save_model(app_workspace, model)
    image = sample_image(digit)
    app = open_app(monkeypatch, image)
    assert_class_scores(app, predict(app), model, image)


@pytest.mark.parametrize(
    "model_name",
    [STANDARD, HIGH_PRECISION, "svc_mnist_digit_classifier_reh_3k.joblib"],
)
def test_committed_model_artifacts(app_workspace, monkeypatch, model_name):
    source = ROOT / "models" / model_name
    shutil.copy2(source, app_workspace / "models" / STANDARD)
    model = joblib.load(source)
    image = sample_image(8)
    app = open_app(monkeypatch, image)
    assert_class_scores(app, predict(app), model, image)


def test_switch_between_standard_and_high_precision(app_workspace, monkeypatch):
    save_model(app_workspace, train_model([3, 5, 6, 8]))
    shutil.copy2(ROOT / "models" / HIGH_PRECISION, app_workspace / "models")
    app = open_app(monkeypatch, sample_image(8))
    for use_high_precision, count in [(False, 4), (True, 10), (False, 4)]:
        text = predict(app, high_precision=use_high_precision, centering=True)
        assert text.count("**Digit ") == count


def test_replaced_model_invalidates_cache(app_workspace, monkeypatch):
    model_path = save_model(app_workspace, train_model([3, 5, 6, 8]))
    app = open_app(monkeypatch, sample_image(8))
    assert predict(app).count("**Digit ") == 4
    old_stat = model_path.stat()
    # Keep the filename and timestamp: the new model's content must matter.
    save_model(app_workspace, train_model(list(range(10))))
    os.utime(model_path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    app.run()
    assert not app.exception
    assert app.sidebar.caption[0].value == (
        "Recognizes digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
    )
    assert predict(app).count("**Digit ") == 10


def test_missing_high_precision_falls_back(app_workspace, monkeypatch):
    save_model(app_workspace, train_model([3, 5, 6, 8]))
    app = open_app(monkeypatch, sample_image(8))
    assert predict(app, high_precision=True).count("**Digit ") == 4
    assert "Falling back to standard model" in app.sidebar.info[0].value


def test_missing_models_stop_cleanly(app_workspace, monkeypatch):
    app = open_app(monkeypatch, sample_image(8))
    assert app.error[0].value.startswith("No valid model file found!")
    assert len(app.button) == 0


def test_canvas_not_ready_shows_warning(app_workspace, monkeypatch):
    save_model(app_workspace, train_model([3, 5, 6, 8]))
    app = open_app(monkeypatch, None)
    predict(app)
    assert app.warning[0].value == "Please draw a digit on the canvas before predicting."
