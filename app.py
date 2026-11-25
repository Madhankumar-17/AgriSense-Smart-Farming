
import base64
import json
import os
import pickle
import joblib
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    g,
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)


# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "AGRISENSE_SECRET_KEY",
    "change-this-to-a-long-random-secret-before-deployment",
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DB_PATH = BASE_DIR / "agrisense.db"

DISEASE_MODEL_PATH = MODEL_DIR / "agrisense_disease_model_fp16.tflite"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
YIELD_MODEL_PATH = MODEL_DIR / "yield_model_20trees.joblib"

IMAGE_SIZE = (224, 224)

GUEST_PREDICTION_LIMIT = 5

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

CLIP_MIN_MARGIN = float(
    os.environ.get("CLIP_MIN_MARGIN", "2.0")
)

# One hectare = 2.47105 acres.
HECTARE_TO_ACRE = 2.47105


# =========================================================
# YIELD MODEL FEATURES
# =========================================================

YIELD_CATEGORICAL = [
    "crop",
    "state",
    "season",
]

YIELD_NUMERIC = [
    "rainfall_mm",
    "temperature_c",
    "soil_ph",
    "nitrogen_kg_ha",
    "phosphorus_kg_ha",
    "potassium_kg_ha",
]

YIELD_FEATURES = YIELD_CATEGORICAL + YIELD_NUMERIC


# =========================================================
# MODEL VARIABLES
# =========================================================

disease_interpreter = None
disease_input_details = None
disease_output_details = None
class_names = []

yield_model = None

clip_model = None
clip_processor = None

clip_error = None
disease_model_error = None
yield_model_error = None


# =========================================================
# CLIP IMAGE PROMPTS
# =========================================================

LEAF_PROMPTS = [
    "a close-up photograph of a single green plant leaf",
    "a close-up photograph of a crop leaf",
    "a photograph of a diseased plant leaf with visible spots",
    "a photograph of a healthy agricultural leaf",
    "a photograph of a tomato plant leaf",
    "a photograph of a potato plant leaf",
    "a photograph of a leaf attached to a plant",
    "a photograph of a plant leaf showing yellowing",
    "a photograph of a plant leaf showing brown patches",
    "a photograph of a green leaf with visible veins",
    "a close-up photograph of a plant leaf surface",
]

NON_LEAF_PROMPTS = [
    "a photograph of a human face",
    "a photograph of a person",
    "a photograph of a dog or cat",
    "a photograph of a bird or animal",
    "a photograph of a car or vehicle",
    "a photograph of a building or house",
    "a photograph of a road or street",
    "a photograph of a computer or mobile phone",
    "a photograph of food on a plate",
    "a photograph of fruits or vegetables",
    "a photograph of a document or book",
    "a photograph of a landscape or mountain",
    "a photograph of the sky or clouds",
    "a photograph of furniture or household objects",
    "a photograph of an object without any plant leaf",
    "a photograph of a flower without a visible leaf",
]


# =========================================================
# DATABASE
# =========================================================

def get_db():
    """Get SQLite connection for current request."""

    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")

    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    """Close database connection."""

    db = g.pop("db", None)

    if db is not None:
        db.close()


def init_db():
    """Create database tables."""

    with sqlite3.connect(DB_PATH) as db:
        db.execute("PRAGMA foreign_keys = ON")

        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction_type TEXT NOT NULL
                    CHECK(prediction_type IN ('disease', 'yield')),
                summary TEXT NOT NULL,
                confidence REAL,
                details TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id)
                    REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_history_user_date
            ON prediction_history(user_id, created_at DESC);
            """
        )


# =========================================================
# USER SESSION
# =========================================================

@app.before_request
def load_current_user():
    """Load logged-in user."""

    user_id = session.get("user_id")

    g.current_user = None

    if user_id:
        g.current_user = get_db().execute(
            """
            SELECT id, name, email, created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()

        if g.current_user is None:
            session.clear()


@app.context_processor
def inject_global_template_values():
    """Provide common template variables."""

    guest_predictions = int(
        session.get("guest_predictions", 0)
    )

    is_guest = not session.get("user_id")

    return {
        "current_user": getattr(g, "current_user", None),
        "guest_predictions": guest_predictions,
        "guest_limit": GUEST_PREDICTION_LIMIT,
        "login_required_now": (
            is_guest
            and guest_predictions >= GUEST_PREDICTION_LIMIT
        ),
    }


def guest_limit_reached():
    """Check guest prediction limit."""

    return (
        not session.get("user_id")
        and int(session.get("guest_predictions", 0))
        >= GUEST_PREDICTION_LIMIT
    )


def record_prediction(
    prediction_type,
    summary,
    details,
    confidence=None,
):
    """Save prediction history or increment guest count."""

    if session.get("user_id"):
        get_db().execute(
            """
            INSERT INTO prediction_history
            (
                user_id,
                prediction_type,
                summary,
                confidence,
                details,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                prediction_type,
                summary,
                confidence,
                json.dumps(details, ensure_ascii=False),
                datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
            ),
        )

        get_db().commit()

    elif prediction_type == "disease":
        # Count one accepted uploaded leaf image as one guest prediction.
        # Automatic yield output from the same image does not consume another slot.
        session["guest_predictions"] = (
            int(session.get("guest_predictions", 0)) + 1
        )

        session.modified = True


# =========================================================
# TENSORFLOW LITE DISEASE MODEL LOADING
# =========================================================

def load_disease_model():
    """Load the tested FP16 TensorFlow Lite disease model."""

    global disease_interpreter
    global disease_input_details
    global disease_output_details
    global class_names
    global disease_model_error

    try:
        if not DISEASE_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"TFLite disease model not found: {DISEASE_MODEL_PATH}"
            )

        if not CLASS_NAMES_PATH.is_file():
            raise FileNotFoundError(
                f"Class names file not found: {CLASS_NAMES_PATH}"
            )

        # Load the class names using the same order as model training.
        with CLASS_NAMES_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:
            labels = json.load(file)

        if isinstance(labels, list):
            names = labels

        elif isinstance(labels, dict):
            try:
                names = [
                    labels[key]
                    for key in sorted(
                        labels,
                        key=lambda key: int(key)
                    )
                ]
            except (ValueError, TypeError):
                names = list(labels.values())

        else:
            raise ValueError(
                "class_names.json must contain a list or dictionary."
            )

        if not names:
            raise ValueError("No disease class names found.")

        interpreter = tf.lite.Interpreter(
            model_path=str(DISEASE_MODEL_PATH)
        )
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if len(input_details) != 1 or len(output_details) != 1:
            raise ValueError(
                "Disease TFLite model must have one input and one output."
            )

        input_shape = tuple(input_details[0]["shape"])
        output_shape = tuple(output_details[0]["shape"])

        if len(input_shape) != 4 or input_shape[1:] != (224, 224, 3):
            raise ValueError(
                f"Unexpected TFLite input shape: {input_shape}"
            )

        if len(output_shape) != 2:
            raise ValueError(
                f"Unexpected TFLite output shape: {output_shape}"
            )

        if output_shape[-1] != len(names):
            raise ValueError(
                "TFLite model output does not match class names."
            )

        disease_interpreter = interpreter
        disease_input_details = input_details
        disease_output_details = output_details
        class_names = [str(name) for name in names]
        disease_model_error = None

        print("\nTFLITE DISEASE MODEL LOADED SUCCESSFULLY")
        print("Model:", DISEASE_MODEL_PATH.name)
        print("Input:", input_shape)
        print("Input dtype:", input_details[0]["dtype"])
        print("Output:", output_shape)
        print("Disease classes:", len(class_names))

    except Exception as exc:
        disease_interpreter = None
        disease_input_details = None
        disease_output_details = None
        class_names = []
        disease_model_error = str(exc)

        app.logger.exception("TFLite disease model loading failed")


# =========================================================
# CLIP MODEL LOADING
# =========================================================

def load_clip_validator():
    """Load pretrained CLIP image validator."""

    global clip_model
    global clip_processor
    global clip_error

    try:
        print("\n========== CLIP VALIDATOR ==========")
        print("Loading:", CLIP_MODEL_NAME)

        clip_processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_NAME
        )

        clip_model = CLIPModel.from_pretrained(
            CLIP_MODEL_NAME
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        clip_model.to(device)
        clip_model.eval()

        clip_error = None

        print("CLIP LOADED SUCCESSFULLY")
        print("Device:", device)
        print("===================================\n")

    except Exception as exc:
        clip_model = None
        clip_processor = None
        clip_error = str(exc)

        app.logger.exception("CLIP validator loading failed")
        print("CLIP LOADING FAILED:", clip_error)


# =========================================================
# CLIP LEAF IMAGE VALIDATION
# =========================================================

def validate_leaf_image(image):
    """Check whether uploaded image appears to contain a plant leaf."""

    if clip_model is None or clip_processor is None:
        raise RuntimeError(
            "CLIP image validator is unavailable. "
            "Check the server terminal."
        )

    try:
        image = image.convert("RGB")

        prompts = LEAF_PROMPTS + NON_LEAF_PROMPTS

        inputs = clip_processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        device = next(
            clip_model.parameters()
        ).device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = clip_model(**inputs)
            similarities = outputs.logits_per_image[0]

        leaf_count = len(LEAF_PROMPTS)

        leaf_scores = similarities[:leaf_count]
        non_leaf_scores = similarities[leaf_count:]

        leaf_top = torch.topk(
            leaf_scores,
            k=min(3, len(leaf_scores))
        ).values.mean()

        non_leaf_top = torch.topk(
            non_leaf_scores,
            k=min(3, len(non_leaf_scores))
        ).values.mean()

        leaf_score = float(leaf_top.item())
        non_leaf_score = float(non_leaf_top.item())

        score_margin = leaf_score - non_leaf_score

        print("\n========== IMAGE VALIDATION ==========")
        print("Leaf score:", round(leaf_score, 4))
        print("Non-leaf score:", round(non_leaf_score, 4))
        print("Score margin:", round(score_margin, 4))

        if leaf_score <= non_leaf_score:
            raise ValueError(
                "Invalid image. This does not appear to be "
                "a plant leaf. Please upload a clear crop leaf photo."
            )

        if score_margin < CLIP_MIN_MARGIN:
            raise ValueError(
                "The uploaded image could not be confidently "
                "identified as a plant leaf. Please upload a "
                "clear, close-up photograph of a crop leaf."
            )

        print("RESULT: ACCEPTED")
        print("=====================================\n")

        return True

    except ValueError:
        raise

    except Exception as exc:
        app.logger.exception("CLIP image validation failed")

        raise RuntimeError(
            "Unable to validate this image. "
            "Please try another leaf photo."
        ) from exc


# =========================================================
# YIELD MODEL LOADING
# =========================================================

def load_yield_model():
    """Load existing trained nine-feature crop yield model."""

    global yield_model
    global yield_model_error

    try:
        if not YIELD_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Yield model not found: {YIELD_MODEL_PATH}"
            )

        loaded_model = joblib.load(YIELD_MODEL_PATH)

        if not callable(
            getattr(loaded_model, "predict", None)
        ):
            raise ValueError(
                "The saved yield model does not support prediction."
            )

        expected_features = getattr(
            loaded_model,
            "feature_names_in_",
            None
        )

        if expected_features is not None:
            if list(expected_features) != YIELD_FEATURES:
                raise ValueError(
                    "Yield model feature mismatch. Expected: "
                    + ", ".join(YIELD_FEATURES)
                )

        yield_model = loaded_model
        yield_model_error = None

        print("\nYIELD MODEL LOADED SUCCESSFULLY")
        print("Model:", type(yield_model).__name__)
        print("Features:", YIELD_FEATURES)

    except Exception as exc:
        yield_model = None
        yield_model_error = str(exc)

        app.logger.exception("Yield model loading failed")


# =========================================================
# INITIALIZE MODELS
# =========================================================

def initialize_models():
    """Initialize database and existing ML models."""

    init_db()
    load_disease_model()
    load_yield_model()

# =========================================================
# HOME PAGE RENDERING
# =========================================================

def render_home(
    disease_result=None,
    disease_error=None,
    yield_result=None,
    yield_error=None,
    auth_notice=None,
):
    """Render the Home page with prediction results."""

    return render_template(
        "index.html",
        disease_result=disease_result,
        disease_error=disease_error,
        yield_result=yield_result,
        yield_error=yield_error,
        auth_notice=auth_notice,
        last_disease_result=session.get("last_disease_result"),
        disease_model_ready=(disease_interpreter is not None),
        leaf_validator_ready=(
            clip_model is not None and clip_processor is not None
        ),
        yield_model_ready=(yield_model is not None),
        disease_model_error=disease_model_error,
        yield_model_error=yield_model_error,
    )


# =========================================================
# DISEASE PREDICTION USING TENSORFLOW LITE
# =========================================================

def predict_disease(image_file):
    """Validate leaf image and predict disease using TFLite."""

    if disease_interpreter is None:
        raise RuntimeError(
            disease_model_error or "TFLite disease model is unavailable."
        )

    validate_leaf_image(display_image)

    try:
        image_file.stream.seek(0)
        image_bytes = image_file.stream.read()

        if not image_bytes:
            raise ValueError("The uploaded image is empty.")

        with Image.open(BytesIO(image_bytes)) as original_image:
            original_image.verify()

        with Image.open(BytesIO(image_bytes)) as original_image:
            display_image = original_image.convert("RGB")
            display_image.thumbnail((500, 500))

        # Keep the existing CLIP leaf-image validation.
        validate_leaf_image(display_image)

        output_buffer = BytesIO()

        display_image.save(
            output_buffer,
            format="JPEG",
            quality=85,
        )

        image_data = (
            "data:image/jpeg;base64,"
            + base64.b64encode(
                output_buffer.getvalue()
            ).decode("utf-8")
        )

        model_image = display_image.resize(IMAGE_SIZE)

        image_array = np.asarray(
            model_image,
            dtype=np.float32
        )

    except (
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise ValueError(
            "Invalid image. Please upload JPG, PNG or WEBP."
        ) from exc

    # Preserve the original preprocessing used by the Keras model.
    image_array = np.expand_dims(
        image_array / 255.0,
        axis=0
    )

    input_detail = disease_input_details[0]
    output_detail = disease_output_details[0]

    # Match the input dtype required by the TFLite model.
    image_array = image_array.astype(
        input_detail["dtype"]
    )

    disease_interpreter.set_tensor(
        input_detail["index"],
        image_array
    )

    disease_interpreter.invoke()

    scores = np.asarray(
        disease_interpreter.get_tensor(
            output_detail["index"]
        )
    )

    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError(
            "Disease model returned unexpected output."
        )

    scores = scores[0]

    if len(scores) != len(class_names):
        raise ValueError(
            "Prediction count does not match disease classes."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "Disease model returned invalid scores."
        )

    index = int(np.argmax(scores))
    raw_label = class_names[index]

    confidence = round(
        float(scores[index]) * 100,
        2
    )

    label = (
        raw_label
        .replace("___", " - ")
        .replace("_", " ")
    )

    crop_from_label = (
        raw_label
        .split("___", 1)[0]
        .replace("_", " ")
        .strip()
    )

    healthy = "healthy" in raw_label.lower()

    if healthy:
        note = (
            "The model classified this leaf as healthy. "
            "Continue monitoring the plant."
        )
    else:
        note = (
            "This is an AI-based screening result, not a "
            "confirmed diagnosis. Consult an agricultural "
            "expert before choosing treatment."
        )

    return {
        "label": label,
        "raw_label": raw_label,
        "crop": crop_from_label,
        "confidence": confidence,
        "healthy": healthy,
        "image_data": image_data,
        "note": note,
    }


# =========================================================
# AUTOMATIC YIELD PREDICTION
# =========================================================

def predict_yield_automatically(disease_result):
    """Predict yield using the existing model and convert to acres."""

    if yield_model is None:
        raise RuntimeError(
            yield_model_error or "Yield model is unavailable."
        )

    crop = str(
        disease_result.get("crop", "")
    ).strip()

    if not crop:
        raise ValueError(
            "Could not identify crop from the uploaded image."
        )

    expected_features = getattr(
        yield_model,
        "feature_names_in_",
        None
    )

    if expected_features is not None:
        if list(expected_features) != YIELD_FEATURES:
            raise ValueError(
                "The saved yield model does not match "
                "the required nine features."
            )

    features = pd.DataFrame(
        [{
            "crop": crop,
            "state": np.nan,
            "season": np.nan,
            "rainfall_mm": np.nan,
            "temperature_c": np.nan,
            "soil_ph": np.nan,
            "nitrogen_kg_ha": np.nan,
            "phosphorus_kg_ha": np.nan,
            "potassium_kg_ha": np.nan,
        }],
        columns=YIELD_FEATURES,
    )

    try:
        prediction = yield_model.predict(features)

    except Exception as exc:
        app.logger.exception(
            "Yield model prediction failed"
        )

        raise RuntimeError(
            "Unable to calculate yield using the existing model. "
            "Check model compatibility and training features."
        ) from exc

    predicted_yield_hectare = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    if not np.isfinite(predicted_yield_hectare):
        raise ValueError(
            "Yield model returned an invalid prediction."
        )

    predicted_yield_acre = (
        predicted_yield_hectare / HECTARE_TO_ACRE
    )

    return {
        "crop": crop,
        "value": round(predicted_yield_acre, 3),
        "unit": "tonnes/acre",
        "type": "Preliminary AI baseline estimate",
        "accuracy_note": (
            "The existing yield model predicts tonnes per hectare. "
            "The result has been converted into tonnes per acre. "
            "Unavailable agricultural measurements are handled "
            "using the existing model's trained imputation pipeline."
        ),
        "note": (
            "This is a preliminary baseline estimate, not a "
            "farm-specific or disease-adjusted yield forecast. "
            "The existing model does not use disease information "
            "or disease severity."
        ),
        "original_value": round(
            predicted_yield_hectare,
            3
        ),
        "original_unit": "tonnes/hectare",
    }


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/")
def index():
    return render_home()


# =========================================================
# SMART FARMING TIPS
# =========================================================

def get_smart_farming_tip(disease_label):
    """Return general farming guidance based on detected condition."""

    label = str(disease_label or "").lower()

    if not label:
        return (
            "Upload a leaf image to receive crop health analysis "
            "and a relevant farming tip."
        )

    if "healthy" in label:
        return (
            "Your latest analysed leaf was classified as healthy. "
            "Continue regular crop monitoring, maintain appropriate "
            "watering, and inspect leaves regularly."
        )

    if "blight" in label:
        return (
            "Remove heavily affected leaves where appropriate. "
            "Avoid overhead watering and maintain sufficient spacing "
            "between plants to improve airflow."
        )

    if "rust" in label:
        return (
            "Monitor nearby leaves for rust-like spots. Remove "
            "severely affected plant material where appropriate and "
            "avoid wetting foliage during irrigation."
        )

    if "mildew" in label:
        return (
            "Improve airflow around plants, avoid overcrowding, "
            "and water near the soil rather than directly over "
            "the leaves."
        )

    if "spot" in label:
        return (
            "Check surrounding plants for similar leaf spots. "
            "Remove severely affected leaves where appropriate "
            "and avoid splashing water between plants."
        )

    return (
        "Your latest analysis detected a possible leaf disease. "
        "Inspect nearby plants, remove severely affected leaves "
        "where appropriate, and consult a local agricultural "
        "expert for disease-specific treatment."
    )


# =========================================================
# DASHBOARD ROUTE
# =========================================================

@app.route("/dashboard")
def dashboard():
    """Display crop health, yield insights and farming tips."""

    if not g.current_user:
        flash(
            "Please log in to access your dashboard.",
            "error"
        )
        return redirect(url_for("login"))

    db = get_db()
    user_id = g.current_user["id"]

    # -----------------------------------------------------
    # FETCH DISEASE PREDICTIONS
    # -----------------------------------------------------

    disease_rows = db.execute(
        """
        SELECT details, summary, created_at
        FROM prediction_history
        WHERE user_id = ?
        AND prediction_type = 'disease'
        ORDER BY created_at DESC
        """,
        (user_id,),
    ).fetchall()

    # -----------------------------------------------------
    # FETCH YIELD PREDICTIONS
    # -----------------------------------------------------

    yield_rows = db.execute(
        """
        SELECT details, summary, created_at
        FROM prediction_history
        WHERE user_id = ?
        AND prediction_type = 'yield'
        ORDER BY created_at DESC
        """,
        (user_id,),
    ).fetchall()

    # -----------------------------------------------------
    # CROP HEALTH OVERVIEW
    # -----------------------------------------------------

    healthy_count = 0
    diseased_count = 0
    latest_disease = None

    for row in disease_rows:

        try:
            details = json.loads(row["details"] or "{}")

            if not isinstance(details, dict):
                details = {}

        except (TypeError, ValueError):
            details = {}

        label = (
            details.get("label")
            or row["summary"]
            or "Unknown"
        )

        if latest_disease is None:
            latest_disease = label

        is_healthy = details.get("healthy")

        # Support older database records without healthy field.
        if is_healthy is None:
            is_healthy = "healthy" in str(label).lower()

        if is_healthy:
            healthy_count += 1
        else:
            diseased_count += 1

    # -----------------------------------------------------
    # YIELD INSIGHTS
    # -----------------------------------------------------

    yield_values = []

    for row in yield_rows:

        try:
            details = json.loads(row["details"] or "{}")

            if not isinstance(details, dict):
                continue

            value = float(details.get("value"))

            if np.isfinite(value):
                yield_values.append(value)

        except (TypeError, ValueError):
            continue

    # Rows are already ordered newest first.
    latest_yield = (
        yield_values[0]
        if yield_values
        else None
    )

    average_yield = (
        sum(yield_values) / len(yield_values)
        if yield_values
        else None
    )

    # -----------------------------------------------------
    # SMART FARMING TIPS
    # -----------------------------------------------------

    farming_tip = get_smart_farming_tip(
        latest_disease
    )

    # -----------------------------------------------------
    # RENDER DASHBOARD
    # -----------------------------------------------------

    return render_template(
        "dashboard.html",

        # Crop health
        total_analyses=len(disease_rows),
        healthy_count=healthy_count,
        diseased_count=diseased_count,

        # Yield insights
        yield_analyses=len(yield_rows),
        latest_yield=latest_yield,
        average_yield=average_yield,

        # Farming guidance
        latest_disease=latest_disease,
        farming_tip=farming_tip,
    )


# =========================================================
# DISEASE + AUTOMATIC YIELD PREDICTION ROUTE
# =========================================================

@app.route(
    "/predict-disease",
    methods=["POST"]
)
def predict_disease_route():

    if guest_limit_reached():
        return render_home(
            disease_error=(
                "You have used your five free predictions. "
                "Please log in or register to continue."
            )
        )

    if disease_interpreter is None:
        return render_home(
            disease_error=(
                "Disease model unavailable: "
                + str(disease_model_error)
            )
        )

    if clip_model is None:
        return render_home(
            disease_error=(
                "CLIP image validator is unavailable. "
                "Check the server terminal."
            )
        )

    image_file = request.files.get("leaf_image")

    if image_file is None or not image_file.filename:
        return render_home(
            disease_error="Please select a plant leaf image."
        )

    try:
        disease_result = predict_disease(image_file)

        session["last_disease_result"] = {
            "label": disease_result["label"],
            "raw_label": disease_result["raw_label"],
            "crop": disease_result["crop"],
            "healthy": disease_result["healthy"],
        }

        session.modified = True

        record_prediction(
            "disease",
            disease_result["label"],
            disease_result,
            disease_result["confidence"],
        )

        yield_result = None
        yield_error = None

        try:
            yield_result = predict_yield_automatically(
                disease_result
            )

            if session.get("user_id"):
                record_prediction(
                    "yield",
                    (
                        f"{yield_result['crop']}: "
                        f"{yield_result['value']} "
                        f"{yield_result['unit']}"
                    ),
                    yield_result,
                )

        except Exception as exc:
            yield_error = str(exc)

            app.logger.exception(
                "Automatic yield prediction failed"
            )

        notice = None

        if guest_limit_reached():
            notice = (
                "You have used your fifth free prediction. "
                "Register or log in to continue."
            )

        return render_home(
            disease_result=disease_result,
            yield_result=yield_result,
            yield_error=yield_error,
            auth_notice=notice,
        )

    except (
        ValueError,
        RuntimeError,
    ) as exc:
        return render_home(
            disease_error=str(exc)
        )

    except Exception:
        app.logger.exception(
            "Unexpected disease prediction error"
        )

        return render_home(
            disease_error=(
                "An unexpected error occurred during "
                "disease prediction. Please try again."
            )
        )


# =========================================================
# OLD MANUAL YIELD ROUTE
# =========================================================

@app.route(
    "/predict-yield",
    methods=["POST"]
)
def predict_yield_route():

    return render_home(
        yield_error=(
            "Yield prediction now runs automatically after "
            "successful disease detection. Please upload "
            "a plant leaf image."
        )
    )


# =========================================================
# USER REGISTRATION
# =========================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if g.current_user:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email:
            flash("Enter your name and email.", "error")
            return render_template("register.html")

        if len(password) < 8:
            flash(
                "Password must contain at least 8 characters.",
                "error"
            )
            return render_template("register.html")

        if confirm_password and password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        try:
            guest_predictions_before_auth = int(
                session.get("guest_predictions", 0)
            )
            db = get_db()

            cursor = db.execute(
                """
                INSERT INTO users
                (name, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    name,
                    email,
                    generate_password_hash(password),
                    datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                ),
            )

            db.commit()

            session.clear()
            session["guest_predictions"] = guest_predictions_before_auth
            session["user_id"] = cursor.lastrowid

            flash(
                "Your account has been created.",
                "success"
            )

            return redirect(url_for("dashboard"))

        except sqlite3.IntegrityError:
            flash(
                "An account with that email already exists.",
                "error"
            )

    return render_template("register.html")


# =========================================================
# USER LOGIN
# =========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if g.current_user:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = get_db().execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

        if (
            user
            and check_password_hash(
                user["password_hash"],
                password
            )
        ):
            guest_predictions_before_auth = int(
                session.get("guest_predictions", 0)
            )
            session.clear()
            session["guest_predictions"] = guest_predictions_before_auth
            session["user_id"] = user["id"]

            flash("Welcome back!", "success")

            return redirect(url_for("dashboard"))

        flash(
            "Email or password is incorrect.",
            "error"
        )

    return render_template("login.html")


# =========================================================
# USER LOGOUT
# =========================================================

@app.route(
    "/logout",
    methods=["POST"]
)
def logout():

    # Keep the guest allowance when a member logs out in the same browser.
    guest_predictions = int(session.get("guest_predictions", 0))
    session.clear()
    session["guest_predictions"] = guest_predictions

    flash("You have been logged out.", "success")

    return redirect(url_for("index"))


# =========================================================
# PREDICTION HISTORY
# =========================================================

@app.route("/history")
def history():

    if not g.current_user:
        flash(
            "Please log in to view prediction history.",
            "error"
        )
        return redirect(url_for("login"))

    rows = get_db().execute(
        """
        SELECT *
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (g.current_user["id"],),
    ).fetchall()

    entries = []

    for row in rows:
        item = dict(row)

        try:
            item["details_obj"] = json.loads(item["details"])

        except (
            ValueError,
            TypeError,
        ):
            item["details_obj"] = {}

        entries.append(item)

    return render_template(
        "history.html",
        entries=entries
    )


# =========================================================
# USER PROFILE
# =========================================================

@app.route("/profile")
def profile():

    if not g.current_user:
        return redirect(url_for("login"))

    count = get_db().execute(
        """
        SELECT COUNT(*)
        FROM prediction_history
        WHERE user_id = ?
        """,
        (g.current_user["id"],),
    ).fetchone()[0]

    return render_template(
        "profile.html",
        prediction_count=count
    )


# =========================================================
# FILE UPLOAD ERROR
# =========================================================

@app.errorhandler(413)
def file_too_large(_error):

    return (
        render_home(
            disease_error=(
                "Uploaded image is too large. "
                "Maximum file size is 10 MB."
            )
        ),
        413,
    )


# =========================================================
# APPLICATION STARTUP
# =========================================================

initialize_models()


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
