# AgriSense AI — CSP Web Application

A Flask web app for:
1. Plant leaf disease classification using the supplied 38-class Keras model.
2. Crop yield estimation using a scikit-learn regression pipeline trained on your own agricultural dataset.

## Project structure

```text
AgriSense_CSP/
├── app.py
├── train_yield_model.py
├── requirements.txt
├── models/
│   ├── agrisense_disease_model.keras
│   ├── class_names.json
│   └── yield_model.pkl        # created after training
├── data/
│   └── yield_dataset_template.csv
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## 1. Requirements

Python 3.10 or 3.11 is recommended. TensorFlow can be platform/version-sensitive; if installation fails, use a fresh virtual environment with a TensorFlow-supported Python version.

### Windows (PowerShell)

```powershell
cd AgriSense_CSP
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### macOS / Linux

```bash
cd AgriSense_CSP
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## 2. Disease detection

The supplied `agrisense_disease_model.keras` and `class_names.json` are included. The app resizes the uploaded image to the model's declared dimensions and scales pixel values to `[0, 1]`, as used by many image classifiers.

If loading fails:
- Check the terminal error printed during startup.
- Confirm TensorFlow and Pillow installed successfully.
- Confirm the model and `class_names.json` files remain inside `models/`.
- Confirm the model's training preprocessing matches the `/255` normalization used here. If your training pipeline used a different preprocessing method, update `app.py` accordingly.

The class order in `class_names.json` must match the model output order. Do not reorder it manually.

## 3. Crop-yield prediction

The disease model does not predict yield. A separate regression model is required, and no yield model is bundled because a real training dataset and validated yield model were not provided.

### Expected CSV columns

The CSV must contain these exact column names:

- `crop`
- `state`
- `season`
- `area_hectares`
- `rainfall_mm`
- `temperature_c`
- `soil_ph`
- `nitrogen_kg_ha`
- `phosphorus_kg_ha`
- `potassium_kg_ha`
- `yield_tonnes_per_hectare` (target)

See `data/yield_dataset_template.csv` for the header. Replace it with actual, reliable observations; the template contains no invented training rows.

Train the model:

```bash
python train_yield_model.py path/to/your_agricultural_dataset.csv
```

The script prints MAE, RMSE, and R² on a held-out test set and saves `models/yield_model.pkl`. Restart `app.py` after training.

**Important:** The dataset's yield target must be in tonnes per hectare. If your dataset contains total production or another unit, convert it correctly before training. Avoid using made-up sample values for project evaluation. A larger and geographically representative dataset is preferable; metrics from a small dataset may not generalize.

## 4. Common issues

- **`ModuleNotFoundError`:** activate the virtual environment and run `pip install -r requirements.txt`.
- **Model loading error:** check TensorFlow version compatibility and the terminal traceback.
- **Yield model not trained:** the yield form will show a message until `models/yield_model.pkl` exists.
- **Wrong predictions:** verify the class order, image preprocessing, training labels, input units, dataset quality, and held-out evaluation results.
- **Port in use:** change `port=5000` at the bottom of `app.py`.

## 5. Project limitations

This is a CSP prototype and decision-support tool. Confidence is not the same as correctness. Leaf disease predictions are not a confirmed diagnosis or treatment recommendation. Yield estimates are only as reliable as the dataset and model used to train them. Validate both modules with suitable real-world data before presenting them as dependable agricultural advice.
