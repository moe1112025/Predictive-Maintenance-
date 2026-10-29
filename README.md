# Predictive Maintenance & Machine Failure Pattern System

**Junior Machine Learning Project**  
**Author:** Moe Htet Ar Kar (Phoe Cho)

An industrial sensor dashboard using Gradient Boosting to classify whether entered sensor values match a synthetic maintenance-risk pattern.

> **Scope:** A learning and portfolio application built with a small, understandable Python script. It is not a production or safety-critical system.

## Dashboard Design

![Illustrative dashboard design preview](screenshots/output.png)

## What the project demonstrates

- Building an interactive user interface with Streamlit.
- Preparing input features in a consistent order and passing them to a scikit-learn model.
- Training and caching a model to avoid retraining on every Streamlit rerun.
- Presenting model output and interpreting it with appropriate caveats.
- Organizing a small Python project with isolated dependencies and Git hygiene.

## Model and data

**Model / workflow:** GradientBoostingClassifier with 100 estimators; probability output from the fitted classifier.

**Training data:** 1,000 generated sensor rows covering air temperature, process temperature, rotational speed, torque, and tool wear. The target is created using a torque > 60 Nm and tool-wear > 180 min rule.

## Output design

A dark industrial dashboard with orange accents, five sensor sliders, a live sensor table, a maintenance-pattern verdict, the model output score, and a description of the synthetic label rule.

The live application is designed to show the entered features, the model output, and an explanation of what the result means. A prediction is only as reliable as its data, labels, features, and evaluation method.

## Technology stack

- Python 3.11
- Streamlit
- pandas / NumPy where used by this project
- scikit-learn
- TensorFlow/Keras where required by this project

## Run locally

### 1. Prerequisite

Install **64-bit Python 3.11**. Confirm it is available in your terminal:

```bash
python --version
```

On Windows, if the `python` command points to another interpreter, use `py -3.11` in place of `python` when creating the environment.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

Run this command from the same folder as `requirements.txt`:

```bash
streamlit run predictive_maintenance.py
```

Streamlit prints a local URL in the terminal (normally `http://localhost:8501`). Open that address in your browser. To stop the app, press `Ctrl+C` in the terminal.

## Project folder structure

```text
03_predictive_maintenance/
├── predictive_maintenance.py
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
    └── output.png           
```

## Current limitations

Training labels are derived entirely from one simple rule, not verified machine-failure histories. The predicted probability reflects this synthetic labeling setup and must not be interpreted as real failure probability or a maintenance schedule. The dataset does not model machine type, operating history, sensor drift, or failure costs.

## Future improvements

1. Replace generated or toy data with a documented, appropriate dataset and record its source and license.
2. Add a repeatable evaluation process, appropriate metrics, baseline comparisons, and model-validation tests.
3. Save a trained model only after validating it, then separate training from app inference if the project grows.
4. Add input validation, clear error messages, and a simple automated test suite.
5. Capture real screenshots and document a reproducible test scenario.

## Portfolio summary

This project demonstrates a junior-level end-to-end workflow: a Python ML model, a simple interactive front end, input handling, result presentation, dependency isolation, and honest documentation of limitations. It should be presented as a learning prototype rather than a production-grade system.

## References

- [Streamlit documentation](https://docs.streamlit.io/)
- [scikit-learn Gradient Boosting documentation](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
