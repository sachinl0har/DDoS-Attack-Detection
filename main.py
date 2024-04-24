from joblib import load
import warnings

# Load the model
model = load(r"OLD_MODELS/SVM.joblib")  # Replace with the path to your model file

# Print the warning message
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sklearn_version_warning = model.__module__
    print("Warning:", sklearn_version_warning)
