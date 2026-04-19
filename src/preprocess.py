# src/preprocess.py
# =============================================================================
# PURPOSE: Load raw IBM HR data, clean it, encode it, split it, apply SMOTE,
#          and save processed train/test CSVs for model training.
# =============================================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# STEP 1: Define all file paths
# =============================================================================

RAW_DATA_PATH       = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROCESSED_TRAIN     = "data/processed/train.csv"
PROCESSED_TEST      = "data/processed/test.csv"
PROCESSED_FULL      = "data/processed/full_processed.csv"
ENCODER_SAVE_PATH   = "models/label_encoders.pkl"
FEATURE_NAMES_PATH  = "models/feature_names.pkl"

# Create output directories if they don't exist yet
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)


# =============================================================================
# STEP 2: Load the raw CSV file
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV and return as DataFrame."""
    print(f"📂 Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"✅ Data loaded — Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}\n")
    return df


# =============================================================================
# STEP 3: Drop constant/useless columns
# These 3 columns have the same value for every single employee
# so they carry zero predictive information
# =============================================================================

def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are constant across all rows."""
    constant_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    
    # Only drop columns that actually exist in the dataframe
    cols_to_drop = [col for col in constant_cols if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    print(f"🗑️  Dropped constant columns: {cols_to_drop}")
    print(f"   New shape: {df.shape}\n")
    return df


# =============================================================================
# STEP 4: Encode the target column
# Attrition: "Yes" → 1 (employee left), "No" → 0 (employee stayed)
# =============================================================================

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Attrition column from Yes/No to 1/0."""
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    
    print(f"🎯 Target encoded — Attrition value counts:")
    print(f"   {df['Attrition'].value_counts().to_dict()}")
    
    # Show class imbalance percentage
    attrition_rate = df["Attrition"].mean() * 100
    print(f"   Attrition rate: {attrition_rate:.1f}% (this is why we need SMOTE)\n")
    return df


# =============================================================================
# STEP 5: Label encode all categorical (object) columns
# scikit-learn models need numbers, not strings like "Male"/"Female"
# =============================================================================

def encode_categorical_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label encode all object-type columns.
    Returns the encoded dataframe and a dict of fitted encoders
    (so we can apply the same encoding to new data in the app).
    """
    # Find all columns with string/object dtype
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    print(f"🔤 Encoding categorical columns: {object_cols}")
    
    encoders = {}  # Store each fitted encoder by column name
    
    for col in object_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le  # Save encoder so we can reverse-transform later
        print(f"   ✅ {col}: {list(le.classes_)} → {list(range(len(le.classes_)))}")
    
    print(f"\n   All categorical columns encoded.\n")
    return df, encoders


# =============================================================================
# STEP 6: Train/Test Split
# 80% training, 20% testing
# stratify=y ensures both splits have same % of attrition (class balance)
# =============================================================================

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets.
    Separates features (X) from target (y) first.
    """
    # Separate features and target
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]
    
    print(f"✂️  Splitting data — Test size: {test_size*100:.0f}%, Random state: {random_state}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,          # Maintain attrition ratio in both splits
        random_state=random_state
    )
    
    print(f"   Train set: {X_train.shape[0]} rows")
    print(f"   Test set : {X_test.shape[0]} rows")
    print(f"   Train attrition rate: {y_train.mean()*100:.1f}%")
    print(f"   Test  attrition rate: {y_test.mean()*100:.1f}%\n")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 7: Apply SMOTE (Synthetic Minority Over-sampling Technique)
# The dataset is imbalanced — ~84% stayed, ~16% left.
# SMOTE creates synthetic "attrition=1" examples to balance the training set.
# IMPORTANT: Apply SMOTE ONLY on training data, NEVER on test data.
# =============================================================================

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """
    Apply SMOTE to balance the training set.
    Returns resampled X and y as DataFrames.
    """
    print(f"⚖️  Applying SMOTE to training set...")
    print(f"   Before SMOTE — Class distribution: {y_train.value_counts().to_dict()}")
    
    smote = SMOTE(random_state=random_state)
    
    # fit_resample returns numpy arrays, so convert back to DataFrame
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name="Attrition")
    
    print(f"   After  SMOTE — Class distribution: {y_resampled.value_counts().to_dict()}")
    print(f"   Training set size: {len(X_resampled)} rows (was {len(X_train)})\n")
    
    return X_resampled, y_resampled


# =============================================================================
# STEP 8: Save processed datasets and encoders
# =============================================================================

def save_artifacts(
    X_train, y_train, X_test, y_test,
    encoders: dict, feature_names: list
):
    """Save all processed data and encoders to disk."""
    
    # Combine X and y back into single DataFrames for saving
    train_df = X_train.copy()
    train_df["Attrition"] = y_train.values
    
    test_df = X_test.copy()
    test_df["Attrition"] = y_test.values
    
    # Save CSVs
    train_df.to_csv(PROCESSED_TRAIN, index=False)
    test_df.to_csv(PROCESSED_TEST, index=False)
    
    print(f"💾 Saved processed train data → {PROCESSED_TRAIN}")
    print(f"💾 Saved processed test  data → {PROCESSED_TEST}")
    
    # Save label encoders dict (needed by app.py for encoding user input)
    joblib.dump(encoders, ENCODER_SAVE_PATH)
    print(f"💾 Saved label encoders       → {ENCODER_SAVE_PATH}")
    
    # Save feature names list (needed by train.py and app.py)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    print(f"💾 Saved feature names         → {FEATURE_NAMES_PATH}\n")


# =============================================================================
# STEP 9: Main pipeline — runs all steps in order
# =============================================================================

def main():
    print("=" * 60)
    print("   IBM HR ATTRITION — PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")
    
    # Step 1: Load raw data
    df = load_data(RAW_DATA_PATH)
    
    # Step 2: Drop useless constant columns
    df = drop_constant_columns(df)
    
    # Step 3: Encode target column (Yes/No → 1/0)
    df = encode_target(df)
    
    # Step 4: Encode all categorical string columns
    df, encoders = encode_categorical_columns(df)
    
    # Step 5: Train/test split (before SMOTE)
    X_train, X_test, y_train, y_test = split_data(
        df, test_size=0.2, random_state=42
    )
    
    # Step 6: Apply SMOTE on train set only
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # Step 7: Save everything to disk
    feature_names = list(X_train.columns)  # Original feature names before SMOTE
    save_artifacts(
        X_train_resampled, y_train_resampled,
        X_test, y_test,
        encoders, feature_names
    )
    
    print("=" * 60)
    print("✅ PREPROCESSING COMPLETE!")
    print(f"   Features used: {len(feature_names)}")
    print(f"   Feature list: {feature_names}")
    print("=" * 60)


# Entry point
if __name__ == "__main__":
    main()