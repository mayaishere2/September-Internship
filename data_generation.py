import argparse
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)
# -------------------------------
# 1) Feature Ranges & Metadata
# -------------------------------
# Typical/illustrative ranges — tune to your plant.
RANGES = {
    #Ambient
    "AT": (-10.00, 40.00, "°C", "Ambient Temperature"),
    "AP": (980.00, 1040.00, "mbar", "Ambient Pressure"),
    "AH": (20.00, 100.00, "%", "Ambient Humidity"),
    #Intake / filter
    "AFDP": (2.0 , 8.0 , "mbar", "Air Filter Delta Pressure"),
    # Core pressures & temps 
    "GTEP": (15.00 , 40.00, "mbar", "Exhaust Pressure"),
    "TIT": (1000.00, 1105.00, "°C", "Turbine Inlet Temperature"),
    "TAT": (500.00, 560.00, "°C", "Turbine Outlet Temperature"),
    "CDP": (9.00, 16.00, "bar", "Compressor Discharge Pressure"), # Corrected Unit from Mbar to bar for consistency
    # Speeds
    "N1": (60.00, 100.00, "%", "Low Pressure Rotor Speed"),
    "N2": (60.00, 100.00, "%", "High Pressure Rotor Speed"),
    # Exhaust gas temps & spread
    "EGT": (450.00, 650.00, "°C", "Exhaust Gas Temperature"),
    "dEGT": (0.00, 40.00, "°C", "Exhaust Gas Temperature Spread"),
    # Vibration & bearings & oil
    "V_shaft": (10.00, 100.00, "µm_pp / mm_s_RMS", "Shaft Vibration"),
    "T_bearing": (60.0, 120.0, "°C", "Bearing Metal Temp"),
    "P_oil": (2.0, 6.0, "bar", "Lube Oil Pressure"),
    "T_oil": (40.0, 90.0, "°C", "Lube Oil Temp"),
    # Flows & Power
    "m_fuel": (1000.0, 5000.0, "kg/h", "Fuel Flow"),
    "P_out": (100.0, 180.0, "MW", "Power Output"),
    "HR": (9000.0, 12000.0, "kJ/KWh", "Heat Rate (computed)"),
    "m_air": (100.0, 500.0, "kg/s", "Air Mass Flow"),
    "SM": (5.0, 30.0, "%","Surge Margin"),
    # Process Train
    "P_suc": (1.0, 5.0, "bar", "Compressor Suction Pressure"),
    "P_dis": (20.0, 60.0, "bar", "Compressor Discharge Pressure"),
    # Emissions 
    "CO": (0.0, 50.0, "mg/m3", "Carbon Monoxide"),
    "NOx": (20.0, 120.0, "mg/m3", "Nitrogen Oxides")
    # Engineered (computed below): EGT_norm, SM_prox, eta_resid, dP_filter
}

FAULT_CLASSES = ["normal", "hot_section", "combustion", "compressor", "bearing tube", "filter fouling"]

# -------------------------------
# Helpers
# -------------------------------

def within_range(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip(x, vmin, vmax)

def synth_series(n: int, lo: float, hi: float, drift: float = 0.0, noise: float = 0.2, seed: int = None) -> np.ndarray:
    # Generate a synthetic time series with drift and noise
    rng = np.random.default_rng(seed)
    base = rng.uniform(lo, hi, size=1).item()
    steps = rng.normal(drift, noise, size=n).cumsum()
    x = base + (hi - lo) * 0.1 * steps
    # A slightly more realistic wander
    x = np.interp(np.linspace(0, 1, n), [0, 1], [base, base + (hi - lo) * 0.1 * rng.uniform(-0.5, 0.5)]) + rng.normal(0, (hi - lo) * 0.02, size=n)
    return within_range(x, lo, hi)

# MODIFIED: New function to load real data and synthesize the rest
# MODIFIED: New function to load real data and synthesize the rest
def build_dataset_from_files(sensor_map: Dict, base_dir: Path) -> Optional[pd.DataFrame]:
    """
    Builds a complete dataset by loading specified sensor files and synthesizing missing features.
    """
    master_df = pd.DataFrame()
    loaded_features = set()
    n_rows = 0

    print("[INFO] Loading real sensor data...")
    
    # --- 1. Special handling for the multi-column 'profile.txt' ---
    # FIX: Correctly check for and handle the multi-column file first.
    try:
        profile_path = base_dir / 'profile.txt'
        profile_df = pd.read_csv(profile_path, sep=r'\s+', header=None)
        n_rows = len(profile_df)
        print(f" - Read {profile_path.name} with {n_rows} rows. This determines the dataset length.")
        
        # Map columns from profile.txt to features
        for feature, source in sensor_map.items():
            # Check if the source is a tuple (filename, col_idx) and filename is 'profile.txt'
            if isinstance(source, tuple) and source[0] == 'profile.txt':
                col_idx = source[1]
                master_df[feature] = profile_df[col_idx]
                loaded_features.add(feature)
                print(f"   - Mapped column {col_idx} from profile.txt to '{feature}'")

    except FileNotFoundError:
        print(f"[WARNING] 'profile.txt' not found in {base_dir}. Cannot load ambient data from it.")
        # We can still proceed if other files exist, but we need to determine n_rows from another file.
    except Exception as e:
        print(f"[ERROR] Could not read 'profile.txt': {e}")
        # If profile.txt is critical and fails, we might need to exit.
        # For now, we'll continue and try to load other files.

    # --- 2. Handle all other single-column sensor files ---
    for feature, source in sensor_map.items():
        if feature in loaded_features:
            continue  # Skip features already loaded from profile.txt

        # FIX: Ensure the source is a string before treating it as a filename.
        if not isinstance(source, str):
            print(f"[DEBUG] Skipping non-string source for feature '{feature}': {source}")
            continue

        filename = source
        try:
            path = base_dir / filename
            sensor_data = pd.read_csv(path, sep=r'\s+', header=None)
            
            # If n_rows hasn't been set yet (e.g., profile.txt was missing), set it from the first successful file read.
            if n_rows == 0:
                n_rows = len(sensor_data)
                print(f" - Dataset length determined from {filename}: {n_rows} rows.")

            if len(sensor_data) != n_rows:
                warnings.warn(f"Length mismatch for {filename} ({len(sensor_data)}) vs expected ({n_rows}). Data will be aligned.")
                # Align data to the established n_rows
                aligned_data = np.full(n_rows, np.nan)
                common_length = min(len(sensor_data), n_rows)
                aligned_data[:common_length] = sensor_data.iloc[:common_length, 0].values
                master_df[feature] = aligned_data
            else:
                 master_df[feature] = sensor_data[0].values

            loaded_features.add(feature)
            print(f" - Loaded '{feature}' from {filename}")

        except FileNotFoundError:
            warnings.warn(f"File '{filename}' for feature '{feature}' not found. This feature will be synthesized.")
        except Exception as e:
            warnings.warn(f"Could not process {filename} for {feature}: {e}. This feature will be synthesized.")

    if n_rows == 0:
        print("[ERROR] Could not determine dataset length from any sensor file. Exiting.")
        return None

    # --- 3. Synthesize missing features ---
    print("\n[INFO] Synthesizing remaining features...")
    for feature, (lo, hi, *_) in RANGES.items():
        if feature not in loaded_features:
            print(f" - Synthesizing '{feature}'")
            seed = hash(feature) % (2**32 - 1)
            master_df[feature] = synth_series(n_rows, lo, hi, seed=seed)
    
    # --- 4. Add a timestamp index ---
    master_df.index = pd.date_range("2021-01-01", periods=n_rows, freq="min")
    
    # --- 5. Handle potential missing values from file loading/alignment ---
    if master_df.isnull().sum().sum() > 0:
        print("\n[INFO] Forward-filling missing values that may have resulted from file loading...")
        master_df.fillna(method='ffill', inplace=True)
        master_df.fillna(method='bfill', inplace=True) # Back-fill for any NaNs at the beginning

    return master_df
# MODIFIED: New function to apply physical relationships
def apply_physics_relations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies some basic physical correlations to the dataset columns.
    This makes the synthetic data more realistic.
    """
    master = df.copy()
    print("\n[INFO] Applying physics-based relationships to data...")
    
    # EGT relates to TIT and load (P_out)
    master["EGT"] = 0.6 * master["TIT"] + 0.1 * master["P_out"] + RNG.normal(0, 5, len(master)) + 100.0
    master["EGT"] = within_range(master["EGT"].values, *RANGES["EGT"][:2])

    # EGT spread small by default
    master["dEGT"] = within_range(np.abs(RNG.normal(4, 2, len(master))), *RANGES["dEGT"][:2])

    # Heat rate inversely proportional to efficiency (coarse proxy using P_out & TIT)
    master["HR"] = 11000 - 5.0 * (master["P_out"] - 140.0) - 2.0 * (master["TIT"] - 1050) + RNG.normal(0, 100, len(master))
    master["HR"] = within_range(master["HR"].values, *RANGES["HR"][:2])

    # Emissions influenced by combustion and TIT
    master["NOx"] = within_range(0.15 * (master["TIT"] - 950.0) + RNG.normal(50, 8, len(master)), *RANGES["NOx"][:2])
    master["CO"]  = within_range(np.maximum(0, 25 - 0.05 * (master["TIT"] - 950.0)) + RNG.normal(0, 3, len(master)), *RANGES["CO"][:2])

    # Final clip to ensure all values are within their defined ranges
    for col, (lo, hi, *_rest) in RANGES.items():
        if col in master.columns:
            master[col] = within_range(master[col].values, lo, hi)
            
    return master

def inject_anamolies(df: pd.DataFrame, rate: float, seed: int=42) -> pd.Series:
    """
    Inject anomalies for different fault types.
    Returns the fault_type series.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    labels = np.array(["normal"] * n, dtype=object)
    k = int(max(1, n * rate))
    # types of faults
    idx_hot = rng.choice(n, k , replace=False)
    idx_comb = rng.choice(np.setdiff1d(np.arange(n), idx_hot), k, replace=False)
    idx_comp = rng.choice(np.setdiff1d(np.arange(n), np.concatenate([idx_hot, idx_comb])), k, replace=False)
    idx_bear = rng.choice(np.setdiff1d(np.arange(n), np.concatenate([idx_hot, idx_comb, idx_comp])), k, replace=False)
    idx_filt = rng.choice(np.setdiff1d(np.arange(n), np.concatenate([idx_hot, idx_comb, idx_comp, idx_bear])), k, replace=False)
    
    # Hot section: raise TIT, EGT, dEGT a bit, NOx up, HR up
    df.loc[idx_hot, "TIT"] += np.abs(rng.normal(10, 5, size=k))
    df.loc[idx_hot, "EGT"] += np.abs(rng.normal(15, 8, size=k))
    df.loc[idx_hot, "dEGT"] += np.abs(rng.normal(5, 3, size=k))
    df.loc[idx_hot, "NOx"] += np.abs(rng.normal(10, 6, size=k))
    df.loc[idx_hot, "HR"] += np.abs(rng.normal(300, 150, size=k))
    labels[idx_hot] = "hot_section"
    # Combustion issues: irregular EGT spread, CO spike, slight TIT up/down
    df.loc[idx_comb, "CO"] += np.abs(rng.normal(8, 4, size=k))
    df.loc[idx_comb, "dEGT"] += np.abs(rng.normal(10, 6, size=k))
    df.loc[idx_comb, "TIT"] += rng.normal(0, 5, size=k)
    labels[idx_comb] = "combustion" # Corrected case from original
    # Compressor issues: lower SM, oscillatory CDP, HR up, m_air down
    df.loc[idx_comp, "SM"] -= np.abs(rng.normal(5, 2, size=k))
    df.loc[idx_comp, "CDP"] += rng.normal(0, 0.5, size=k)
    df.loc[idx_comp, "HR"] += np.abs(rng.normal(400, 200, size=k))
    df.loc[idx_comp, "m_air"] -= np.abs(rng.normal(10, 5, size=k))
    labels[idx_comp] = "compressor"
    # Bearing/lube: increase V_shaft, T_bearing, lower P_oil, raise T_oil
    df.loc[idx_bear, "V_shaft"] += np.abs(rng.normal(15, 7, size=k))
    df.loc[idx_bear, "T_bearing"] += np.abs(rng.normal(10, 5, size=k))
    df.loc[idx_bear, "P_oil"] -= np.abs(rng.normal(0.5, 0.3, size=k))
    df.loc[idx_bear, "T_oil"] += np.abs(rng.normal(8, 4, size=k))
    labels[idx_bear] = "bearing tube" # Corrected case from original
    # Filter fouling: AFDP up, CDP slightly down, P_out down
    df.loc[idx_filt, "AFDP"] += np.abs(rng.normal(1.0, 0.5, size=k)) # Reduced std dev for stability
    df.loc[idx_filt, "CDP"] -= np.abs(rng.normal(0.4, 0.2, size=k))
    df.loc[idx_filt, "P_out"] -= np.abs(rng.normal(5.0, 2.0, size=k))
    labels[idx_filt] = "filter fouling"

    for col, (lo, hi , *_rest) in RANGES.items():
        if col in df.columns:
            df[col] = within_range(df[col], lo, hi)
    
    return pd.Series(labels, name="fault_type", index=df.index)


def Compute_Engineered(df: pd.DataFrame) -> pd.DataFrame:
    """ Compute engineered features and add to df"""
    out = df.copy()
    # Expected EGT proxy (simple affine model of TIT and load)
    out["EGT_Expected"] = 0.55 * out["TIT"] + 0.12 * out["P_out"] + 120.0
    out["EGT_norm"] = (out["EGT"]/out["EGT_Expected"]).clip(lower = 0.5, upper = 1.5)

    # Surge proximity (lower is closer to surge). Here we map from SM (%).
    out["SM_prox"] = (30.0 - out["SM"]) / 30.0  # 0 (safe) .. 1 (at surge line)
    out["SM_prox"] = out["SM_prox"].clip(0, 1)

    # Efficiency residual proxy: HR higher than nominal -> positive residual
    nominal_hr = 10000.0
    out["eta_resid"] = (out["HR"] - nominal_hr) / 1000.0

    # Filter dP trend proxy: if AFDP high relative to moving baseline
    out["AFDP_ma"] = out["AFDP"].rolling(60, min_periods=1).mean()
    out["dP_filter"] = (out["AFDP"] - out["AFDP_ma"]).fillna(0.0)

    # Simple vibration spectral proxy: harmonics ~ function of V_shaft
    out["V_harm_1x"] = out["V_shaft"] * (1.0 + np.random.default_rng(0).normal(0, 0.05, len(out)))
    out["V_harm_2x"] = 0.5 * out["V_shaft"] * (1.0 + np.random.default_rng(1).normal(0, 0.07, len(out)))

    return out

def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic gas turbine dataset, using real sensor data where available.")
    ap.add_argument("--base_dir", type=str, default=".", help="Directory containing your sensor TXT files")
    ap.add_argument("--output", type=str, default="lng_pdm_dataset.csv", help="Output CSV path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inject_rate", type=float, default=0.06, help="Fraction per-fault to inject")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_path = Path(args.output)
    
    # MODIFIED: Define the mapping from features to your sensor files.
    # For multi-column files like 'profile.txt', specify a tuple: (filename, column_index)
    # For single-column files, just provide the filename string.
    SENSOR_MAP = {
        'AT': ('profile.txt', 0),  # Ambient Temperature
        'AP': ('profile.txt', 1),  # Ambient Pressure
        'AH': ('profile.txt', 2),  # Ambient Humidity
        'AFDP': 'PS1.txt',         # Air Filter Delta Pressure
        'CDP': 'PS2.txt',          # Compressor Discharge Pressure
        'GTEP': 'PS3.txt',         # Gas Turbine Exhaust Pressure
        'P_oil': 'PS4.txt',        # Lube Oil Pressure
        'P_suc': 'PS5.txt',        # Compressor Suction Pressure
        'P_dis': 'PS6.txt',        # Compressor Discharge Pressure
        'TIT': 'TS1.txt',          # Turbine Inlet Temperature
        'TAT': 'TS2.txt',          # Turbine Outlet Temperature
        'T_bearing': 'TS3.txt',    # Bearing Temperature
        'T_oil': 'TS4.txt',        # Lube Oil Temperature
        'm_fuel': 'FS1.txt',       # Fuel Flow
        'm_air': 'FS2.txt',        # Air Mass Flow
        'V_shaft': 'VS1.txt',      # Shaft Vibration
        'N1': 'SE.txt',            # LP Rotor Speed (assuming SE = Speed Engine)
        'P_out': 'EPS1.txt',       # Power Output (assuming EPS = Electrical Power System)
        'CO': 'CE.txt'             # Carbon Monoxide (assuming CE = CO Emissions)
    }

    # 1) Build the master dataframe from real and synthetic data
    master = build_dataset_from_files(SENSOR_MAP, base_dir)
    
    if master is None:
        print("[FAIL] Could not build dataset. Exiting.")
        sys.exit(1)

    # 2) Apply physics-based relationships to improve realism
    master = apply_physics_relations(master)

    # ... (previous lines)

    # 3) Compute engineered features
    master = Compute_Engineered(master)

    # FIX: Reset the index here, BEFORE injecting anomalies
    master.reset_index(drop=True, inplace=True)

    # 4) Inject anomalies and get labels
    labels = inject_anamolies(master, rate=args.inject_rate, seed=args.seed)
    master["fault_type"] = labels

    # 5) Shuffle the data (the index has already been reset)
    master = master.sample(frac=1.0, random_state=args.seed).sort_index()

    # ... (rest of the code)
    # 6) Save output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False)

    meta = {
        "rows": len(master), 
        "columns": list(master.columns),
        "seed": args.seed,
        "fault_distribution": master["fault_type"].value_counts().to_dict()
    }
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Saved dataset to: {out_path}")
    print(f"[OK] Saved metadata to: {out_path.with_suffix('.meta.json')}")
    print("\nPreview of the final dataset:")
    print(master.head(5))
    print("\nFault type distribution:")
    print(master['fault_type'].value_counts())

if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 200)
    # The original script used a context manager to ignore warnings, which is fine,
    # but showing them can be useful during debugging.
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    main()