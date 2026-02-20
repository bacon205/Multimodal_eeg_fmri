"""EEG data loading utilities for the bridge pipeline.

Extracted from bridge notebook Cell 5. Pure data I/O — no dependency
on crossmodal_v4_enhancements or any model code.
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.io import loadmat

logger = logging.getLogger(__name__)


def load_eeg_labels(label_dir, binary=True):
    """Load EEG clinical labels from medical_score.csv.

    Args:
        label_dir: Directory (str or Path) containing medical_score.csv.
        binary: If True, binarise scores (<=2 → 0, else → 1).

    Returns:
        Dict[int, int] mapping subject ID to label.
    """
    csv_path = os.path.join(str(label_dir), 'medical_score.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Label file not found: {csv_path}')
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Postoperative evaluation'])
    if df['Subject'].dtype == object:
        df['subject_id'] = df['Subject'].str.replace('sub', '', regex=False).astype(int)
    else:
        df['subject_id'] = df['Subject'].astype(int)
    label_dict = {}
    for _, row in df.iterrows():
        subj = int(row['subject_id'])
        score = row['Postoperative evaluation']
        label_dict[subj] = 0 if score <= 2 else 1 if binary else score
    return label_dict


def load_eeg_conn_features(conn_dir, subject_list, band_list, cond_list):
    """Load EEG connectivity features from .mat files.

    Args:
        conn_dir: Path to connectivity feature directory.
        subject_list: List of subject IDs (ints).
        band_list: Dict mapping band keys to band names (e.g. {'alpha': 'Alpha'}).
        cond_list: List of condition strings (e.g. ['open', 'close']).

    Returns:
        Dict keyed by (subject, band_key, condition, label_placeholder).
    """
    conn_dir = Path(conn_dir)
    conn_features = {}
    for subj in subject_list:
        subj_str = f'{subj:02d}'
        for band_key, band_name in band_list.items():
            for cond in cond_list:
                pattern = conn_dir / f'conn_{band_name}_{cond}_sub{subj_str}.mat'
                files = sorted(glob.glob(str(pattern)))
                if not files:
                    pattern_lower = conn_dir / f'conn_{band_key}_{cond}_sub{subj_str}.mat'
                    files = sorted(glob.glob(str(pattern_lower)))
                for f in files:
                    try:
                        mat = loadmat(f)
                        for k in mat:
                            if not k.startswith('_'):
                                data = np.array(mat[k], dtype=np.float32).flatten()
                                data = np.nan_to_num(data, nan=0.0)
                                label_val = 0  # placeholder
                                conn_key = (subj, band_key, cond, label_val)
                                conn_features[conn_key] = data
                                break
                    except Exception as e:
                        logger.warning(f'Error loading {f}: {e}')
    logger.info(f'Loaded {len(conn_features)} EEG connectivity samples')
    return conn_features


def load_eeg_pw_features(pw_dir, subject_list, band_list, freq_list):
    """Load EEG power spectrum features from .mat files.

    Args:
        pw_dir: Path to power spectrum feature directory.
        subject_list: List of subject IDs (ints).
        band_list: List of band keys (e.g. ['alpha', 'beta', 'theta']).
        freq_list: List of frequency strings (e.g. ['1_Hz', '2_Hz', ...]).

    Returns:
        Dict keyed by (subject, band, freq, label_placeholder).
    """
    pw_dir = Path(pw_dir)
    pw_features = {}
    for subj in subject_list:
        subj_str = f'{subj:02d}'
        for band in band_list:
            for freq in freq_list:
                pattern = str(pw_dir / f'powspctrm_{band}_{freq}_sub{subj_str}.mat')
                for f in sorted(glob.glob(pattern)):
                    try:
                        mat = loadmat(f)
                        for k in mat:
                            if not k.startswith('_'):
                                data = np.array(mat[k], dtype=np.float32).flatten()
                                data = np.nan_to_num(data, nan=0.0)
                                label_val = 0
                                pw_key = (subj, band, freq, label_val)
                                pw_features[pw_key] = data
                                break
                    except Exception as e:
                        logger.warning(f'Error loading {f}: {e}')
    logger.info(f'Loaded {len(pw_features)} EEG power spectrum samples')
    return pw_features


def load_eeg_erp_features(erp_dir, subject_list, band_list, freq_list):
    """Load EEG ERP features from .mat/.h5 files.

    Args:
        erp_dir: Path to ERP feature directory.
        subject_list: List of subject IDs (ints).
        band_list: List of band keys.
        freq_list: List of frequency strings.

    Returns:
        Dict keyed by (subject, band, freq, label_placeholder).
    """
    erp_dir = Path(erp_dir)
    erp_features = {}
    for subj in subject_list:
        subj_str = f'{subj:02d}'
        for band in band_list:
            for freq in freq_list:
                pattern = erp_dir / f'ERP_sub{subj_str}_{band}_{freq}*.mat'
                erp_files = sorted(glob.glob(str(pattern)))
                for f in erp_files:
                    try:
                        with h5py.File(f, 'r') as hf:
                            if 'erp_struct' in hf:
                                erp_group = hf['erp_struct']
                            elif 'erp' in hf:
                                erp_group = hf['erp']
                            else:
                                erp_group = hf[list(hf.keys())[0]]

                            if 'avg' in erp_group:
                                data = np.array(erp_group['avg'], dtype=np.float32)
                            elif 'trial' in erp_group:
                                data = np.array(erp_group['trial'], dtype=np.float32)
                                if data.ndim == 3:
                                    data = np.mean(data, axis=0)
                            else:
                                for dk in erp_group.keys():
                                    candidate = erp_group[dk]
                                    if hasattr(candidate, 'shape') and len(candidate.shape) >= 2:
                                        data = np.array(candidate, dtype=np.float32)
                                        break
                                else:
                                    continue

                            data = np.nan_to_num(data, nan=0.0)
                            label_val = 0
                            erp_key = (subj, band, freq, label_val)
                            erp_features[erp_key] = data
                    except Exception as e:
                        # Try scipy loadmat fallback
                        try:
                            mat = loadmat(f)
                            for k in mat:
                                if not k.startswith('_'):
                                    data = np.array(mat[k], dtype=np.float32)
                                    data = np.nan_to_num(data, nan=0.0)
                                    label_val = 0
                                    erp_key = (subj, band, freq, label_val)
                                    erp_features[erp_key] = data
                                    break
                        except Exception:
                            logger.warning(f'Error loading ERP {f}: {e}')
    logger.info(f'Loaded {len(erp_features)} EEG ERP samples')
    return erp_features
