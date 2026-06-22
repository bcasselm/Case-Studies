"""
First-level GLM for a multi-run, multi-subject word-presentation fMRI experiment.
Single-trial betas are estimated with Least-Squares-Separate (LSS; Mumford et al., 2012):
one GLM per trial, with the target trial modelled on its own and all other trials collapsed
into a single regressor. The single-trial betas of each word are then averaged into one
beta map per word.
"""

#####################################################################
# Imports
#####################################################################
import warnings
warnings.filterwarnings("ignore")

import gc
import re
import shutil
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from nilearn import plotting, image
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
import nibabel as nib

#####################################################################
# Configuration
#####################################################################
PROJECT_ROOT   = Path("/Volumes/T9/Birgit")

DATA_DIR       = PROJECT_ROOT / "data"
BIDS_DIR       = DATA_DIR / "ds004301"
FMRIPREP_DIR   = BIDS_DIR / "derivatives" / "preprocessed_data"
OUT_DIR        = BIDS_DIR / "derivatives" / "betas"
MASK_PATH      = DATA_DIR / "brain_parcellations" / "emotion_parcellation_rsa_union.nii.gz"

# GLM parameters
T_R            = 2.0          # repetition time in seconds (ds004301; Wang et al. 2022)
SLICE_TIME_REF = 0.5          # reference slice for slice-timing (fraction of TR)
HRF_MODEL      = "glover"     # canonical HRF ('spm' is equivalent for RSA)
DRIFT_MODEL    = None         # high-pass handled via cosine regressors in the confounds TSV
HIGH_PASS      = None
SMOOTHING_FWHM = None         # mm; None = no smoothing, to preserve multivariate patterns
NOISE_MODEL    = "ar1"        # AR(1) prewhitening for temporal autocorrelation
SIGNAL_SCALING = 0            # 0 = percent signal change
MAX_RUNS       = 0            # 0 = use all runs, >0 = use first N runs

# Which fMRIPrep confounds to include
CONFOUND_STRATEGY = ("motion", "compcor", "cosines", "spikes")
MOTION_PARAMS     = "full"    # 'basic'=6, 'full'=24
N_COMPCOR         = 6         # number of leading aCompCor components

# Memory/performance controls
N_JOBS           = 4          # parallel LSS fits within a run (joblib); set to 1 to disable
MINIMIZE_MEMORY  = True

# List of subjects to process (get_run_files prepends 'sub-', as in the main pipeline)
SUBJECTS = [f"{i:02d}" for i in range(1, 12)]

#####################################################################
# Helper functions
#####################################################################
def get_run_files(subject: str):
    """
    Return sorted lists of (bold_img, events_df, confounds_df) for every run.
    """
    bold_files = list((FMRIPREP_DIR / f"sub-{subject}" / "func").glob(
        f"sub-{subject}_task-listening_run-*_bold.nii.gz"
    ))
    def _run_sort_key(p):
        m = re.search(r"run-(\d+)", p.stem)
        return int(m.group(1)) if m else p.stem
    bold_files = sorted(bold_files, key=_run_sort_key)
    if not bold_files:
        raise FileNotFoundError(f"No BOLD files found for sub-{subject}")

    if MAX_RUNS > 0:
        bold_files = bold_files[:MAX_RUNS]
        print(f"  Using first {len(bold_files)} runs (MAX_RUNS={MAX_RUNS})")

    def _select_confounds(confounds_df: pd.DataFrame) -> pd.DataFrame:
        selected = []

        if "motion" in CONFOUND_STRATEGY:
            if MOTION_PARAMS == "basic":
                selected.extend([
                    "trans_x", "trans_y", "trans_z",
                    "rot_x", "rot_y", "rot_z",
                ])
            else:
                selected.extend([
                    "trans_x", "trans_x_derivative1", "trans_x_power2", "trans_x_derivative1_power2",
                    "trans_y", "trans_y_derivative1", "trans_y_power2", "trans_y_derivative1_power2",
                    "trans_z", "trans_z_derivative1", "trans_z_power2", "trans_z_derivative1_power2",
                    "rot_x", "rot_x_derivative1", "rot_x_power2", "rot_x_derivative1_power2",
                    "rot_y", "rot_y_derivative1", "rot_y_power2", "rot_y_derivative1_power2",
                    "rot_z", "rot_z_derivative1", "rot_z_power2", "rot_z_derivative1_power2",
                ])

        if "compcor" in CONFOUND_STRATEGY:
            selected.extend(sorted(c for c in confounds_df.columns if c.startswith("a_comp_cor_"))[:N_COMPCOR])

        if "cosines" in CONFOUND_STRATEGY:
            selected.extend([c for c in confounds_df.columns if c.startswith("cosine")])

        if "spikes" in CONFOUND_STRATEGY:
            selected.extend([c for c in confounds_df.columns
                             if c.startswith("motion_outlier") or c.startswith("non_steady_state_outlier")])

        selected = [column for column in selected if column in confounds_df.columns]
        if not selected:
            raise ValueError("No requested confound columns found in confounds TSV.")

        return confounds_df[selected].fillna(0.0)

    events_list, confounds_list, imgs = [], [], []
    for bold in bold_files:
        run_label = [p for p in bold.stem.split("_") if p.startswith("run-")][0]
        events_path = (
            BIDS_DIR / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-listening_{run_label}_events.tsv"
        )
        events = pd.read_csv(events_path, sep="\t", encoding="utf-8-sig")
        if "trial_type" not in events.columns:
            if "stim_file" not in events.columns:
                raise ValueError(f"Events file must contain 'trial_type' or 'stim_file': {events_path}")

            def _stim_to_trial_type(stim_value: str) -> str:
                text = str(stim_value)
                match = re.search(r"word\d+", text)
                if match:
                    return match.group(0)
                stem = Path(text).stem
                return stem if stem else "unknown"

            events = events.copy()
            events["trial_type"] = events["stim_file"].astype(str).map(_stim_to_trial_type)

        confounds_path = bold.with_name(bold.name.replace("_bold.nii.gz", "_desc-confounds.tsv"))
        if not confounds_path.exists():
            raise FileNotFoundError(f"Confounds file not found: {confounds_path}")
        confounds_raw = pd.read_csv(confounds_path, sep="\t")
        n_scans = nib.load(str(bold)).shape[3]
        if len(confounds_raw) > n_scans:
            confounds_raw = confounds_raw.iloc[-n_scans:].reset_index(drop=True)
        confounds = _select_confounds(confounds_raw)

        imgs.append(str(bold))
        events_list.append(events)
        confounds_list.append(confounds)

    return imgs, events_list, confounds_list


def make_lss_events(events: pd.DataFrame, target_trial: int) -> pd.DataFrame:
    """
    Build the events table for one LSS fit: the target trial is named 'trial',
    all other trials are collapsed into 'other'.
    """
    lss = events[["onset", "duration"]].copy()
    lss["trial_type"] = "other"
    lss.iloc[target_trial, lss.columns.get_loc("trial_type")] = "trial"
    return lss


def make_model(mask_img) -> FirstLevelModel:
    """
    A fresh single-trial GLM.  The caller pre-resamples the 3-D parcellation
    mask to BOLD native space (cheap, nearest-neighbour) and passes it here so
    nilearn never touches the 4-D BOLD data.  N_JOBS=1 because parallelism is
    handled at the trial level by joblib in fit_first_level.
    """
    return FirstLevelModel(
        t_r=T_R,
        slice_time_ref=SLICE_TIME_REF,
        hrf_model=HRF_MODEL,
        drift_model=DRIFT_MODEL,
        high_pass=HIGH_PASS,
        smoothing_fwhm=SMOOTHING_FWHM,
        mask_img=mask_img,
        noise_model=NOISE_MODEL,
        standardize=False,
        signal_scaling=SIGNAL_SCALING,
        minimize_memory=MINIMIZE_MEMORY,
        n_jobs=1,
        verbose=0,
    )


def _fit_one_trial(bold_img, mask_img, events, confounds, trial, trial_path,
                   run_idx, subject):
    """Fit a single LSS trial and save the beta map. Returns trial_path."""
    import numpy as np
    model = make_model(mask_img)
    try:
        model.fit(
            run_imgs=bold_img,
            events=make_lss_events(events, trial),
            confounds=confounds,
        )
    except np.linalg.LinAlgError as e:
        print(f"  WARNING: SVD/LinAlg failure on sub-{subject} run{run_idx} trial{trial} — skipping ({e})")
        return None
    beta_img = model.compute_contrast(
        contrast_def="trial",
        stat_type="t",
        output_type="effect_size",
    )
    nib.save(beta_img, trial_path)

    if run_idx == 0 and trial == 0:
        dm_dir = Path("/Volumes/T9/Birgit") / "reports" / "plots" / "design_matrix"
        dm_dir.mkdir(parents=True, exist_ok=True)
        plotting.plot_design_matrix(
            model.design_matrices_[0],
            output_file=str(dm_dir / f"example_lss_design_matrix_sub-{subject}.png"),
        )
    return trial_path


def fit_first_level(subject: str):
    """
    Fit one LSS GLM per trial across ALL runs of one subject, then average the
    single-trial betas of each word into one beta map per word.
    """
    print(f"\n{'='*60}")
    print(f"  sub-{subject}")
    print(f"{'='*60}")

    imgs, events_list, confounds_list = None, None, None
    model = None
    try:
        out_sub = OUT_DIR / f"sub-{subject}"
        existing_betas = list(out_sub.glob("beta_*.nii.gz"))
        if existing_betas and not (out_sub / "trials").exists():
            conditions = sorted(f.stem.replace("beta_", "") for f in existing_betas)
            print(f"  Already complete ({len(conditions)} beta maps) — skipping")
            return conditions

        imgs, events_list, confounds_list = get_run_files(subject)
        print(f"  Found {len(imgs)} runs")

        if not MASK_PATH.exists():
            raise FileNotFoundError(f"Mask file not found: {MASK_PATH}")

        trials_dir = out_sub / "trials"
        trials_dir.mkdir(parents=True, exist_ok=True)

        # Fit every trial of every run; group the single-trial beta paths by word.
        # If a trial beta already exists on disk, skip refitting it.
        # The 4-D BOLD is never resampled.  Instead, the 3-D parcellation mask is
        # resampled (nearest-neighbour, once per run) to the BOLD's native grid
        # and passed to every trial's GLM.
        word_files = {}
        skipped = 0
        for run_idx, (img, events, confounds) in enumerate(zip(imgs, events_list, confounds_list)):

            # Determine which trials still need fitting
            pending = []
            for trial in range(len(events)):
                word = str(events["trial_type"].iloc[trial])
                trial_path = trials_dir / f"run{run_idx:02d}_trial{trial:03d}_{word}.nii.gz"
                word_files.setdefault(word, []).append(trial_path)
                if trial_path.exists():
                    skipped += 1
                else:
                    pending.append(trial)

            if not pending:
                continue  # entire run already done — skip entirely

            # Lazy-load the BOLD (ArrayProxy — no data read yet) and resample
            # the small 3-D mask to match it
            bold_img = nib.load(img)
            mask_resampled = image.resample_to_img(
                str(MASK_PATH), bold_img, interpolation="nearest"
            )

            # Build (trial, trial_path) pairs for pending trials
            pending_paths = [
                (trial,
                 trials_dir / f"run{run_idx:02d}_trial{trial:03d}_{str(events['trial_type'].iloc[trial])}.nii.gz")
                for trial in pending
            ]

            # Fit pending trials — in parallel when N_JOBS > 1
            if N_JOBS > 1:
                Parallel(n_jobs=N_JOBS, prefer="processes")(
                    delayed(_fit_one_trial)(
                        bold_img, mask_resampled, events, confounds,
                        trial, trial_path, run_idx, subject,
                    )
                    for trial, trial_path in pending_paths
                )
            else:
                for trial, trial_path in pending_paths:
                    _fit_one_trial(
                        bold_img, mask_resampled, events, confounds,
                        trial, trial_path, run_idx, subject,
                    )

            del bold_img, mask_resampled
            gc.collect()

        if skipped:
            print(f"  Resumed: skipped {skipped} already-computed trial betas")

        # Average the single-trial betas of each word into one beta per word.
        # Filter out trial paths that were never written
        for word, files in word_files.items():
            existing = [str(f) for f in files if f.exists()]
            if not existing:
                print(f"  WARNING: no trials saved for word '{word}' — skipping beta")
                continue
            mean_beta = image.mean_img(existing, copy_header=True)
            nib.save(mean_beta, out_sub / f"beta_{word}.nii.gz")

        # Remove the single-trial maps now that the per-word betas are written
        shutil.rmtree(trials_dir, ignore_errors=True)

        conditions = sorted(word_files)
        print(f"  Saved {len(conditions)} beta maps → {out_sub}")
        return conditions
    finally:
        del model, imgs, events_list, confounds_list
        gc.collect()

#######################################################################
# Main execution
#######################################################################
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_conditions = None
    for sub in tqdm(SUBJECTS, desc="Fitting LSS first-level models"):
        try:
            conditions = fit_first_level(sub)
            if all_conditions is None:
                all_conditions = conditions
        except FileNotFoundError as e:
            print(f"  SKIPPING sub-{sub}: {e}")
        finally:
            gc.collect()

    if not all_conditions:
        raise RuntimeError("No subjects were processed successfully; no condition list to write.")

    # Save condition list (shared across subjects) for reference in RSA step
    cond_path = OUT_DIR / "conditions.txt"
    cond_path.write_text("\n".join(all_conditions))
    print(f"\nAll subjects done. Condition list saved → {cond_path}")