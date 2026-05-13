"""
First-level GLM for a multi-run, multi-subject word-presentation fMRI experiment.
Produces one beta map (NIfTI) per word/condition per subject.

Assumptions
-----------
- Data is organised in BIDS format and preprocessed with fMRIPrep.
- Each run has an events file with at least columns: onset, duration, stim_file.
- stim_file contains the word label.
- Confounds come from fMRIPrep's *_desc-confounds.tsv files.

Outputs
-------
betas/<sub>/beta_<word>.nii.gz   – one 3-D NIfTI per word per subject
"""

#####################################################################
# Imports
#####################################################################
import warnings
warnings.filterwarnings("ignore")

import gc
import re
from pathlib import Path
from tqdm import tqdm
from nilearn import plotting
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
import nibabel as nib
from bids import BIDSLayout

#####################################################################
# Configuration
#####################################################################
BIDS_DIR       = Path("/home/f_moldovan/projects/case_studies/data/bids")
FMRIPREP_DIR   = Path("/home/f_moldovan/projects/case_studies/data/bids/derivatives/preprocessed_data")
OUT_DIR        = Path("/home/f_moldovan/projects/case_studies/data/bids/derivatives/betas")
MASK_PATH      = Path("/home/f_moldovan/projects/case_studies/data/brain_parcellations/emotion_parcellation_rsa_union.nii.gz")

# GLM parameters
T_R            = BIDSLayout(BIDS_DIR).get_tr()          # repetition time in seconds (should be 2.0 for this dataset) - we read it from BIDS metadata to avoid hardcoding
SLICE_TIME_REF = 0.5                                    # reference slice for slice-timing (0–1, fraction of TR; should be in the middle of TR according to fMRIPrep documentation)
HRF_MODEL      = "glover"                               # for now a rather simple canonical HRF model; we could consider more flexible basis sets if we want to capture more complex response shapes, but this is a good starting point and keeps the model simpler (and less likely to overfit) given our limited data per condition. Note that for RSA, we care more about relative differences between conditions than absolute effect sizes, so a simple HRF model is often sufficient.
DRIFT_MODEL    = "cosine"                               # high-pass filtering via discrete cosines
HIGH_PASS      = 1/128                                  # Hz  (128 s period, SPM default)
SMOOTHING_FWHM = 6                                      # mm (None = no smoothing)
MAX_RUNS       = 0                                      # 0 = use all runs, >0 = use first N runs

# Which fMRIPrep confounds to include (now we only include motion parameters and WM/CSF signals, but we could add more if needed; sometimes some confounds have their own parameters, e.g. motion can be 'basic' (6 parameters) or 'full' (24 parameters including derivatives and squares))
CONFOUND_STRATEGY = ("motion", "wm_csf")
MOTION_PARAMS     = "full"    # 'basic'=6, 'full'=24
WM_CSF_PARAMS     = "full"    # 'basic'=2, 'full'=8; change this to 'full' to include derivatives and squares of WM/CSF signals, but for now we stick with 'basic' to keep the model simpler 

# Memory/performance controls
N_JOBS           = 1
MINIMIZE_MEMORY  = True
NOISE_MODEL      = "ols"
RESAMPLE_TO_MASK = True  # if True, resample data to mask space/affine during GLM fitting (saves memory)

# List of subjects to process (should match BIDS directory names)
SUBJECTS = BIDSLayout(BIDS_DIR).get_subjects()

#####################################################################
# Helper functions
#####################################################################
def get_run_files(subject: str):
    """
    Return sorted lists of (bold_img, events_df, confounds_df) for every run.
    """
    # Collect BOLD files and sort by numeric run index (run-1, run-2, ...)
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
        print(f"  Using first {len(bold_files)} runs (GLM_MAX_RUNS={MAX_RUNS})")

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

        if "wm_csf" in CONFOUND_STRATEGY:
            if WM_CSF_PARAMS == "basic":
                selected.extend(["white_matter", "csf"])
            else:
                selected.extend([
                    "white_matter", "white_matter_derivative1", "white_matter_power2", "white_matter_derivative1_power2",
                    "csf", "csf_derivative1", "csf_power2", "csf_derivative1_power2"
                ])

        selected = [column for column in selected if column in confounds_df.columns]
        if not selected:
            raise ValueError("No requested confound columns found in confounds TSV.")

        return confounds_df[selected].fillna(0.0)

    events_list, confounds_list, imgs = [], [], []
    for bold in bold_files:
        # Events file (from raw BIDS, not derivatives)
        run_label = [p for p in bold.stem.split("_") if p.startswith("run-")][0]
        events_path = (
            BIDS_DIR / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-listening_{run_label}_events.tsv"
        )
        events = pd.read_csv(events_path, sep="\t")
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
        confounds = _select_confounds(pd.read_csv(confounds_path, sep="\t"))

        imgs.append(str(bold))
        events_list.append(events)
        confounds_list.append(confounds)

    return imgs, events_list, confounds_list


def fit_first_level(subject: str):
    """
    Fit a single FirstLevelModel across ALL runs of one subject,
    then extract one beta map per word/condition.
    """
    print(f"\n{'='*60}")
    print(f"  sub-{subject}")
    print(f"{'='*60}")

    imgs, events_list, confounds_list = None, None, None
    model = None
    try:
        imgs, events_list, confounds_list = get_run_files(subject)
        print(f"  Found {len(imgs)} runs")

        # Brain mask for GLM fitting (nilearn will apply this mask to all runs)
        # Our mask is in the same space (MNI152) but 2mm resolution, while data is 1mm, so we rely on nilearn to handle resampling (to data) internally.
        mask_path = MASK_PATH
        if not Path(mask_path).exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask_img = nib.load(str(mask_path))

        target_affine = None
        target_shape = None
        if RESAMPLE_TO_MASK:
            target_affine = mask_img.affine
            target_shape = mask_img.shape[:3]

        # Fit the GLM across all runs together, with run-specific intercepts and confounds
        # Passing a list of imgs + a list of events DataFrames + a list of confounds
        # DataFrames causes nilearn to:
        #   1. Concatenate all runs into one long timeseries.
        #   2. Automatically add a separate constant (intercept) regressor for each
        #      run — this soaks up between-run baseline differences.
        #   3. Apply high-pass filtering WITHIN each run (cosine drift regressors
        #      are generated per-run).

        model = FirstLevelModel(
            t_r=T_R,
            slice_time_ref=SLICE_TIME_REF,
            hrf_model=HRF_MODEL,
            drift_model=DRIFT_MODEL,
            high_pass=HIGH_PASS,
            smoothing_fwhm=SMOOTHING_FWHM,
            mask_img=str(mask_path),
            noise_model=NOISE_MODEL,
            standardize=False,
            minimize_memory=MINIMIZE_MEMORY,
            n_jobs=N_JOBS,
            target_affine=target_affine,
            target_shape=target_shape,
            verbose=1,
        )

        model.fit(
            run_imgs=imgs,
            events=events_list,
            confounds=confounds_list,
        )

        # number of in-mask voxels used by the masker:
        inmask_vox = int((model.masker_.mask_img_.get_fdata() != 0).sum())
        print("in-mask voxels:", inmask_vox, flush=True)

        # Save the design matrix (first run) for this subject for documentation
        Path("reports/plots/design_matrix").mkdir(parents=True, exist_ok=True)
        fig_path = f"reports/plots/design_matrix/example_design_matrix_sub-{subject}_run-01.png"
        dm0 = model.design_matrices_[0]
        plotting.plot_design_matrix(dm0, output_file=fig_path)
        print(f"Saved example design matrix for first run to: {fig_path}", flush=True)

        # Extract beta maps for each condition (word) and save to disk
        # The design matrix columns named after stim_file values are the conditions
        # we want; everything else (confounds, drift, run intercepts) we skip.
        conditions = _get_condition_names(model)
        print(f"  Conditions ({len(conditions)}): {conditions[:5]} ...")

        out_sub = OUT_DIR / f"sub-{subject}"
        out_sub.mkdir(parents=True, exist_ok=True)

        for cond in conditions:
            contrast_vectors = _build_runwise_contrast_vectors(model, cond)
            beta_img = model.compute_contrast(
                contrast_def=contrast_vectors,
                stat_type="t",          # use 'effect_size' for raw betas (Cohen's d)
                output_type="effect_size",   # raw beta (more natural for RSA)
            )
            out_path = out_sub / f"beta_{cond}.nii.gz"
            nib.save(beta_img, out_path)
            del beta_img

        print(f"  Saved {len(conditions)} beta maps → {out_sub}")

        return conditions
    finally:
        del model, imgs, events_list, confounds_list
        gc.collect()


def _get_condition_names(model: FirstLevelModel):
    """
    Extract condition (trial_type) names from the fitted model's design matrices,
    excluding confound regressors, drift regressors, and run constants.
    """
    skip_prefixes = ("drift_", "constant", "trans_", "rot_", "framewise",
                     "std_dvars", "wm", "csf", "global_signal", "acomp", "white_matter")
    condition_names = set()
    for dm in model.design_matrices_:
        for col in dm.columns:
            if not any(col.lower().startswith(p) for p in skip_prefixes):
                condition_names.add(col)
    return sorted(condition_names)


def _build_runwise_contrast_vectors(model: FirstLevelModel, condition: str):
    """
    Build one numeric contrast vector per run design matrix.
    If a condition column is absent in a run, its run contrast is all zeros.
    """
    vectors = []
    for dm in model.design_matrices_:
        vector = np.zeros(len(dm.columns), dtype=float)
        if condition in dm.columns:
            vector[dm.columns.get_loc(condition)] = 1.0
        vectors.append(vector)
    return vectors

#######################################################################
# Main execution
#######################################################################
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_conditions = None
    first_successful_subject = None
    for sub in tqdm(SUBJECTS, desc="Fitting first-level models"):
        try:
            conditions = fit_first_level(sub) # driver of the whole process for one subject
            if all_conditions is None:
                all_conditions = conditions
            if first_successful_subject is None:
                first_successful_subject = sub
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