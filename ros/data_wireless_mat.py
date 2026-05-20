# data-wireless-mat.py
# Create a synthetic wireless dataset and save in .mat format for Julia
# Compatible with the Julia loading function structure

import numpy as np
import os
from scipy.io import savemat

def percentiles(x, ps=(1, 10, 50, 90, 99)):
    x = np.asarray(x).ravel()
    return {p: np.percentile(x, p) for p in ps}

def pstr(dct, fmt="{: .3e}"):
    return ", ".join(f"{int(p)}%={fmt.format(v)}" for p, v in dct.items())

def uniform_in_disk(n, R, rng):
    u = rng.random(n)
    r = R * np.sqrt(u)
    th = 2.0 * np.pi * rng.random(n)
    return r * np.cos(th), r * np.sin(th)

def pairwise_distance(ax, ay, bx, by):
    dx = ax[:, None] - bx[None, :]
    dy = ay[:, None] - by[None, :]
    return np.hypot(dx, dy)

def link_gain_linear(d_m, fc_Hz, alpha, sigma_sh_dB, rng, d0_m=1.0):
    # Path gain (linear) with lognormal shadowing.
    # PL_dB(d) = FSPL_1m_dB + 10*alpha*log10(max(d, d0)) - X_sh
    # gain = 10^(-PL_dB/10)
    c = 299_792_458.0
    lam = c / fc_Hz
    FSPL_1m_dB = 20.0 * np.log10(4.0 * np.pi / lam)
    d_eff = np.maximum(d_m, d0_m)
    Xsh = rng.normal(0.0, sigma_sh_dB, size=d_eff.shape)
    PL_dB = FSPL_1m_dB + 10.0 * alpha * np.log10(d_eff) - Xsh
    return 10.0 ** (-PL_dB / 10.0)

def worst_case_I_from_C(C, Umax, umax, s_sparse):
    """
    Maximize u^T C under ||u||_2 <= Umax, |u_j| <= umax, ||u||_0 <= s.
    C: (mUE, nJ) >= 0, returns (mUE,)
    Closed-form for nonnegative C: saturate top-s at umax until L2 budget used.
    """
    mUE, nJ = C.shape
    if s_sparse <= 0 or Umax <= 0 or umax <= 0 or nJ == 0:
        return np.zeros(mUE, dtype=C.dtype)

    # sort C per-UE descending
    C_sorted = np.sort(C, axis=1)[:, ::-1]
    # number of fully saturated entries at umax allowed by l2 budget
    t_full = int(np.floor((Umax / umax) ** 2))
    t_full = max(0, min(s_sparse, t_full))
    # fully saturated contribution
    I = umax * (C_sorted[:, :t_full].sum(axis=1) if t_full > 0 else 0.0)

    rem = Umax**2 - t_full * umax**2
    if t_full < s_sparse and rem > 1e-14:
        amp = min(umax, np.sqrt(rem))
        I += amp * C_sorted[:, t_full]
    return I

def summarize(title, arr, ps=(1, 10, 50, 90, 99), fmt="{: .3e}"):
    print(f"{title}: " + pstr(percentiles(arr, ps), fmt=fmt))

def generate_instance(seed, nBS, nJ, mUE, mat_dir="instances"):
    """Generate a single instance with given parameters."""
    rng = np.random.default_rng(seed)

    # -------------------------
    # Configuration
    # -------------------------
    cfg = {
        "seed": seed,
        "R_m": 1000.0,      # cell radius [m]
        "nBS": nBS,         # number of base stations (n_BS in Julia)
        "nJ": nJ,           # number of jammers (n_J in Julia)
        "mUE": mUE,         # number of user devices (m_UE in Julia)
        "fc_GHz": 2.0,      # carrier [GHz]
        "B_Hz": 10e6,       # bandwidth [Hz]
        "NF_dB": 7.0,       # noise figure [dB]
        # Pathloss and shadowing
        "alpha_BU": 3.5,    # BS->UE exponent
        "alpha_BJ": 2.2,    # BS->J exponent (LOS-leaning)
        "alpha_JU": 2.0,    # J->UE exponent (LOS-leaning)
        "sigma_sh_BU_dB": 8.0,
        "sigma_sh_BJ_dB": 3.0,
        "sigma_sh_JU_dB": 3.0,
        # Power and targets
        "gamma_linear": 0.2,          # SINR target (linear) = ~7 dB (reduced from 10.0)
        "p_max_W": 150.0,             # maximum power per BS [W] (increased from 40.0)
        "p_test_W": 50.0,             # equal-power per BS [W] used for diagnostics (increased)
        "target_IoverN_dB": 2.0,     # desired median worst-case I/N in dB (reduced from 15.0)
        # Jammer uncertainty set
        "s_sparse": 3,                # at most s active jammers
        "Umax": 2.0,                #2 l2 bound
        "umax": 1.6,                 #1 l_inf bound
        # Clipping
        "clip_g_quantile": 0.999,     # clip extreme g before/after calibration
        # Rescale for readability (keeps SINR unchanged)
        "rescale_noise_median_to": 0.1,  # Reduced from 1.0 to make noise smaller relative to signal
        "mat_dir": mat_dir,
    }

    fc_Hz = cfg["fc_GHz"] * 1e9

    # -------------------------
    # Geometry
    # -------------------------
    bs_x, bs_y = uniform_in_disk(cfg["nBS"], cfg["R_m"], rng)
    ue_x, ue_y = uniform_in_disk(cfg["mUE"], cfg["R_m"], rng)
    jm_x, jm_y = uniform_in_disk(cfg["nJ"],  cfg["R_m"], rng)

    d_BU = pairwise_distance(ue_x, ue_y, bs_x, bs_y)      # (mUE, nBS)
    d_BJ = pairwise_distance(bs_x, bs_y, jm_x, jm_y)      # (nBS, nJ)
    d_JU = pairwise_distance(ue_x, ue_y, jm_x, jm_y)      # (mUE, nJ)

    # -------------------------
    # Link gains (linear)
    # -------------------------
    h = link_gain_linear(d_BU, fc_Hz, cfg["alpha_BU"], cfg["sigma_sh_BU_dB"], rng)  # (mUE, nBS)
    h_BJ = link_gain_linear(d_BJ, fc_Hz, cfg["alpha_BJ"], cfg["sigma_sh_BJ_dB"], rng)  # (nBS, nJ)
    h_JU = link_gain_linear(d_JU, fc_Hz, cfg["alpha_JU"], cfg["sigma_sh_JU_dB"], rng)  # (mUE, nJ)

    # Reactive jammer cascade (before global amplification)
    g0 = h_JU[:, None, :] * h_BJ[None, :, :]  # (mUE, nBS, nJ), nonnegative

    # -------------------------
    # Noise per UE
    # -------------------------
    kB = 1.380649e-23
    T = 290.0
    N0 = kB * T * cfg["B_Hz"] * 10.0 ** (cfg["NF_dB"] / 10.0)  # baseline noise power
    # Small per-UE NF variations (lognormal around 1)
    sigma2 = N0 * rng.lognormal(mean=0.0, sigma=0.2, size=cfg["mUE"])

    # -------------------------
    # Calibrate jammer gain to hit target median I/N (equal-power, given s-sparse),
    # honoring clipping consistently.
    # -------------------------
    p_vec = np.full(cfg["nBS"], cfg["p_test_W"])
    target_ratio = 10.0 ** (cfg["target_IoverN_dB"] / 10.0)

    # Clip threshold computed on g0 first
    if cfg["clip_g_quantile"] < 1.0:
        clip_thr0 = np.quantile(g0, cfg["clip_g_quantile"])
        g0_cal = np.minimum(g0, clip_thr0)
    else:
        clip_thr0 = None
        g0_cal = g0

    # Base C and worst-case I (with Gjam=1) for calibration
    C0 = np.tensordot(g0_cal, p_vec, axes=([1], [0]))  # (mUE, nJ)
    I_base = worst_case_I_from_C(C0, cfg["Umax"], cfg["umax"], cfg["s_sparse"])
    base_ratio = np.median(I_base / sigma2)
    Gjam_lin = target_ratio / base_ratio if base_ratio > 0 else 1.0

    # Apply calibrated amplification and clip consistently
    g = Gjam_lin * g0
    if clip_thr0 is not None:
        g = np.minimum(g, Gjam_lin * clip_thr0)

    # Achieved severity (pre-rescale)
    C = np.tensordot(g, p_vec, axes=([1], [0]))  # (mUE, nJ)
    I_rob_eq = worst_case_I_from_C(C, cfg["Umax"], cfg["umax"], cfg["s_sparse"])
    achieved_ratio_dB = 10.0 * np.log10(np.median(I_rob_eq / sigma2))

    # -------------------------
    # Global rescale so median noise = 1 (readability only)
    # -------------------------
    s = cfg["rescale_noise_median_to"] / np.median(sigma2)
    sigma2 *= s
    h *= s
    g *= s
    # Recompute with scaled values
    S_eq = h @ p_vec
    C_eq = np.tensordot(g, p_vec, axes=([1], [0]))  # (mUE, nJ)
    I_rob_eq = worst_case_I_from_C(C_eq, cfg["Umax"], cfg["umax"], cfg["s_sparse"])

    # -------------------------
    # Margins (nominal vs robust)
    # margin = S / (gamma * (I + N)) - 1
    # -------------------------
    gamma = cfg["gamma_linear"]
    margin_nom = S_eq / (gamma * sigma2) - 1.0
    margin_rob = S_eq / (gamma * (I_rob_eq + sigma2)) - 1.0
    
    # -------------------------
    # Feasibility Check for Nominal Problem
    # -------------------------
    # Check if nominal problem is feasible with equal power
    feasibility_check_passed = np.all(margin_nom > -0.1)  # Allow small negative tolerance
    if not feasibility_check_passed:
        print("\n⚠️ WARNING: Nominal problem may be infeasible!")
        print(f"   Fraction of UEs with negative nominal margin: {np.mean(margin_nom < 0):.3f}")
        print(f"   Minimum nominal margin: {np.min(margin_nom):.3e}")
        print("   Consider:")
        print("   - Reducing gamma_linear (SINR target)")
        print("   - Increasing p_max_W (power budget)")
        print("   - Adjusting pathloss exponents")

    # -------------------------
    # Reporting
    # -------------------------
    print("=== Configuration (key knobs) ===")
    print(f"seed: {cfg['seed']}")
    print(f"R_m: {cfg['R_m']}")
    print(f"nBS: {cfg['nBS']}")
    print(f"nJ: {cfg['nJ']}")
    print(f"mUE: {cfg['mUE']}")
    print(f"fc_GHz: {cfg['fc_GHz']}")
    print(f"B_Hz: {cfg['B_Hz']:.0f}")
    print(f"NF_dB: {cfg['NF_dB']}")
    print(f"alpha_BU: {cfg['alpha_BU']}, alpha_BJ: {cfg['alpha_BJ']}, alpha_JU: {cfg['alpha_JU']}")
    print(f"sigma_sh_BU_dB: {cfg['sigma_sh_BU_dB']}, sigma_sh_BJ_dB: {cfg['sigma_sh_BJ_dB']}, sigma_sh_JU_dB: {cfg['sigma_sh_JU_dB']}")
    print(f"gamma (linear): {gamma}")
    print(f"p_max_W: {cfg['p_max_W']}")
    print(f"p_test_W: {cfg['p_test_W']}")
    print(f"s_sparse: {cfg['s_sparse']}, Umax: {cfg['Umax']}, umax: {cfg['umax']}")
    print(f"clip_g_quantile: {cfg['clip_g_quantile']}")
    print(f"target_IoverN_dB: {cfg['target_IoverN_dB']}")
    print("Units after rescale are normalized; SINR and feasibility are unchanged by the rescale.")
    print(f"Calibrated jammer gain: Gjam_lin ≈ {Gjam_lin:.3e} (≈ {10*np.log10(Gjam_lin):.2f} dB)")
    print(f"Achieved median I_rob/noise (equal-power, s={cfg['s_sparse']}) ≈ {achieved_ratio_dB:.2f} dB")

    # Geometry stats
    print("\n=== Geometry stats ===")
    summarize("UE-to-nearest-BS distance [m]", d_BU.min(axis=1), fmt="{: .3e}")
    summarize("Jammer-to-nearest-BS distance [m]", d_BJ.min(axis=0), fmt="{: .3e}")
    summarize("UE-to-nearest-jammer distance [m]", d_JU.min(axis=1), fmt="{: .3e}")

    # Gain stats
    print("\n=== Gain stats (after rescale; linear) ===")
    summarize("h[k,i] (all pairs)", h, fmt="{: .3e}")
    summarize("max_i h[k,i] (best BS per UE)", h.max(axis=1), fmt="{: .3e}")
    summarize("g[k,i,j] (all triples)", g, fmt="{: .3e}")
    summarize("max_j g[k,i,j] / h[k,i]", (g.max(axis=2) / np.maximum(h, 1e-30)), fmt="{: .3e}")

    # Noise stats
    print("\n=== Noise stats (after rescale) ===")
    summarize("sigma2[k]", sigma2, fmt="{: .3e}")

    # Equal-power diagnostics: nominal vs robust
    print("\n=== Nominal vs Robust (equal-power) ===")
    summarize("Signal S_eq [linear]", S_eq, fmt="{: .3e}")
    summarize("Worst-case interference I_rob_eq [linear]", I_rob_eq, fmt="{: .3e}")
    summarize("I_rob_eq / noise [linear]", I_rob_eq / sigma2, fmt="{: .3e}")
    frac_nom = np.mean(margin_nom > 0.0)
    frac_rob = np.mean(margin_rob > 0.0)
    print(f"Frac UEs with nominal margin > 0: {frac_nom:.3f}")
    print(f"Frac UEs with robust  margin > 0: {frac_rob:.3f}")
    summarize("Nominal margin", margin_nom, fmt="{: .3e}")
    summarize("Robust  margin", margin_rob, fmt="{: .3e}")

    # -------------------------
    # Save to .mat file (compatible with Julia)
    # -------------------------
    os.makedirs(cfg["mat_dir"], exist_ok=True)
    
    # Build filename matching Julia's expected format: RJ_nBS_nJ_mUE_seedX.mat
    mat_filename = f"RJ_{cfg['nBS']}BS_{cfg['nJ']}J_{cfg['mUE']}UE_seed{cfg['seed']}.mat"
    mat_path = os.path.join(cfg["mat_dir"], mat_filename)
    
    # Prepare data dictionary for .mat file
    # Julia expects: gains_h (m_UE x n_BS), gains_g (m_UE x n_BS x n_J), noise, gamma, p_max
    mat_data = {
        'gains_h': h,           # (mUE, nBS) - matches Julia expectation
        'gains_g': g,           # (mUE, nBS, nJ) - matches Julia expectation
        'noise': sigma2,        # (mUE,) - noise vector
        'gamma': gamma,         # scalar SINR target
        'p_max': cfg['p_max_W'],  # scalar maximum power per BS
        # Additional metadata
        'seed': cfg['seed'],
        's_sparse': cfg['s_sparse'],
        'Umax': cfg['Umax'],
        'umax': cfg['umax'],
        'n_BS': cfg['nBS'],
        'n_J': cfg['nJ'],
        'm_UE': cfg['mUE'],
    }
    
    savemat(mat_path, mat_data)
    print(f"\nSaved .mat file to: {mat_path}")
    print(f"Format: RJ_{{n_BS}}BS_{{n_J}}J_{{m_UE}}UE_seed{{seed}}.mat")
    print("\nData fields in .mat file:")
    print("  - gains_h: (m_UE x n_BS) channel gains")
    print("  - gains_g: (m_UE x n_BS x n_J) reactive jammer gains")
    print("  - noise: (m_UE,) noise power per UE")
    print("  - gamma: SINR target (linear)")
    print("  - p_max: maximum power per BS [W]")
    
    return mat_path


def main():
    """Generate multiple instances for the wireless problem configurations."""
    
    # Define problem configurations matching Julia code
    WIRELESS_CONFIGS = [
        {"n_BS": 5, "n_J": 20, "m_UE": 50},
        {"n_BS": 10, "n_J": 50, "m_UE": 200},
        {"n_BS": 20, "n_J": 100, "m_UE": 300}
    ]
    
    # Seeds matching Julia code
    SEEDS = [1, 2, 3, 4, 5]
    
    mat_dir = "instances"
    os.makedirs(mat_dir, exist_ok=True)
    
    print("="*80)
    print("🚀 GENERATING WIRELESS INSTANCES FOR ALL CONFIGURATIONS")
    print("="*80)
    print(f"📁 Output directory: {mat_dir}/")
    print(f"🌱 Seeds: {SEEDS}")
    print(f"📊 Configurations: {len(WIRELESS_CONFIGS)}")
    print("="*80)
    
    total_instances = 0
    
    for config_idx, config in enumerate(WIRELESS_CONFIGS, 1):
        n_BS = config["n_BS"]
        n_J = config["n_J"]
        m_UE = config["m_UE"]
        
        print(f"\n{'='*80}")
        print(f"📏 CONFIGURATION {config_idx}/{len(WIRELESS_CONFIGS)}: n_BS={n_BS} | n_J={n_J} | m_UE={m_UE}")
        print(f"{'='*80}")
        
        for seed in SEEDS:
            print(f"\n🌱 Generating instance with seed={seed}...")
            mat_path = generate_instance(seed, n_BS, n_J, m_UE, mat_dir)
            total_instances += 1
            print(f"✅ Generated: {os.path.basename(mat_path)}")
    
    print(f"\n{'='*80}")
    print(f"🎉 GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"📊 Total instances generated: {total_instances}")
    print(f"📁 All files saved in: {mat_dir}/")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()