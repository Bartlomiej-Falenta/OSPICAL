"""
optomod_dashboard.py
====================
Interactive dashboard for the optomod SRS propagator.

Layout (single window):
  Row 0 – sliders (all parameters)
  Row 1 – [Propagation I_s(z) & I_p(z)]  [I_c linear sweep: gain A vs B]
  Row 2 – [Surface: I_out(I_s, I_c)]      [Surface: Gain(I_s, I_c)]

All plots update live when any slider moves.
I_p is also slider-controlled.

Authors : Kamil R. W. Kozak, Machlian Django-Kozak (optomod project)
          assisted by Claude (Anthropic), March 2026.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')          # change to 'Qt5Agg' if TkAgg unavailable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.colors import LogNorm
from dataclasses import dataclass


# ============================================================
# Physics core  (self-contained, no external imports needed)
# ============================================================

@dataclass
class P:
    """Parameter bundle — all floats."""
    g_R:     float = 5e-11
    L:       float = 5e-3       # interaction length [m]
    I_SAT:   float = 5e11
    beta:    float = 5e-14
    phi:     float = np.pi / 2
    eta:     float = 0.03
    alpha_s: float = 5.0
    alpha_p: float = 6.0
    lam_p:   float = 455e-9     # pump wavelength [m]
    lam_s:   float = 522e-9     # signal wavelength [m]
    n_z:     int   = 800        # ODE z-points


def mzi_T(I_c, beta, phi):
    return np.sin((beta * I_c + phi) / 2.0) ** 2


def model_A(I_s, I_p, I_c, p: P):
    T   = mzi_T(I_c, p.beta, p.phi)
    G   = p.g_R * p.L * (I_p * T + p.eta * I_c)
    Gsat = G / (1.0 + I_s / p.I_SAT)
    return I_s * np.exp(Gsat)


def _odes(z, y, g_eff, a_s, a_p, fr):
    Is, Ip = max(y[0], 0.0), max(y[1], 0.0)
    return [
         g_eff * Ip * Is - a_s * Is,
        -fr    * g_eff * Ip * Is - a_p * Ip,
    ]


def model_B(I_s0, I_p0, I_c, p: P):
    eps  = 1e-30
    T    = mzi_T(I_c, p.beta, p.phi)
    g_eff = p.g_R * (T + p.eta * I_c / max(I_p0, eps))
    fr   = (p.lam_s / p.lam_p)          # ω_p/ω_s = λ_s/λ_p
    sol  = solve_ivp(
        _odes, (0.0, p.L),
        [I_s0, I_p0],
        t_eval=np.linspace(0.0, p.L, p.n_z),
        args=(g_eff, p.alpha_s, p.alpha_p, fr),
        method='Radau', rtol=1e-7, atol=1e-11,
        dense_output=False,
    )
    return sol.t, sol.y[0], sol.y[1], g_eff


# ============================================================
# Compute all data for current parameters
# ============================================================

N_IC   = 18      # points in linear I_c sweep (propagation panel)
N_SURF = 40      # grid size for surface plots
I_C_MAX_DEFAULT = 1e15


def compute_all(p: P, I_s0, I_p0,
                I_c_max=I_C_MAX_DEFAULT,
                I_c_test=None):
    """Return dict with all data needed to update every panel."""

    # --- 1. Propagation at single I_c_test (Model B) ---
    if I_c_test is None:
        I_c_test = I_c_max / 2.0
    z, Is_z, Ip_z, g_eff = model_B(I_s0, I_p0, I_c_test, p)

    # --- 2. Linear I_c sweep ---
    I_c_lin = np.linspace(0.0, I_c_max, N_IC)
    gainA_lin = np.array([model_A(I_s0, I_p0, ic, p) / max(I_s0, 1e-30)
                          for ic in I_c_lin])
    gainB_lin = np.zeros(N_IC)
    for i, ic in enumerate(I_c_lin):
        _, Is_end, _, _ = model_B(I_s0, I_p0, ic, p)
        gainB_lin[i] = Is_end[-1] / max(I_s0, 1e-30)

    # --- 3. 2-D surfaces over (I_s, I_c) grid ---
    Is_vec = np.logspace(7, 13, N_SURF)
    Ic_vec = np.logspace(7, 15, N_SURF)
    IS, IC = np.meshgrid(Is_vec, Ic_vec)

    Iout_surf = np.vectorize(lambda s, c: model_A(s, I_p0, c, p))(IS, IC)
    Gain_surf = Iout_surf / np.maximum(IS, 1e-30)

    return dict(
        z=z, Is_z=Is_z, Ip_z=Ip_z, g_eff=g_eff, I_c_test=I_c_test,
        I_c_lin=I_c_lin, gainA_lin=gainA_lin, gainB_lin=gainB_lin,
        IS=IS, IC=IC, Iout_surf=Iout_surf, Gain_surf=Gain_surf,
        Is_vec=Is_vec, Ic_vec=Ic_vec,
    )


# ============================================================
# Build figure
# ============================================================

def build_dashboard():
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('#0f0f1a')
    plt.rcParams.update({
        'text.color': 'white', 'axes.labelcolor': 'white',
        'xtick.color': 'white', 'ytick.color': 'white',
        'axes.edgecolor': '#555577', 'grid.color': '#333355',
        'axes.facecolor': '#12122a', 'figure.facecolor': '#0f0f1a',
    })

    # ---- outer grid: sliders top, plots bottom ----
    outer = gridspec.GridSpec(2, 1, figure=fig,
                              height_ratios=[1, 2.8],
                              hspace=0.35, left=0.06, right=0.97,
                              top=0.96, bottom=0.04)

    slider_area = outer[0]          # top band
    plot_area   = gridspec.GridSpecFromSubplotSpec(
                      2, 2, subplot_spec=outer[1],
                      hspace=0.42, wspace=0.35)

    ax_prop  = fig.add_subplot(plot_area[0, 0])   # propagation
    ax_sweep = fig.add_subplot(plot_area[0, 1])   # linear Ic sweep
    ax_surf1 = fig.add_subplot(plot_area[1, 0], projection='3d')  # I_out surface
    ax_surf2 = fig.add_subplot(plot_area[1, 1], projection='3d')  # Gain surface

    for ax in [ax_prop, ax_sweep]:
        ax.set_facecolor('#12122a')
        ax.grid(True, alpha=0.25)

    # ---- Sliders ----
    # We create a sub-axes grid just for sliders
    sl_gs = gridspec.GridSpecFromSubplotSpec(
                3, 5, subplot_spec=slider_area,
                hspace=0.15, wspace=0.37)

    def make_slider(row, col, label, vmin, vmax, vinit, log=False):
        ax_sl = fig.add_subplot(sl_gs[row, col])
        ax_sl.set_facecolor('#1a1a2e')
        if log:
            sl = Slider(ax_sl, label, np.log10(vmin), np.log10(vmax),
                        valinit=np.log10(vinit), color='#5566cc',
                        handle_style={'facecolor': '#aabbff', 'size': 8})
            sl._log = True
        else:
            sl = Slider(ax_sl, label, vmin, vmax, valinit=vinit,
                        color='#5566cc',
                        handle_style={'facecolor': '#aabbff', 'size': 8})
            sl._log = False
        ax_sl.tick_params(colors='#8899cc', labelsize=6)
        ax_sl.xaxis.label.set_color('#aabbff')
        ax_sl.xaxis.label.set_size(6)
        return sl

    sl_gR    = make_slider(0, 0, 'log g_R [m/W]',  1e-13, 1e-8,  5e-11, log=True)
    sl_L     = make_slider(0, 1, 'L [mm]',          0.1,   20.0,  5.0)
    sl_ISAT  = make_slider(0, 2, 'log I_SAT',       1e9,   1e14,  5e11, log=True)
    sl_beta  = make_slider(0, 3, 'log β [m²/W]',   1e-16, 1e-11, 5e-14, log=True)
    sl_phi   = make_slider(0, 4, 'φ/π',            0.0,   2.0,   0.5)

    sl_eta   = make_slider(1, 0, 'η (overlap)',    0.0,   1.0,   0.03)
    sl_alpS  = make_slider(1, 1, 'α_s [1/m]',      0.0,   100.0, 5.0)
    sl_alpP  = make_slider(1, 2, 'α_p [1/m]',      0.0,   100.0, 6.0)
    sl_lamP  = make_slider(1, 3, 'λ_p [nm]',       200,   2000,  455)
    sl_lamS  = make_slider(1, 4, 'λ_s [nm]',       200,   2000,  522)

    sl_Is0   = make_slider(2, 0, 'log I_s0',       1e6,   1e13,  1e9,  log=True)
    sl_Ip0   = make_slider(2, 1, 'log I_p0',       1e9,   1e14,  5e11, log=True)
    sl_IcMax = make_slider(2, 2, 'log I_c_max',    1e9,   1e16,  1e15, log=True)
    sl_IcTst = make_slider(2, 3, 'log I_c (prop)', 1e9,   1e16,  5e14, log=True)
    # empty slot [2,4] — could be used later

    def get_val(sl):
        return 10 ** sl.val if sl._log else sl.val

    # ---- Initial compute ----
    p0   = P()
    I_s0 = 1e9
    I_p0 = 5e11
    d    = compute_all(p0, I_s0, I_p0)

    # ---- Plot helpers ----
    CYAN   = '#00e5ff'
    ORANGE = '#ff9100'
    GREEN  = '#69ff47'
    RED    = '#ff4444'

    # -- Propagation --
    lp_Is, = ax_prop.semilogy(d['z']*1e3, d['Is_z'], color=CYAN,  lw=2,
                               label='Signal $I_s(z)$')
    lp_Ip, = ax_prop.semilogy(d['z']*1e3, d['Ip_z'], color=ORANGE, lw=2,
                               ls='--', label='Pump $I_p(z)$')
    ax_prop.set_xlabel('z [mm]')
    ax_prop.set_ylabel('Intensity [W/m²]')
    ax_prop.set_title('Propagation profile (Model B)', color='white', pad=4)
    ax_prop.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    prop_title = ax_prop.title

    # -- Linear Ic sweep --
    ls_A, = ax_sweep.plot(d['I_c_lin'], d['gainA_lin'],
                           color='royalblue', lw=2, label='Model A')
    ls_B, = ax_sweep.plot(d['I_c_lin'], d['gainB_lin'],
                           color=GREEN, lw=2, ls='--', label='Model B')
    ax_sweep.axhline(2.0, color=RED, ls='-.', lw=1.8, label='Gain = 2')
    ax_sweep.set_xlabel('$I_c$ [W/m²]')
    ax_sweep.set_ylabel('Gain')
    ax_sweep.set_title('Gain vs $I_c$ (linear sweep)', color='white', pad=4)
    ax_sweep.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    # -- Surfaces --
    surf1_obj = [None]
    surf2_obj = [None]

    def draw_surfaces(d):
        # clear and redraw 3D axes
        for ax3d in [ax_surf1, ax_surf2]:
            ax3d.cla()
            ax3d.set_facecolor('#12122a')
            ax3d.tick_params(colors='white', labelsize=6)
            ax3d.xaxis.label.set_color('white')
            ax3d.yaxis.label.set_color('white')
            ax3d.zaxis.label.set_color('white')

        logIS = np.log10(d['IS'])
        logIC = np.log10(d['IC'])

        # I_out surface
        Iout_safe = np.clip(d['Iout_surf'], 1e-30, None)
        ax_surf1.plot_surface(logIS, logIC, np.log10(Iout_safe),
                              cmap='plasma', alpha=0.85, linewidth=0,
                              antialiased=True)
        ax_surf1.set_xlabel('log $I_s$ [W/m²]', labelpad=2)
        ax_surf1.set_ylabel('log $I_c$ [W/m²]', labelpad=2)
        ax_surf1.set_zlabel('log $I_{out}$',     labelpad=2)
        ax_surf1.set_title('$I_{out}(I_s, I_c)$ — Model A',
                            color='white', pad=2, fontsize=9)

        # Gain surface
        G_safe = np.clip(d['Gain_surf'], 1e-6, None)
        ax_surf2.plot_surface(logIS, logIC, np.log10(G_safe),
                              cmap='viridis', alpha=0.85, linewidth=0,
                              antialiased=True)
        # mark gain=2 plane
        ax_surf2.plot_surface(logIS, logIC,
                              np.full_like(G_safe, np.log10(2.0)),
                              alpha=0.18, color='red')
        ax_surf2.set_xlabel('log $I_s$ [W/m²]', labelpad=2)
        ax_surf2.set_ylabel('log $I_c$ [W/m²]', labelpad=2)
        ax_surf2.set_zlabel('log Gain',          labelpad=2)
        ax_surf2.set_title('Gain$(I_s, I_c)$ — Model A  [red plane = gain 2]',
                            color='white', pad=2, fontsize=9)

    draw_surfaces(d)

    # ---- Update callback ----
    def update(_):
        p = P(
            g_R     = get_val(sl_gR),
            L       = get_val(sl_L) * 1e-3,      # mm → m
            I_SAT   = get_val(sl_ISAT),
            beta    = get_val(sl_beta),
            phi     = get_val(sl_phi) * np.pi,
            eta     = get_val(sl_eta),
            alpha_s = get_val(sl_alpS),
            alpha_p = get_val(sl_alpP),
            lam_p   = get_val(sl_lamP) * 1e-9,   # nm → m
            lam_s   = get_val(sl_lamS) * 1e-9,
            n_z     = 800,
        )
        I_s0   = get_val(sl_Is0)
        I_p0   = get_val(sl_Ip0)
        I_c_max = get_val(sl_IcMax)
        I_c_test = get_val(sl_IcTst)

        try:
            d = compute_all(p, I_s0, I_p0,
                            I_c_max=I_c_max,
                            I_c_test=I_c_test)
        except Exception as e:
            print(f"[compute error] {e}")
            return

        # propagation
        lp_Is.set_xdata(d['z'] * 1e3)
        lp_Is.set_ydata(np.maximum(d['Is_z'], 1e-30))
        lp_Ip.set_xdata(d['z'] * 1e3)
        lp_Ip.set_ydata(np.maximum(d['Ip_z'], 1e-30))
        ax_prop.relim(); ax_prop.autoscale_view()
        gain_end = d['Is_z'][-1] / max(I_s0, 1e-30)
        ax_prop.set_title(
            f'Propagation (Model B) — gain={gain_end:.3f}  '
            f'g_eff={d["g_eff"]:.2e}',
            color='white', pad=4, fontsize=8)

        # linear sweep
        ls_A.set_xdata(d['I_c_lin'])
        ls_A.set_ydata(d['gainA_lin'])
        ls_B.set_xdata(d['I_c_lin'])
        ls_B.set_ydata(d['gainB_lin'])
        ax_sweep.relim(); ax_sweep.autoscale_view()

        # surfaces
        draw_surfaces(d)

        fig.canvas.draw_idle()

    for sl in [sl_gR, sl_L, sl_ISAT, sl_beta, sl_phi,
               sl_eta, sl_alpS, sl_alpP, sl_lamP, sl_lamS,
               sl_Is0, sl_Ip0, sl_IcMax, sl_IcTst]:
        sl.on_changed(update)

    # ---- Reset button ----
    ax_reset = fig.add_axes([0.92, 0.005, 0.06, 0.025])
    ax_reset.set_facecolor('#1a1a2e')
    btn_reset = Button(ax_reset, 'Reset',
                       color='#1a1a2e', hovercolor='#3344aa')
    btn_reset.label.set_color('white')

    def reset(_):
        for sl, v in zip(
            [sl_gR, sl_L, sl_ISAT, sl_beta, sl_phi,
             sl_eta, sl_alpS, sl_alpP, sl_lamP, sl_lamS,
             sl_Is0, sl_Ip0, sl_IcMax, sl_IcTst],
            [5e-11, 5.0, 5e11, 5e-14, 0.5,
             0.03, 5.0, 6.0, 455, 522,
             1e9, 5e11, 1e15, 5e14]
        ):
            if sl._log:
                sl.set_val(np.log10(v))
            else:
                sl.set_val(v)

    btn_reset.on_clicked(reset)

    fig.text(0.5, 0.995,
             'Optomod SRS Dashboard  —  K.R.W. Kozak & M. Django-Kozak',
             ha='center', va='top', color='#8899dd', fontsize=10)

    plt.show()


if __name__ == '__main__':
    build_dashboard()
