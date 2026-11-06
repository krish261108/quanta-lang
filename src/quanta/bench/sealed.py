"""Sealed law classes for the novel-task suite (family D).

These classes were written blind: the author did not see the system under test,
its hypothesis machinery or its experiments. Each class hides a law y = f(x)
that is not one of the benchmark's familiar, gap or compose laws, and most
sampled parameter vectors cannot be imitated closely by any of them.

API
---
    CLASSES                      name -> {"description": str, "n_params": int}
    sample_params(name, rng)     draw a parameter vector (uses only `rng`)
    truth(name, params, x)       noise-free law value at x

Every law is finite on [0.1, 13] and keeps its characteristic structure inside
the measurable domain [0.1, 10]. Behaviour on [10, 13] follows from the law
itself, so extrapolation rewards understanding rather than curve description.

PROTECTED: part of the grader; the system under test must never read it.
"""
import math
import random

_TAU = 2.0 * math.pi


def _sign(rng: random.Random) -> float:
    return 1.0 if rng.random() < 0.5 else -1.0


# ----------------------------------------------------------------- special functions
# Bessel J0 from its integral representation J0(z) = (1/pi) * int_0^pi cos(z sin t) dt.
# The midpoint rule on a periodic analytic integrand converges geometrically; 128 nodes
# (32 after symmetry) give ~1e-15 absolute accuracy for |z| <= 60.
_J0_NODES = 128
_J0_SINES = tuple(math.sin(_TAU * (j + 0.5) / _J0_NODES) for j in range(_J0_NODES // 4))


def _bessel_j0(z: float) -> float:
    return 4.0 / _J0_NODES * math.fsum(math.cos(z * s) for s in _J0_SINES)


# Airy function Ai: Maclaurin series for moderate |z|, asymptotic expansions beyond.
_AI0 = 0.355028053887817239     # Ai(0)
_AIP0 = 0.258819403792806798    # -Ai'(0)
_AIRY_U = [1.0]
for _k in range(1, 40):
    _AIRY_U.append(_AIRY_U[-1] * (6 * _k - 5) * (6 * _k - 3) * (6 * _k - 1) / ((2 * _k - 1) * 216.0 * _k))
_AIRY_U = tuple(_AIRY_U)
_AI_MAX = 0.5356566560156999    # max Ai(z), attained at z = -1.0188


def _airy_series(z: float) -> float:
    z3 = z * z * z
    t, s = 1.0, z
    f, g = 1.0, z
    for k in range(1, 400):
        t *= z3 / ((3 * k - 1) * (3 * k))
        s *= z3 / ((3 * k + 1) * (3 * k))
        f += t
        g += s
        if k > 3 and abs(t) <= 1e-18 * max(1.0, abs(f)) and abs(s) <= 1e-18 * max(1.0, abs(g)):
            break
    return _AI0 * f - _AIP0 * g


def _airy_ai(z: float) -> float:
    if -7.0 <= z <= 6.0:
        return _airy_series(z)
    a = abs(z)
    zeta = 2.0 / 3.0 * a ** 1.5
    if z > 0:                       # exponentially small side
        total, prev = 0.0, math.inf
        for k, u in enumerate(_AIRY_U):
            term = u / zeta ** k
            if term > prev:
                break
            prev = term
            total += term if k % 2 == 0 else -term
        return math.exp(-zeta) * total / (2.0 * math.sqrt(math.pi) * a ** 0.25)
    p_sum = q_sum = 0.0             # oscillatory side
    prev = math.inf
    for k, u in enumerate(_AIRY_U):
        term = u / zeta ** k
        if term > prev:
            break
        prev = term
        signed = term if (k // 2) % 2 == 0 else -term
        if k % 2 == 0:
            p_sum += signed
        else:
            q_sum += signed
    phase = zeta - math.pi / 4.0
    return (math.cos(phase) * p_sum + math.sin(phase) * q_sum) / (math.sqrt(math.pi) * a ** 0.25)


def _hermite(n: int, xi: float) -> float:
    """Physicists' Hermite polynomial H_n(xi) by its three-term recurrence."""
    h_prev, h = 1.0, 2.0 * xi
    if n == 0:
        return h_prev
    for k in range(1, n):
        h_prev, h = h, 2.0 * xi * h - 2.0 * k * h_prev
    return h


def _ho_shape(n: int, xi: float) -> float:
    h = _hermite(n, xi)
    return h * h * math.exp(-xi * xi)


def _ho_peak(n: int) -> float:
    """Maximum of H_n(xi)^2 exp(-xi^2): coarse scan, then golden-section refinement."""
    step = 0.0025
    i_best = max(range(2001), key=lambda i: _ho_shape(n, i * step))
    lo, hi = max(0.0, (i_best - 1) * step), (i_best + 1) * step
    r = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(80):
        m1, m2 = hi - r * (hi - lo), lo + r * (hi - lo)
        if _ho_shape(n, m1) < _ho_shape(n, m2):
            lo = m1
        else:
            hi = m2
    return _ho_shape(n, 0.5 * (lo + hi))


_HO_LEVELS = (3, 4, 5, 6)
_HO_PEAK = {n: _ho_peak(n) for n in _HO_LEVELS}


def _kepler_eccentric_anomaly(mean_anomaly: float, e: float) -> float:
    """Solve E - e sin E = M by safeguarded Newton iteration (bracket [-pi, pi])."""
    m = math.remainder(mean_anomaly, _TAU)
    lo, hi = -math.pi, math.pi
    big_e = m + 0.85 * e * (1.0 if math.sin(m) >= 0.0 else -1.0)
    big_e = min(max(big_e, lo), hi)
    for _ in range(100):
        f = big_e - e * math.sin(big_e) - m
        if f > 0.0:
            hi = big_e
        else:
            lo = big_e
        nxt = big_e - f / (1.0 - e * math.cos(big_e))
        if not lo <= nxt <= hi:
            nxt = 0.5 * (lo + hi)
        if abs(nxt - big_e) < 1e-14:
            return nxt
        big_e = nxt
    return big_e


# ----------------------------------------------------------------- laws
def _two_tone_beats(p, x):
    offset, amp, ratio, w1, dw, ph1, ph2 = p
    return offset + amp * (math.sin(w1 * x + ph1) + ratio * math.sin((w1 + dw) * x + ph2))


def _linear_chirp(p, x):
    offset, amp, w0, rate, ph = p
    return offset + amp * math.sin(ph + w0 * x + 0.5 * rate * x * x)


def _multi_slit(p, x):
    background, peak, k, phase, slits = p
    n = int(round(slits))
    half = 0.5 * (k * x + phase)
    s = math.sin(half)
    if abs(s) < 1e-12:
        intensity = 1.0
    else:
        intensity = (math.sin(n * half) / (n * s)) ** 2
    return background + peak * intensity


def _fano(p, x):
    background, amp, x0, width, q = p
    eps = (x - x0) / width
    return background + amp * (q + eps) ** 2 / ((1.0 + q * q) * (1.0 + eps * eps))


def _inverse_field_oscillation(p, x):
    offset, amp, damping, freq, ph = p
    return offset + amp * math.exp(-damping / x) * math.cos(_TAU * freq / x + ph)


def _quantum_wire_dos(p, x):
    background, scale, e_first, e_unit, gamma = p
    n_bands = max(1, int(math.floor(math.sqrt((25.0 - e_first) / e_unit + 1.0))))
    total = 0.0
    for n in range(1, n_bands + 1):
        u = x - (e_first + e_unit * (n * n - 1))
        r = math.hypot(u, gamma)
        total += math.sqrt(0.5 * (r + u)) / r   # Re (u + i gamma)^(-1/2)
    return background + scale * math.sqrt(gamma) * total


def _kepler_radial_velocity(p, x):
    v_sys, semi_amp, period, ecc, omega, phase = p
    big_e = _kepler_eccentric_anomaly(_TAU * (x / period - phase), ecc)
    nu = 2.0 * math.atan2(math.sqrt(1.0 + ecc) * math.sin(0.5 * big_e),
                          math.sqrt(1.0 - ecc) * math.cos(0.5 * big_e))
    return v_sys + semi_amp * (math.cos(nu + omega) + ecc * math.cos(omega))


def _bouncing_ball(p, x):
    floor, height, fall_time, rest, release = p
    s = x - release
    if s <= 0.0:
        return floor + height
    if s < fall_time:
        return floor + height * (1.0 - (s / fall_time) ** 2)
    s -= fall_time
    q = s * (1.0 - rest) / (2.0 * fall_time * rest)
    if q >= 1.0:                      # infinitely many bounces have finished: at rest
        return floor
    n = int(math.floor(math.log1p(-q) / math.log(rest))) + 1
    u = 0.0
    for _ in range(4):                # guard against rounding at bounce boundaries
        start = 2.0 * fall_time * rest * (1.0 - rest ** (n - 1)) / (1.0 - rest)
        u = (s - start) / (2.0 * fall_time * rest ** n)
        if u < 0.0 and n > 1:
            n -= 1
        elif u > 1.0:
            n += 1
        else:
            break
    u = min(max(u, 0.0), 1.0)
    return floor + 4.0 * height * rest ** (2 * n) * u * (1.0 - u)


def _repeated_bolus(p, x):
    baseline, dose, k_elim, interval, first = p
    if x < first:
        return baseline
    n = int(math.floor((x - first) / interval)) + 1
    since = (x - first) - (n - 1) * interval
    r = math.exp(-k_elim * interval)
    return baseline + dose * math.exp(-k_elim * since) * (1.0 - r ** n) / (1.0 - r)


def _half_wave_rectifier(p, x):
    offset, amp, w, ph = p
    return offset + amp * max(0.0, math.sin(w * x + ph))


def _two_fluid_heat_capacity(p, x):
    offset, scale, t_c = p
    t = x / t_c
    return offset + scale * (3.0 * t ** 3 if t < 1.0 else t)


def _harvest_fold(p, x):
    offset, capacity, h_crit = p
    if x < h_crit:
        return offset + 0.5 * capacity * (1.0 + math.sqrt(1.0 - x / h_crit))
    return offset


def _airy_caustic(p, x):
    background, peak, x_caustic, length = p
    a = _airy_ai((x_caustic - x) / length) / _AI_MAX
    return background + peak * a * a


def _bessel_standing_wave(p, x):
    offset, amp, k, center = p
    return offset + amp * _bessel_j0(k * (x - center))


def _chebyshev_lowpass(p, x):
    offset, gain, cutoff, ripple, order = p
    n = int(round(order))
    u = x / cutoff
    if u <= 1.0:
        t_n = math.cos(n * math.acos(max(-1.0, u)))
    else:
        t_n = math.cosh(n * math.acosh(u))
    return offset + gain / math.sqrt(1.0 + (ripple * t_n) ** 2)


def _oscillator_density(p, x):
    background, peak, center, width, level = p
    n = int(round(level))
    return background + peak * _ho_shape(n, (x - center) / width) / _HO_PEAK[n]


# ----------------------------------------------------------------- samplers
def _s_two_tone_beats(rng):
    u = rng.uniform
    return (u(-1, 1), u(1.0, 2.5), u(0.6, 1.0), u(2.0, 4.0), u(0.8, 1.6), u(0, _TAU), u(0, _TAU))


def _s_linear_chirp(rng):
    u = rng.uniform
    return (u(-1, 1), u(1.0, 3.0), u(0.5, 1.5), u(0.2, 0.5), u(0, _TAU))


def _s_multi_slit(rng):
    u = rng.uniform
    return (u(-0.5, 0.5), u(2.0, 5.0), u(1.8, 3.2), u(0, _TAU), float(rng.randint(3, 5)))


def _s_fano(rng):
    u = rng.uniform
    return (u(-1, 1), _sign(rng) * u(2.0, 5.0), u(3.0, 7.0), u(0.25, 0.8), _sign(rng) * u(0.5, 2.0))


def _s_inverse_field_oscillation(rng):
    u = rng.uniform
    return (u(-1, 1), u(2.0, 5.0), u(5.0, 10.0), u(6.0, 15.0), u(0, _TAU))


def _s_quantum_wire_dos(rng):
    u = rng.uniform
    return (u(-1, 1), u(2.0, 4.0), u(0.3, 1.5), u(0.35, 0.8), u(0.02, 0.1))


def _s_kepler_radial_velocity(rng):
    u = rng.uniform
    return (u(-2, 2), u(1.0, 4.0), u(2.0, 4.5), u(0.3, 0.7), u(0, _TAU), u(0, 1))


def _s_bouncing_ball(rng):
    u = rng.uniform
    return (u(-1, 1), u(2.0, 5.0), u(0.9, 1.8), u(0.7, 0.9), u(-0.5, 1.0))


def _s_repeated_bolus(rng):
    u = rng.uniform
    return (u(-0.5, 0.5), u(1.0, 3.0), u(0.15, 0.7), u(1.5, 3.2), u(0.3, 2.0))


def _s_half_wave_rectifier(rng):
    u = rng.uniform
    return (u(-1, 1), _sign(rng) * u(2.0, 5.0), u(1.3, 3.0), u(0, _TAU))


def _s_two_fluid_heat_capacity(rng):
    u = rng.uniform
    return (u(-1, 1), u(1.0, 2.5), u(3.5, 7.5))


def _s_harvest_fold(rng):
    u = rng.uniform
    return (u(-1, 1), u(2.0, 5.0), u(4.0, 8.0))


def _s_airy_caustic(rng):
    u = rng.uniform
    return (u(-0.5, 0.5), u(2.0, 5.0), u(0.8, 3.0), u(0.6, 1.2))


def _s_bessel_standing_wave(rng):
    u = rng.uniform
    return (u(-1, 1), _sign(rng) * u(2.0, 5.0), u(2.0, 4.0), u(2.5, 7.5))


def _s_chebyshev_lowpass(rng):
    u = rng.uniform
    return (u(-1, 1), u(2.0, 5.0), u(5.5, 8.5), u(0.8, 1.5), float(rng.randint(5, 8)))


def _s_oscillator_density(rng):
    u = rng.uniform
    background, peak, center = u(-1, 1), u(2.0, 5.0), u(4.5, 5.5)
    level = rng.randint(3, 6)
    width = u(2.8, 4.0) / math.sqrt(2 * level + 1)   # classical turning points at +-2.8..4.0
    return (background, peak, center, width, float(level))


# ----------------------------------------------------------------- registry
_LAWS = {
    "two_tone_beats": (
        "Acoustic beats: superposition of two pure tones with nearby frequencies",
        _two_tone_beats, _s_two_tone_beats),
    "linear_chirp": (
        "Frequency-swept tone whose instantaneous frequency rises linearly with x",
        _linear_chirp, _s_linear_chirp),
    "multi_slit_interference": (
        "Intensity pattern of an ideal grating with a few narrow slits (principal and secondary maxima)",
        _multi_slit, _s_multi_slit),
    "fano_resonance": (
        "Asymmetric Fano resonance: a discrete state interfering with a continuum",
        _fano, _s_fano),
    "inverse_field_oscillation": (
        "Quantum (Shubnikov-de Haas type) oscillations periodic in 1/x with Dingle-type damping",
        _inverse_field_oscillation, _s_inverse_field_oscillation),
    "quantum_wire_dos": (
        "Density of states of a quantum wire: broadened inverse-square-root subband edges",
        _quantum_wire_dos, _s_quantum_wire_dos),
    "kepler_radial_velocity": (
        "Radial velocity of a star on an eccentric Keplerian orbit (Kepler's equation solved numerically)",
        _kepler_radial_velocity, _s_kepler_radial_velocity),
    "bouncing_ball": (
        "Height of a dropped ball bouncing with a constant coefficient of restitution",
        _bouncing_ball, _s_bouncing_ball),
    "repeated_bolus_dosing": (
        "Drug concentration under repeated intravenous doses with first-order elimination",
        _repeated_bolus, _s_repeated_bolus),
    "half_wave_rectifier": (
        "Output of an ideal half-wave rectifier driven by a sinusoid",
        _half_wave_rectifier, _s_half_wave_rectifier),
    "two_fluid_heat_capacity": (
        "Superconductor heat capacity in the two-fluid model: cubic below Tc, jump, linear above",
        _two_fluid_heat_capacity, _s_two_fluid_heat_capacity),
    "harvest_fold_collapse": (
        "Equilibrium of a logistic population under constant harvesting, collapsing at a fold",
        _harvest_fold, _s_harvest_fold),
    "airy_caustic_intensity": (
        "Wave intensity near a caustic (Airy function squared): shadow side and fringes",
        _airy_caustic, _s_airy_caustic),
    "bessel_standing_wave": (
        "Circular standing-wave amplitude along a line through its centre (Bessel J0)",
        _bessel_standing_wave, _s_bessel_standing_wave),
    "chebyshev_lowpass": (
        "Magnitude response of a Chebyshev type-I low-pass filter (equiripple passband)",
        _chebyshev_lowpass, _s_chebyshev_lowpass),
    "oscillator_eigenstate_density": (
        "Position probability density of an excited quantum harmonic-oscillator eigenstate",
        _oscillator_density, _s_oscillator_density),
}

_N_PARAMS = {
    "two_tone_beats": 7, "linear_chirp": 5, "multi_slit_interference": 5, "fano_resonance": 5,
    "inverse_field_oscillation": 5, "quantum_wire_dos": 5, "kepler_radial_velocity": 6,
    "bouncing_ball": 5, "repeated_bolus_dosing": 5, "half_wave_rectifier": 4,
    "two_fluid_heat_capacity": 3, "harvest_fold_collapse": 3, "airy_caustic_intensity": 4,
    "bessel_standing_wave": 4, "chebyshev_lowpass": 5, "oscillator_eigenstate_density": 5,
}

CLASSES: dict[str, dict] = {
    name: {"description": desc, "n_params": _N_PARAMS[name]} for name, (desc, _, _) in _LAWS.items()
}


def sample_params(name: str, rng: random.Random) -> tuple[float, ...]:
    """Draw a parameter vector for class `name`; deterministic given the state of `rng`."""
    if name not in _LAWS:
        raise KeyError(name)
    return tuple(float(v) for v in _LAWS[name][2](rng))


def truth(name: str, params: tuple[float, ...], x: float) -> float:
    """Noise-free value of the hidden law `name` with parameters `params` at x."""
    if name not in _LAWS:
        raise KeyError(name)
    if len(params) != _N_PARAMS[name]:
        raise ValueError(f"{name} expects {_N_PARAMS[name]} parameters, got {len(params)}")
    return float(_LAWS[name][1](tuple(params), float(x)))
