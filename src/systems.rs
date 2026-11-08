//! Fast generators for the dynamical systems of the original MATLAB
//! `dynamical-systems` toolbox (same equations, same parameter order).
//! Named regimes live on the Python side (`ordinal_networks.systems`).

pub const MAXD: usize = 10;

// ------------------------------------------------------------------ maps ---

/// (state dimension, number of parameters)
pub fn map_spec(name: &str) -> Option<(usize, usize)> {
    Some(match name {
        "logistic" => (1, 1),
        "cubic" => (1, 1),
        "tent" => (1, 1),
        "dyadic" => (1, 1),
        "bent_baker" => (1, 0),
        "gauss" => (1, 2),
        "piecewise_linear" => (1, 0),
        "henon" => (2, 2),
        "lozi" => (2, 2),
        "ikeda" => (2, 1),
        "holmes" => (2, 2),
        "tinkerbell" => (2, 4),
        "gingerbread_man" => (2, 0),
        "henon_area" => (2, 1),
        "chirikov" => (2, 1),
        "arnold_cat" => (2, 0),
        "bakers" => (2, 0),
        "stiletto" => (2, 0),
        "zaslavsky" => (2, 4),
        "solenoid" => (3, 0),
        "general_henon" => (3, 2),
        "folded_towel" => (3, 2),
        _ => return None,
    })
}

#[inline]
pub fn map_step(name: &str, p: &[f64], x: &mut [f64]) {
    use std::f64::consts::PI;
    match name {
        "logistic" => x[0] = p[0] * x[0] * (1.0 - x[0]),
        "cubic" => x[0] = (p[0] - 1.0) * x[0] - p[0] * x[0] * x[0] * x[0],
        "tent" => x[0] = if x[0] < 0.5 { p[0] * x[0] } else { p[0] * (1.0 - x[0]) },
        "dyadic" => x[0] = (p[0] * x[0]).rem_euclid(1.0),
        "bent_baker" => {
            let s6 = 6f64.sqrt();
            let v = x[0];
            x[0] = ((4.0 * s6 / 3.0) * v * v * v - 2.0 * s6 * v * v + (2.0 + 2.0 * s6 / 3.0) * v).rem_euclid(1.0);
        }
        "gauss" => x[0] = (-p[0] * x[0] * x[0]).exp() + p[1],
        "piecewise_linear" => x[0] = x[0] * (1.0 / x[0]).ln(),
        "henon" => {
            let x1 = 1.0 + x[1] - p[0] * x[0] * x[0];
            x[1] = p[1] * x[0];
            x[0] = x1;
        }
        "lozi" => {
            // MATLAB form: (x2, 1 + b x1 - a|x2|)
            let x2 = 1.0 + p[1] * x[0] - p[0] * x[1].abs();
            x[0] = x[1];
            x[1] = x2;
        }
        "ikeda" => {
            let u = p[0];
            let t = 0.4 - 6.0 / (1.0 + x[0] * x[0] + x[1] * x[1]);
            let (s, c) = t.sin_cos();
            let x1 = 1.0 + u * (x[0] * c - x[1] * s);
            x[1] = u * (x[0] * s + x[1] * c);
            x[0] = x1;
        }
        "holmes" => {
            let y = p[0] * x[1] - p[1] * x[0] - x[1] * x[1] * x[1];
            x[0] = x[1];
            x[1] = y;
        }
        "tinkerbell" => {
            let (a, b, c, d) = (p[0], p[1], p[2], p[3]);
            let x1 = x[0] * x[0] - x[1] * x[1] + a * x[0] + b * x[1];
            x[1] = 2.0 * x[0] * x[1] + c * x[0] + d * x[1];
            x[0] = x1;
        }
        "gingerbread_man" => {
            let x1 = 1.0 - x[1] + x[0].abs();
            x[1] = x[0];
            x[0] = x1;
        }
        "henon_area" => {
            let (s, c) = p[0].sin_cos();
            let q = x[1] - x[0] * x[0];
            // Henon (1969): y' = x sin(a) + (y - x^2) cos(a). The MATLAB file had
            // "- (y - x^2) cos(a)", which is not area-preserving (det J = -cos 2a).
            let x1 = x[0] * c - q * s;
            x[1] = x[0] * s + q * c;
            x[0] = x1;
        }
        "chirikov" => {
            let tp = 2.0 * PI;
            let x1 = (x[0] + p[0] * x[1].sin()).rem_euclid(tp);
            x[1] = (x[1] + x1).rem_euclid(tp);
            x[0] = x1;
        }
        "arnold_cat" => {
            let x1 = (2.0 * x[0] + x[1]).rem_euclid(1.0);
            x[1] = (x[0] + x[1]).rem_euclid(1.0);
            x[0] = x1;
        }
        "bakers" => {
            if x[0] < 0.5 {
                x[0] *= 2.0;
                x[1] *= 0.5;
            } else {
                x[0] = 2.0 - 2.0 * x[0];
                x[1] = 1.0 - 0.5 * x[1];
            }
        }
        "stiletto" => {
            let x1 = (x[0] + 1.0 / 3.0) * (-3.0 * x[0] + 2.0).exp() - 1.0 / 3.0 + x[1];
            x[1] = 3.0 * x[0] / 10.0;
            x[0] = x1;
        }
        "zaslavsky" => {
            // p = (Omega, alpha, Gamma, epsilon)
            let (om, al, ga, ep) = (p[0], p[1], p[2], p[3]);
            let mu = ((ga).exp() - 1.0) / ga;
            let y = (-ga).exp() * (x[1] + ep * x[0]);
            x[0] = (x[0] + om + al * mu * y).rem_euclid(2.0 * PI);
            x[1] = y;
        }
        "solenoid" => {
            // state (x, y, theta)
            let th = x[2];
            x[0] = x[0] / 4.0 + 0.5 * (2.0 * PI * th).cos();
            x[1] = x[1] / 4.0 + 0.5 * (2.0 * PI * th).sin();
            x[2] = (2.0 * th).rem_euclid(2.0 * PI);
        }
        "general_henon" => {
            let x1 = p[0] - x[1] * x[1] - p[1] * x[2];
            x[2] = x[1];
            x[1] = x[0];
            x[0] = x1;
        }
        "folded_towel" => {
            let (a, b) = (p[0], p[1]);
            let (x1, x2, x3) = (x[0], x[1], x[2]);
            x[0] = a * x1 * (1.0 - x1) - 0.05 * (x2 + 0.35) * (1.0 - 2.0 * x3);
            x[1] = 0.1 * ((x2 + 0.35) * (1.0 + 2.0 * x3) - 1.0) * (1.0 - 1.9 * x1);
            x[2] = 3.78 * x3 * (1.0 - x3) + b * x2;
        }
        _ => unreachable!(),
    }
}

/// Iterate a map: discard `transient` iterates, then store `n` states (row-major n x d).
/// Returns the trajectory and the next state (to continue in chunks).
pub fn iterate_map(name: &str, p: &[f64], x0: &[f64], n: usize, transient: usize) -> (Vec<f64>, Vec<f64>) {
    let d = x0.len();
    let mut x = x0.to_vec();
    for _ in 0..transient {
        map_step(name, p, &mut x);
    }
    let mut out = vec![0f64; n * d];
    for i in 0..n {
        out[i * d..(i + 1) * d].copy_from_slice(&x);
        map_step(name, p, &mut x);
    }
    (out, x)
}

// ----------------------------------------------------------------- flows ---

/// (state dimension, number of parameters). Non-autonomous systems carry
/// time (or phase) as their last state variable.
pub fn flow_spec(name: &str) -> Option<(usize, usize)> {
    Some(match name {
        "lorenz" => (3, 3),
        "rossler" => (3, 3),
        "rossler4d" => (4, 4),
        "chua" => (3, 5),
        "muthuswamy" => (3, 4),
        "rabinovich_fabrikant" => (3, 2),
        "duffing" => (3, 5),
        "vanderpol" => (3, 3),
        "reiterer" => (9, 3),
        "abc" => (3, 3),
        "double_gyre" => (3, 3),
        "double_gyre_auto" => (2, 0),
        "double_gyre_roto" => (3, 0),
        "bickley" => (3, 11),
        _ => return None,
    })
}

#[inline]
fn vf(name: &str, p: &[f64], x: &[f64], dx: &mut [f64]) {
    use std::f64::consts::PI;
    match name {
        "lorenz" => {
            // p = (sigma, rho, beta)
            dx[0] = p[0] * (x[1] - x[0]);
            dx[1] = x[0] * (p[1] - x[2]) - x[1];
            dx[2] = x[0] * x[1] - p[2] * x[2];
        }
        "rossler" => {
            dx[0] = -x[1] - x[2];
            dx[1] = x[0] + p[0] * x[1];
            dx[2] = p[1] + x[2] * (x[0] - p[2]);
        }
        "rossler4d" => {
            dx[0] = -x[1] - x[2];
            dx[1] = x[0] + p[0] * x[1] + x[3];
            dx[2] = p[1] + x[0] * x[2];
            dx[3] = -p[2] * x[2] + p[3] * x[3];
        }
        "chua" => {
            // p = (alpha, beta, m0, m1, E)
            let (al, be, m0, m1, e) = (p[0], p[1], p[2], p[3], p[4]);
            let f = m1 * x[0] + 0.5 * (m0 - m1) * ((x[0] + e).abs() - (x[0] - e).abs());
            dx[0] = al * (x[1] - x[0] - f);
            dx[1] = x[0] - x[1] + x[2];
            dx[2] = -be * x[1];
        }
        "muthuswamy" => {
            let (c, l, al, be) = (p[0], p[1], p[2], p[3]);
            dx[0] = x[1] / c;
            dx[1] = -(x[0] + be * (x[2] * x[2] - 1.0) * x[1]) / l;
            dx[2] = -x[1] - al * x[2] + x[1] * x[2];
        }
        "rabinovich_fabrikant" => {
            let (al, ga) = (p[0], p[1]);
            dx[0] = x[1] * (x[2] - 1.0 + x[0] * x[0]) + ga * x[0];
            dx[1] = x[0] * (3.0 * x[2] + 1.0 - x[0] * x[0]) + ga * x[1];
            dx[2] = -2.0 * x[2] * (al + x[0] * x[1]);
        }
        "duffing" => {
            // p = (alpha, beta, gamma, delta, omega); x[2] = phase
            let (al, be, ga, de, om) = (p[0], p[1], p[2], p[3], p[4]);
            dx[0] = x[1];
            dx[1] = -de * x[1] - be * x[0] - al * x[0] * x[0] * x[0] + ga * x[2].cos();
            dx[2] = om;
        }
        "vanderpol" => {
            let (mu, a, om) = (p[0], p[1], p[2]);
            dx[0] = x[1];
            dx[1] = mu * (1.0 - x[0] * x[0]) * x[1] - x[0] + a * x[2].sin();
            dx[2] = om;
        }
        "reiterer" => {
            let (a, s, r) = (p[0], p[1], p[2]);
            let a2 = a * a;
            let b1 = 4.0 * (1.0 + a2) / (1.0 + 2.0 * a2);
            let b2 = (1.0 + 2.0 * a2) / (2.0 * (1.0 + a2));
            let b3 = 2.0 * (1.0 - a2) / (1.0 + a2);
            let b4 = a2 / (1.0 + a2);
            let b5 = 8.0 * a2 / (1.0 + 2.0 * a2);
            let b6 = 4.0 / (1.0 + 2.0 * a2);
            let (x1, x2, x3, x4, x5, x6, x7, x8, x9) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]);
            dx[0] = -s * b1 * x1 - x2 * x4 + b4 * x4 * x4 + b3 * x3 * x5 - s * b2 * x7;
            dx[1] = -s * x2 + x1 * x4 - x2 * x5 + x4 * x5 - 0.5 * s * x9;
            dx[2] = -s * b1 * x3 + x2 * x4 - b4 * x2 * x2 - b3 * x1 * x5 + s * b2 * x8;
            dx[3] = -s * x4 - x2 * x3 - x2 * x5 + x4 * x5 + 0.5 * s * x9;
            dx[4] = -s * b5 * x5 + 0.5 * x2 * x2 - x4 * x4 / 2.0;
            dx[5] = -b6 * x6 + x2 * x9 - x4 * x9;
            dx[6] = -b1 * x7 - r * x1 + 2.0 * x5 * x8 - x4 * x9;
            dx[7] = -b1 * x8 + r * x3 - 2.0 * x5 * x7 + x2 * x9;
            dx[8] = -x9 - r * x2 + r * x4 - 2.0 * x2 * x6 + 2.0 * x4 * x6 + x4 * x7 - x2 * x8;
        }
        "abc" => {
            let (a, b, c) = (p[0], p[1], p[2]);
            dx[0] = a * x[2].sin() + c * x[1].cos();
            dx[1] = b * x[0].sin() + a * x[2].cos();
            dx[2] = c * x[1].sin() + b * x[0].cos();
        }
        "double_gyre" => {
            // state (x, y, t)
            let (a, ep, om) = (p[0], p[1], p[2]);
            let st = (om * x[2]).sin();
            let f = ep * st * x[0] * x[0] + (1.0 - 2.0 * ep * st) * x[0];
            let dfdx = 1.0 + 2.0 * (x[0] - 1.0) * ep * st;
            dx[0] = -PI * a * (PI * f).sin() * (PI * x[1]).cos();
            dx[1] = PI * a * (PI * f).cos() * (PI * x[1]).sin() * dfdx;
            dx[2] = 1.0;
        }
        "double_gyre_auto" => {
            dx[0] = -PI * (PI * x[0]).sin() * (PI * x[1]).cos();
            dx[1] = PI * (PI * x[0]).cos() * (PI * x[1]).sin();
        }
        "double_gyre_roto" => {
            let t = x[2];
            let st = if t > 0.0 && t < 1.0 { t * t * (3.0 - 2.0 * t) } else if t > 1.0 { 1.0 } else { 0.0 };
            let dxpp = 2.0 * PI * (2.0 * PI * x[0]).cos() * (PI * x[1]).sin();
            let dypp = PI * (2.0 * PI * x[0]).sin() * (PI * x[1]).cos();
            let dxpf = PI * (PI * x[0]).cos() * (2.0 * PI * x[1]).sin();
            let dypf = 2.0 * PI * (PI * x[0]).sin() * (2.0 * PI * x[1]).cos();
            dx[0] = -((1.0 - st) * dypp + st * dypf);
            dx[1] = (1.0 - st) * dxpp + st * dxpf;
            dx[2] = 1.0;
        }
        "bickley" => {
            let (u0, l0) = (p[0], p[1]);
            let a = [p[2], p[3], p[4]];
            let k = [p[5], p[6], p[7]];
            let c = [p[8], p[9], p[10]];
            let t = x[2];
            let sech2 = 1.0 / (x[1] / l0).cosh().powi(2);
            let th = (x[1] / l0).tanh();
            let mut dxpsi = 0.0;
            let mut dypsi = -u0 * sech2;
            for i in 0..3 {
                let ph = k[i] * (x[0] - c[i] * t);
                dxpsi -= k[i] * a[i] * u0 * l0 * ph.sin() * sech2;
                dypsi -= 2.0 * a[i] * u0 * ph.cos() * th * sech2;
            }
            dx[0] = -dypsi;
            dx[1] = dxpsi;
            dx[2] = 1.0;
        }
        _ => unreachable!(),
    }
}

#[inline]
fn rk4(name: &str, p: &[f64], x: &[f64], h: f64, out: &mut [f64]) {
    let d = x.len();
    let mut k1 = [0f64; MAXD];
    let mut k2 = [0f64; MAXD];
    let mut k3 = [0f64; MAXD];
    let mut k4 = [0f64; MAXD];
    let mut t = [0f64; MAXD];
    vf(name, p, x, &mut k1[..d]);
    for i in 0..d { t[i] = x[i] + 0.5 * h * k1[i]; }
    vf(name, p, &t[..d], &mut k2[..d]);
    for i in 0..d { t[i] = x[i] + 0.5 * h * k2[i]; }
    vf(name, p, &t[..d], &mut k3[..d]);
    for i in 0..d { t[i] = x[i] + h * k3[i]; }
    vf(name, p, &t[..d], &mut k4[..d]);
    for i in 0..d {
        out[i] = x[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

/// Fixed-step RK4. Stores `n` samples, one every `every` steps, after
/// `transient` steps. Returns (row-major n x d, final state).
pub fn integrate(name: &str, p: &[f64], x0: &[f64], dt: f64, n: usize, every: usize, transient: usize) -> (Vec<f64>, Vec<f64>) {
    let d = x0.len();
    let mut x = [0f64; MAXD];
    let mut y = [0f64; MAXD];
    x[..d].copy_from_slice(x0);
    for _ in 0..transient {
        rk4(name, p, &x[..d], dt, &mut y[..d]);
        x = y;
    }
    let mut out = vec![0f64; n * d];
    for i in 0..n {
        out[i * d..(i + 1) * d].copy_from_slice(&x[..d]);
        for _ in 0..every.max(1) {
            rk4(name, p, &x[..d], dt, &mut y[..d]);
            x = y;
        }
    }
    (out, x[..d].to_vec())
}

/// Poincare surface of section {g(x) = normal . x - offset = 0}, keeping
/// crossings where g goes from negative to non-negative (direction = +1) or
/// from positive to non-positive (direction = -1). The crossing point inside
/// each RK4 step is found by bisection on the cubic Hermite interpolant.
/// Returns (row-major n x d crossing states, crossing times, final state).
pub fn poincare(
    name: &str, p: &[f64], x0: &[f64], dt: f64, n: usize, normal: &[f64], offset: f64,
    direction: f64, transient: usize, max_steps: u64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let d = x0.len();
    let g = |x: &[f64]| -> f64 { (0..d).map(|i| normal[i] * x[i]).sum::<f64>() - offset };
    let mut x = [0f64; MAXD];
    let mut y = [0f64; MAXD];
    x[..d].copy_from_slice(x0);
    for _ in 0..transient {
        rk4(name, p, &x[..d], dt, &mut y[..d]);
        x = y;
    }
    let mut out = Vec::with_capacity(n * d);
    let mut times = Vec::with_capacity(n);
    let mut f0 = [0f64; MAXD];
    let mut f1 = [0f64; MAXD];
    vf(name, p, &x[..d], &mut f0[..d]);
    let mut steps: u64 = 0;
    let herm = |s: f64, y0: f64, y1: f64, d0: f64, d1: f64| {
        let s2 = s * s;
        let s3 = s2 * s;
        (2.0 * s3 - 3.0 * s2 + 1.0) * y0 + (s3 - 2.0 * s2 + s) * dt * d0 + (-2.0 * s3 + 3.0 * s2) * y1 + (s3 - s2) * dt * d1
    };
    while times.len() < n {
        if steps >= max_steps {
            return Err(format!("only {} crossings found in {} steps", times.len(), steps));
        }
        rk4(name, p, &x[..d], dt, &mut y[..d]);
        vf(name, p, &y[..d], &mut f1[..d]);
        steps += 1;
        let a = g(&x[..d]);
        let b = g(&y[..d]);
        let crossed = if direction >= 0.0 { a < 0.0 && b >= 0.0 } else { a > 0.0 && b <= 0.0 };
        if crossed {
            let gs = |s: f64| -> f64 {
                (0..d).map(|i| normal[i] * herm(s, x[i], y[i], f0[i], f1[i])).sum::<f64>() - offset
            };
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let v = gs(mid);
                let before = if direction >= 0.0 { v < 0.0 } else { v > 0.0 };
                if before { lo = mid } else { hi = mid }
            }
            let s = 0.5 * (lo + hi);
            for i in 0..d {
                out.push(herm(s, x[i], y[i], f0[i], f1[i]));
            }
            times.push((steps as f64 - 1.0 + s) * dt);
        }
        x = y;
        f0 = f1;
    }
    Ok((out, times, x[..d].to_vec()))
}

/// Mackey-Glass delay equation dx/dt = beta x_tau/(1 + alpha x_tau^n) - gamma x,
/// p = (beta, gamma, n, tau, alpha) as in the MATLAB toolbox. Fixed-step RK4
/// with the delayed term linearly interpolated on the step grid; constant
/// initial history x0. Returns `n_out` samples every `every` steps after `transient`.
pub fn mackey_glass(p: &[f64], x0: f64, dt: f64, n_out: usize, every: usize, transient: usize) -> Vec<f64> {
    let (be, ga, nn, tau, al) = (p[0], p[1], p[2], p[3], p[4]);
    let lag = (tau / dt).round().max(1.0) as usize;
    let total = transient + n_out * every.max(1);
    let mut hist = vec![x0; lag + total + 1];
    let f = |x: f64, xt: f64| be * xt / (1.0 + al * xt.powf(nn)) - ga * x;
    for k in 0..total {
        let i = lag + k; // current index
        let x = hist[i];
        let xt0 = hist[i - lag];
        let xt1 = hist[i - lag + 1];
        let xth = 0.5 * (xt0 + xt1);
        let k1 = f(x, xt0);
        let k2 = f(x + 0.5 * dt * k1, xth);
        let k3 = f(x + 0.5 * dt * k2, xth);
        let k4 = f(x + dt * k3, xt1);
        hist[i + 1] = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    (0..n_out).map(|j| hist[lag + transient + j * every.max(1)]).collect()
}

// ---------------------------------------------------------- random walk ---

pub struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut r = z;
            r = (r ^ (r >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            r = (r ^ (r >> 27)).wrapping_mul(0x94D049BB133111EB);
            *v = r ^ (r >> 31);
        }
        Xoshiro256 { s }
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Random walk on a weighted directed graph in CSR form (weights need not be
/// normalised: the next node is drawn proportionally to the out-weights).
/// Stops early at a node without out-links (the returned path is shorter).
pub fn random_walk(indptr: &[i64], indices: &[i64], weights: &[f64], start: usize, steps: usize, seed: u64) -> Vec<i64> {
    let mut rng = Xoshiro256::new(seed);
    let mut path = Vec::with_capacity(steps + 1);
    let mut cur = start;
    path.push(cur as i64);
    for _ in 0..steps {
        let (a, b) = (indptr[cur] as usize, indptr[cur + 1] as usize);
        if a == b {
            break;
        }
        let tot: f64 = weights[a..b].iter().sum();
        let mut u = rng.next_f64() * tot;
        let mut nxt = indices[b - 1];
        for k in a..b {
            u -= weights[k];
            if u < 0.0 {
                nxt = indices[k];
                break;
            }
        }
        cur = nxt as usize;
        path.push(nxt);
    }
    path
}
