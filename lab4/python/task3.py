import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

# HG(z) = 0.03 (z + 0.75) / (z^2 - 1.5 z + 0.5)
B = np.array([0.03, 0.03*0.75])  # b0 + b1 z^{-1}
A = np.array([1.0, -1.5, 0.5])   # 1 + a1 z^{-1} + a2 z^{-2}

# Вариант 8
zeta = 0.78
wd = 4.0
T = 0.45
Kv_target = 0.14

# Желаемые полюса (пара) + интегратор для типа 1
wn = wd / np.sqrt(1 - zeta**2)
r = np.exp(-zeta * wn * T)
phi = wd * T
p1 = r * np.exp(1j*phi)
p2 = r * np.exp(-1j*phi)
A_pair = np.poly([p1, p2]).real  # [1, a1p, a2p]
A_int = np.array([1.0, -1.0])    # (1 - z^{-1})
A_d = np.convolve(A_int, A_pair) # степень 3

# Коэффициенты растения
a0, a1, a2 = A
b0, b1 = B

# Синтез R,S: S(z)=s0 + s1 z^{-1}, R(z)=r0 + r1 z^{-1} + r2 z^{-2}
# Нормируем s0=1
s0 = 1.0
# Система уравнений по коэффициентам z^0..z^-3: A*S + B*R = A_d
# Неизвестные x = [s1, r0, r1, r2]
M = np.array([
    [0.0, b0,   0.0,  0.0],                 # d0 = a0*s0 + b0*r0
    [a0,  b1,   b0,   0.0],                 # d1 = a1*s0 + a0*s1 + b1*r0 + b0*r1
    [a1,  0.0,  b1,   b0 ],                 # d2 = a2*s0 + a1*s1 + b1*r1 + b0*r2
    [a2,  0.0,  0.0,  b1 ],                 # d3 = a2*s1 + b1*r2
], dtype=float)
Y = np.array([
    A_d[0] - a0*s0,
    A_d[1] - a1*s0,
    A_d[2] - a2*s0,
    A_d[3],
], dtype=float)

s1, r0, r1, r2 = np.linalg.solve(M, Y)

# Проверка диофантова равенства
A_poly = np.poly1d(A)
B_poly = np.poly1d(B)
S_poly = np.poly1d([s0, s1])
R_poly = np.poly1d([r0, r1, r2])
left = (A_poly * S_poly + B_poly * R_poly).c
assert np.allclose(left, A_d, atol=1e-6), "Диофантово уравнение не решено"

# Подбор T(z)=t0+t1 z^{-1}+t2 z^{-2}: H(1)=1 и Kv=0.14
A_d_poly = np.poly1d(A_d)
T1_req = A_d_poly(1.0) / B_poly(1.0)  # t0 + t1 + t2 = T1_req


def evaluate_t01(t0: float, t1: float):
    t2 = float(T1_req - t0 - t1)
    num = np.polymul(B, np.array([t0, t1, t2]))
    den = A_d
    N = 5000
    k = np.arange(N)
    ramp = k.astype(float)
    y = signal.lfilter(num/den[0], den/den[0], ramp)
    e = ramp - y
    e_ss = np.mean(e[-600:])
    Kv_est = np.inf if e_ss <= 0 else 1.0 / e_ss
    return abs(Kv_est - Kv_target), Kv_est, (t0, t1, t2)

# Грубая сетка и локальное уточнение
best = (np.inf, None, None)
for t0 in np.linspace(-100.0, 100.0, 801):
    for t1 in np.linspace(-100.0, 100.0, 801):
        err, Kv_est, coeffs = evaluate_t01(t0, t1)
        if np.isfinite(Kv_est) and err < best[0]:
            best = (err, Kv_est, coeffs)

err, Kv_est, (t0_best, t1_best, t2_best) = best
span = 10.0
for _ in range(6):
    improved = False
    grid0 = np.linspace(t0_best - span, t0_best + span, 161)
    grid1 = np.linspace(t1_best - span, t1_best + span, 161)
    for t0 in grid0:
        for t1 in grid1:
            e2, Kv2, coeffs2 = evaluate_t01(t0, t1)
            if np.isfinite(Kv2) and e2 < err:
                err, Kv_est = e2, Kv2
                t0_best, t1_best, t2_best = coeffs2
                improved = True
    if not improved:
        break
    span *= 0.35

print(f"RST: S=[{s0:.4f}, {s1:.4f}], R=[{r0:.4f}, {r1:.4f}, {r2:.4f}], T=[{t0_best:.5f}, {t1_best:.5f}, {t2_best:.5f}], Kv≈{Kv_est:.5f}, |Δ|={err:.3e}")

# Переходная на ступень
num_step = np.polymul(B, np.array([t0_best, t1_best, t2_best]))
den = A_d
N = 800
step = np.ones(N)
y_step = signal.lfilter(num_step/den[0], den/den[0], step)

THIS = Path(__file__).resolve()
IMG = THIS.parent.parent / 'images' / 'task3'
IMG.mkdir(parents=True, exist_ok=True)

t = np.arange(N) * T
plt.figure(figsize=(8,4))
plt.plot(t, np.ones_like(t), 'k--', lw=0.8, label='r=1')
plt.plot(t, y_step, label='y')
plt.xlabel('t, s'); plt.ylabel('y'); plt.grid(True); plt.legend();
plt.tight_layout()
plt.savefig(IMG / 'closed_hg.png', dpi=150)
plt.close()
print('Saved:', IMG / 'closed_hg.png')
