import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from pathlib import Path

# Вариант 8
T1 = 0.9
T2 = 1.05

# Пути
THIS = Path(__file__).resolve()
LAB = THIS.parent.parent
IMG2 = LAB / 'images' / 'task2'
IMG2.mkdir(parents=True, exist_ok=True)

# Непрерывная модель объекта: два апериодических звена
A_c = np.array([[-1.0/T1, 0.0],
                [ 1.0/T2, -1.0/T2]])
B_c = np.array([[1.0/T1],
                [0.0]])
C = np.array([[0.0, 1.0]])
D = np.array([[0.0]])


def simulate(Ts: float, q0: float, N: int = 400):
    Ad, Bd, Cd, Dd, _ = cont2discrete((A_c, B_c, C, D), Ts, method='zoh')
    x = np.zeros(2)
    y_hist = []
    for k in range(N):
        r = 1.0
        y = float(C @ x)
        u = q0 * (r - y)
        x = (Ad @ x) + (Bd.flatten() * u)
        y_hist.append(y)
    t = np.arange(N) * Ts
    return t, np.array(y_hist)


def step_quality(y: np.ndarray, r: float = 1.0, Ts: float = 1.0):
    overshoot = max(0.0, (y.max() - r) / max(r, 1e-9))
    # Время установления до 2% от задания
    tol = 0.02 * r if r != 0 else 0.02
    settled_idx = len(y) - 1
    for i in range(len(y) - 1, -1, -1):
        if abs(y[i] - r) > tol:
            break
        settled_idx = i
    t_s = settled_idx * Ts
    # Интегральная квадратичная ошибка
    ise = np.mean((y - r) ** 2)
    return overshoot, t_s, ise


def tune_q0(Ts: float, q_min: float = 0.1, q_max: float = 5.0, nq: int = 60):
    best = None
    q_grid = np.linspace(q_min, q_max, nq)
    for q in q_grid:
        t, y = simulate(Ts, q)
        os, t_s, ise = step_quality(y, Ts=Ts)
        # Критерий: ограниченная перерегулировка 5..15% и минимальное t_s,
        # если не удаётся — минимальный ISE при устойчивости (y не NaN/Inf)
        if not np.isfinite(y).all():
            continue
        if 0.05 <= os <= 0.15:
            cand = (0, t_s, ise, q)  # приоритет 0 — удовлетворяет окну по осцилляциям
        else:
            cand = (1, ise, t_s, q)  # приоритет 1 — fallback по ISE
        if best is None or cand < best:
            best = cand
    if best is None:
        return 1.0  # безопасное значение по умолчанию
    return float(best[-1])


if __name__ == '__main__':
    Ts12 = T1 / 2
    Ts14 = T1 / 4

    q12 = tune_q0(Ts12)
    q14 = tune_q0(Ts14)

    print(f"Подобранные q0: T=T1/2 => {q12:.4f}, T=T1/4 => {q14:.4f}")

    t12, y12 = simulate(Ts12, q12)
    t14, y14 = simulate(Ts14, q14)

    plt.figure(figsize=(8,4))
    plt.plot(t12, np.ones_like(t12), 'k--', lw=0.8, label='r=1')
    plt.plot(t12, y12, label=f'y, T=T1/2, q0={q12:.2f}')
    plt.plot(t14, y14, label=f'y, T=T1/4, q0={q14:.2f}')
    plt.xlabel('t, s'); plt.ylabel('y'); plt.grid(True); plt.legend();
    plt.tight_layout()
    plt.savefig(IMG2 / 'compare_T.png', dpi=150)
    plt.close()

    # Сохраним q12 в файл для использования в задании 3
    (LAB / 'python' / 'q0_T12.txt').write_text(f"{q12:.6f}\n")
