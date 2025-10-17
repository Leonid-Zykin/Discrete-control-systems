import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from pathlib import Path

# Вариант 8
T1 = 0.9
T2_nom = 1.05

# Пути
THIS = Path(__file__).resolve()
LAB = THIS.parent.parent
IMG3 = LAB / 'images' / 'task3'
IMG3.mkdir(parents=True, exist_ok=True)

# Загрузка q0 для T=T1/2, найденного в task2
q_file = LAB / 'python' / 'q0_T12.txt'
if q_file.exists():
    q0 = float(q_file.read_text().strip())
else:
    q0 = 4.0

Ts = T1 / 2


def simulate(T2: float, N: int = 400):
    A_c = np.array([[-1.0/T1, 0.0],
                    [ 1.0/T2, -1.0/T2]])
    B_c = np.array([[1.0/T1],
                    [0.0]])
    C = np.array([[0.0, 1.0]])
    D = np.array([[0.0]])
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


if __name__ == '__main__':
    T2_minus = T2_nom * 0.8
    T2_plus = T2_nom * 1.2

    t0, y0 = simulate(T2_nom)
    t1, y1 = simulate(T2_minus)
    t2, y2 = simulate(T2_plus)

    plt.figure(figsize=(8,4))
    plt.plot(t0, y0, label=f'T2={T2_nom:.2f}')
    plt.plot(t1, y1, label=f'T2={T2_minus:.2f} (-20%)')
    plt.plot(t2, y2, label=f'T2={T2_plus:.2f} (+20%)')
    plt.plot(t0, np.ones_like(t0), 'k--', lw=0.8, label='r=1')
    plt.xlabel('t, s'); plt.ylabel('y'); plt.grid(True); plt.legend();
    plt.tight_layout()
    plt.savefig(IMG3 / 'compare_T2_perturb.png', dpi=150)
    plt.close()

    print('График сохранён:', IMG3 / 'compare_T2_perturb.png')
