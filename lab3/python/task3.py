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

# Загрузка q0 для T=T1/2
q_file = LAB / 'python' / 'q0_T12.txt'
if q_file.exists():
    q0 = float(q_file.read_text().strip())
else:
    q0 = 4.0

Ts = T1 / 2


def get_controller_params(Ts: float, T2: float):
    """Расчёт по формуле методички: Wc(z)=q0 (z-d1)(z-d2)/(z(z-1))"""
    A_c = np.array([[-1.0/T1, 0.0],
                    [ 1.0/T2, -1.0/T2]])
    B_c = np.array([[1.0/T1],
                    [0.0]])
    C = np.array([[0.0, 1.0]])
    D = np.array([[0.0]])
    
    Ad, Bd, Cd, Dd, _ = cont2discrete((A_c, B_c, C, D), Ts, method='zoh')
    
    d1 = np.exp(-Ts / T1)
    d2 = np.exp(-Ts / T2)
    num_ctrl = np.array([1.0, -(d1 + d2), d1 * d2])
    den_ctrl = np.array([1.0, -1.0, 0.0])
    
    return num_ctrl, den_ctrl, Ad, Bd, Cd, Dd


def simulate(T2: float, N: int = 400):
    """Моделирование реакции на возмущение при разных T2"""
    num_ctrl, den_ctrl, Ad, Bd, Cd, Dd = get_controller_params(Ts, T2)
    
    if len(num_ctrl) > 0 and num_ctrl[0] != 0:
        num_ctrl = num_ctrl / num_ctrl[0]
    if len(den_ctrl) > 0 and den_ctrl[0] != 0:
        den_ctrl = den_ctrl / den_ctrl[0]
    
    n_ctrl = max(len(num_ctrl), len(den_ctrl))
    e_buf = np.zeros(n_ctrl)
    u_ctrl_buf = np.zeros(n_ctrl)
    
    x = np.zeros(Ad.shape[0])
    y_hist, d_hist = [], []
    
    for k in range(N):
        r = 0.0  # Задание равно нулю (реакция на возмущение)
        d = 0.5 if k >= 20 else 0.0  # Ступенчатое возмущение
        
        y = float(Cd @ x)
        
        e = r - y
        
        # Регулятор
        u = 0.0
        for i in range(len(num_ctrl)):
            if i < len(e_buf):
                u += q0 * num_ctrl[i] * e_buf[i]
        for i in range(1, len(den_ctrl)):
            if i < len(u_ctrl_buf):
                u -= den_ctrl[i] * u_ctrl_buf[i]
        
        e_buf = np.roll(e_buf, 1)
        e_buf[0] = e
        u_ctrl_buf = np.roll(u_ctrl_buf, 1)
        u_ctrl_buf[0] = u
        
        # Возмущение добавляется к входу объекта
        u_obj = u + d
        x = Ad @ x + Bd.flatten() * u_obj
        
        y_hist.append(y)
        d_hist.append(d)
    
    t = np.arange(N) * Ts
    return t, np.array(y_hist), np.array(d_hist)


if __name__ == '__main__':
    T2_minus = T2_nom * 0.8
    T2_plus = T2_nom * 1.2

    t0, y0, d0 = simulate(T2_nom)
    t1, y1, d1 = simulate(T2_minus)
    t2, y2, d2 = simulate(T2_plus)

    plt.figure(figsize=(8,4))
    plt.step(t0, d0, where='post', color='k', lw=1.0, label='возмущение')
    plt.step(t0, y0, where='post', color='C0', lw=1.8, label=f'T2={T2_nom:.2f} (номинальное)')
    plt.step(t1, y1, where='post', color='C1', lw=1.8, label=f'T2={T2_minus:.2f} (-20%)')
    plt.step(t2, y2, where='post', color='C2', lw=1.8, label=f'T2={T2_plus:.2f} (+20%)')
    plt.xlabel('t, s'); plt.ylabel('y'); plt.grid(True); plt.legend();
    plt.tight_layout()
    plt.savefig(IMG3 / 'compare_T2_perturb.png', dpi=150)
    plt.close()

    print('График сохранён:', IMG3 / 'compare_T2_perturb.png')
