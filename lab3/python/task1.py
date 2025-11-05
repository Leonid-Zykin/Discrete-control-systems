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
IMG1 = LAB / 'images' / 'task1'
IMG1.mkdir(parents=True, exist_ok=True)

# Непрерывная модель объекта: G(s) = 1/((T1*s+1)*(T2*s+1))
A_c = np.array([[-1.0/T1, 0.0],
                [ 1.0/T2, -1.0/T2]])
B_c = np.array([[1.0/T1],
                [0.0]])
C = np.array([[0.0, 1.0]])
D = np.array([[0.0]])


def get_controller_params(Ts: float):
    """Расчет параметров регулятора по методике:
    Wc(z) = q0 (z - d1)(z - d2) / (z (z - 1)),
    где d1 = e^{-T/T1}, d2 = e^{-T/T2}.
    """
    # Дискретизация объекта (для моделирования)
    Ad, Bd, Cd, Dd, _ = cont2discrete((A_c, B_c, C, D), Ts, method='zoh')

    # Полюса приведенной непрерывной части (по методичке): d1, d2
    d1 = np.exp(-Ts / T1)
    d2 = np.exp(-Ts / T2)

    # Числитель регулятора: (z - d1)(z - d2) = z^2 - (d1 + d2) z + d1 d2
    num_ctrl = np.array([1.0, -(d1 + d2), d1 * d2])
    # Знаменатель регулятора: z^2 - z
    den_ctrl = np.array([1.0, -1.0, 0.0])

    return num_ctrl, den_ctrl, Ad, Bd, Cd, Dd


def simulate(Ts: float, q0: float = 1.0, N: int = 300, mode: str = 'set', seed: int = 0):
    """Моделирование системы с регулятором"""
    num_ctrl, den_ctrl, Ad, Bd, Cd, Dd = get_controller_params(Ts)
    
    # Нормализация регулятора
    if len(num_ctrl) > 0 and num_ctrl[0] != 0:
        num_ctrl = num_ctrl / num_ctrl[0]
    if len(den_ctrl) > 0 and den_ctrl[0] != 0:
        den_ctrl = den_ctrl / den_ctrl[0]
    
    # Буферы для регулятора
    n_ctrl = max(len(num_ctrl), len(den_ctrl))
    e_buf = np.zeros(n_ctrl)
    u_ctrl_buf = np.zeros(n_ctrl)
    
    # Состояние объекта
    x = np.zeros(Ad.shape[0])
    
    y_hist, u_hist, r_hist, d_hist = [], [], [], []
    rng = np.random.default_rng(seed)
    
    for k in range(N):
        # Формирование входных сигналов
        if mode == 'set':
            r = 1.0
            d = 0.0
        elif mode == 'dist_step':
            r = 0.0
            d = 0.5 if k >= 20 else 0.0
        elif mode == 'noise':
            r = 0.0
            d = 0.2 * rng.standard_normal()
        else:
            r = 0.0
            d = 0.0
        
        # Выход объекта
        y = float(Cd @ x) + (Dd[0, 0] if Dd.size > 0 and Dd[0, 0] != 0 else 0.0) * (u_hist[-1] if len(u_hist) > 0 else 0.0)
        
        # Ошибка
        e = r - y
        
        # Регулятор: W(z) = q0 * num_ctrl(z) / den_ctrl(z)
        # den_ctrl(z) * U(z) = q0 * num_ctrl(z) * E(z)
        # В разностном уравнении:
        u = 0.0
        # Вклад от числителя
        for i in range(len(num_ctrl)):
            if i < len(e_buf):
                u += q0 * num_ctrl[i] * e_buf[i]
        # Вклад от знаменателя (со сдвигом на 1)
        for i in range(1, len(den_ctrl)):
            if i < len(u_ctrl_buf):
                u -= den_ctrl[i] * u_ctrl_buf[i]
        
        # Обновление буферов регулятора
        e_buf = np.roll(e_buf, 1)
        e_buf[0] = e
        u_ctrl_buf = np.roll(u_ctrl_buf, 1)
        u_ctrl_buf[0] = u
        
        # Управление объектом (с возмущением)
        u_obj = u + d
        
        # Обновление состояния объекта
        x = Ad @ x + Bd.flatten() * u_obj
        
        y_hist.append(y)
        u_hist.append(u)
        r_hist.append(r)
        d_hist.append(d)
    
    t = np.arange(N) * Ts
    return t, np.array(r_hist), np.array(y_hist), np.array(u_hist), np.array(d_hist)


def plot_and_save(Ts: float, mode: str, fname: str, q0: float = 1.0):
    t, r, y, u, d = simulate(Ts, q0=q0, mode=mode)
    plt.figure(figsize=(8,4))
    if mode == 'set':
        plt.step(t, r, where='post', color='k', lw=1.0, label='r')
    if mode != 'set':
        plt.step(t, d, where='post', color='k', lw=1.0, label='возмущение')
    plt.step(t, y, where='post', color='C0', lw=1.8, label='y')
    plt.xlabel('t, s'); plt.grid(True); plt.legend();
    plt.tight_layout()
    plt.savefig(IMG1 / fname, dpi=150)
    plt.close()


if __name__ == '__main__':
    Ts12 = T1/2
    Ts14 = T1/4
    
    # Загружаем q0 из task2
    q_file = LAB / 'python' / 'q0_T12.txt'
    if q_file.exists():
        q0_12 = float(q_file.read_text().strip())
    else:
        q0_12 = 4.0
    
    q_file = LAB / 'python' / 'q0_T14.txt'
    if q_file.exists():
        q0_14 = float(q_file.read_text().strip())
    else:
        q0_14 = 4.9
    
    # T = T1/2
    plot_and_save(Ts12, 'set', 'step_set_T12.png', q0_12)
    plot_and_save(Ts12, 'dist_step', 'step_dist_T12.png', q0_12)
    plot_and_save(Ts12, 'noise', 'noise_T12.png', q0_12)
    
    # T = T1/4
    plot_and_save(Ts14, 'set', 'step_set_T14.png', q0_14)
    plot_and_save(Ts14, 'dist_step', 'step_dist_T14.png', q0_14)
    plot_and_save(Ts14, 'noise', 'noise_T14.png', q0_14)
    
    print('Сохранены графики task1 для T=T1/2 и T1/4')
