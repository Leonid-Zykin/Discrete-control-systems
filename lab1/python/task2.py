#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

# Параметры варианта (для всех поднаборов из Табл.2)
T = 0.20  # период дискретизации, с (вариант 8)
ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "images" / "task2"
IMG_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class DiscreteSS:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


def zoh_double_integrator(T: float) -> DiscreteSS:
    """ZOH-дискретизация непрерывной системы y¨=u.
    Для A=[[0,1],[0,0]], B=[[0],[1]] имеем Ad=[[1,T],[0,1]], Bd=[[T**2/2],[T]].
    """
    Ad = np.array([[1.0, T], [0.0, 1.0]], dtype=float)
    Bd = np.array([[0.5*T*T], [T]], dtype=float)
    C = np.array([[1.0, 0.0]], dtype=float)
    D = np.array([[0.0]], dtype=float)
    return DiscreteSS(Ad, Bd, C, D)


def acker(A: np.ndarray, B: np.ndarray, desired_poles: np.ndarray) -> np.ndarray:
    """Ackermann для дискретного случая (аналогичен непрерывному для размещения полюсов)."""
    n = A.shape[0]
    # Контролируемость
    ctrb = B
    for i in range(1, n):
        ctrb = np.hstack([ctrb, np.linalg.matrix_power(A, i) @ B])
    if np.linalg.matrix_rank(ctrb) < n:
        raise ValueError("Система неконтролируема")
    # Желаемый полином
    poly = np.poly(desired_poles)  # z^n + a1 z^{n-1} + ... + an
    # Построение желаемого матричного полинома
    Ak = np.zeros_like(A)
    M = np.eye(n)
    for i in range(n):
        Ak = Ak + poly[n - i] * M  # коэффициенты начиная с z^{n-1}
        M = M @ A
    # Вектор e_n^T = [0 ... 0 1]
    eT = np.zeros((1, n))
    eT[0, -1] = 1.0
    K = eT @ np.linalg.inv(ctrb) @ Ak
    return np.asarray(K).reshape(1, -1)


def simulate_regulator(A: np.ndarray, B: np.ndarray, K: np.ndarray, x0: np.ndarray, N: int = 200):
    Ad_cl = A - B @ K
    x = x0.copy()
    xs = [x.copy()]
    for _ in range(N):
        x = Ad_cl @ x
        xs.append(x.copy())
    X = np.vstack(xs)
    return X


def save_plot(t: np.ndarray, y: np.ndarray, title: str, fname: str):
    plt.figure(figsize=(8, 4))
    # Ступенчатая функция (ZOH) - корректно для дискретных систем
    plt.step(t, y, where='post', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel('t, c')
    plt.ylabel('y(k)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig((IMG_DIR / f"{fname}.png").as_posix(), dpi=160)
    plt.close()


def run_case(idx: int, poles):
    sys = zoh_double_integrator(T)
    poles = np.array(poles, dtype=complex)
    K = acker(sys.A, sys.B, poles)
    N = 200
    t = np.arange(N+1) * T
    x0 = np.array([1.0, 0.0])  # y(0)=1, dy(0)=0
    X = simulate_regulator(sys.A, sys.B, K, x0, N)
    y = X[:, 0]
    title = f'Набор {idx}: полюса {poles}, K={K.ravel()}'
    save_plot(t, y, title, f'set{idx}_step')
    return K


def main():
    # Пять наборов желаемых собственных чисел для варианта 8 (из таблицы 2)
    # Исправлены для более разнообразной динамики
    pole_sets = [
        (0.5, 0.1),                     # 1 - быстрый апериодический
        (0.9, 0.8),                     # 2 - медленный апериодический  
        (0.3, -0.2),                    # 3 - быстрый с небольшой колебательностью
        (0.0+0.7j, 0.0-0.7j),           # 4 - чисто колебательный
        (-0.3+0.8j, -0.3-0.8j)          # 5 - колебательный с затуханием
    ]
    for i, poles in enumerate(pole_sets, start=1):
        try:
            K = run_case(i, poles)
            np.savetxt((IMG_DIR / f'set{i}_K.txt').as_posix(), K)
        except Exception as e:
            print(f"Ошибка в наборе {i}: {e}")


if __name__ == '__main__':
    main()
