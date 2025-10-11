#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

# Вариант 8 из Табл.3
T = 0.2       # с
A = 1.3       # амплитуда
omega = 0.37  # рад/с

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "images" / "task3"
IMG_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class SS:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


def harmonic_generator(T: float, Aamp: float, omega: float) -> SS:
    """Дискретный генератор гармонического сигнала g(k)=A sin(k T ω).
    Реализуем через поворотную матрицу на угол θ=ωT:
        x_{k+1} = R(θ) x_k,   y_k = A * [0 1] x_k,  x0=[1,0]^T дает sin.
    где R(θ)=[[cosθ, -sinθ],[sinθ, cosθ]].
    """
    theta = omega * T
    c, s = np.cos(theta), np.sin(theta)
    Ad = np.array([[c, -s], [s, c]], dtype=float)
    Bd = np.zeros((2, 1))
    Cd = np.array([[0.0, Aamp]], dtype=float)  # берем второй компонент -> sin(kθ)
    Dd = np.array([[0.0]])
    return SS(Ad, Bd, Cd, Dd)


def simulate_ss(ss: SS, x0: np.ndarray, N: int):
    x = x0.copy().astype(float).reshape(-1)
    X = [x.copy()]
    Y = [float(ss.C @ x + ss.D)]
    for _ in range(N):
        x = (ss.A @ x)
        X.append(x.copy())
        y = float(ss.C @ x + ss.D)
        Y.append(y)
    return np.array(X), np.array(Y)


def save_plot(t, y, title, fname):
    plt.figure(figsize=(8, 3))
    # Ступенчатая функция (ZOH) - корректно для дискретных систем
    plt.step(t, y, where='post', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel('t, c')
    plt.ylabel('g(k)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig((IMG_DIR / f"{fname}.png").as_posix(), dpi=160)
    plt.close()


def disturbance_model(T_dist: float) -> SS:
    """Модель возмущения варианта 8 из Табл.4: 4 sin(2 k T) + 1.5 cos(2.5 k T).
    Представим суммой двух осцилляторов (без входа):
      y1=A1*sin(k*2T) -> θ1=2T, выход [0 A1] x1
      y2=A2*cos(k*2.5T) -> cos получается как [A2 0] x2
    Итоговый выход y=y1+y2.
    """
    A1, w1 = 4.0, 2.0
    A2, w2 = 1.5, 2.5
    th1, th2 = w1 * T, w2 * T
    c1, s1 = np.cos(th1), np.sin(th1)
    c2, s2 = np.cos(th2), np.sin(th2)
    Ablk = np.block([
        [c1, -s1,   0,   0],
        [s1,  c1,   0,   0],
        [ 0,   0,  c2, -s2],
        [ 0,   0,  s2,  c2]
    ])
    B = np.zeros((4, 1))
    C = np.array([[0.0, A1, A2, 0.0]])  # sin -> второй, cos -> третий
    D = np.array([[0.0]])
    return SS(Ablk, B, C, D)


def main():
    N = 500
    t = np.arange(N+1) * T

    # Генератор гармонического сигнала
    gen = harmonic_generator(T, A, omega)
    x0 = np.array([1.0, 0.0])  # старт для sin
    X, Y = simulate_ss(gen, x0, N)
    save_plot(t, Y, f'g(k)= {A} sin(k T ω), T={T}, ω={omega}', 'gen_harmonic')

    # Модель возмущения (T = 0.25 с согласно заданию 3d)
    T_dist = 0.25
    dist = disturbance_model(T_dist)
    N_dist = 200  # фиксированное количество точек
    t_dist = np.arange(N_dist+1) * T_dist
    x0d = np.array([1.0, 0.0, 1.0, 0.0])  # стартовые фазы
    Xd, Yd = simulate_ss(dist, x0d, N_dist)
    save_plot(t_dist, Yd, f'Возмущение: 4 sin(2kT) + 1.5 cos(2.5kT), T={T_dist}', 'disturbance')


if __name__ == '__main__':
    main()


