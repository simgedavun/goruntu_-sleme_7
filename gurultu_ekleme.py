import numpy as np
import matplotlib.pyplot as plt


def ikinin_kuvveti_mi(n):
    return n > 0 and (n & (n - 1)) == 0


def sinus_doldur(vektor, genlik, faz, frekans=None, periyot=None, fs=1.0):
    if not ikinin_kuvveti_mi(len(vektor)):
        raise ValueError("Vektor boyutu 2'nin kuvveti olmali!")

    if periyot is not None:
        frekans = 1 / periyot

    for n in range(len(vektor)):
        t = n / fs
        vektor[n] = genlik * np.sin(2 * np.pi * frekans * t + faz)

    return vektor


# Her elemana [-oran*genlik, +oran*genlik] araliginda rastgele gurultu ekler
def gurultu_ekle(vektor, oran):
    if oran < 0 or oran > 1:
        raise ValueError("Gurultu orani 0 ile 1 arasinda olmali!")

    genlik = np.max(np.abs(vektor))
    gurultulu = np.zeros(len(vektor))

    for i in range(len(vektor)):
        gurultu = np.random.uniform(-1, 1) * oran * genlik
        gurultulu[i] = vektor[i] + gurultu

    return gurultulu


N = 64
fs = 64
genlik = 2
faz = np.pi / 4
frekans = 3

vektor = np.zeros(N)
sinus_doldur(vektor, genlik, faz, frekans=frekans, fs=fs)

oranlar = [0.1, 0.3, 0.6, 1.0]
t = np.arange(N) / fs

plt.figure(figsize=(12, 8))

for i, oran in enumerate(oranlar):
    gurultulu = gurultu_ekle(vektor, oran)

    plt.subplot(2, 2, i + 1)
    plt.plot(t, vektor, "b-", label="Orijinal")
    plt.plot(t, gurultulu, "r.-", label="Gurultulu")
    plt.title(f"Gurultu orani = {oran}")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Genlik")
    plt.ylim(-4.5, 4.5)
    plt.legend(loc="upper right")
    plt.grid()

plt.tight_layout()
plt.savefig("gurultu_grafik.png")
plt.show()