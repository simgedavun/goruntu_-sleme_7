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


def gurultu_ekle(vektor, oran):
    if oran < 0 or oran > 1:
        raise ValueError("Gurultu orani 0 ile 1 arasinda olmali!")

    genlik = np.max(np.abs(vektor))
    gurultulu = np.zeros(len(vektor))

    for i in range(len(vektor)):
        gurultu = np.random.uniform(-1, 1) * oran * genlik
        gurultulu[i] = vektor[i] + gurultu

    return gurultulu


def dft(vektor):
    N = len(vektor)
    if not ikinin_kuvveti_mi(N):
        raise ValueError("Vektor boyutu 2'nin kuvveti olmali!")

    harmonikler = np.zeros(N, dtype=complex)
    for k in range(N):
        toplam = 0
        for n in range(N):
            toplam += vektor[n] * np.exp(-2j * np.pi * k * n / N)
        harmonikler[k] = toplam

    return harmonikler


# fs = N secildigi icin k. harmonik k Hz'e karsilik gelir
N = 256
fs = 256
oran = 0.3

np.random.seed(2)

sinyal1 = np.zeros(N)
sinus_doldur(sinyal1, genlik=12, faz=0, frekans=50, fs=fs)
sinyal1 = gurultu_ekle(sinyal1, oran)

sinyal2 = np.zeros(N)
sinus_doldur(sinyal2, genlik=5, faz=0, frekans=30, fs=fs)
sinyal2 = gurultu_ekle(sinyal2, oran)

toplam = sinyal1 + sinyal2

X1 = dft(sinyal1)
X2 = dft(sinyal2)
X_toplam = dft(toplam)

print("En buyuk harmonikler (k < N/2), genlik tahmini = 2*|X[k]|/N")
for isim, X in [("Sinyal 1", X1), ("Sinyal 2", X2), ("Toplam", X_toplam)]:
    yari = np.abs(X[:N // 2])
    enbuyuk = np.argsort(yari)[::-1][:2]
    for k in sorted(enbuyuk):
        print(f"{isim:9s} k = {k:3d} Hz   |X[k]| = {yari[k]:7.1f}   genlik = {2 * yari[k] / N:5.2f}")

t = np.arange(N) / fs
k = np.arange(N)

plt.figure(figsize=(13, 11))

plt.subplot(3, 2, 1)
plt.plot(t, sinyal1, "b-")
plt.title("Sinyal 1: f = 50 Hz, A = 12 (gurultulu)")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.xlim(0, 0.25)
plt.grid()

plt.subplot(3, 2, 2)
plt.stem(k, np.abs(X1), markerfmt=" ")
plt.title("Sinyal 1 Harmonikleri |X[k]|")
plt.xlabel("k (Hz)")
plt.ylabel("|X[k]|")
plt.grid()

plt.subplot(3, 2, 3)
plt.plot(t, sinyal2, "g-")
plt.title("Sinyal 2: f = 30 Hz, A = 5 (gurultulu)")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.xlim(0, 0.25)
plt.grid()

plt.subplot(3, 2, 4)
plt.stem(k, np.abs(X2), linefmt="g-", markerfmt=" ")
plt.title("Sinyal 2 Harmonikleri |X[k]|")
plt.xlabel("k (Hz)")
plt.ylabel("|X[k]|")
plt.grid()

plt.subplot(3, 2, 5)
plt.plot(t, toplam, "r-")
plt.title("Toplam Sinyal (Sinyal 1 + Sinyal 2)")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.xlim(0, 0.25)
plt.grid()

plt.subplot(3, 2, 6)
plt.stem(k, np.abs(X_toplam), linefmt="r-", markerfmt=" ")
plt.title("Toplam Sinyalin Harmonikleri |X[k]|")
plt.xlabel("k (Hz)")
plt.ylabel("|X[k]|")
plt.grid()

plt.tight_layout()
plt.savefig("iki_sinyal_dft_grafik.png")
plt.show()