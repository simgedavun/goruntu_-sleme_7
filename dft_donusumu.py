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


# X[k] = sum( x[n] * e^(-j*2*pi*k*n/N) ),  k = 0..N-1
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


N = 64
fs = 64
genlik = 2
faz = np.pi / 4
frekans = 3

gurultusuz = np.zeros(N)
sinus_doldur(gurultusuz, genlik, faz, frekans=frekans, fs=fs)

np.random.seed(0)
gurultu = np.random.normal(0, 0.8, N)
gurultulu = gurultusuz + gurultu

X_gurultusuz = dft(gurultusuz)
X_gurultulu = dft(gurultulu)

# Kendi DFT'mizi numpy'nin FFT'si ile karsilastir
print("Gurultusuz DFT dogru mu:", np.allclose(X_gurultusuz, np.fft.fft(gurultusuz)))
print("Gurultulu DFT dogru mu: ", np.allclose(X_gurultulu, np.fft.fft(gurultulu)))

print("\nHarmonik genlikleri |X[k]| (ilk 8):")
for k in range(8):
    print(f"k={k}:  gurultusuz = {abs(X_gurultusuz[k]):6.2f}   gurultulu = {abs(X_gurultulu[k]):6.2f}")

k = np.arange(N)
t = np.arange(N) / fs

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, gurultusuz, "b.-")
plt.title("Gurultusuz Sinyal")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.grid()

plt.subplot(2, 2, 2)
plt.stem(k, np.abs(X_gurultusuz))
plt.title("Gurultusuz Sinyalin Harmonikleri |X[k]|")
plt.xlabel("k (harmonik)")
plt.ylabel("|X[k]|")
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(t, gurultulu, "r.-")
plt.title("Gurultulu Sinyal")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.grid()

plt.subplot(2, 2, 4)
plt.stem(k, np.abs(X_gurultulu), linefmt="r-", markerfmt="ro")
plt.title("Gurultulu Sinyalin Harmonikleri |X[k]|")
plt.xlabel("k (harmonik)")
plt.ylabel("|X[k]|")
plt.grid()

plt.tight_layout()
plt.savefig("dft_grafik.png")
plt.show()
