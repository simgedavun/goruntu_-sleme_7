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


# x[n] = (1/N) * sum( X[k] * e^(j*2*pi*k*n/N) )
def ters_dft(harmonikler):
    N = len(harmonikler)
    sinyal = np.zeros(N, dtype=complex)
    for n in range(N):
        toplam = 0
        for k in range(N):
            toplam += harmonikler[k] * np.exp(2j * np.pi * k * n / N)
        sinyal[n] = toplam / N

    return sinyal.real


def frekans_filtresi(vektor, esik):
    X = dft(vektor)
    X_filtreli = X.copy()

    for k in range(len(X)):
        if np.abs(X[k]) < esik:
            X_filtreli[k] = 0

    yeni_sinyal = ters_dft(X_filtreli)
    return yeni_sinyal, X, X_filtreli


def hata(a, b):
    return np.mean((a - b) ** 2)


N = 256
fs = 256
oran = 0.3

np.random.seed(2)

temiz1 = np.zeros(N)
sinus_doldur(temiz1, genlik=12, faz=0, frekans=50, fs=fs)
sinyal1 = gurultu_ekle(temiz1, oran)

temiz2 = np.zeros(N)
sinus_doldur(temiz2, genlik=5, faz=0, frekans=30, fs=fs)
sinyal2 = gurultu_ekle(temiz2, oran)

temiz_toplam = temiz1 + temiz2
toplam = sinyal1 + sinyal2

# isim, gurultulu sinyal, temiz sinyal, esik
testler = [
    ("Sinyal 1 (50 Hz, A=12)", sinyal1, temiz1, 200),
    ("Sinyal 2 (30 Hz, A=5)", sinyal2, temiz2, 200),
    ("Toplam sinyal", toplam, temiz_toplam, 200),
    ("Toplam sinyal (yuksek esik)", toplam, temiz_toplam, 1000),
]

t = np.arange(N) / fs
k = np.arange(N)

plt.figure(figsize=(14, 14))

for i, (isim, sinyal, temiz, esik) in enumerate(testler):
    yeni, X, X_filtreli = frekans_filtresi(sinyal, esik)
    kalan = [j for j in range(N // 2) if X_filtreli[j] != 0]

    print(f"{isim}, esik = {esik}")
    print(f"  Kalan harmonikler (k < N/2): {kalan}")
    print(f"  MSE filtre oncesi = {hata(sinyal, temiz):.3f}   filtre sonrasi = {hata(yeni, temiz):.3f}")

    plt.subplot(4, 2, 2 * i + 1)
    plt.plot(t, sinyal, color="lightgray", label="Gurultulu")
    plt.plot(t, temiz, "b--", label="Orijinal")
    plt.plot(t, yeni, "r-", label="Filtrelenmis")
    plt.title(f"{isim}, esik = {esik}")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Genlik")
    plt.xlim(0, 0.25)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid()

    plt.subplot(4, 2, 2 * i + 2)
    plt.stem(k, np.abs(X), linefmt="gray", markerfmt=" ", basefmt="k-", label="Filtre oncesi")
    kalan_tum = np.nonzero(X_filtreli)[0]
    plt.stem(kalan_tum, np.abs(X_filtreli[kalan_tum]), linefmt="r-", markerfmt="ro", basefmt=" ", label="Filtre sonrasi")
    plt.axhline(esik, color="g", linestyle="--", label=f"Esik = {esik}")
    plt.title(f"Harmonik vektoru - {isim}")
    plt.xlabel("k (Hz)")
    plt.ylabel("|X[k]|")
    plt.legend(loc="upper center", fontsize=8)
    plt.grid()

plt.tight_layout()
plt.savefig("frekans_filtresi_grafik.png")
plt.show()