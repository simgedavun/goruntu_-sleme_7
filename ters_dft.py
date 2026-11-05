import os
import numpy as np
import matplotlib.pyplot as plt


def ikinin_kuvveti_mi(n):
    return n > 0 and (n & (n - 1)) == 0


def dft_matrisi(N, isaret):
    W = np.zeros((N, N), dtype=complex)
    for a in range(N):
        for b in range(N):
            W[a, b] = np.exp(isaret * 2j * np.pi * a * b / N)
    return W


def dft_2d(resim):
    M, N = resim.shape
    if not ikinin_kuvveti_mi(M) or not ikinin_kuvveti_mi(N):
        raise ValueError("Resim boyutlari 2'nin kuvveti olmali!")
    return dft_matrisi(M, -1) @ resim @ dft_matrisi(N, -1)


# f[x, y] = (1/MN) * sum_u sum_v  X[u, v] * e^(+j*2*pi*(u*x/M + v*y/N))
def ters_dft_2d(harmonikler):
    M, N = harmonikler.shape
    resim = dft_matrisi(M, +1) @ harmonikler @ dft_matrisi(N, +1) / (M * N)
    return resim.real


def resim_yukle(dosya, boyut=128):
    resim = plt.imread(dosya).astype(float)
    if resim.ndim == 3:
        resim = resim[:, :, :3].mean(axis=2)
    if resim.max() > 1:
        resim = resim / 255
    satirlar = (np.arange(boyut) * resim.shape[0] / boyut).astype(int)
    sutunlar = (np.arange(boyut) * resim.shape[1] / boyut).astype(int)
    return resim[satirlar][:, sutunlar]


def gauss_gurultu_ekle(resim, kuvvet):
    gurultu = np.random.normal(0, kuvvet, resim.shape)
    return np.clip(resim + gurultu, 0, 1)


def salt_and_pepper_ekle(resim, oran):
    if oran < 0 or oran > 1:
        raise ValueError("Oran 0 ile 1 arasinda olmali!")
    gurultulu = resim.copy()
    M, N = resim.shape
    for i in range(M):
        for j in range(N):
            r = np.random.rand()
            if r < oran / 2:
                gurultulu[i, j] = 0
            elif r < oran:
                gurultulu[i, j] = 1
    return gurultulu


# Gauss alcak geciren filtre: merkezden uzaklastikca harmonikleri zayiflatir
# H(u, v) = e^(-D^2 / (2 * D0^2)),  D = merkeze uzaklik
def alcak_geciren_filtre(resim, D0):
    M, N = resim.shape
    X = dft_2d(resim)

    u = np.arange(M)
    v = np.arange(N)
    u = np.minimum(u, M - u)          # k ve N-k ayni frekans oldugu icin
    v = np.minimum(v, N - v)
    D2 = u[:, None] ** 2 + v[None, :] ** 2
    H = np.exp(-D2 / (2 * D0 ** 2))

    return np.clip(ters_dft_2d(X * H), 0, 1)


# ---------------- Benzerlik metrikleri ----------------
def zncc(a, b):
    a0 = a - a.mean()
    b0 = b - b.mean()
    return np.sum(a0 * b0) / np.sqrt(np.sum(a0 ** 2) * np.sum(b0 ** 2))


def mse(a, b):
    return np.mean((a - b) ** 2)


def mae(a, b):
    return np.mean(np.abs(a - b))


def psnr(a, b):
    return 10 * np.log10(1 / mse(a, b))     # pikseller 0-1 araliginda


def ssim(a, b, pencere=8):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    degerler = []
    for i in range(0, a.shape[0], pencere):
        for j in range(0, a.shape[1], pencere):
            x = a[i:i + pencere, j:j + pencere]
            y = b[i:i + pencere, j:j + pencere]
            mx, my = x.mean(), y.mean()
            vx, vy = x.var(), y.var()
            cxy = np.mean((x - mx) * (y - my))
            degerler.append(((2 * mx * my + C1) * (2 * cxy + C2)) /
                            ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2)))
    return np.mean(degerler)


def metrikler(orijinal, r):
    return (f"ZNCC = {zncc(orijinal, r):.4f}   MSE = {mse(orijinal, r):.4f}   "
            f"MAE = {mae(orijinal, r):.4f}   PSNR = {psnr(orijinal, r):5.2f} dB   "
            f"SSIM = {ssim(orijinal, r):.4f}")


boyut = 128
np.random.seed(0)

if os.path.exists("resim.jpg"):
    orijinal = resim_yukle("resim.jpg", boyut)
else:
    orijinal = np.zeros((boyut, boyut))
    orijinal[44:84, 34:94] = 1

gurultuler = [
    ("Gauss gurultu", gauss_gurultu_ekle(orijinal, kuvvet=0.1)),
    ("Salt & Pepper", salt_and_pepper_ekle(orijinal, oran=0.05)),
]

D0_degerleri = [10, 20, 30, 40, 60]
sonuclar = []

for isim, gurultulu in gurultuler:
    print(f"\n===== {isim} =====")
    print(f"Gurultulu resim      : {metrikler(orijinal, gurultulu)}")

    # Filtresiz ters DFT: harmonik matrisi degistirilmeden geri donulur
    geri = ters_dft_2d(dft_2d(gurultulu))
    print(f"Filtresiz ters DFT   : {metrikler(orijinal, geri)}")

    en_iyi_D0, en_iyi_zncc = None, -1
    for D0 in D0_degerleri:
        filtreli = alcak_geciren_filtre(gurultulu, D0)
        z = zncc(orijinal, filtreli)
        print(f"Alcak geciren D0={D0:2d} : {metrikler(orijinal, filtreli)}")
        if z > en_iyi_zncc:
            en_iyi_D0, en_iyi_zncc = D0, z

    sonuclar.append((isim, gurultulu, geri, en_iyi_D0, alcak_geciren_filtre(gurultulu, en_iyi_D0)))
    print(f"En iyi sonuc: D0 = {en_iyi_D0}")

plt.figure(figsize=(16, 9))

for i, (isim, gurultulu, geri, D0, filtreli) in enumerate(sonuclar):
    gorseller = [
        ("Orijinal", orijinal),
        (f"{isim}\nZNCC = {zncc(orijinal, gurultulu):.4f}", gurultulu),
        (f"Filtresiz ters DFT\nZNCC = {zncc(orijinal, geri):.4f}", geri),
        (f"Alcak geciren filtre + ters DFT (D0={D0})\nZNCC = {zncc(orijinal, filtreli):.4f}", filtreli),
    ]
    for j, (baslik, r) in enumerate(gorseller):
        plt.subplot(2, 4, 4 * i + j + 1)
        plt.imshow(r, cmap="gray", vmin=0, vmax=1)
        plt.title(baslik, fontsize=10)
        plt.axis("off")

plt.tight_layout()
plt.savefig("ters_dft_grafik.png")
plt.show()