import os
import numpy as np
import matplotlib.pyplot as plt


def ikinin_kuvveti_mi(n):
    return n > 0 and (n & (n - 1)) == 0


def dft_2d(resim):
    M, N = resim.shape
    if not ikinin_kuvveti_mi(M) or not ikinin_kuvveti_mi(N):
        raise ValueError("Resim boyutlari 2'nin kuvveti olmali!")

    W_M = np.zeros((M, M), dtype=complex)
    for u in range(M):
        for x in range(M):
            W_M[u, x] = np.exp(-2j * np.pi * u * x / M)

    W_N = np.zeros((N, N), dtype=complex)
    for y in range(N):
        for v in range(N):
            W_N[y, v] = np.exp(-2j * np.pi * y * v / N)

    return W_M @ resim @ W_N


def resim_yukle(dosya, boyut=128):
    resim = plt.imread(dosya).astype(float)

    if resim.ndim == 3:
        resim = resim[:, :, :3].mean(axis=2)

    # Piksel degerlerini 0-1 araligina getir
    if resim.max() > 1:
        resim = resim / 255

    satirlar = (np.arange(boyut) * resim.shape[0] / boyut).astype(int)
    sutunlar = (np.arange(boyut) * resim.shape[1] / boyut).astype(int)
    return resim[satirlar][:, sutunlar]


def ortala(harmonikler):
    M, N = harmonikler.shape
    return np.roll(np.roll(harmonikler, M // 2, axis=0), N // 2, axis=1)


# Her piksele ortalamasi 0, standart sapmasi "kuvvet" olan rastgele deger ekler
def gauss_gurultu_ekle(resim, kuvvet):
    gurultu = np.random.normal(0, kuvvet, resim.shape)
    gurultulu = resim + gurultu
    return np.clip(gurultulu, 0, 1)


# Piksellerin "oran" kadarini rastgele secip yarisini siyah (0), yarisini beyaz (1) yapar
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


# Merkezden uzaklasan (yuksek frekans) harmoniklerdeki enerjinin yuzdesi
def yuksek_frekans_orani(harmonikler, yaricap=10):
    E = np.abs(ortala(harmonikler)) ** 2
    M, N = E.shape
    u, v = np.mgrid[-M // 2:M // 2, -N // 2:N // 2]
    yuksek = (u ** 2 + v ** 2) > yaricap ** 2
    return 100 * E[yuksek].sum() / E.sum()


boyut = 128
np.random.seed(0)

if os.path.exists("resim.jpg"):
    resim = resim_yukle("resim.jpg", boyut)
else:
    resim = np.zeros((boyut, boyut))
    resim[44:84, 34:94] = 1

gauss_resim = gauss_gurultu_ekle(resim, kuvvet=0.1)
sp_resim = salt_and_pepper_ekle(resim, oran=0.05)

X_orijinal = dft_2d(resim)
X_gauss = dft_2d(gauss_resim)
X_sp = dft_2d(sp_resim)

resimler = [
    ("Orijinal", resim, X_orijinal),
    ("Gauss gurultu (kuvvet = 0.1)", gauss_resim, X_gauss),
    ("Salt & Pepper (oran = 0.05)", sp_resim, X_sp),
]

print("Yuksek frekanslardaki enerji orani (merkezden 10 birimden uzak):")
for isim, r, X in resimler:
    print(f"  {isim:30s}: %{yuksek_frekans_orani(X):.2f}")

ust = boyut // 2
eksen = [-ust, ust, ust, -ust]

# Uc spektrumu ayni renk araliginda gostermek icin ortak sinirlar
log_spektrumlar = [np.log(1 + np.abs(ortala(X))) for _, _, X in resimler]
alt_sinir = min(s.min() for s in log_spektrumlar)
ust_sinir = max(s.max() for s in log_spektrumlar)

plt.figure(figsize=(15, 10))

for i, (isim, r, X) in enumerate(resimler):
    plt.subplot(2, 3, i + 1)
    plt.imshow(r, cmap="gray", vmin=0, vmax=1)
    plt.title(isim)
    plt.axis("off")

    plt.subplot(2, 3, i + 4)
    plt.imshow(log_spektrumlar[i], cmap="gray", extent=eksen, vmin=alt_sinir, vmax=ust_sinir)
    plt.title(f"Harmonik matrisi - {isim.split(' (')[0]}")
    plt.xlabel("v")
    plt.ylabel("u")

plt.tight_layout()
plt.savefig("resim_gurultu_grafik.png")
plt.show()