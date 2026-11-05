import os
import numpy as np
import matplotlib.pyplot as plt


def ikinin_kuvveti_mi(n):
    return n > 0 and (n & (n - 1)) == 0


# X[u, v] = sum_x sum_y  f[x, y] * e^(-j*2*pi*(u*x/M + v*y/N))
# Bu toplam matris carpimi ile yazilabilir:  X = W_M * f * W_N
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

    harmonikler = W_M @ resim @ W_N
    return harmonikler


def resim_yukle(dosya, boyut=128):
    resim = plt.imread(dosya).astype(float)

    if resim.ndim == 3:
        resim = resim[:, :, :3].mean(axis=2)

    # En yakin komsu yontemi ile boyut x boyut olacak sekilde yeniden boyutlandir
    satirlar = (np.arange(boyut) * resim.shape[0] / boyut).astype(int)
    sutunlar = (np.arange(boyut) * resim.shape[1] / boyut).astype(int)
    return resim[satirlar][:, sutunlar]


# Sifir frekansi (X[0,0]) resmin ortasina tasir
def ortala(harmonikler):
    M, N = harmonikler.shape
    return np.roll(np.roll(harmonikler, M // 2, axis=0), N // 2, axis=1)


boyut = 128

# Test 1: dikey cizgili yapay resim (yatayda 8 periyot)
x = np.arange(boyut)
cizgili = np.tile(np.sin(2 * np.pi * 8 * x / boyut), (boyut, 1))

# Test 2: gercek resim (klasorde resim.jpg varsa o kullanilir)
if os.path.exists("resim.jpg"):
    resim = resim_yukle("resim.jpg", boyut)
    baslik = "Resim (resim.jpg)"
else:
    resim = np.zeros((boyut, boyut))
    resim[44:84, 34:94] = 1
    baslik = "Resim (beyaz dikdortgen)"

for isim, r in [("Cizgili resim", cizgili), (baslik, resim)]:
    X = dft_2d(r)
    print(f"{isim}: kendi DFT'miz numpy fft2 ile ayni mi? {np.allclose(X, np.fft.fft2(r))}")

X_cizgili = dft_2d(cizgili)
X_resim = dft_2d(resim)

ust = boyut // 2
eksen = [-ust, ust, ust, -ust]

plt.figure(figsize=(11, 10))

plt.subplot(2, 2, 1)
plt.imshow(cizgili, cmap="gray")
plt.title("Cizgili resim (8 periyot)")

plt.subplot(2, 2, 2)
plt.imshow(np.log(1 + np.abs(ortala(X_cizgili))), cmap="gray", extent=eksen)
plt.title("Harmonik matrisi log(1 + |X[u,v]|)")
plt.xlabel("v")
plt.ylabel("u")

plt.subplot(2, 2, 3)
plt.imshow(resim, cmap="gray")
plt.title(baslik)

plt.subplot(2, 2, 4)
plt.imshow(np.log(1 + np.abs(ortala(X_resim))), cmap="gray", extent=eksen)
plt.title("Harmonik matrisi log(1 + |X[u,v]|)")
plt.xlabel("v")
plt.ylabel("u")

plt.tight_layout()
plt.savefig("dft_2d_grafik.png")
plt.show()