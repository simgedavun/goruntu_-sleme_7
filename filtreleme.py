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


def gauss_filtre(vektor, boyut=5, sigma=1.0):
    yari = boyut // 2
    x = np.arange(-yari, yari + 1)
    cekirdek = np.exp(-(x ** 2) / (2 * sigma ** 2))
    cekirdek = cekirdek / np.sum(cekirdek)

    # Kenarlarda tasma olmamasi icin bas ve son elemanlar tekrarlaniyor
    genis = np.pad(vektor, yari, mode="edge")
    sonuc = np.zeros(len(vektor))

    for i in range(len(vektor)):
        pencere = genis[i:i + boyut]
        sonuc[i] = np.sum(pencere * cekirdek)

    return sonuc


def medyan_filtre(vektor, boyut=5):
    yari = boyut // 2
    genis = np.pad(vektor, yari, mode="edge")
    sonuc = np.zeros(len(vektor))

    for i in range(len(vektor)):
        pencere = genis[i:i + boyut]
        sonuc[i] = np.median(pencere)

    return sonuc


def hata(a, b):
    return np.mean((a - b) ** 2)


N = 64
fs = 64
genlik = 2
faz = np.pi / 4
frekans = 3

np.random.seed(1)

vektor = np.zeros(N)
sinus_doldur(vektor, genlik, faz, frekans=frekans, fs=fs)

gurultulu = gurultu_ekle(vektor, 0.5)
gauss_sonuc = gauss_filtre(gurultulu, boyut=5, sigma=1.0)
medyan_sonuc = medyan_filtre(gurultulu, boyut=5)

print("Ortalama kare hata (MSE) - orijinal sinyale gore:")
print(f"Gurultulu  : {hata(gurultulu, vektor):.4f}")
print(f"Gauss      : {hata(gauss_sonuc, vektor):.4f}")
print(f"Medyan     : {hata(medyan_sonuc, vektor):.4f}")

t = np.arange(N) / fs

plt.figure(figsize=(12, 9))

plt.subplot(2, 2, 1)
plt.plot(t, vektor, "b-", label="Orijinal")
plt.plot(t, gurultulu, "r.-", label="Gurultulu")
plt.title(f"Gurultulu Sinyal (MSE = {hata(gurultulu, vektor):.3f})")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.legend(loc="upper right")
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(t, vektor, "b-", label="Orijinal")
plt.plot(t, gauss_sonuc, "g.-", label="Gauss filtre")
plt.title(f"Gauss Filtre (MSE = {hata(gauss_sonuc, vektor):.3f})")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.legend(loc="upper right")
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(t, vektor, "b-", label="Orijinal")
plt.plot(t, medyan_sonuc, "m.-", label="Medyan filtre")
plt.title(f"Medyan Filtre (MSE = {hata(medyan_sonuc, vektor):.3f})")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.legend(loc="upper right")
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(t, vektor, "b-", label="Orijinal")
plt.plot(t, gauss_sonuc, "g-", label="Gauss")
plt.plot(t, medyan_sonuc, "m-", label="Medyan")
plt.title("Karsilastirma")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.legend(loc="upper right")
plt.grid()

plt.tight_layout()
plt.savefig("filtre_grafik.png")
plt.show()