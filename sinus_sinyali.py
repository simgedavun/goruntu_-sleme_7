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


N = 64
fs = 64
genlik = 2
faz = np.pi / 4
frekans = 3

vektor = np.zeros(N)
sinus_doldur(vektor, genlik, faz, frekans=frekans, fs=fs)
# periyot ile: sinus_doldur(vektor, genlik, faz, periyot=1/3, fs=fs)

print("Vektor degerleri:")
print(vektor)

t = np.arange(N) / fs
t_surekli = np.linspace(0, N / fs, 1000)
y_surekli = genlik * np.sin(2 * np.pi * frekans * t_surekli + faz)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.stem(np.arange(N), vektor)
plt.title("Vektor (N = 64)")
plt.xlabel("n")
plt.ylabel("Genlik")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t_surekli, y_surekli, label="Sinus dalgasi")
plt.plot(t, vektor, "ro", markersize=4, label="Vektor ornekleri")
plt.title("Sinuzoidal Dalga")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.legend()
plt.grid()


plt.tight_layout()
plt.savefig("sinus_grafik.png")
plt.show()
plt.show()