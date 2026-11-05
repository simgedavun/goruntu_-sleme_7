
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Görüntü yükleme
# ----------------------------------------------------------------------
def goruntu_yukle():
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"Resim okunamadı: {sys.argv[1]}")
        return img.astype(np.float64)
    try:
        from skimage import data
        return data.camera().astype(np.float64)
    except ImportError:                                   # sentetik yedek
        img = np.full((512, 512), 60.0)
        cv2.rectangle(img, (80, 80), (250, 250), 200, -1)
        cv2.circle(img, (370, 160), 90, 150, -1)
        cv2.putText(img, "TEST", (90, 420), cv2.FONT_HERSHEY_SIMPLEX, 4, 230, 12)
        return img


# ----------------------------------------------------------------------
# Gürültüler
# ----------------------------------------------------------------------
def gauss_gurultu(img, sigma=20, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255)


def tuz_biber_gurultu(img, oran=0.05, seed=1):
    rng = np.random.default_rng(seed)
    out = img.copy()
    r = rng.random(img.shape)
    out[r < oran / 2] = 0
    out[r > 1 - oran / 2] = 255
    return out


PERIYOT_U, PERIYOT_V = 40, 40           # periyodik gürültünün frekansı

def periyodik_gurultu(img, genlik=40):
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W]
    desen = genlik * np.sin(2 * np.pi * (PERIYOT_U * x / W + PERIYOT_V * y / H))
    return np.clip(img + desen, 0, 255)


# ----------------------------------------------------------------------
# Filtre maskeleri H(u,v)   (merkez = sıfır frekans)
# ----------------------------------------------------------------------
def uzaklik_matrisi(shape):
    H, W = shape
    u = np.arange(H) - H / 2
    v = np.arange(W) - W / 2
    V, U = np.meshgrid(v, u)
    return np.sqrt(U ** 2 + V ** 2)          # D(u,v)


def low_pass(shape, D0, tip="gauss", n=2):
    D = uzaklik_matrisi(shape)
    if tip == "ideal":
        return (D <= D0).astype(np.float64)
    if tip == "butterworth":
        return 1.0 / (1.0 + (D / D0) ** (2 * n))
    return np.exp(-(D ** 2) / (2 * D0 ** 2))  # gauss


def high_pass(shape, D0, tip="gauss", n=2):
    return 1.0 - low_pass(shape, D0, tip, n)


def band_reject(shape, D0, Wb, tip="gauss", n=2):
    """D0: bandın merkez yarıçapı, Wb: bant genişliği"""
    D = uzaklik_matrisi(shape)
    D = np.where(D == 0, 1e-8, D)
    if tip == "ideal":
        return np.where(np.abs(D - D0) <= Wb / 2, 0.0, 1.0)
    if tip == "butterworth":
        return 1.0 / (1.0 + ((D * Wb) / (D ** 2 - D0 ** 2 + 1e-8)) ** (2 * n))
    return 1.0 - np.exp(-(((D ** 2 - D0 ** 2) / (D * Wb)) ** 2))  # gauss


def band_pass(shape, D0, Wb, tip="gauss", n=2):
    return 1.0 - band_reject(shape, D0, Wb, tip, n)


# ----------------------------------------------------------------------
# Frekans uzayında filtreleme
# ----------------------------------------------------------------------
def fft_filtrele(img, H):
    F = np.fft.fftshift(np.fft.fft2(img))     # merkezlenmiş spektrum
    G = F * H                                 # filtre uygulama
    g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    return g


def spektrum(img):
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(F))


# ----------------------------------------------------------------------
# Ölçütler
# ----------------------------------------------------------------------
def psnr(ref, img):
    mse = np.mean((ref - np.clip(img, 0, 255)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def kenar_siddeti(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.hypot(gx, gy))


# ----------------------------------------------------------------------
def main():
    orijinal = goruntu_yukle()
    shape = orijinal.shape

    goruntuler = {
        "Orijinal": orijinal,
        "Gauss gürültü": gauss_gurultu(orijinal),
        "Tuz-biber": tuz_biber_gurultu(orijinal),
        "Periyodik": periyodik_gurultu(orijinal),
    }

    # Periyodik gürültünün spektrumdaki yarıçapı (band-reject bunu hedefler)
    D_gurultu = np.hypot(PERIYOT_U, PERIYOT_V)

    TIP = "gauss"
    filtreler = {
        "Low-pass (D0=30)":          low_pass(shape, 30, TIP),
        "High-pass (D0=30)":         high_pass(shape, 30, TIP),
        "Band-pass (D0=60, W=40)":   band_pass(shape, 60, 40, TIP),
        f"Band-reject (D0={D_gurultu:.0f}, W=10)": band_reject(shape, D_gurultu, 10, TIP),
    }

    # ---------------- 1) Filtre maskeleri ----------------
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, (ad, H) in zip(ax, filtreler.items()):
        a.imshow(H, cmap="gray", vmin=0, vmax=1)
        a.set_title(ad); a.axis("off")
    fig.suptitle("Filtre maskeleri H(u,v) (Gauss tipi, beyaz = geçen frekans)")
    fig.tight_layout(); fig.savefig("1_filtre_maskeleri.png", dpi=110)

    # ---------------- 2) Spektrumlar ----------------
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, (ad, img) in zip(ax, goruntuler.items()):
        a.imshow(spektrum(img), cmap="magma")
        a.set_title(f"{ad} - spektrum"); a.axis("off")
    fig.tight_layout(); fig.savefig("2_spektrumlar.png", dpi=110)

    # ---------------- 3) Sonuç ızgarası + ölçütler ----------------
    satir, sutun = len(goruntuler), len(filtreler) + 1
    fig, ax = plt.subplots(satir, sutun, figsize=(4 * sutun, 4 * satir))
    tablo = []
    for i, (g_ad, img) in enumerate(goruntuler.items()):
        ax[i, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        ax[i, 0].set_title(f"{g_ad}\nort={img.mean():.0f}  PSNR={psnr(orijinal, img):.1f}")
        tablo.append((g_ad, "-", img.mean(), psnr(orijinal, img), kenar_siddeti(img)))
        for j, (f_ad, H) in enumerate(filtreler.items(), start=1):
            out = fft_filtrele(img, H)
            p = psnr(orijinal, out)
            tablo.append((g_ad, f_ad, out.mean(), p, kenar_siddeti(out)))
            ax[i, j].imshow(np.clip(out, 0, 255), cmap="gray", vmin=0, vmax=255)
            ax[i, j].set_title(f"{f_ad}\nort={out.mean():.0f}  PSNR={p:.1f}")
        for a in ax[i]:
            a.axis("off")
    fig.tight_layout(); fig.savefig("3_sonuclar.png", dpi=90)

    # ---------------- 4) High-pass / band-pass kontrast germe ----------------
    # (Ortalaması ~0 olduğu için yukarıda neredeyse siyah görünürler.)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    hp = fft_filtrele(orijinal, filtreler["High-pass (D0=30)"])
    bp = fft_filtrele(orijinal, filtreler["Band-pass (D0=60, W=40)"])
    bp_per = fft_filtrele(goruntuler["Periyodik"], band_pass(shape, D_gurultu, 10, TIP))
    for a, x, t in zip(ax, [hp, bp, bp_per],
                       ["High-pass (gerilmiş)", "Band-pass (gerilmiş)",
                        "Periyodik resim: gürültü bandı (band-pass)"]):
        a.imshow(cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX), cmap="gray")
        a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig("4_hp_bp_gerilmis.png", dpi=110)

    # ---------------- 5) İdeal vs Gauss low-pass (ringing) ----------------
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, tip in zip(ax, ["ideal", "butterworth", "gauss"]):
        out = fft_filtrele(orijinal, low_pass(shape, 30, tip))
        a.imshow(np.clip(out, 0, 255), cmap="gray", vmin=0, vmax=255)
        a.set_title(f"Low-pass {tip} (D0=30)  PSNR={psnr(orijinal, out):.1f}")
        a.axis("off")
    fig.tight_layout(); fig.savefig("5_lowpass_tip_karsilastirma.png", dpi=110)

    # ---------------- Tablo ----------------
    print(f"{'Görüntü':<15}{'Filtre':<28}{'Ort.parlaklık':>14}{'PSNR(dB)':>10}{'Kenar':>8}")
    print("-" * 75)
    for g, f, m, p, k in tablo:
        print(f"{g:<15}{f:<28}{m:>14.1f}{p:>10.1f}{k:>8.1f}")

    print("\nKaydedildi: 1_filtre_maskeleri.png, 2_spektrumlar.png, 3_sonuclar.png, "
          "4_hp_bp_gerilmis.png, 5_lowpass_tip_karsilastirma.png")
    plt.show()


if __name__ == "__main__":
    main()