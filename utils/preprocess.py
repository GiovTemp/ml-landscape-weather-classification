import cv2
import numpy as np

# Ridimensiona l'immagine a una dimensione specifica (ad esempio 256x256) e ritaglia eventuali bordi neri
def resize_and_crop(img, size):
    h, w = img.shape[:2]
    if h > w:
        img = cv2.resize(img, (size, int(h * size / w)))
    else:
        img = cv2.resize(img, (int(w * size / h), size))
    h, w = img.shape[:2]
    h_margin = (h - size) // 2
    w_margin = (w - size) // 2
    return img[h_margin:h_margin+size, w_margin:w_margin+size]


# Applica l'equalizzazione dell'istogramma all'immagine
def equalize_hist(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)


# Applica il filtro di mediana all'immagine
def median_filter(img, kernel_size):
    return cv2.medianBlur(img, kernel_size)

# Applica la segmentazione basata sulla soglia all'immagine per separare il cielo dal terreno
def threshold_segmentation(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return thresh

# Applica una rotazione casuale all'immagine
def random_rotate(img, angle_range):
    angle = np.random.uniform(-angle_range, angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def preprocess_image(img_path, size=256, threshold=150, kernel_size=3, angle_range=10):
    # Leggi l'immagine dal file
    img = cv2.imread(img_path)
    
    #Convert the image into RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ridimensiona e ritaglia l'immagine
    img = resize_and_crop(img, size)

    # Applica l'equalizzazione dell'istogramma
    img = equalize_hist(img)

    # Applica il filtro di mediana
    #img = median_filter(img, kernel_size)

    # Applica la segmentazione basata sulla soglia
    img = threshold_segmentation(img, threshold)

    # Applica una rotazione casuale
    img = random_rotate(img, angle_range)

    # Normalizza l'immagine
    img = img.astype(np.float32) / 255.

    # Aggiungi una dimensione per rappresentare il batch
    #img = np.expand_dims(img, axis=-1)

    return img
