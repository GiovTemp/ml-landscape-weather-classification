import cv2
import numpy as np

# Carica l'immagine
img = cv2.imread("./test_image.jpg")

# Converte l'immagine in scala di grigi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applica un filtro di Sobel per rilevare i bordi
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
abs_sobelx = cv2.convertScaleAbs(sobelx)
abs_sobely = cv2.convertScaleAbs(sobely)
sobel = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

# Applica una soglia per creare un'immagine binaria
ret, thresh = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)

# Applica un kernel per ridurre il rumore
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Rileva i contorni
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Disegna i contorni sull'immagine originale
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

# Mostra l'immagine risultante
cv2.imshow("Raindrops Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()