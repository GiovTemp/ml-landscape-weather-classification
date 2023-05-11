import preprocess
import os
import cv2


# Percorso della directory di origine delle immagini

#original_dir = 'dataset_train'
#preprocessed_dir = 'dataset_train_preprocessed'

original_dir = 'dataset_test'
preprocessed_dir = 'dataset_test_preprocessed'

# Crea una nuova directory per le immagini pre-processate
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

# Ottieni una lista di tutti i file presenti nella directory di origine
dirnames = os.listdir(original_dir)

for dirname in dirnames :
    filenames= os.listdir(os.path.join(original_dir,dirname))
    for filename in filenames :
        preprocessed_img = preprocess.preprocess_image(os.path.join(original_dir,dirname,filename))
        if not os.path.exists(os.path.join(preprocessed_dir,dirname)):
            os.makedirs(os.path.join(preprocessed_dir,dirname))
        cv2.imwrite(os.path.join(preprocessed_dir,dirname,filename),preprocessed_img)









