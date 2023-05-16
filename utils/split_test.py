import os
import random
import shutil

def move_files(source_dir, destination_dir, percentage):
    # Ottieni la lista di tutti i file nella cartella di origine
    file_list = os.listdir(source_dir)
    
    # Calcola il numero di file da selezionare
    num_files = int(len(file_list) * percentage)
    
    # Seleziona casualmente i file da spostare
    files_to_move = random.sample(file_list, num_files)
    
    # Sposta i file nella cartella di destinazione
    for file_name in files_to_move:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        shutil.move(source_path, destination_path)
        print(f"Spostato il file '{file_name}' in '{destination_dir}'.")

# Imposta le cartelle di origine e destinazione
source_folder = 'datasets/dataset2/train/shine'
destination_folder = 'datasets/dataset2/test/shine'
# Create the destination directory if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
# Imposta la percentuale dei file da selezionare (20%)
percentage_to_move = 0.2

# Chiama la funzione per spostare i file
move_files(source_folder, destination_folder, percentage_to_move)
