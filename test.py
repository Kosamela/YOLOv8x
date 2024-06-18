import os
import cv2
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

model = YOLO('yolov8x.pt')
dict_classes = model.model.names

def wczytywanko_folderow(folder, label):
    data = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            data.append((path, label))
    return data

def przetworz_i_zapisz_obraz(sciezka_obrazu, folder_wyjsciowy, folder_wyjsciowy_niewykryto):
    obraz = cv2.imread(sciezka_obrazu)
    results = model(obraz)

    wykryto_auto = False
    if isinstance(results, list):
        results = results[0]

    if 'boxes' in results:
        for box in results.xyxy[0]:
            klasa = int(box[5])
            if dict_classes[klasa] == 'car':
                wykryto_auto = True
                x1, y1, x2, y2 = map(int, box[:4])
                etykieta = f'{dict_classes[klasa]}: {box[4]:.2f}'
                # Rysowanie ramki wokół wykrytego obiektu
                cv2.rectangle(obraz, (x1, y1), (x2, y2), (255, 0, 0), 4)
                # Wielkość czcionki - warto wziąć pod uwagę skalowanie obrazu i dobrać pod wielkość
                skala_czcionki = 1.2
                cv2.putText(obraz, etykieta, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, skala_czcionki, (255, 0, 0), 3)

    # Zapis przetworzonego obrazu
    if wykryto_auto:
        if not os.path.exists(folder_wyjsciowy):
            os.makedirs(folder_wyjsciowy)
        sciezka_obrazu_wyjsciowego = os.path.join(folder_wyjsciowy, os.path.basename(sciezka_obrazu))
        cv2.imwrite(sciezka_obrazu_wyjsciowego, obraz)
    else:
        if not os.path.exists(folder_wyjsciowy_niewykryto):
            os.makedirs(folder_wyjsciowy_niewykryto)
        sciezka_obrazu_wyjsciowego = os.path.join(folder_wyjsciowy_niewykryto, os.path.basename(sciezka_obrazu))
        cv2.imwrite(sciezka_obrazu_wyjsciowego, obraz)

folder_uczenia = './images/train'

car_folder = os.path.join(folder_uczenia, 'car')
other_folder = os.path.join(folder_uczenia, 'other')
car_data = wczytywanko_folderow(car_folder, 'car')
other_data = wczytywanko_folderow(other_folder, 'other')

# Połączenie danych i podział na zbiór treningowy i walidacyjny
data = car_data + other_data
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

folder_wyjsciowy = './processed_images/wykryto'
folder_wyjsciowy_niewykryto = './processed_images/niewykryto'

print("Rozpoczynam uczenie...")

# Zmienna do śledzenia postępu
liczba_obrazow = len(train_data)
obrazy_przetworzone = 0

# Trenowanie modelu
for sciezka_obrazu_wejsciowego, _ in train_data:
    przetworz_i_zapisz_obraz(sciezka_obrazu_wejsciowego, folder_wyjsciowy, folder_wyjsciowy_niewykryto)
    obrazy_przetworzone += 1
    if obrazy_przetworzone % 100 == 0:
        print(f"Przetworzono {obrazy_przetworzone}/{liczba_obrazow} obrazów.")

print("Uczenie zakończone.")

