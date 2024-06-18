import os
import cv2
from ultralytics import YOLO

model = YOLO('yolov8x.pt')
dict_classes = model.model.names

# --------Skalujemy obraz - w funkcji ustalamy wartosc skalowania
def powieksz_obraz(image, wspolczynnik_skalowania=2):
    szerokosc = int(image.shape[1] * wspolczynnik_skalowania)
    wysokosc = int(image.shape[0] * wspolczynnik_skalowania)
    wymiary = (szerokosc, wysokosc)
    powiekszony = cv2.resize(image, wymiary, interpolation=cv2.INTER_CUBIC)
    return powiekszony

# Przetwarzamy pojedynczy obraz
def przetworz_obraz(sciezka_obrazu, folder_wyjsciowy, folder_wyjsciowy_niewykryto, wspolczynnik_skalowania=2):
    obraz = cv2.imread(sciezka_obrazu)
    obraz = powieksz_obraz(obraz, wspolczynnik_skalowania)
    results = model(obraz)

    # Czasem nie chce stworzyć listy?
    if isinstance(results, list):
        results = results[0]

    # Przetwarzanie wyników i dodawanie do obrazu
    wykryto_auto = False
    for box in results.boxes:
        box_data = box.data.squeeze()
        if box_data.numel() >= 6:
            klasa = int(box_data[5])
            if dict_classes[klasa] == 'car':
                wykryto_auto = True
                x1, y1, x2, y2 = map(int, box_data[:4])
                etykieta = f'{dict_classes[klasa]}: {box_data[4]:.2f}'
                # --------Rysowanie ramek + określenie ich koloru i grubości
                cv2.rectangle(obraz, (x1, y1), (x2, y2), (255, 0, 0), 4)
                # --------Wielkość czcionki - warto wziąć pod uwagę skalowanie obrazu i dobrać pod wielkość
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

# Przetwarzanie wszystkich obrazów w folderze
def przetworz_folder(folder_wejsciowy, folder_wyjsciowy, folder_wyjsciowy_niewykryto, wspolczynnik_skalowania=2):
    if not os.path.exists(folder_wyjsciowy):
        os.makedirs(folder_wyjsciowy)
    if not os.path.exists(folder_wyjsciowy_niewykryto):
        os.makedirs(folder_wyjsciowy_niewykryto)

    for podfolder, _, pliki in os.walk(folder_wejsciowy):
        for plik in pliki:
            if plik.lower().endswith(('.png', '.jpg', '.jpeg')):
                sciezka_obrazu_wejsciowego = os.path.join(podfolder, plik)
                przetworz_obraz(sciezka_obrazu_wejsciowego, folder_wyjsciowy, folder_wyjsciowy_niewykryto, wspolczynnik_skalowania)
#-------- Ściezki folderu do walidacji
folder_walidacyjny = './images/validation'

#-------- Ścieżki folderów wtyjściowych
folder_wyjsciowy = './processed_images/wykryto'
folder_wyjsciowy_niewykryto = './processed_images/niewykryto'

przetworz_folder(folder_walidacyjny, folder_wyjsciowy, folder_wyjsciowy_niewykryto)
