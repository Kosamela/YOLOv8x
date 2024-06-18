---

## Projekt pierwszy: main.py
Wykorzystuje YOLOv8 i wcześniej wyuczone modele do rozpoznawania obrazów.
Stwórz foldery walidacyjne 
mkdir \images\validation\car
mkdir \images\validation\other

### Funkcjonalności
- Program zapisuje zdjęcia, na których wykryto auta w folderze `.\processed_images\wykryto` z obrysowaną niebieską ramką.
- Zdjęcia, na których aut nie wykryto, są zapisywane w `.\processed_images\niewykryto`.

### Biblioteki
- `opencv-python`
- `ultralytics`

### Konfiguracja
Aby zainstalować wymagane biblioteki, użyj następującej komendy:
pip install -r requirements.txt

Projekt drugi: test.py

## Funkcjonalności
- Program przeprowadza trening na danych z folderu .\images\train.
- Walidacja odbywa się na danych z folderu .\images\validation.
- Zdjęcia, na których wykryto auta, są zapisywane w folderze .\processed_images\wykryto z obrysowaną niebieską ramką.
- Zdjęcia, na których aut nie wykryto, są zapisywane w .\processed_images\niewykryto.


### Biblioteki
- `opencv-python`
- `ultralytics`
- `scikit-learn`

### Konfiguracja
Aby zainstalować wymagane biblioteki, użyj następującej komendy:
pip install -r requirements.txt

Aby włączyć jeden z programów z poziomu cmd:
python main.py
python test.py
