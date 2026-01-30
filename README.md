# Modelowanie agentowe klasycznej gry w nazywanie - implementacja



## Struktura repozytorium

```
repozytorium/
├── data/ <- przechowywanie danych
├── logs/ <- logi skryptów działających w tle
├── plots/ <- wygenerowane wygresy
├── pyproject.toml <- definicja środowiska
├── run_script.sh <- uruchamianie skryptów w tle
├── src/
│   ├── analysis/ <- analiza danych
│   ├── games/ <- implementacja modelu
│   ├── simulations/ <- symulacje 
│   └── utils/ <- funkcje pomocnicze
├── uv.lock <- stan środowiska
```


## Środowisko
Środowisko inicjalizujemy za pomocą menadżera środowiska [uv](https://github.com/astral-sh/uv). Instrukcja instalacji znajduje się w podanym repozytorium.


### Inicjalizacja
Aby zainicjalizować środowisko
```bash
uv sync
```

Aby uruchomić skrypt
```bash
uv run main.py
```

## Implementacja modelu i modyfikacji
W `src/games' znajdują się pliki:
- base_game.py - implementacja podstawowego modelu
- fuzzy_object.py - błąd detekcji
- fuzzy_word.py - błąd transmisji
- fuzzy_word_object.py - błąd detekcji i transmisji

### Model podstawowy
Implementacja modelu przedstawionego w rozdziale 3 znajduję się w pliku `src/games/base_game.py`. Model jest reprezentowany przez klasę `BaseGame`.

Aby uruchomić model inicjalizujemy obiekt z argumentami i wywołujemy metodę `play`
```py
model = BaseGame(
    game_instances, <- liczba gier jednocześnie.
    agents, <- liczba agentów
    objects, <- liczba obiektów
    memory, <- rozmiar pamięci dla pojedyńczego obiektu
    vocab_size, <- maksymalna ilość słów
    context_size, <- rozmiar kontekstu
    device, <- urządzenie na którym będzie działać symulacja
)


model.play(
    rounds, <- liczba kroków czasowych
    tqdm_desc, <- opis dla paska progresu
    sample_freq, <- częstotliwość zapisu statystyk
    disable_tqdm, <- włącz/wyłącz pasek progresu
    tqdm_position, <- poyzcje paska progresu (w przypadku wielu pasków)
    ma_window, <- rozmiar okna czasowego dla średniej ruchomej
)
```

## Modyfikacja 1 - błąd detekcji
Implementacja modelu przedstawionego w rozdziale 4 znajduję się w pliku `src/games/fuzzy_object.py`. Model jest reprezentowany przez klasę `FuzzyObjectGame`.
```py

model = FuzzyObjectGame(
    ...,
    confusion_prob <- prawdopodobieństwo błędu detekcji
)
```

## Modyfikacja 2 - błąd transmisji
Implementacja modelu przedstawionego w rozdziale 5 znajduję się w pliku `src/games/fuzzy_word.py`. Model jest reprezentowany przez klasę `FuzzyObjectWord`.
```py

model = FuzzyWordGame(
    ...,
    flip_prob <- prawdopodobieństwo błędu transmisji
)
```
## Połączenie modyfikacji 
Implementacja modelu przedstawionego w rozdziale 6 znajduję się w pliku `src/games/fuzzy_word_object.py`. Model jest reprezentowany przez klasę `FuzzyWordObject`.
```py

model = FuzzyWordObjectGame(
    ...,
    confusion_prob <- prawdopodobieństwo błędu detekcji
    flip_prob <- prawdopodobieństwo błędu transmisji
)
```


## Wykonane symulacje
Wszystkie symulacje zostały przeprowadzone przy użyciu skryptów
```
simulations/
├── base_game.py <- symulacja modelu bazowego wraz z wariacjami poszczególnych parametrów
├── fuzzy_object.py <- symulacja modyfikacji 2 
├── fuzzy_word_object.py <- symulacja łączonych modyfikacji
├── fuzzy_word.py <- symlacja modyfikacji 1
├── parameters.py <- parametry bazowe
├── profile_game.py <- porównanie wydajności CPU vs GPU

```