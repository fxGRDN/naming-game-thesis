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


## Implementacja modelu
W `src/games' znajdują się pliki:
- base_game.py - implementacja podstawowego modelu
- fuzzy_object.py - błąd detekcji
- fuzzy_word.py - błąd transmisji
- fuzzy_word_object.py - błąd detekcji i transmisji

### Model podstawowy
...