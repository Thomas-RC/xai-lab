# Obrazy testowe

Zestaw 9 obrazów dobranych pod kątem różnorodności scenariuszy XAI.
Wszystkie w formacie `.webp` — apka i `run_batch.py` obsługują też `.jpg`, `.jpeg`, `.png`.

## Bieżący zestaw

| Plik | Treść | Spodziewana klasa ImageNet | Cel testowy |
|---|---|---|---|
| `brown_bears_family.webp` | niedźwiedzica + 2 młode w trawie | `brown bear` (294) | **multi-instance** — Grad-CAM++ powinien zaznaczyć wszystkie trzy, Grad-CAM zwykle tylko największego |
| `cat_with_food_bowl.webp` | kot kaliko + ludzka ręka z miską | `tabby` / `Egyptian cat` (281/285) | **ambiguous context** — kot, ręka, miska, drewniana podłoga; ciekawe na której czynności XAI się skupi |
| `platypus_in_towel.webp` | dziobak na ręczniku, trzymany przez człowieka | `platypus` (103) | **rzadka klasa** + nietypowa morfologia (ptasi dziób, futro) — model patrzy na dziób czy ogon? |
| `bird_singing_branch.webp` | mały ptak (drozd / słowik) na gałęzi | `robin` (15) / `goldfinch` (11) / inne | **kontekst gałęzi** — gałąź zajmuje 30% kadru; LIME powinien ją odrzucić |
| `beaver_on_ice.webp` | bóbr na lodzie | `beaver` (337) | **shortcut learning** — czy lód wpływa na klasyfikację? |
| `scorpion_on_sand.webp` | skorpion na czerwonym piasku | `scorpion` (71) | **shortcut learning** — pustynia jako kontekst gatunkowy |
| `concorde_skyborne.webp` | Concorde Air France w locie | `airliner` (404) / `warplane` (895) | **napis na boku** — czy model wykorzystuje "AIR FRANCE" do klasyfikacji? |
| `porsche_911_road.webp` | Porsche 911 w ruchu, droga w górach | `sports car` (817) / `convertible` (511) | **motion blur** — czy gradient łapie się na rozmytym tle? |
| `lightning_storm.webp` | piorun nad polem, droga gruntowa | brak idealnej klasy | **failure case** — model zmuszony do nietrafnej predykcji; XAI pokazuje "skąd ją wziął" |

## Strategiczne pary do dyskusji w sprawozdaniu

- **`scorpion_on_sand` vs `beaver_on_ice`** — oba zwierzęta na "tle gatunkowym" (pustynia / lód).
  Dwa różne podejścia do shortcut learning; sprawdzamy, czy XAI rzeczywiście pokazuje
  zwierzę a nie tło.

- **`brown_bears_family`** osobno — najmocniejszy argument dla Grad-CAM++ vs Grad-CAM.
  Trzy obiekty tej samej klasy w jednym kadrze.

- **`lightning_storm`** osobno — failure case. Sprawozdanie sekcja 7
  ("Przypadki ciekawe") może otworzyć tym przykładem: *gdy model nie ma dobrej
  klasy, jak się broni? co XAI mówi o jego "rozpaczliwym" wyborze?*

- **`platypus_in_towel`** — klasa `platypus` istnieje w ImageNet (103), ale jest
  rzadko widoczna w treningu. Spodziewamy się niskiej pewności i ciekawej heatmapy
  (zwykle skupionej na dziobie — najbardziej "endemicznej" cesze).

## Wymagania techniczne

- Format: JPG / JPEG / PNG / WEBP
- Rozdzielczość: dowolna (apka skaluje do 224×224 z central crop)
- Treść: dowolna z 1000 klas ImageNet (lista: `imagenet_categories()` w
  [`src/utils/imagenet.py`](../../src/utils/imagenet.py))

## Licencje

Obrazy w tym katalogu pochodzą ze stocków (Getty, Shutterstock) oraz publikacji.
**Użyte w ramach edukacyjnego fair use** dla projektu zaliczeniowego.
**Nie commitować** do publicznego repozytorium GitHub bez weryfikacji licencji.
