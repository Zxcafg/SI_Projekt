# 1)Opis rzeczywistego problemu.

## _Cel:_

Celem projektu jest opracowanie modelu sztucznej inteligencji (SI), który będzie przewidywał przyszłe wartości na podstawie analizy trendów w danych. 
Model ten będzie wykorzystywał technologię Long Short-Term Memory (LSTM), która jest jednym z typów rekurencyjnych sieci neuronowych (RNN). 
Model będzie analizował dane wejściowe przedstawiające zmieniające się wartości w czasie (np. zmiany ceny mieszkań, temperatury, itp.) i na ich podstawie przewidywał wartość na przyszłość.

Przykład: mamy dane o cenie mieszkań z ostatnich kwartałów, w odpowiedzi chcemy przewidzieć cenę na następny kwatał.

![image](https://github.com/user-attachments/assets/002d1bbd-0141-441d-9a93-301ca9b197b1)


## _Motywacja:_

Zrozumienie i prognozowanie trendów w różnych dziedzinach życia, takich jak ekonomia, meteorologia czy analiza rynku, ma ogromne znaczenie praktyczne. 
Przewidywanie przyszłych wartości na podstawie wcześniejszych danych może pomóc w podejmowaniu lepszych decyzji, takich jak optymalizacja procesów biznesowych,
prognozy popytu na produkty, monitorowanie zmian klimatycznych czy przewidywanie cen na rynkach finansowych. Projekt oparty na sieciach neuronowych LSTM jest istotny,
ponieważ LSTM jest bardziej odporny na błędy niż klasyczne rekurencyjne sieci neuronowe, a jego zastosowanie w analizie szeregów czasowych może zwiększyć dokładność prognoz.

## _Dane wejściowe:_

Dane wejściowe do modelu będą pochodzić z serii średnich cen transakcyjnych na mieszkania (z rynku pierwotnego, z ostanich 15 lat) wartości zmieniających się w czasie(kwartał), które będą wprowadzane do algorytmu (dla uproszczenia w przedziale "high" , "Medium" i "Low").
Dane te będą reprezentować zmieniający się trend w czasie (np. zmiany cen, temperatury).
Model będzie otrzymywał dane na przestrzeni "x" czasu, a potem model wygeneruje prognozę wartości na podstawie tych danych.

![image](https://github.com/user-attachments/assets/4cc6afc0-4922-4f0b-8066-b88bce4703a0)


## _Zastosowanie sztucznej inteligencji:_

Projekt wykorzystuje algorytm Long Short-Term Memory (LSTM), który należy do rodziny rekurencyjnych sieci neuronowych (RNN).
LSTM jest szczególnie skuteczną metodą w analizie danych szeregów czasowych, ponieważ jest w stanie przechowywać długoterminowe zależności w danych, co pozwala na dokładniejsze przewidywania. 
W tym przypadku, model LSTM będzie analizował zmieniające się wartości i na ich podstawie przewidywał przyszłe dane.

# 2)State of art
## 1. Recurrent Neural Networks (RNN)

<img width="812" alt="image" src="https://github.com/user-attachments/assets/4a73f72a-d225-4b21-8380-1dbdc991898a">

RNN to podstawowa struktura sieci neuronowej, która jest szczególnie skuteczna w przetwarzaniu danych sekwencyjnych.
Główna cecha RNN to fakt, że posiadają one "pamięć", czyli informacje z poprzednich kroków mogą być wykorzystywane do przetwarzania kolejnych danych w sekwencji.
Jednak tradycyjne RNN mają problem z utrzymywaniem długoterminowych zależności ,tzw. problem zanikania gradientu(Vanishing/Exploding Gradient Problem).

### _Zalety:_

* Dobry do analizy danych sekwencyjnych (np. tekstu, danych czasowych).

* Prosta architektura.

### _Wady:_

* Problemy z przechowywaniem długoterminowych zależności.

* Skłonność do problemu zanikania gradientu.


## 2. Long Short-Term Memory (LSTM)
   
LSTM to rozszerzenie klasycznego RNN, które zostało zaprojektowane, aby rozwiązać problem zanikania gradientu w długich sekwencjach. Dzięki swojej strukturze (trzy główne "bramki": wejścia, zapomnienia i wyjścia), LSTM jest w stanie przechowywać informacje przez długie okresy czasu, co czyni go bardziej skutecznym w rozwiązywaniu problemów z długoterminowymi zależnościami.

<img width="450" alt="image" src="https://github.com/user-attachments/assets/aad364e4-2181-449a-aea0-0ded00d3b93d">

### _Zalety:_

* Skuteczne w modelowaniu długoterminowych zależności.

* Dobrze radzi sobie z nieliniowymi wzorcami i sekwencjami o dużych odległościach czasowych.

### _Wady:_

* Złożona architektura, co prowadzi do większych wymagań obliczeniowych.

* Trudniejsza do wytrenowania niż tradycyjne RNN.


## 3. Gated Recurrent Unit (GRU)
GRU to uproszczona wersja LSTM, która osiąga podobne rezultaty przy mniejszej liczbie parametrów. GRU ma dwie główne bramki (Update Gate i Reset Gate),w odróżnieniu od LSTM która ma trzy bramki (Input , Output , Forget), co czyni je bardziej efektywnymi obliczeniowo. GRU jest również w stanie przechowywać długoterminowe zależności, ale bez tak złożonej struktury jak LSTM.

<img width="450" alt="image" src="https://github.com/user-attachments/assets/35ccb965-609b-4e4a-aa3c-0754c9f2d1b4">

### _Zalety:_

* Mniejsza liczba parametrów w porównaniu do LSTM.

* Szybsze trenowanie przy zachowaniu wysokiej wydajności.

* Wydajniejsze w przypadku mniejszych zbiorów danych.

### _Wady:_

* Może nie być tak elastyczne jak LSTM w rozwiązywaniu niektórych bardziej złożonych problemów.

* Czasami może oferować gorszą jakość wyników w bardziej złożonych zadaniach.

## _Podsumowanie:_
RNN to podstawowy model do przetwarzania sekwencji, ale ma trudności z długoterminowymi zależnościami.
LSTM to bardziej zaawansowany model, który rozwiązuje problem zanikania gradientu i świetnie radzi sobie z długoterminowymi zależnościami.
GRU to uproszczona wersja LSTM, która oferuje podobną wydajność, ale jest bardziej efektywna obliczeniowo.

# 3) Opis wybranej koncepcji – LSTM (Long Short-Term Memory)

## Opis ogólny LSTM:

LSTM to szczególny typ sieci neuronowej rekurencyjnej (RNN), zaprojektowany do rozwiązywania problemu przechowywania długoterminowej zależności w danych czasowych. Jest szeroko stosowany do analizy sekwencyjnych danych, takich jak przewidywanie wartości czasowych.LSTM zawiera tzw. cell state (stan komórki), który jest używany do przechowywania informacji przez długi czas, co pozwala na skuteczną obsługę długoterminowych zależności, które są wyzwaniem w tradycyjnych RNN.

## Wzory LSTM i struktura algorytmu:

### Struktura LSTM

LSTM składa się z trzech głównych składników:

1. **Bramka zapomnienia (Forget Gate)** – kontroluje, które informacje mają zostać zapomniane w stanie komórki.
2. **Bramka wejściowa (Input Gate)** – decyduje, które nowe informacje będą zapisywane w stanie komórki.
3. **Bramka wyjściowa (Output Gate)** – kontroluje, które informacje będą przekazane jako wyjście sieci.

Wzory dla tych bramek:

### 1. **Bramka zapomnienia (Forget Gate)**

Decyduje, które informacje powinny być zapomniane w stanie komórki.

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

- **$f_t$** – wartość bramki zapomnienia.
- **$\sigma$** – funkcja aktywacji sigmoidalnej.
- **$W_f$** – wagi bramki zapomnienia.
- **$h_{t-1}$** – ukryte wyjście z poprzedniego kroku czasowego.
- **$x_t$** – wejście w czasie t.
- **$b_f$** – bias bramki zapomnienia.

### 2. **Bramka wejściowa (Input Gate)**

Decyduje, które nowe informacje będą zapisywane w stanie komórki.

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

Nowe informacje, które mogą zostać zapisane:

$$
\tilde{C_t} = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$


- \( **$i_t$** \) – wartość bramki wejściowej.
- \(**$\tilde{C}_t$** \) – nowe informacje, które mogą zostać zapisane.

### 3. **Aktualizacja stanu komórki (Cell State)**

Stan komórki jest aktualizowany na podstawie bramki zapomnienia oraz bramki wejściowej.

$$
C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t
$$

- **$C_t$** – nowy stan komórki.
- **$C_{t-1}$** – stan komórki z poprzedniego kroku.

### 4. **Bramka wyjściowa (Output Gate)**

Decyduje, co zostanie wyjściem sieci i jakie informacje będą przekazywane dalej.

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

Finalne wyjście sieci:

$$
h_t = o_t \cdot \tanh(C_t)
$$

- **$o_t$**  – wartość bramki wyjściowej.
- **$h_t$**  – ukryte wyjście w czasie t.

## Przewodnik krok po kroku po LSTM

### 1. Zapominanie zbędnych informacji

Pierwszym krokiem w LSTM jest podjęcie decyzji, jakie informacje wyrzucić z aktualnego stanu komórki. Decyzję tę podejmuje warstwa sigmoidalna, nazywana **„warstwą bramki zapominania”** (*forget gate layer*). Analizuje ona **$h_{t-1}$** oraz **$x_t$** , a następnie zwraca wartość pomiędzy \( 0 \) a \( 1 \) dla każdej liczby w stanie komórki **$C_{t-1}$** Wartość \( 1 \) oznacza „zachowaj to w całości,” a \( 0 \) „całkowicie to usuń.”

<img width="508" alt="image" src="https://github.com/user-attachments/assets/fba20ee7-3eea-4ab9-bc31-0a4c98be3612">

### 2. Wprowadzanie nowych informacji

Kolejnym krokiem jest decyzja, jakie nowe informacje zapisać w stanie komórki. Składa się to z dwóch części:

- Warstwa sigmoidalna (**„warstwa bramki wejścia”**) decyduje, które wartości zaktualizować.
- Warstwa tangensa hiperbolicznego tworzy wektor nowych potencjalnych wartości **$\tilde{C}_t$** , które mogą zostać dodane do stanu.

Następnie obie decyzje są łączone, aby zaktualizować stan komórki.

<img width="520" alt="image" src="https://github.com/user-attachments/assets/214abca8-7557-40dc-9283-9c147c5f8d28">

### 3. Aktualizacja stanu komórki

Stan komórki **$C_{t-1}$** jest aktualizowany do **$C_t$** poprzez:

- Mnożenie starego stanu przez **$f_t$** , aby zapomnieć wybrane informacje.
- Dodanie  **$i_t \cdot \tilde{C}_t$** , czyli nowych wartości skalowanych decyzjami bramki wejścia.

Stan komórki **$C_t$** jest kluczowym elementem architektury LSTM. Przechowuje on informacje przez długi czas, umożliwiając sieci "zapamiętanie" istotnych danych z przeszłości, które są następnie wykorzystywane do prognozowania w przyszłości. Aktualizacja stanu komórki odbywa się na podstawie **bramki zapomnienia**  **$f_t$** oraz **bramki wejściowej** **$i_t$**, co pozwala na efektywne zarządzanie długoterminową pamięcią.

<img width="472" alt="image" src="https://github.com/user-attachments/assets/837b5524-569d-4e5f-ac1e-97107a8eff53">

### 4. Generowanie wyjścia

Na koniec LSTM generuje wyjście oparte na stanie komórki, przefiltrowane przez:

- Warstwę sigmoidalną decydującą, które części stanu komórki wyprowadzić.
- Tangens hiperboliczny **$(\tanh)$** ograniczający wartości między \( -1 \) a \( 1 \).

To pozwala wyprowadzać tylko te informacje, które są potrzebne w danym kroku.

<img width="498" alt="image" src="https://github.com/user-attachments/assets/a34ed59a-4a06-4688-9e9f-c6775b922d2c">

## Zastosowanie LSTM w projekcie

W projekcie sieć LSTM analizuje przesłane dane i przewiduje trend na następny czas(kwartał). Dzięki opisanym mechanizmom (zapominanie, aktualizacja, generowanie wyjścia) model potrafi uwzględniać istotne zależności czasowe i ignorować zbędne dane.LSTM "przechodzi" przez dane sekwencyjne, na każdym kroku dokonując aktualizacji stanu komórki i wyjścia. W przeciwieństwie do klasycznych RNN, LSTM pozwala na przechowywanie informacji na dłuższe okresy, dzięki bramkom, które kontrolują przepływ informacji. Z tego powodu LSTM może lepiej radzić sobie z długoterminowymi zależnościami, np. w analizie trendów czasowych.

---

### Przykład działania LSTM w kontekście prognozowania wartości na podstawie 4 kwartałów

Dla czterech kwartałów wejściowych, LSTM analizuje każdy kwartał jako część sekwencji. Wartości z wcześniejszych kwartałów wpływają na decyzje podejmowane przez bramki, co pozwala na przewidywanie wartości na następny kwartał. 

- Stan komórki przechowuje informacje o zależnościach występujących w danych (np. zmiany wartości w trendzie). 
- Te informacje są następnie wykorzystywane do prognozowania kolejnej wartości.

Przykładowe obliczenia dla jednego kwartału:

<img width="957" alt="image" src="https://github.com/user-attachments/assets/6879f8fc-2f7c-4ff9-9287-e405d54814bf">

Takie obliczenia zgodnie z algorytmem zostaną zrobione dla każdego kwartału:

<img width="934" alt="image" src="https://github.com/user-attachments/assets/82979938-5f75-4198-a963-9c425df1aa8b">


Pod koniec obliczeń otrzymujemy konieczną wartośc z "Short Term Memory"(na zdjęciu oznaczona jako "x"), która i będzie przewidywaną wartością na następny kwartał.

---

## Co jest potrzebne do realizacji w rzeczywistym świecie?

### Wymagania sprzętowe i środowiskowe

1. **Komputer z odpowiednią mocą obliczeniową**  
   Do uruchamiania i trenowania modeli LSTM potrzebny jest komputer z wystarczającą ilością pamięci RAM i procesorem obsługującym obliczenia równoległe (np. z GPU). W przypadku mniejszych danych wystarczy standardowy laptop.

2. **Środowisko programistyczne**  
   - Python w wersji 3.x.  
   - Biblioteki do obliczeń i analizy danych:  
     - NumPy (przetwarzanie danych).  
     - PyTorch (implementacja i trenowanie LSTM).  
     - Matplotlib (wizualizacja wyników).  

3. **Źródło danych**  
   - W rzeczywistym świecie dane mogą pochodzić z baz danych, plików CSV lub interfejsów API, takich jak dane giełdowe, dane pogodowe, logi systemów IT itp.  
   - W naszym przypadku dane są brane ze zródeł które posiadają dane zmian cen na mieszkania w zakresie ostatnich 15 lat(z różnych miast Polski).  

4. **Dane historyczne do trenowania**  
   W prawdziwych zastosowaniach wymagane są duże zbiory danych historycznych, aby nauczyć model rozpoznawać wzorce.

---

### Procedura testowania rozwiązania

1. **Testy funkcjonalne**  
   - Sprawdzenie poprawności generowania danych wejściowych: czy dane są w odpowiednim formacie.  
   - Upewnienie się, że model LSTM poprawnie przewiduje wartość na podstawie wcześniejszych dannych.  

2. **Testy wydajnościowe**  
   - Testowanie szybkości trenowania modelu na większych zbiorach danych.  
   - Monitorowanie zużycia zasobów, takich jak pamięć RAM i moc obliczeniowa procesora/GPU.

3. **Testy dokładności**  
   - Porównanie prognozowanych wartości z rzeczywistymi w celu oceny dokładności przewidywań.  
   - Obliczenie błędów takich jak MSE (Mean Squared Error) lub MAE (Mean Absolute Error).  

4. **Testy użytkowe**  
   - Weryfikacja, czy użytkownik może intuicyjnie korzystać z aplikacji (np. czy kliknięcie myszy powoduje poprawne generowanie przewidywań i aktualizację wykresu).  
   - Testowanie interaktywności wykresów.  

---

### Identyfikacja potencjalnych problemów

1. **Brak wystarczających danych historycznych**  
   W prawdziwych projektach ograniczona liczba danych historycznych może wpłynąć na zdolność modelu do nauki i przewidywań.

2. **Przeuczenie modelu (overfitting)**  
   Jeśli model jest zbyt skomplikowany w stosunku do ilości danych, może „zapamiętać” dane zamiast uczyć się ogólnych wzorców.

3. **Wydajność obliczeniowa**  
   Trenowanie sieci LSTM na dużych zbiorach danych może być czasochłonne i wymagać dużej mocy obliczeniowej.

4. **Interpretacja wyników**  
   Prognozy modelu mogą być trudne do interpretacji, zwłaszcza w sytuacjach, gdy przewidywania są błędne lub nieintuicyjne.

5. **Zarządzanie błędami**  
   Wprowadzenie mechanizmów radzenia sobie z brakującymi danymi, nietypowymi wartościami lub problemami z generowaniem danych.

---

Rozwiązanie tych problemów oraz przygotowanie dokładnych testów zapewni, że model będzie działał poprawnie w rzeczywistym świecie.

# Proof of concept 
## Opis Kodu projektu

### Importowanie bibliotek

Biblioteki wykorzystywane w projekcie to: numpy do obliczeń numerycznych, torch do budowania i trenowania modelu LSTM, matplotlib do wizualizacji wyników, oraz mplcursors do interakcji z wykresami.
```python
import numpy as np  # Importowanie biblioteki do obliczeń numerycznych (np. operacje na tablicach)
import torch  # Importowanie Pytorch (biblioteka do uczenia maszynowego)
import torch.nn as nn  # Importowanie modułu z sieciami neuronowymi Pytorch
import torch.optim as optim  # Importowanie optymalizatorów w Pytorch
import matplotlib.pyplot as plt  # Importowanie biblioteki do tworzenia wykresów
import mplcursors  # Importowanie biblioteki do interakcji z wykresami (dodanie kursora)
```
### Przygotowanie danych

W tym kroku wczytujemy dane o cenach mieszkań w Warszawie i przygotowujemy je do dalszej analizy. Dane są zapisane w postaci słownika, gdzie kluczami są nazwy miast, a wartościami listy zawierające ceny mieszkań.

```python
# Przykładowe dane: ceny mieszkań od 4 kwartału 2006 do 4 kwartału 2021 roku (zł/m^2)
data = {
    "Warszawa": [7143, 7730, 8696, 9137, 9034, 8921, 8546, 8528, 9046, 8406, 8406, 7949, 8497, 8620, 8933, 8493, 8024,
                 7915, 7920, 7920, 7889, 7601, 7522, 7247, 7238, 6687, 6867, 6971, 7189, 7170, 7332, 7365, 7387, 7335,
                 7308, 7447, 7401, 7434, 7355, 7309, 7556, 7429, 7821, 8020, 8054, 8047, 8394, 8604, 8762, 8985, 9243,
                 9457, 9812, 10003, 10288, 10470, 10671, 10895, 11103, 10931],  # Ceny mieszkań w Warszawie
                                                                                # Następna cena:11288
}
```
### Normalizacja danych (min-max scaling)

```python
data_min = np.min([np.min(data[city]) for city in data])  # Obliczanie minimalnej wartości we wszystkich miastach
data_max = np.max([np.max(data[city]) for city in data])  # Obliczanie maksymalnej wartości we wszystkich miastach
data_normalized = {city: (np.array(data[city]) - data_min) / (data_max - data_min) for city in data}  # Normalizacja danych
```
Dane są normalizowane do zakresu [0, 1] za pomocą skalowania min-max, co ułatwia proces uczenia maszynowego.

### Przygotowanie danych do uczenia
```python
def create_dataset(data, time_step=1):
    X, Y = [], []  # Tworzymy listy na dane wejściowe (X) i wyniki (Y)
    for i in range(len(data) - time_step):  # Przechodzimy po danych z opóźnieniem o 'time_step'
        X.append(data[i:(i + time_step)])  # Dodajemy dane wejściowe
        Y.append(data[i + time_step])  # Dodajemy wynik (wartość po 'time_step')
    return np.array(X), np.array(Y)  # Zwracamy dane wejściowe i wyniki jako tablice NumPy
```
Funkcja create_dataset tworzy dane wejściowe i wyniki na podstawie danych o cenach mieszkań. Używa opóźnienia (time_step) do tworzenia sekwencji, która będzie wykorzystywana w modelu LSTM.
```python
time_step = 4  # Liczba kwartałów w historii, które będziemy wykorzystywać do przewidywania
X_all = []  # Lista do przechowywania danych wejściowych
Y_all = []  # Lista do przechowywania wyników

for city in data_normalized:  # Iterujemy po wszystkich miastach
    X, Y = create_dataset(data_normalized[city], time_step)  # Tworzymy dane wejściowe i wyniki dla danego miasta
    X_all.append(X)  # Dodajemy dane wejściowe do listy X_all
    Y_all.append(Y)  # Dodajemy wyniki do listy Y_all
X_all = np.concatenate(X_all, axis=0)  # Łączenie wszystkich danych wejściowych w jedną tablicę
Y_all = np.concatenate(Y_all, axis=0)  # Łączenie wszystkich wyników w jedną tablicę
```
Dane z różnych miast są łączone w jedną tablicę, aby stworzyć zbiór danych do trenowania modelu.

### Przekształcanie danych na tensor
```python
X = torch.tensor(X_all).float()  # Konwertowanie danych wejściowych na tensor Pytorch
Y = torch.tensor(Y_all).float()  # Konwertowanie wyników na tensor Pytorch
```
Dane wejściowe i wyniki są konwertowane na tensory Pytorch, co jest wymagane do trenowania modelu.

### Model LSTM
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()  # Inicjalizacja klasy bazowej
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)  # Warstwa LSTM
        self.fc = nn.Linear(hidden_layer_size, output_size)  # Warstwa w pełni połączona (fully connected)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Przepuszczanie danych przez warstwę LSTM
        predictions = self.fc(lstm_out[:, -1, :])  # Wybieramy ostatni element (z ostatniej próbki) z wyjścia LSTM
        return predictions  # Zwracamy prognozowane wartości
```
Definiowany jest model LSTM, który składa się z warstwy LSTM i warstwy w pełni połączonej. Model ten będzie przewidywał ceny mieszkań.

### Inicjalizacja modelu, funkcji kosztu i optymalizatora
```python
model = LSTMModel()  # Tworzymy model LSTM
criterion = nn.MSELoss()  # Funkcja kosztu (średni błąd kwadratowy)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optymalizator Adam z określoną szybkością uczenia
```
Model jest inicjalizowany, a funkcja kosztu i optymalizator Adam są przygotowywane do treningu.

### Trenowanie modelu
```python
epochs = 1000  # Liczba epok
losses = []  # Lista do przechowywania wartości funkcji kosztu podczas treningu
for epoch in range(epochs):  # Pętla po liczbie epok
    model.train()  # Ustawienie modelu w tryb treningu
    optimizer.zero_grad()  # Zerowanie gradientów
    y_pred = model(X.unsqueeze(-1))  # Przewidywanie cen na podstawie danych wejściowych
    loss = criterion(y_pred, Y.unsqueeze(-1))  # Obliczanie błędu
    loss.backward()  # Obliczanie gradientów wstecz
    optimizer.step()  # Aktualizacja wag modelu
    losses.append(loss.item())  # Dodanie wartości funkcji kosztu do listy
    if (epoch + 1) % 50 == 0:  # Co 50 epok wypisujemy postęp
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
```
Model jest trenowany przez 1000 epok. W każdej epoce gradienty są obliczane i wagi są aktualizowane za pomocą optymalizatora. Po każdej epoce wartość funkcji kosztu (loss) jest zapisywana, aby śledzić postęp treningu.

### Wizualizacja postępu treningu
```python
plt.plot(range(epochs), losses)  # Rysowanie wykresu funkcji kosztu
plt.xlabel('Epoki')  # Etykieta osi X
plt.ylabel('Loss')  # Etykieta osi Y
plt.title('Postęp funkcji kosztu')  # Tytuł wykresu
plt.show()  # Wyświetlanie wykresu
```
Na wykresie przedstawiany jest postęp treningu modelu, pokazując zmiany wartości funkcji kosztu w kolejnych epokach.

### Predykcja na podstawie ostatnich danych
```python
model.eval()  # Ustawienie modelu w tryb oceny
last_data = torch.tensor(data_normalized["Warszawa"][-time_step:]).float().unsqueeze(0).unsqueeze(-1)  # Pobieranie ostatnich danych
predicted_price = model(last_data).item()  # Przewidywanie ceny na podstawie ostatnich danych
predicted_price = predicted_price * (data_max - data_min) + data_min  # Przekształcanie prognozy na pierwotną skalę
```
Model dokonuje prognozy ceny na podstawie ostatnich danych o cenach mieszkań w Warszawie, a wynik jest przekształcany z powrotem na pierwotną skalę.

### Wyświetlanie wyników
```python
plt.plot(range(len(data["Warszawa"])), data["Warszawa"], label='Dane rzeczywiste')  # Rysowanie wykresu danych rzeczywistych
plt.plot(len(data["Warszawa"]), predicted_price, 'ro', label=f'Przewidywana cena ({predicted_price:.2f})')  # Rysowanie przewidywanej ceny
plt.xlabel('Czas (kwartały)')  # Etykieta osi X
plt.ylabel('Cena')  # Etykieta osi Y
plt.legend()  # Dodanie legendy
mplcursors.cursor(hover=True).connect(
    "add", lambda sel: (sel.annotation.set_text(
        f"Cena: {data['Warszawa'][int(sel.target[0])]}\n"
        f"Rok: {2006 + (int(sel.target[0]) // 4)}\n"
        f"Kwartał: {int(sel.target[0]) % 4 + 1}"
        if 0 <= int(sel.target[0]) < len(data['Warszawa'])
        else "Index out of range"
    )))
plt.show()  # Wyświetlanie wykresu
```
Na wykresie wyświetlana jest zarówno seria rzeczywistych cen, jak i przewidywana cena na ostatni kwartał. Interaktywny kursor umożliwia wyświetlanie szczegółowych informacji o danych punktach na wykresie.
## Testy do wykonania
### Test poprawności działania modelu
Po uruchomieniu kodu mamy pierwsze wyniki.

_**Wykres funkcji kosztu:**_

![Figure_3](https://github.com/user-attachments/assets/4b8c6b05-4673-45ef-a9a7-17f42db44c84)

_**Wykres predykcji:**_

![Figure_4](https://github.com/user-attachments/assets/38a84818-4f6f-4afc-b19f-33070efaea91)

Jak widać, koszt staje się minimalny, a przewidywana cena **(11052)** jest przybliżona do tej którą oczekujemy **(11288)**. Po ponownych uruchomieniach przewidywana cena jest zawsze większa od ceny za poprzedni kwartał i znajduje się w okresie od 11050 do 11180, co jest dobrym wynikiem. 
### Test na różnych miastach
Podobne wyniki otrzymaliśmy dla Krakowa i innych miast. Predykcja działa dobrze.Dla przykładu zostanie pokazany test predykcji dla krakowa.

_**Wykres funkcji kosztu:**_

![Figure_6](https://github.com/user-attachments/assets/aa803294-1f7e-43b1-8a47-f427e49df446)

_**Wykres predykcji:**_

![Figure_7](https://github.com/user-attachments/assets/a34c3227-d679-41e4-b88f-50e2fc167f10)

Jak widać, koszt staje się minimalny, a przewidywana cena **(9519)** jest przybliżona do tej którą oczekujemy **(9785)**. 
### Test generalizacji
Sprawdzenie jak model radzi sobie z danymi spoza treningu. Rozdzielanie dane na zestawy treningowe i testowe (np. 80% do treningu, 20% do testów) i ocenianie, jak model generalizuje.

```python
# Podział danych na treningowe i testowe
split_ratio = 0.8
train_size = int(len(X_all) * split_ratio)

X_train, Y_train = X_all[:train_size], Y_all[:train_size]
X_test, Y_test = X_all[train_size:], Y_all[train_size:]

# Trening modelu na zestawie treningowym
X_train_tensor = torch.tensor(X_train).float().unsqueeze(-1)
Y_train_tensor = torch.tensor(Y_train).float()

# Przewidywanie na zestawie testowym
X_test_tensor = torch.tensor(X_test).float().unsqueeze(-1)
Y_test_tensor = torch.tensor(Y_test).float()

with torch.no_grad():
    y_pred = model(X_test_tensor)

# Oblicz MSE na zestawie testowym
mse_test = torch.mean((y_pred.squeeze() - Y_test_tensor) ** 2).item()
print(f"Test generalizacji (MSE): {mse_test:.4f}")
```
W odpowiedzi dostajemy następny komunikat: 

Test generalizacji (MSE): 0.0025

Wynik testu generalizacji (MSE) na poziomie 0.0025 oznacza, że nasz model LSTM osiąga stosunkowo niewielki błąd średniokwadratowy na danych testowych, co wskazuje na dobrą jakość predykcji.Można postarać się go zmniejszyć.

Najlepszy wyniki predykcji były otrzymany dla nastaw lr=0.0005 , epochs = 2000 , time_step = 25. 

Wyniki z optymalizowanymi nastawami:

![Figure_9_step25_lr0dot005_2000epochs](https://github.com/user-attachments/assets/54909112-4edc-4742-9bf0-3d0b0e13efa8)

![Figure_10](https://github.com/user-attachments/assets/1416418f-e4b3-49ba-a70e-fbab2aee2c71)

Jak widać, otrzymane wyniki są dokładniejsze niż wcześniejsze ( przewidywana cena 9785). Znacznie zmniejszyły się koszty funkcji (do poziomu ~0.0006) w porównaniu do wcześniejszych wyników (~0.0019)

### Test na większej liczbie danych

W przypadku problemów z danymi czasowymi o długoterminowych zależnościach warto eksperymentować z wartością time_step. W tym projekcie optymalną wartością dla przewidywania ceny mieszkania w Krakowie okazało się time_step = 25, ale w innych przypadkach parametr ten powinien być dobierany empirycznie, biorąc pod uwagę charakterystykę danych.

### Test wydajności

**1.Testowanie wydajności:**

* Dodano funkcje do pomiaru czasu trwania różnych etapów procesu (trenowanie modelu, predykcja) z wykorzystaniem modułu $time$ w Pythonie.

```python
import time # moduł czasu
```
* Zaimplementowano raportowanie czasów w celu analizy wydajności.
  Raportowanie czasu wykonania poszczególnych etapów jest realizowane poprzez obliczenie różnicy między czasem końcowym a początkowym i wyświetlenie wyniku
```Python
start_time = time.time()  # Początek pomiaru czasu
# kod do wykonania (np. normalizacja danych)
end_time = time.time()  # Koniec pomiaru czasu
print(f"Normalizacja danych: {end_time - start_time:.6f} s")  # Raportowanie wyniku
```

**2.Raportowanie pamięci:**

* Dodano kod, który monitoruje wykorzystanie pamięci GPU za pomocą $torch.cuda.memory_allocated()$ (jeśli GPU jest dostępne).
Jeśli kod działa na GPU, dodano możliwość monitorowania wykorzystania pamięci GPU przed i po treningu. To pozwala na analizę wpływu użycia GPU na pamięć w trakcie trenowania modelu.
```Python
if torch.cuda.is_available():
    print(f"Zajęta pamięć GPU przed treningiem: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    # kod do treningu modelu
    print(f"Zajęta pamięć GPU po treningu: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
```
**3.Porównanie CPU i GPU:**

* Dodano opcję wyboru urządzenia obliczeniowego (CPU lub GPU) i porównania wydajności między nimi.
Przed rozpoczęciem treningu kod sprawdza, czy dostępne jest GPU, i przełącza urządzenie obliczeniowe na GPU, jeśli jest dostępne. To umożliwia porównanie wydajności na różnych urządzeniach.
```Python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_all = torch.tensor(X_all).float().to(device)
Y_all = torch.tensor(Y_all).float().to(device)
```
* Sprawdzano, czy GPU jest dostępne, i automatyczne przełączanie na GPU, jeśli to możliwe.
Jeśli GPU jest dostępne, kod automatycznie używa GPU do obliczeń, a jeśli nie, obliczenia są wykonywane na CPU.
```Python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
**4.Dane wyjściowe dla testu wydajności:**

* Wydrukowano czas trwania treningu modelu oraz predykcji.
Czas wykonania dla treningu modelu i predykcji jest raportowany w czasie rzeczywistym, dzięki czemu użytkownik może zobaczyć, ile czasu zajmują te operacje.
```Python
print(f"Trening modelu: {end_time - start_time:.6f} s")
```

* Porównano wydajność dla różnych urządzeń (jeśli GPU jest dostępne).
Jeśli model jest uruchomiony na obu urządzeniach (CPU i GPU), kod drukuje czas wykonania na każdym z urządzeń, co pozwala na porównanie ich wydajności.
```Python
print(f"Predykcja: {end_time - start_time:.6f} s")
```


**Podsumowanie:**

Kod mierzy czas wykonywania dla etapów takich jak:

* Normalizacja danych(zajmuje bardo mały czas z powody małej ilości dannych ,0.001040 s)

* Tworzenie zestawów danych(prawie błyskawicznie 0.000000 s)

* Trening modelu(trwa znacznie dłużej niż inne kroki, 3.426518 s)

* Predykcja (działa bardzo szybko 0.015540 s)

Dodatkowo, monitoruje wykorzystanie pamięci GPU (nie jest dostępne u mnie), umożliwia przełączanie na GPU w przypadku jego dostępności, oraz porównuje wydajność między CPU i GPU, raportując czas wykonania dla obu urządzeń.

**Wyniki:**

Dla dannyh cen mieszkań w Krakowie z nastawami epochs = 1000, lr=0.001 oraz stepsize = 25 mamy następujące wyniki:

**Postęp trenowania:**
* Epoka [10/1000]: Loss: 0.111736
* Epoka [50/1000]: Loss: 0.046849
* Epoka [100/1000]: Loss: 0.007081
* Epoka [200/1000]: Loss: 0.001709
* Epoka [400/1000]: Loss: 0.001457
* Epoka [600/1000]: Loss: 0.001237
* Epoka [800/1000]: Loss: 0.001031
* Epoka [1000/1000]: Loss: 0.000898

**Wykres:**


![Figure_12](https://github.com/user-attachments/assets/4e61d10b-e0ce-4d5b-8acf-957f1e8937cd)



Literatura, zdjęcia, dane:

https://www.bankier.pl/wiadomosc/Od-boomu-do-boomu-tak-zmienialy-sie-ceny-mieszkan-od-2006-roku-8277516.html

https://colah.github.io/posts/2015-08-Understanding-LSTMs/

https://www.youtube.com/watch?v=YCzL96nL7j0&t=98s&ab_channel=StatQuestwithJoshStarmer
