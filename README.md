**1)Opis rzeczywistego problemu.**  

_Cel:_

Celem projektu jest opracowanie modelu sztucznej inteligencji (SI), który będzie przewidywał przyszłe wartości na podstawie analizy trendów w danych. 
Model ten będzie wykorzystywał technologię Long Short-Term Memory (LSTM), która jest jednym z typów rekurencyjnych sieci neuronowych (RNN). 
Model będzie analizował dane wejściowe przedstawiające zmieniające się wartości w czasie (np. zmiany ceny akcji, temperatury, itp.) i na ich podstawie przewidywał wartość na przyszłość.

Przykład: mamy dane o cenie akcji spółki z ostatnich czterech dni, w odpowiedzi chcemy przewidzieć jutrzejszą cenę.
![Zrzut ekranu 2024-12-07 191340](https://github.com/user-attachments/assets/4ea31326-595b-4b9b-a383-4d65b0b43b53)

_Motywacja:_

Zrozumienie i prognozowanie trendów w różnych dziedzinach życia, takich jak ekonomia, meteorologia czy analiza rynku, ma ogromne znaczenie praktyczne. 
Przewidywanie przyszłych wartości na podstawie wcześniejszych danych może pomóc w podejmowaniu lepszych decyzji, takich jak optymalizacja procesów biznesowych,
prognozy popytu na produkty, monitorowanie zmian klimatycznych czy przewidywanie cen na rynkach finansowych. Projekt oparty na sieciach neuronowych LSTM jest istotny,
ponieważ LSTM jest bardziej odporny na błędy niż klasyczne rekurencyjne sieci neuronowe, a jego zastosowanie w analizie szeregów czasowych może zwiększyć dokładność prognoz.

_Dane wejściowe:_

Dane wejściowe do modelu będą pochodzić z serii wartości zmieniających się w czasie(dniach), które będą generowane losowo (dla uproszczenia w przedziale "high" , "Medium" i "Low").
Dane te będą reprezentować zmieniający się trend w czasie (np. zmiany cen, temperatury, wskaźników ekonomicznych).
Model będzie otrzymywał dane na przestrzeni "x" dni, a potem model wygeneruje prognozę wartości na podstawie tych danych.
<img width="637" alt="image" src="https://github.com/user-attachments/assets/b958f90e-3c27-416e-a53f-04987830de88">

_Zastosowanie sztucznej inteligencji:_

Projekt wykorzystuje algorytm Long Short-Term Memory (LSTM), który należy do rodziny rekurencyjnych sieci neuronowych (RNN).
LSTM jest szczególnie skuteczną metodą w analizie danych szeregów czasowych, ponieważ jest w stanie przechowywać długoterminowe zależności w danych, co pozwala na dokładniejsze przewidywania. 
W tym przypadku, model LSTM będzie analizował zmieniające się wartości i na ich podstawie przewidywał przyszłe dane.

**2)State of art**
1. **Recurrent Neural Networks (RNN)** <img width="812" alt="image" src="https://github.com/user-attachments/assets/4a73f72a-d225-4b21-8380-1dbdc991898a">

RNN to podstawowa struktura sieci neuronowej, która jest szczególnie skuteczna w przetwarzaniu danych sekwencyjnych.
Główna cecha RNN to fakt, że posiadają one "pamięć", czyli informacje z poprzednich kroków mogą być wykorzystywane do przetwarzania kolejnych danych w sekwencji.
Jednak tradycyjne RNN mają problem z utrzymywaniem długoterminowych zależności ,tzw. problem zanikania gradientu(Vanishing/Exploding Gradient Problem).

_Zalety:_

* Dobry do analizy danych sekwencyjnych (np. tekstu, danych czasowych).

* Prosta architektura.

_Wady:_

* Problemy z przechowywaniem długoterminowych zależności.

* Skłonność do problemu zanikania gradientu.


2. **Long Short-Term Memory (LSTM)**
   
LSTM to rozszerzenie klasycznego RNN, które zostało zaprojektowane, aby rozwiązać problem zanikania gradientu w długich sekwencjach. Dzięki swojej strukturze (trzy główne "bramki": wejścia, zapomnienia i wyjścia), LSTM jest w stanie przechowywać informacje przez długie okresy czasu, co czyni go bardziej skutecznym w rozwiązywaniu problemów z długoterminowymi zależnościami.

<img width="450" alt="image" src="https://github.com/user-attachments/assets/aad364e4-2181-449a-aea0-0ded00d3b93d">

_Zalety:_

* Skuteczne w modelowaniu długoterminowych zależności.

* Dobrze radzi sobie z nieliniowymi wzorcami i sekwencjami o dużych odległościach czasowych.

_Wady:_

* Złożona architektura, co prowadzi do większych wymagań obliczeniowych.

* Trudniejsza do wytrenowania niż tradycyjne RNN.


3. **Gated Recurrent Unit (GRU)**
GRU to uproszczona wersja LSTM, która osiąga podobne rezultaty przy mniejszej liczbie parametrów. GRU ma dwie główne bramki (Update Gate i Reset Gate),w odróżnieniu od LSTM która ma trzy bramki (Input , Output , Forget), co czyni je bardziej efektywnymi obliczeniowo. GRU jest również w stanie przechowywać długoterminowe zależności, ale bez tak złożonej struktury jak LSTM.
<img width="450" alt="image" src="https://github.com/user-attachments/assets/35ccb965-609b-4e4a-aa3c-0754c9f2d1b4">

_Zalety:_

* Mniejsza liczba parametrów w porównaniu do LSTM.

* Szybsze trenowanie przy zachowaniu wysokiej wydajności.

* Wydajniejsze w przypadku mniejszych zbiorów danych.

_Wady:_

* Może nie być tak elastyczne jak LSTM w rozwiązywaniu niektórych bardziej złożonych problemów.

* Czasami może oferować gorszą jakość wyników w bardziej złożonych zadaniach.

_Podsumowanie:_
RNN to podstawowy model do przetwarzania sekwencji, ale ma trudności z długoterminowymi zależnościami.
LSTM to bardziej zaawansowany model, który rozwiązuje problem zanikania gradientu i świetnie radzi sobie z długoterminowymi zależnościami.
GRU to uproszczona wersja LSTM, która oferuje podobną wydajność, ale jest bardziej efektywna obliczeniowo.

**3) Opis wybranej koncepcji – LSTM (Long Short-Term Memory)**

**_Opis ogólny LSTM:_**

LSTM to szczególny typ sieci neuronowej rekurencyjnej (RNN), zaprojektowany do rozwiązywania problemu przechowywania długoterminowej zależności w danych czasowych. Jest szeroko stosowany do analizy sekwencyjnych danych, takich jak przewidywanie wartości czasowych.LSTM zawiera tzw. cell state (stan komórki), który jest używany do przechowywania informacji przez długi czas, co pozwala na skuteczną obsługę długoterminowych zależności, które są wyzwaniem w tradycyjnych RNN.

**_Wzory LSTM i struktura algorytmu:_**

LSTM składa się z trzech głównych składników: bramki wejściowej (input gate), bramki zapomnienia (forget gate) i bramki wyjściowej (output gate), które decydują o tym, jak informacje będą przechowywane, zapominane i wydobywane w komórkach LSTM.

Formuły dla tych bramek:
# LSTM - Long Short-Term Memory

LSTM (Long Short-Term Memory) to specjalny typ sieci neuronowej rekurencyjnej (RNN), zaprojektowany do rozwiązywania problemu przechowywania długoterminowej zależności w danych czasowych.

## Wzory i algorytm LSTM

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
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

- \( i_t \) – wartość bramki wejściowej.
- \( \tilde{C}_t \) – nowe informacje, które mogą zostać zapisane.

### 3. **Aktualizacja stanu komórki (Cell State)**

Stan komórki jest aktualizowany na podstawie bramki zapomnienia oraz bramki wejściowej.

$$
C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t
$$

- \( C_t \) – nowy stan komórki.
- \( C_{t-1} \) – stan komórki z poprzedniego kroku.

### 4. **Bramka wyjściowa (Output Gate)**

Decyduje, co zostanie wyjściem sieci i jakie informacje będą przekazywane dalej.

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

Finalne wyjście sieci:

$$
h_t = o_t \cdot \tanh(C_t)
$$

- \( o_t \) – wartość bramki wyjściowej.
- \( h_t \) – ukryte wyjście w czasie \( t \).

## Struktura LSTM

LSTM składa się z trzech głównych składników:

1. **Bramka zapomnienia (Forget Gate)** – kontroluje, które informacje mają zostać zapomniane w stanie komórki.
2. **Bramka wejściowa (Input Gate)** – decyduje, które nowe informacje będą zapisywane w stanie komórki.
3. **Bramka wyjściowa (Output Gate)** – kontroluje, które informacje będą przekazane jako wyjście sieci.

### Cell State (Stan komórki)

Stan komórki \( C_t \) jest kluczowym elementem architektury LSTM. Przechowuje on informacje przez długi czas, umożliwiając sieci "zapamiętanie" istotnych danych z przeszłości, które są następnie wykorzystywane do prognozowania w przyszłości.

## Przykład zastosowania

Dla czterech dni wejściowych, LSTM będzie analizować każdy dzień jako część sekwencji. Wartości z wcześniejszych dni wpływają na decyzje podejmowane przez bramki, co pozwala na przewidywanie wartości na dzień 5.

