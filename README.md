# 1)Opis rzeczywistego problemu.

## _Cel:_

Celem projektu jest opracowanie modelu sztucznej inteligencji (SI), który będzie przewidywał przyszłe wartości na podstawie analizy trendów w danych. 
Model ten będzie wykorzystywał technologię Long Short-Term Memory (LSTM), która jest jednym z typów rekurencyjnych sieci neuronowych (RNN). 
Model będzie analizował dane wejściowe przedstawiające zmieniające się wartości w czasie (np. zmiany ceny akcji, temperatury, itp.) i na ich podstawie przewidywał wartość na przyszłość.

Przykład: mamy dane o cenie akcji spółki z ostatnich dni, w odpowiedzi chcemy przewidzieć jutrzejszą cenę.

![Zrzut ekranu 2024-12-07 191340](https://github.com/user-attachments/assets/554e7fe9-b65a-45c8-bba3-da9c1f26b315)


## _Motywacja:_

Zrozumienie i prognozowanie trendów w różnych dziedzinach życia, takich jak ekonomia, meteorologia czy analiza rynku, ma ogromne znaczenie praktyczne. 
Przewidywanie przyszłych wartości na podstawie wcześniejszych danych może pomóc w podejmowaniu lepszych decyzji, takich jak optymalizacja procesów biznesowych,
prognozy popytu na produkty, monitorowanie zmian klimatycznych czy przewidywanie cen na rynkach finansowych. Projekt oparty na sieciach neuronowych LSTM jest istotny,
ponieważ LSTM jest bardziej odporny na błędy niż klasyczne rekurencyjne sieci neuronowe, a jego zastosowanie w analizie szeregów czasowych może zwiększyć dokładność prognoz.

## _Dane wejściowe:_

Dane wejściowe do modelu będą pochodzić z serii wartości zmieniających się w czasie(dniach), które będą generowane losowo (dla uproszczenia w przedziale "high" , "Medium" i "Low").
Dane te będą reprezentować zmieniający się trend w czasie (np. zmiany cen, temperatury, wskaźników ekonomicznych).
Model będzie otrzymywał dane na przestrzeni "x" dni, a potem model wygeneruje prognozę wartości na podstawie tych danych.

<img width="637" alt="image" src="https://github.com/user-attachments/assets/b958f90e-3c27-416e-a53f-04987830de88">

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

W projekcie sieć LSTM analizuje wygenerowane dane i przewiduje trend na następny dzień. Dzięki opisanym mechanizmom (zapominanie, aktualizacja, generowanie wyjścia) model potrafi uwzględniać istotne zależności czasowe i ignorować zbędne dane.LSTM "przechodzi" przez dane sekwencyjne, na każdym kroku dokonując aktualizacji stanu komórki i wyjścia. W przeciwieństwie do klasycznych RNN, LSTM pozwala na przechowywanie informacji na dłuższe okresy, dzięki bramkom, które kontrolują przepływ informacji. Z tego powodu LSTM może lepiej radzić sobie z długoterminowymi zależnościami, np. w analizie trendów czasowych.

---

### Przykład działania LSTM w kontekście prognozowania wartości na podstawie 4 dni

Dla czterech dni wejściowych, LSTM analizuje każdy dzień jako część sekwencji. Wartości z wcześniejszych dni wpływają na decyzje podejmowane przez bramki, co pozwala na przewidywanie wartości na dzień 5. 

- Stan komórki przechowuje informacje o zależnościach występujących w danych (np. zmiany wartości w trendzie). 
- Te informacje są następnie wykorzystywane do prognozowania kolejnej wartości.

Przykładowe obliczenia dla jednego dnia:

<img width="957" alt="image" src="https://github.com/user-attachments/assets/6879f8fc-2f7c-4ff9-9287-e405d54814bf">

Takie obliczenia zgodnie z algorytmem zostaną zrobione dla każdego dnia:

<img width="934" alt="image" src="https://github.com/user-attachments/assets/6f26adb9-48be-41d7-a546-ec9667289ca9">

Pod koniec obliczeń otrzymujemy konieczną wartośc z "Short Term Memory"(na zdjęciu oznaczona jako "x"), która i będzie przewidywaną wartością na 5 dzień.

---

## Co jest potrzebne do realizacji w rzeczywistym świecie?

### Wymagania sprzętowe i środowiskowe

1. **Komputer z odpowiednią mocą obliczeniową**  
   Do uruchamiania i trenowania modeli LSTM potrzebny jest komputer z wystarczającą ilością pamięci RAM i procesorem obsługującym obliczenia równoległe (np. z GPU). W przypadku mniejszych danych wystarczy standardowy laptop.

2. **Środowisko programistyczne**  
   - Python w wersji 3.x.  
   - Biblioteki do obliczeń i analizy danych:  
     - NumPy i Pandas (przetwarzanie danych).  
     - TensorFlow lub PyTorch (implementacja i trenowanie LSTM).  
     - Matplotlib lub Plotly (wizualizacja wyników).  

3. **Źródło danych**  
   - W rzeczywistym świecie dane mogą pochodzić z baz danych, plików CSV lub interfejsów API, takich jak dane giełdowe, dane pogodowe, logi systemów IT itp.  
   - W naszym przypadku dane są generowane automatycznie w zakresie od 0 do 1.  

4. **Dane historyczne do trenowania**  
   W prawdziwych zastosowaniach wymagane są duże zbiory danych historycznych, aby nauczyć model rozpoznawać wzorce.

---

### Procedura testowania rozwiązania

1. **Testy funkcjonalne**  
   - Sprawdzenie poprawności generowania danych wejściowych: czy dane są w odpowiednim formacie (np. zakres od 0 do 1).  
   - Upewnienie się, że model LSTM poprawnie przewiduje wartość na podstawie wcześniejszych dni.  

2. **Testy wydajnościowe**  
   - Testowanie szybkości trenowania modelu na większych zbiorach danych.  
   - Monitorowanie zużycia zasobów, takich jak pamięć RAM i moc obliczeniowa procesora/GPU.

3. **Testy dokładności**  
   - Porównanie prognozowanych wartości z rzeczywistymi (wygenerowanymi wcześniej) w celu oceny dokładności przewidywań.  
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

