import numpy as np  
import torch  
import torch.nn as nn  
import torch.optim as optim 
import matplotlib.pyplot as plt  
import mplcursors 

# Przykładowe dane: ceny mieszkań od 4 kwartału 2006 do 4 kwartału 2021 roku (zł/m^2)
data = {
    "Warszawa": [7143, 7730, 8696, 9137, 9034, 8921, 8546, 8528, 9046, 8406, 8406, 7949, 8497, 8620, 8933, 8493, 8024,
                 7915, 7920, 7920, 7889, 7601, 7522, 7247, 7238, 6687, 6867, 6971, 7189, 7170, 7332, 7365, 7387, 7335,
                 7308, 7447, 7401, 7434, 7355, 7309, 7556, 7429, 7821, 8020, 8054, 8047, 8394, 8604, 8762, 8985, 9243,
                 9457, 9812, 10003, 10288, 10470, 10671, 10895, 11103, 10931],  
    "Kraków": [6349, 6267, 7309, 7193, 6715, 6777, 6629, 6294, 6530, 6166, 6065, 5975, 6155, 6182, 6115, 6296, 6284,
               6394, 6478, 6369, 6224, 6489, 6011, 6096, 6115, 5931, 5680, 5884, 5827, 5731, 5937, 5930, 5834, 6197,
               6157, 6095, 5939, 5884, 5740, 5820, 5951, 5979, 6030, 6347, 6337, 6215, 6400, 6621, 6849, 6626, 6956,
               7135, 7414, 7766, 8061, 8100, 8188, 8366, 9103, 9372]
}

data_min = np.min([np.min(data[city]) for city in data])  
data_max = np.max([np.max(data[city]) for city in data])  
data_normalized = {city: (np.array(data[city]) - data_min) / (data_max - data_min) for city in data}  # Normalizacja danych

# Funkcja do tworzenia danych wejściowych i wyników
def create_dataset(data, time_step=1):
    X, Y = [], []  
    for i in range(len(data) - time_step): 
        X.append(data[i:(i + time_step)])  
        Y.append(data[i + time_step])  
    return np.array(X), np.array(Y)  

time_step = 4  # Liczba kwartałów w historii, które będziemy wykorzystywać do przewidywania
X_all = []  
Y_all = [] 


for city in data_normalized:
    X, Y = create_dataset(data_normalized[city], time_step)  
    X_all.append(X)  
    Y_all.append(Y) 

X_all = np.concatenate(X_all, axis=0)  
Y_all = np.concatenate(Y_all, axis=0) 

X = torch.tensor(X_all).float()  
Y = torch.tensor(Y_all).float() 

# Model LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()  # Inicjalizacja klasy bazowej
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)  # Warstwa LSTM
        self.fc = nn.Linear(hidden_layer_size, output_size)  # Warstwa fully connected

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        predictions = self.fc(lstm_out[:, -1, :]) 
        return predictions  # Zwracamy prognozowane wartości

model = LSTMModel()  
criterion = nn.MSELoss()  # Funkcja kosztu (średni błąd kwadratowy)
optimizer = optim.Adam(model.parameters(), lr=0.001)  
epochs = 1000  
losses = []  

# Trening modelu
for epoch in range(epochs):  
    model.train()  # Ustawienie modelu w tryb treningu
    optimizer.zero_grad()  # Zerowanie gradientów
    y_pred = model(X.unsqueeze(-1))  # Przewidywanie cen na podstawie danych wejściowych
    loss = criterion(y_pred, Y.unsqueeze(-1))  # Obliczanie błędu
    loss.backward()  # Obliczanie gradientów wstecz
    optimizer.step()  # Aktualizacja wag modelu
    losses.append(loss.item())  # Dodanie wartości funkcji kosztu do listy
    if (epoch + 1) % 50 == 0:  # Co 50 epok wypisujemy postęp
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Rysowanie wykresu funkcji kosztu
plt.plot(range(epochs), losses)
plt.xlabel('Epoki')  
plt.ylabel('Loss')  
plt.title('Postęp funkcji kosztu')  
plt.show() 

# Ocena modelu
model.eval()  
last_data = torch.tensor(data_normalized["Warszawa"][-time_step:]).float().unsqueeze(0).unsqueeze(-1)  # Pobieranie ostatnich danych
predicted_price = model(last_data).item()  # Przewidywanie ceny na podstawie ostatnich danych
predicted_price = predicted_price * (data_max - data_min) + data_min  # Przekształcanie prognozy na pierwotną skalę

# Rysowanie wykresu
plt.plot(range(len(data["Warszawa"])), data["Warszawa"], label='Dane rzeczywiste')  
plt.plot(len(data["Warszawa"]), predicted_price, 'ro', label=f'Przewidywana cena ({predicted_price:.2f})')  
plt.xlabel('Czas (kwartały)')  
plt.ylabel('Cena')  
plt.legend() 

# Interakcja z wykresem
mplcursors.cursor(hover=True).connect(
    "add", lambda sel: (sel.annotation.set_text(
        f"Cena: {data['Warszawa'][int(sel.target[0])]}\n"
        f"Rok: {2006 + (int(sel.target[0]) // 4)}\n"
        f"Kwartał: {int(sel.target[0]) % 4 + 1}"
        if 0 <= int(sel.target[0]) < len(data['Warszawa'])
        else "Index out of range"
    ))
)

plt.show()  # Wyświetlanie wykresu



# Test generalizacji:
# # Podział danych na treningowe i testowe
# split_ratio = 0.8
# train_size = int(len(X_all) * split_ratio)
#
# X_train, Y_train = X_all[:train_size], Y_all[:train_size]
# X_test, Y_test = X_all[train_size:], Y_all[train_size:]
#
# # Trening modelu na zestawie treningowym
# X_train_tensor = torch.tensor(X_train).float().unsqueeze(-1)
# Y_train_tensor = torch.tensor(Y_train).float()
#
# # Przewidywanie na zestawie testowym
# X_test_tensor = torch.tensor(X_test).float().unsqueeze(-1)
# Y_test_tensor = torch.tensor(Y_test).float()
#
# with torch.no_grad():
#     y_pred = model(X_test_tensor)
#
# # Oblicz MSE na zestawie testowym
# mse_test = torch.mean((y_pred.squeeze() - Y_test_tensor) ** 2).item()
# print(f"Test generalizacji (MSE): {mse_test:.4f}")

# TEST wydajności
# import time # moduł czasu
# start_time = time.time()  # Początek pomiaru czasu
# # kod do wykonania (np. normalizacja danych)
# end_time = time.time()  # Koniec pomiaru czasu
# print(f"Normalizacja danych: {end_time - start_time:.6f} s")  # Raportowanie wyniku
# if torch.cuda.is_available():
#     print(f"Zajęta pamięć GPU przed treningiem: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
#     # kod do treningu modelu
#     print(f"Zajęta pamięć GPU po treningu: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# X_all = torch.tensor(X_all).float().to(device)
# Y_all = torch.tensor(Y_all).float().to(device)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Trening modelu: {end_time - start_time:.6f} s")
# print(f"Predykcja: {end_time - start_time:.6f} s")
