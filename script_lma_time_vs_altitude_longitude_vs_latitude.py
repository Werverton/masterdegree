import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Caminho do arquivo de dados
filename = 'C:/Users/suporte/GIT/masterdegree/LL_240720_234000_0600.dat/LL_240720_234000_0600.dat'

# Configurações para leitura do arquivo
headerlength = 55  # Cabeçalho do arquivo
datatypes = (float, float, float, float, float, int, float)

# Carregar os dados do arquivo
sfm, lat, lon, alt, Xisq, nstn, dBW = np.genfromtxt(
    filename, dtype=datatypes, unpack=True, skip_header=headerlength, comments="#", usecols=[0, 1, 2, 3, 4, 5, 6]
)
data = {'sfm': sfm, 'lat': lat, 'lon': lon, 'alt': alt, 'Xisq': Xisq, 'nstn': nstn, 'dBW': dBW}

# Criar DataFrame
df = pd.DataFrame(data)
print("Shape do dataframe:", df.shape)
print(df.head())
print(df.describe())


# Converter segundos desde a meia-noite para horas, minutos e segundos
m, s = divmod(df['sfm'], 60)
h, m = divmod(m, 60)
df['time'] = h + m / 60 + s / 3600  # Tempo em horas

# Plot 1: Time vs Altitude
plt.figure(figsize=(10, 5))
plt.scatter(df['time'], df['alt'] / 1000, c=df['dBW'], cmap='viridis', s=10)
plt.colorbar(label='dBW (Power)')
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Altitude (km)', fontsize=12)
plt.title('LMA Data: Altitude vs Time', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('LMA_Time_vs_Altitude.png', dpi=200)
plt.show()

# Plot 2: Longitude vs Latitude
plt.figure(figsize=(8, 6))
plt.scatter(df['lon'], df['lat'], c=df['dBW'], cmap='plasma', s=10)
plt.colorbar(label='dBW (Power)')
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('LMA Data: Longitude vs Latitude', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('LMA_Longitude_vs_Latitude.png', dpi=200)
plt.show()