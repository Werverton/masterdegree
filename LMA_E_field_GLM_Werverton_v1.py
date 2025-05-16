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

# Depuração: Verificar os dados carregados
print("Dados carregados:")
print(df.head())

# Filtrar os dados
filtered_df = df.query('nstn < 0')

# Depuração: Verificar os dados filtrados
print("Dados filtrados:")
print(filtered_df.head())

# Verificar se há dados após a filtragem
if filtered_df.empty:
    print("Nenhum dado após a filtragem. Verifique os critérios de filtragem.")
else:
    # Converter segundos desde a meia-noite para horas, minutos e segundos
    m, s = divmod(filtered_df['sfm'], 60)
    h, m = divmod(m, 60)
    filtered_df['time'] = h + m / 60 + s / 3600  # Tempo em horas

    # Plotar os dados
    plt.figure(figsize=(10, 6))

    # Plotar altitude em função do tempo
    plt.scatter(filtered_df['time'], filtered_df['alt'] / 1000, c=filtered_df['dBW'], cmap='viridis', s=10)
    plt.colorbar(label='dBW (Power)')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Altitude (km)', fontsize=12)
    plt.title('LMA Data: Altitude vs Time', fontsize=14)
    plt.grid(True)

    # Salvar e mostrar o gráfico
    plt.savefig('LMA_plot.png', dpi=200)
    plt.show()