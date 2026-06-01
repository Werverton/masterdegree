import pandas as pd
import matplotlib.pyplot as plt

# Caminho do arquivo CSV gerado pelo chargepol
csv_file = r"C:\Users\werve\Downloads\chargepol\chargepol.csv"

# Ler o CSV, informando que linhas iniciadas com '#' são comentários
df = pd.read_csv(csv_file, comment='#')

# Remove linhas de cabeçalho duplicadas no meio do arquivo
df = df[df['charge'].isin(['pos', 'neg'])]

# Converte a coluna de altitude para numérico
df['zmin'] = pd.to_numeric(df['zmin'], errors='coerce')

# Remove valores nulos gerados na conversão
df = df.dropna(subset=['zmin'])

# Separar os dados de cargas positivas e negativas
df_pos = df[df['charge'] == 'pos']
df_neg = df[df['charge'] == 'neg']

# Criar a figura para o plot
plt.figure(figsize=(10, 6))

# Plotar os histogramas (bins=20 divide os dados em 20 faixas de altitude)
plt.hist(df_pos['zmin'], bins=20, alpha=0.6, color='red', label='Carga Positiva', edgecolor='black')
plt.hist(df_neg['zmin'], bins=20, alpha=0.6, color='blue', label='Carga Negativa', edgecolor='black')

# Configurações de exibição do gráfico
plt.xlabel('Altitude Base da Camada - zmin (km)', fontsize=12)
plt.ylabel('Frequência (Número de Ocorrências)', fontsize=12)
plt.title('Distribuição Vertical das Camadas de Carga', fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Mostrar o resultado
plt.show()