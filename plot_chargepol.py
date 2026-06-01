import pandas as pd
import matplotlib.pyplot as plt

# Caminho do arquivo CSV gerado pelo chargepol
csv_file = r"C:\Users\werve\Downloads\chargepol\chargepol.csv"

# Ler o CSV, informando que linhas iniciadas com '#' são comentários
df = pd.read_csv(csv_file, comment='#')

# Separar os dados de cargas positivas e negativas
df_pos = df[df['charge'] == 'pos']
df_neg = df[df['charge'] == 'neg']

# Criar a figura para o plot
plt.figure(figsize=(12, 6))

# Plotar as camadas de carga positiva (em vermelho)
# O eixo x é o tempo. O y vai de 'zmin' até 'zmin + zwidth'
plt.vlines(x=df_pos['time'], ymin=df_pos['zmin'], ymax=df_pos['zmin'] + df_pos['zwidth'], 
           color='red', label='Carga Positiva', alpha=0.7, linewidth=3)

# Plotar as camadas de carga negativa (em azul)
plt.vlines(x=df_neg['time'], ymin=df_neg['zmin'], ymax=df_neg['zmin'] + df_neg['zwidth'], 
           color='blue', label='Carga Negativa', alpha=0.7, linewidth=3)

# Configurações de exibição do gráfico
plt.xlabel('Tempo (Segundos UT)', fontsize=12)
plt.ylabel('Altitude (km)', fontsize=12)
plt.title('Distribuição de Camadas de Carga (Positiva e Negativa) ao Longo do Tempo', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Mostrar o resultado
plt.show()
