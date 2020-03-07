import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#FUNCTION TO READ DATA FROM FILE
def read_array_data(file):
    file_pointer = open(file)
    array_data = np.array(file_pointer.read().splitlines()).astype(np.float)
    array_media = array_data[2:-4]
    array_dados = array_data[-4:]
    media = np.mean(array_media)

    return array_dados,array_media,media


#Dados de Diferenças entre MYRIAD e CPU

dados_AG_FT16_B1_CPU, media_dados_AG161CPU, media_AG161CPU = read_array_data("age-gender-recognition-retail-0013_62-62.xmlCPU.txt")
dados_AG_FT16_B1_MYRIAD, media_dados_AG161MYRIAD, media_AG161MYRIAD = read_array_data("age-gender-recognition-retail-0013_62-62.xmlMYRIAD.txt")
dados_AG_FT16_B1_MYRIAD2, media_dados_AG161MYRIAD2, media_AG161MYRIAD2 = read_array_data("age-gender-recognition-retail-0013_MYRIAD2.txt")

dados_E_FT16_B1_CPU, media_dados_E161CPU, media_E161CPU = read_array_data("emotions-recognition-retail-0003_64-64.xmlCPU.txt")
dados_E_FT16_B1_MYRIAD, media_dados_E161MYRIAD, media_E161MYRIAD = read_array_data("emotions-recognition-retail-0003_64-64.xmlMYRIAD.txt")
dados_E_FT16_B1_MYRIAD2, media_dados_E161MYRIAD2, media_E161MYRIAD2 = read_array_data("emotions-recognition-retail-0003_MYRIAD2.txt")

dados_FD_FT16_B1_CPU, media_dados_FD161CPU, media_FD161CPU = read_array_data("face-detection-adas-0001_672-384.xmlCPU.txt")
dados_FD_FT16_B1_MYRIAD, media_dados_FD161MYRIAD, media_FD161MYRIAD = read_array_data("face-detection-adas-0001.xmlMYRIAD.txt")
dados_FD_FT16_B1_MYRIAD2, media_dados_FD161MYRIAD2, media_FD161MYRIAD2 = read_array_data("face-detection-adas-0001_MYRIAD2.txt")

dados_FR_FT16_B1_CPU, media_dados_FR161CPU, media_FR161CPU = read_array_data("face-reidentification-retail-0095_182-128.xmlCPU.txt")
dados_FR_FT16_B1_MYRIAD, media_dados_FR161MYRIAD, media_FR161MYRIAD = read_array_data("face-reidentification-retail-0095_128-128.xmlMYRIAD.txt")
dados_FR_FT16_B1_MYRIAD2, media_dados_FR161MYRIAD2, media_FR161MYRIAD2 = read_array_data("face-reidentification-retail-0095_MYRIAD2.txt")

dados_FL_FT16_B1_CPU, media_dados_FL161CPU, media_FL161CPU = read_array_data("facial-landmarks-35-adas-0002_60-60.xmlCPU.txt")
dados_FL_FT16_B1_MYRIAD, media_dados_FL161MYRIAD, media_FL161MYRIAD = read_array_data("facial-landmarks-35-adas-0002_60-60.xmlMYRIAD.txt")
dados_FL_FT16_B1_MYRIAD2, media_dados_FL161MYRIAD2, media_FL161MYRIAD2 = read_array_data("facial-landmarks-35-adas-0002_MYRIAD2.txt")

dados_HP_FT16_B1_CPU, media_dados_HP161CPU, media_HP161CPU = read_array_data("human-pose-estimation-0001_456_256.xmlCPU.txt")
dados_HP_FT16_B1_MYRIAD, media_dados_HP161MYRIAD, media_HP161MYRIAD = read_array_data("human-pose-estimation-0001.xmlMYRIAD.txt")
dados_HP_FT16_B1_MYRIAD2, media_dados_HP161MYRIAD2, media_HP161MYRIAD2 = read_array_data("human-pose-estimation-0001_MYRIAD2.txt")

dados_C_FT16_B1_CPU, media_dados_C161CPU, media_C161CPU = read_array_data("person-vehicle-bike-detection-crossroad-0078.xmlCPU.txt")
dados_C_FT16_B1_MYRIAD, media_dados_C161MYRIAD, media_C161MYRIAD = read_array_data("person-vehicle-bike-detection-crossroad-0078.xmlMYRIAD.txt")
dados_C_FT16_B1_MYRIAD2, media_dados_C161MYRIAD2, media_C161MYRIAD2 = read_array_data("person-vehicle-bike-detection-crossroad-0078_MYRIAD2.txt")

dados_V_FT16_B1_CPU, media_dados_V161CPU, media_V161CPU = read_array_data("vehicle-license-plate-detection-barrier-0106.xmlCPU.txt")
dados_V_FT16_B1_MYRIAD, media_dados_V161MYRIAD, media_V161MYRIAD= read_array_data("vehicle-license-plate-detection-barrier-0106.xmlMYRIAD.txt")
dados_V_FT16_B1_MYRIAD2, media_dados_V161MYRIAD2, media_V161MYRIAD2= read_array_data("vehicle-license-plate-detection-barrier-0106_MYRIAD2.txt")

#Gráfico de MYRIAD e CPU


raw_data = {'model_name': ['Idade/Gênero', 'Emoções', 'Características Faciais', 'Deteção de Matrículas', 'Reconhecimento de Cara',
                           'Deteção de Cara', ' Deteção de Objetos Estrada', 'Pose do Corpo'],
        'cpu_dados': [media_AG161CPU.round(1),media_E161CPU.round(1),media_FL161CPU.round(1),media_V161CPU.round(1),
                      media_FR161CPU.round(1),media_FD161CPU.round(1),media_C161CPU.round(1),media_HP161CPU.round(1)],
        'myriad_dados': [media_AG161MYRIAD.round(1),media_E161MYRIAD.round(1),media_FL161MYRIAD.round(1),media_V161MYRIAD.round(1),
                         media_FR161MYRIAD.round(1),media_FD161MYRIAD.round(1),media_C161MYRIAD.round(1),media_HP161MYRIAD.round(1)],
        'myriad2_dados': [media_AG161MYRIAD2.round(1),media_E161MYRIAD2.round(1),media_FL161MYRIAD2.round(1),media_V161MYRIAD2.round(1),
                          media_FR161MYRIAD2.round(1),media_FD161MYRIAD2.round(1),media_C161MYRIAD2.round(1),media_HP161MYRIAD2.round(1)]}

df = pd.DataFrame(raw_data, columns = ['model_name', 'cpu_dados', 'myriad2_dados', 'myriad_dados'])

# Setting the positions and width for the bars
pos = list(range(len(df['cpu_dados'])))
width = 0.25

# Plotting the bars
fig, ax = plt.subplots(figsize=(12, 7))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos,
        # using df['pre_score'] data,
        df['cpu_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#EE3224',
        # with label the first value in first_name
        label=df['model_name'][0])

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos],
        # using df['mid_score'] data,
        df['myriad2_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#F78F1E',
        # with label the second value in first_name
        label=df['model_name'][1])

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width * 2 for p in pos],
        # using df['post_score'] data,
        df['myriad_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#FFC222',
        # with label the third value in first_name
        label=df['model_name'][2])

# Set the y axis label
ax.set_ylabel('Tempo (ms)')

# Set the chart's title
#ax.set_title('Testes De Desempenho')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['model_name'],rotation=80)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos) - width, max(pos) + width * 4)
plt.ylim([0, 450])

print(pos)
for i in range(len(pos)):
    plt.text(x=pos[i]- 0.1, y=df['cpu_dados'][i] + 0.1, s=df['cpu_dados'][i], size=10)

for i in range(len(pos)):
    plt.text(x=pos[i]+0.15, y=df['myriad2_dados'][i]+0.1, s=df['myriad2_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i]+0.4, y=df['myriad_dados'][i]+0.1, s=df['myriad_dados'][i],size=10)

plt.subplots_adjust(bottom=0.3)
# Adding the legend and showing the plot
plt.legend(['AMD RYZEN 5 2600', 'MYRIAD 2 (DESKTOP)', 'MYRIAD 1 (DESKTOP)'], loc='upper left')
plt.grid(axis='y')
plt.show()
fig.savefig('Gráfico.png')