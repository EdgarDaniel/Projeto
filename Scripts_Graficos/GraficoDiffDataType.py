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


#READ FILES AND SEPARATE DATA IN ARRAYS / CALCULATE MEAN

#Dados de Diferentes Equipamentos com Batch Size 30
"""
dados_AGE_GENDER_16FP_B30_MYRIAD, array_media_AG16B30MYRIAD, media_AG16B30MYRIAD = read_array_data("age-gender-recognition-retail-0013_16_B30xmlMYRIAD.txt")
dados_AGE_GENDER_16FP_B30_CPU, array_media_AG16B30CPU, media_AG16B30CPU= read_array_data("age-gender-recognition-retail-0013_16_B30.xmlCPU.txt")
"""

#Dados de Diferentes Tipos de Dados

dados_AG_FT16, media_dados_AG16, media_AG16 = read_array_data("age-gender-recognition-retail-0013_CPU_16_6262.txt")
dados_AG_FT32, media_dados_AG32, media_AG32 = read_array_data("age-gender-recognition-retail-0013_CPU_32_6262.txt")

dados_HP_FT16, media_dados_HP16, media_HP16 = read_array_data("human-pose-estimation-0001_CPU_16_456-256.txt")
dados_HP_FT32, media_dados_HP32, media_HP32 = read_array_data("human-pose-estimation-0001_CPU_32_456-256.txt")

dados_C_FT16, media_dados_C16, media_C16 = read_array_data("person-vehicle-bike-detection-crossroad-0078_CPU_16_10241024.txt")
dados_C_FT32, media_dados_C32, media_C32 = read_array_data("person-vehicle-bike-detection-crossroad-0078_CPU_32_10241024.txt")

dados_E_FT16, media_dadosE16, media_E16 = read_array_data("emotions-recognition-retail-0003_CPU_16_6464.txt")
dados_E_FT32, media_dadosE32, media_E32 = read_array_data("emotions-recognition-retail-0003_CPU_32_6464.txt")

dados_FD_FT16, media_dadosFD16, media_FD16 = read_array_data("face-detection-adas-0001_CPU_16_672-384.txt")
dados_FD_FT32, media_dadosFD32, media_FD32 = read_array_data("face-detection-adas-0001_CPU_32_672-384.txt")

dados_FR_FT16, media_dados_FR16, media_FR16 = read_array_data("face-reidentification-retail-0095_CPU_16_128128.txt")
dados_FR_FT32, media_dados_FR32, media_FR32 = read_array_data("face-reidentification-retail-0095_CPU_32_128128.txt")

dados_FL_FT16, media_dados_FL16, media_FL16 = read_array_data("facial-landmarks-35-adas-0002_CPU_16_6060.txt")
dados_FL_FT32, media_dados_FL32, media_FL32 = read_array_data("facial-landmarks-35-adas-0002_CPU_32_6060.txt")

dados_V_FT16, media_dados_V16, media_V16 = read_array_data("vehicle-license-plate-detection-barrier-0106_16_CPU_300300.txt")
dados_V_FT32, media_dados_V32, media_V32 = read_array_data("vehicle-license-plate-detection-barrier-0106_32_CPU_300300.txt")


#Gráfico de Diferentes Tipos de Dados


raw_data = {'model_name': ['Idade/Gênero','Emoções','Características Faciais','Deteção de Matrículas','Reconhecimento de Cara','Deteção de Cara',
                           'Deteção de Objetos Estrada','Pose do Corpo'],
        'FT16_dados': [media_AG16.round(1),media_E16.round(1),media_FL16.round(1), media_V16.round(1),media_FR16.round(1),media_FD16.round(1),
                       media_C16.round(1),media_HP16.round(1)],
        'FT32_dados': [media_AG32.round(1),media_E32.round(1),media_FL32.round(1), media_V32.round(1),media_FR32.round(1),media_FD32.round(1),
                    media_C32.round(1),media_HP32.round(1)]}

df = pd.DataFrame(raw_data, columns = ['model_name', 'FT16_dados', 'FT32_dados'])

# Setting the positions and width for the bars
pos = list(range(len(df['FT16_dados'])))
width = 0.25

# Plotting the bars
fig, ax = plt.subplots(figsize=(13,6))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos,
        # using df['pre_score'] data,
        df['FT16_dados'],
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
        df['FT32_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#F78F1E',
        # with label the second value in first_name
        label=df['model_name'][0])

"""
# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width * 2 for p in pos],
        # using df['post_score'] data,
        df['post_score'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#FFC222',
        # with label the third value in first_name
        label=df['first_name'][2])
"""

# Set the y axis label
ax.set_ylabel('Tempo (ms)')

# Set the chart's title
#ax.set_title('Testes De Desempenho em CPU (AMD RYZEN 5 2600)')

# Set the position of the x ticks
ax.set_xticks([p + 0.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['model_name'],rotation = 80)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos) - width, max(pos) + width * 4)
#plt.ylim([0, max(df['FT16_dados'] + df['FT32_dados'])])
plt.ylim([0,70])
print(pos)
for i in range(len(pos)):
    plt.text(x=pos[i]- 0.15, y=df['FT16_dados'][i] + 0.1, s=df['FT16_dados'][i], size=10)

for i in range(len(pos)):
    plt.text(x=pos[i]+0.2, y=df['FT32_dados'][i]+0.1, s=df['FT32_dados'][i],size=10)

plt.subplots_adjust(bottom=0.35)

# Adding the legend and showing the plot
plt.legend(['FT16', 'FT32'], loc='upper right')
plt.grid(axis='y')
plt.show()
fig.savefig('Different_Data_F16_F32_CPU.png')
