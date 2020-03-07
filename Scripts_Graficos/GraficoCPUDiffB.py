#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


#Dados de Diferentes Batch Size CPU


#dados_AG_FT16_B1_CPU,media_dados_AG161C,media_AG161C = read_array_data("age-gender-recognition-retail-0013_16_1_62-62_CPU.txt")
dados_AG_FT16_B1_CPU,media_dados_AG161C,media_AG161C = read_array_data("age-gender-recognition-retail-0013_CPU_1.txt")
dados_AG_FT16_B2_CPU,media_dados_AG162C,media_AG162C = read_array_data("age-gender-recognition-retail-0013_CPU_2.txt")
dados_AG_FT16_B4_CPU,media_dados_AG164C,media_AG164C = read_array_data("age-gender-recognition-retail-0013_CPU_4.txt")
dados_AG_FT16_B8_CPU,media_dados_AG168C,media_AG168C = read_array_data("age-gender-recognition-retail-0013_CPU_8.txt")
dados_AG_FT16_B16_CPU,media_dados_AG1616C,media_AG1616C = read_array_data("age-gender-recognition-retail-0013_CPU_16.txt")
dados_AG_FT16_B30_CPU,media_dados,media_AG1630C = read_array_data("age-gender-recognition-retail-0013_CPU_30.txt")

dados_E_FT16_B30_CPU, media_dados_E1630C, media_E1630C = read_array_data("emotions-recognition-retail-0003_CPU_30.txt")
dados_E_FT16_B16_CPU, media_dados_E1616C, media_E1616C = read_array_data("emotions-recognition-retail-0003_CPU_16.txt")
dados_E_FT16_B8_CPU, media_dados_E168C, media_E168C = read_array_data("emotions-recognition-retail-0003_CPU_8.txt")
dados_E_FT16_B4_CPU, media_dados_E164C, media_E164C = read_array_data("emotions-recognition-retail-0003_CPU_4.txt")
dados_E_FT16_B2_CPU, media_dados_E162C, media_E162C =read_array_data("emotions-recognition-retail-0003_CPU_2.txt")
dados_E_FT16_B1_CPU, media_dados_E161C, media_E161C = read_array_data("emotions-recognition-retail-0003_CPU_1.txt")

dados_FR_FT16_B30_CPU, media_dados_FR1630C, media_FR1630C = read_array_data("face-reidentification-retail-0095_CPU_30.txt")
dados_FR_FT16_B16_CPU, media_dados_FR1616C, media_FR1616C = read_array_data("face-reidentification-retail-0095_CPU_16.txt")
dados_FR_FT16_B8_CPU, media_dados_FR168C, media_FR168C = read_array_data("face-reidentification-retail-0095_CPU_8.txt")
dados_FR_FT16_B4_CPU, media_dados_FR164C, media_FR164C = read_array_data("face-reidentification-retail-0095_CPU_4.txt")
dados_FR_FT16_B2_CPU, media_dados_FR162C, media_FR162C = read_array_data("face-reidentification-retail-0095_CPU_2.txt")
dados_FR_FT16_B1_CPU, media_dados_FR161C, media_FR161C = read_array_data("face-reidentification-retail-0095_CPU_1.txt")

dados_FD_FT16_B30_CPU, media_dados_FD1630C, media_FD1630C = read_array_data("face-detection-adas-0001_CPU_30.txt")
dados_FD_FT16_B16_CPU, media_dados_FD1616C, media_FD1616C = read_array_data("face-detection-adas-0001_CPU_16.txt")
dados_FD_FT16_B8_CPU, media_dados_FD168C, media_FD168C = read_array_data("face-detection-adas-0001_CPU_8.txt")
dados_FD_FT16_B4_CPU, media_dados_FD164C, media_FD164C = read_array_data("face-detection-adas-0001_CPU_4.txt")
dados_FD_FT16_B2_CPU, media_dados_FD162C, media_FD162C = read_array_data("face-detection-adas-0001_CPU_2.txt")
dados_FD_FT16_B1_CPU, media_dados_FD161C, media_FD161C = read_array_data("face-detection-adas-0001_CPU_1.txt")

dados_FL_FT16_B30_CPU, media_dados_FL1630C, media_FL1630C = read_array_data("facial-landmarks-35-adas-0002_CPU_30.txt")
dados_FL_FT16_B16_CPU, media_dados_FL1616C, media_FL1616C = read_array_data("facial-landmarks-35-adas-0002_CPU_16.txt")
dados_FL_FT16_B8_CPU, media_dados_FL168C, media_FL168C = read_array_data("facial-landmarks-35-adas-0002_CPU_8.txt")
dados_FL_FT16_B4_CPU, media_dados_FL164C, media_FL164C = read_array_data("facial-landmarks-35-adas-0002_CPU_4.txt")
dados_FL_FT16_B2_CPU, media_dados_FL162C, media_FL162C = read_array_data("facial-landmarks-35-adas-0002_CPU_2.txt")
dados_FL_FT16_B1_CPU, media_dados_FL161C, media_FL161C = read_array_data("facial-landmarks-35-adas-0002_CPU_1.txt")


#Gráfico Diferentes Batch Size CPU


raw_data = {'model_name': ['Idade/Gênero (62*62)','Emoções (64*64)','Características Faciais (60*60)',
                           'Reconhecimento de Cara (128*128)','Deteção de Cara (672*384)'],
        'B1_dados': [media_AG161C.round(1),media_E161C.round(1), media_FL161C.round(1),media_FR161C.round(1)
            ,media_FD161C.round(1)],
        'B2_dados': [(media_AG162C/2).round(1),(media_E162C/2).round(1),(media_FL162C/2).round(1),(media_FR162C/2).round(1),
                     (media_FD162C/2).round(1)],
        'B4_dados': [(media_AG164C/4).round(1),(media_E164C/4).round(1),(media_FL164C/4).round(1),(media_FR164C/4).round(1),
                     (media_FD164C/4).round(1)],
        'B8_dados': [(media_AG168C/8).round(1),(media_E168C/8).round(1),(media_FL168C/8).round(1),(media_FR168C/8).round(1),
                     (media_FD168C/8).round(1)],
        'B16_dados': [(media_AG1616C/16).round(1),(media_E1616C/16).round(1),(media_FL1616C/16).round(1),(media_FR1616C/16).round(1),
                      (media_FD1616C/16).round(1)],
        'B30_dados': [(media_AG1630C/30).round(1),(media_E1630C/30).round(1),(media_FL1630C/30).round(1),
                      (media_FR1630C/30).round(1),(media_FD1630C/30).round(1)]}

df = pd.DataFrame(raw_data, columns = ['model_name', 'B1_dados', 'B2_dados', 'B4_dados', 'B8_dados', 'B16_dados','B30_dados'])

# Setting the positions and width for the bars
#pos = list(range(len(df['B1_dados'])))
pos = [0,2,4,6,8]
width = 0.30

# Plotting the bars
fig, ax = plt.subplots(figsize=(14,8))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos,
        # using df['pre_score'] data,
        df['B1_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#7293CB',
        # with label the first value in first_name
        label=df['model_name'][0])


# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos],
        # using df['mid_score'] data,
        df['B2_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#E1974C',
        # with label the second value in first_name
        label=df['model_name'][0])


# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + (width * 2) for p in pos],
        # using df['post_score'] data,
        df['B4_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#84BA5B',
        # with label the third value in first_name
        label=df['model_name'][0])

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + (width * 3) for p in pos],
        # using df['post_score'] data,
        df['B8_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#D35E60',
        # with label the third value in first_name
        label=df['model_name'][0])

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + (width * 4) for p in pos],
        # using df['post_score'] data,
        df['B16_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#808585',
        # with label the third value in first_name
        label=df['model_name'][0])

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + (width * 5) for p in pos],
        # using df['post_score'] data,
        df['B30_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#9067A7',
        # with label the third value in first_name
        label=df['model_name'][0])

# Set the y axis label
ax.set_ylabel('Tempo (ms)')

# Set the chart's title
#ax.set_title('Testes De Desempenho em CPU')

# Set the position of the x ticks
ax.set_xticks([p + 2.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['model_name'],rotation=10)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos) - width, max(pos) + width * 6)
#plt.ylim([0, max(df['FT16_dados'] + df['FT32_dados'])])
plt.ylim([0,20])
print(pos)


for i in range(len(pos)):
    plt.text(x=pos[i] - 0.08 , y=df['B1_dados'][i] + 0.1, s=df['B1_dados'][i], size=10)

for i in range(len(pos)):
    plt.text(x=pos[i]+0.15, y=df['B2_dados'][i]+0.1, s=df['B2_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i]+0.47, y=df['B4_dados'][i]+0.1, s=df['B4_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i]+0.80, y=df['B8_dados'][i]+0.1, s=df['B8_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i] + 1.10, y=df['B16_dados'][i]+0.1, s=df['B16_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i] + 1.35, y=df['B30_dados'][i]+0.1, s=df['B30_dados'][i],size=10)

#plt.subplots_adjust(bottom=0.35)

# Adding the legend and showing the plot
plt.legend(['BATCH SIZE 1', 'BATCH SIZE 2' , 'BATCH SIZE 4', 'BATCH SIZE 8', 'BATCH SIZE 16', 'BATCH SIZE 30'], loc='upper right')
plt.grid(axis='y')
plt.show()
fig.savefig('Different_BatchSize_CPU.png')
