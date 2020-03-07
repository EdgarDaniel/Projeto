# encoding=utf8


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

#Dados de Diferentes Batch Size MYRIAD


#dados_AG_FT16_B1_CPU,media_dados_AG161C,media_AG161C = read_array_data("age-gender-recognition-retail-0013_16_1_62-62_CPU.txt")
dados_AG_FT16_B1_MYRIAD,media_dados_AG161M,media_AG161M = read_array_data("age-gender-recognition-retail-0013_MYRIAD_1.txt")
dados_AG_FT16_B2_MYRIAD,media_dados_AG162M,media_AG162M = read_array_data("age-gender-recognition-retail-0013_MYRIAD_2.txt")
dados_AG_FT16_B4_MYRIAD,media_dados_AG164M,media_AG164M = read_array_data("age-gender-recognition-retail-0013_MYRIAD_4.txt")
dados_AG_FT16_B8_MYRIAD,media_dados_AG168M,media_AG168M = read_array_data("age-gender-recognition-retail-0013_MYRIAD_8.txt")
dados_AG_FT16_B16_MYRIAD,media_dados_AG1616M,media_AG1616M = read_array_data("age-gender-recognition-retail-0013_MYRIAD_16.txt")
dados_AG_FT16_B30_MYRIAD,media_dados_AG1630M,media_AG1630M = read_array_data("age-gender-recognition-retail-0013_MYRIAD_30.txt")

dados_E_FT16_B30_MYRIAD, media_dados_E1630M, media_E1630M = read_array_data("emotions-recognition-retail-0003_MYRIAD_30.txt")
dados_E_FT16_B16_MYRIAD, media_dados_E1616M, media_E1616M = read_array_data("emotions-recognition-retail-0003_MYRIAD_16.txt")
dados_E_FT16_B8_MYRIAD, media_dados_E168M, media_E168M = read_array_data("emotions-recognition-retail-0003_MYRIAD_8.txt")
dados_E_FT16_B4_MYRIAD, media_dados_E164M, media_E164M = read_array_data("emotions-recognition-retail-0003_MYRIAD_4.txt")
dados_E_FT16_B2_MYRIAD, media_dados_E162M, media_E162M =read_array_data("emotions-recognition-retail-0003_MYRIAD_2.txt")
dados_E_FT16_B1_MYRIAD, media_dados_E161M, media_E161M = read_array_data("emotions-recognition-retail-0003_MYRIAD_1.txt")

dados_FR_FT16_B30_MYRIAD, media_dados_FR1630M, media_FR1630M = read_array_data("face-reidentification-retail-0095_MYRIAD_30.txt")
dados_FR_FT16_B16_MYRIAD, media_dados_FR1616M, media_FR1616M = read_array_data("face-reidentification-retail-0095_MYRIAD_16.txt")
dados_FR_FT16_B8_MYRIAD, media_dados_FR168M, media_FR168M = read_array_data("face-reidentification-retail-0095_MYRIAD_8.txt")
dados_FR_FT16_B4_MYRIAD, media_dados_FR164M, media_FR164M = read_array_data("face-reidentification-retail-0095_MYRIAD_4.txt")
dados_FR_FT16_B2_MYRIAD, media_dados_FR162M, media_FR162M = read_array_data("face-reidentification-retail-0095_MYRIAD_2.txt")
dados_FR_FT16_B1_MYRIAD, media_dados_FR161M, media_FR161M = read_array_data("face-reidentification-retail-0095_MYRIAD_1.txt")

dados_FD_FT16_B30_MYRIAD, media_dados_FD1630M, media_FD1630M = read_array_data("face-detection-adas-0001_MYRIAD_30.txt")
dados_FD_FT16_B16_MYRIAD, media_dados_FD1616M, media_FD1616M = read_array_data("face-detection-adas-0001_MYRIAD_16.txt")
dados_FD_FT16_B8_MYRIAD, media_dados_FD168M, media_FD168M = read_array_data("face-detection-adas-0001_MYRIAD_8.txt")
dados_FD_FT16_B4_MYRIAD, media_dados_FD164M, media_FD164M = read_array_data("face-detection-adas-0001_MYRIAD_4.txt")
dados_FD_FT16_B2_MYRIAD, media_dados_FD162M, media_FD162M = read_array_data("face-detection-adas-0001_MYRIAD_2.txt")
dados_FD_FT16_B1_MYRIAD, media_dados_FD161M, media_FD161M = read_array_data("face-detection-adas-0001_MYRIAD_1.txt")

dados_FL_FT16_B30_MYRIAD, media_dados_FL1630M, media_FL1630M = read_array_data("facial-landmarks-35-adas-0002_MYRIAD_30.txt")
dados_FL_FT16_B16_MYRIAD, media_dados_FL1616M, media_FL1616M = read_array_data("facial-landmarks-35-adas-0002_MYRIAD_16.txt")
dados_FL_FT16_B8_MYRIAD, media_dados_FL168M, media_FL168M = read_array_data("facial-landmarks-35-adas-0002_MYRIAD_8.txt")
dados_FL_FT16_B4_MYRIAD, media_dados_FL164M, media_FL164M = read_array_data("facial-landmarks-35-adas-0002_MYRIAD_4.txt")
dados_FL_FT16_B2_MYRIAD, media_dados_FL162M, media_FL162M = read_array_data("facial-landmarks-35-adas-0002_MYRIAD_2.txt")
dados_FL_FT16_B1_MYRIAD, media_dados_FL161M, media_FL161M = read_array_data("facial-landmarks-35-adas-0002_MYRIAD_1.txt")


#Gráfico Diferentes Batch Size MYRIAD


raw_data = {'model_name': ['Idade/Gênero (62*62)','Emoções (64*64)','Características Faciais (60*60)',
                           'Reconhecimento de Cara (128*128)','Deteção de Cara (672*384)'],
        'B1_dados': [int(media_AG161M),int(media_E161M), int(media_FL161M),int(media_FR161M)
            ,int(media_FD161M)],
        'B2_dados': [int(media_AG162M/2),int(media_E162M/2),int(media_FL162M/2),int(media_FR162M/2),
                     int(media_FD162M/2)],
        'B4_dados': [int(media_AG164M/4),int(media_E164M/4),int(media_FL164M/4), int(media_FR164M/4),
                     int(media_FD164M/4)],
        'B8_dados': [int(media_AG168M/8),int(media_E168M/8),int(media_FL168M/8),int(media_FR168M/8),
                     int(media_FD168M/8)],
        'B16_dados': [int(media_AG1616M/16),int(media_E1616M/16),int(media_FL1616M/16),int(media_FR1616M/16),
                      int(media_FD1616M/16)],
        'B30_dados': [int(media_AG1630M/30),int(media_E1630M/30),int(media_FL1630M/30),
                      int(media_FR1630M/30),int(media_FD1630M/30)]}

df = pd.DataFrame(raw_data, columns = ['model_name', 'B1_dados', 'B2_dados', 'B4_dados', 'B8_dados', 'B16_dados','B30_dados'])

print(int(media_AG162M/2))

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
plt.ylim([0,160])
print(pos)


for i in range(len(pos)):
    plt.text(x=pos[i] - 0.10 , y=df['B1_dados'][i] + 0.1, s=df['B1_dados'][i], size=10)

for i in range(len(pos)):
    plt.text(x=pos[i]+0.18, y=df['B2_dados'][i]+0.1, s=df['B2_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i]+0.47, y=df['B4_dados'][i]+0.1, s=df['B4_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i]+0.77, y=df['B8_dados'][i]+0.1, s=df['B8_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i] + 1.05, y=df['B16_dados'][i]+0.1, s=df['B16_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i] + 1.35, y=df['B30_dados'][i]+0.1, s=df['B30_dados'][i],size=10)

#plt.subplots_adjust(bottom=0.35)

# Adding the legend and showing the plot
plt.legend(['BATCH SIZE 1', 'BATCH SIZE 2' , 'BATCH SIZE 4', 'BATCH SIZE 8', 'BATCH SIZE 16', 'BATCH SIZE 30'], loc='upper right')
plt.grid(axis='y')
plt.show()
fig.savefig('Different_BatchSize_MYRIAD.png')
