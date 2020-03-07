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



#Gráfico de Todos os Equipamentos com VGG16


dados_cpu_t,dados_media_cpu_t,media_cpu_t = read_array_data('GENDER_RECONIGTION_B1_CPU.txt')
dados_myriad,dados_media_myriad,media_myriad = read_array_data('Gender_RecognitionMYRIAD.txt')
dados_cpu_ie,dados_media_cpu_ie,media_cpu_ie = read_array_data('Gender_RecognitionCPU.txt')
dados_gpu,dados_media_gpu,media_gpu = read_array_data("GENDER_RECONIGTION_B1_GPU.txt")
file_pointer = open("Gender_RecognitionMYRIAD_RAS_1.txt")
array_data = np.array(file_pointer.read().splitlines()).astype(np.float)
array_media = array_data[:-1]
print(array_media)
media = np.mean(array_media)
file_pointer_2 = open("Gender_RecognitionMYRIAD_2.txt")
array_data_2 = np.array(file_pointer_2.read().splitlines()).astype(np.float)
array_media_2 = array_data_2[:-1]
print(array_media_2)
media_2 = np.mean(array_media_2)
file_pointer_3 = open("Gender_RecognitionMYRIAD_2_Desktop.txt")
array_data_3 = np.array(file_pointer_3.read().splitlines()).astype(np.float)
array_media_3 = array_data_3[:-1]
media_3 = np.mean(array_media_3)
file_pointer_4 = open("EdgeTPUD_Results.txt")
array_data_4 = np.array(file_pointer_4.read().splitlines()).astype(np.float)
array_media_4 = array_data_4[:-1]
media_4 = np.mean(array_media_4)

raw_data = {'model_name': ['Classificação de Gênero (VGG16-D) (224*224)'],
        'myriad_dados' : [media_myriad.round(1)],
        'cpu_t_dados': [media_cpu_t.round(1)],
        'cpu_ie_dados': [media_cpu_ie.round(1)],
        'gpu_dados': [media_gpu.round(1)],
        'rpi_dados': [media.round(1)],
        'rpi4_2': [media_2.round(1)],
        'myriad2_dados': [media_3.round(1)],
        'edge_tpu_dados': [media_4.round(1)]}

df = pd.DataFrame(raw_data, columns = ['model_name','gpu_dados','cpu_ie_dados','myriad2_dados','rpi4_2', 'cpu_t_dados',
                                       'edge_tpu_dados','myriad_dados','rpi_dados'])

# Setting the positions and width for the bars
pos = list(range(len(df['myriad_dados'])))
width = 0.25

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,7))


# Create a bar with pre_score data,
# in position pos,
plt.bar(pos,
        # using df['post_score'] data,
        df['gpu_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#7293CB',
        # with label the third value in first_name
        label=df['model_name'][0])


# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos],
        # using df['post_score'] data,
        df['cpu_ie_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#E1974C',
        # with label the third value in first_name
        label=df['model_name'][0])

plt.bar([p + width * 2 for p in pos],
        df['myriad2_dados'],
        width,
        alpha=0.5,
        color = '#84BA5B',
        label = df['model_name'][0])

plt.bar([p + width * 3 for p in pos],
        df['rpi4_2'],
        width,
        alpha=0.5,
        color= '#D35E60',
        label=df['model_name'][0])


# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width * 4 for p in pos],
        # using df['mid_score'] data,
        df['cpu_t_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#808585',
        # with label the second value in first_name
        label=df['model_name'][0])

plt.bar([p + width * 5 for p in pos],
        # using df['mid_score'] data,
        df['edge_tpu_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#9067A7',
        # with label the second value in first_name
        label=df['model_name'][0])

plt.bar([p + width * 6 for p in pos],
        # using df['pre_score'] data,
        df['myriad_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#AB6857',
        # with label the first value in first_name
        label=df['model_name'][0])

plt.bar([p + width * 7 for p in pos],
        # using df['pre_score'] data,
        df['rpi_dados'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='#CCC210',
        # with label the first value in first_name
        label=df['model_name'][0])

# Set the y axis label
ax.set_ylabel('Tempo (ms)')

# Set the chart's title
#ax.set_title('Testes De Desempenho')

# Set the position of the x ticks
ax.set_xticks([p + 3.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['model_name'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos) - width, max(pos) + width * 8)
plt.ylim([0, max(df['myriad_dados'] + df['cpu_t_dados'] + df['cpu_ie_dados'] + df['gpu_dados'] + df['rpi4_2'] + df['myriad2_dados'])])

print(pos)
for i in range(len(pos)):
    plt.text(x=pos[i] - 0.05, y=df['gpu_dados'][i] + 0.1, s=df['gpu_dados'][i], size=10)

for i in range(len(pos)):
    plt.text(x=pos[i] + 0.2, y=df['cpu_ie_dados'][i] + 0.1, s=df['cpu_ie_dados'][i], size=10)

for i in range(len(pos)):
    plt.text(x=pos[i] + 0.45, y=df['myriad2_dados'][i] +0.1, s=df['myriad2_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i] + 0.7, y=df['rpi4_2'][i] + 0.1, s=df['rpi4_2'][i], size=10)

for i in range(len(pos)):
    plt.text(x=pos[i] + 0.95, y=df['cpu_t_dados'][i]+0.1, s=df['cpu_t_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i] + 1.2, y=df['edge_tpu_dados'][i]+0.1, s=df['edge_tpu_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i] + 1.45, y=df['myriad_dados'][i]+0.1, s=df['myriad_dados'][i],size=10)

for i in range(len(pos)):
    plt.text(x=pos[i] + 1.7, y=df['rpi_dados'][i]+0.1, s=df['rpi_dados'][i],size=10)

plt.subplots_adjust(bottom=0.2)

# Adding the legend and showing the plot
plt.legend(['GEFORCE RTX 2070', 'CPU (OPENVINO)', 'RASPBERRY PI 4 (MYRIAD 2)', 'DESTKOP (MYRIAD 2)', 'CPU (TENSORFLOW)',
            'DESKTOP (GOOGLE EDGE TPU)','DESKTOP (MYRIAD 1)','RASPBERRY PI 4 (MYRIAD 1)'], loc='upper right')
plt.grid(axis='y')
plt.show()
fig.savefig('Gender_B1_ALL.png')
