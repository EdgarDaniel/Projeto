#COPYRIGHT EDGAR DANIEL
#SCRIP THAT ALLOWS THE INTEGRATION OF THE API (INFERENCE ENGINE)
#IN COMMENT ARE THE CODE TO CALCULATE THE BENCHMARKS AND WRITE THE RESULTS TO A FILE

from argparse import ArgumentParser, SUPPRESS
import sys
import os
import cv2
import numpy as np
from time import time
import logging as log
from openvino.inference_engine import IENetwork, IECore

#FUNCTION THAT PARSES THE ARGUMENTS THAT BELONGS TO THE EXAMPLES IN THE OPENVINO TOOLKIT
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True,
                      type=str, nargs="+")
    #args.add_argument("-l", "--cpu_extension",
                      #help="Optional. Required for CPU custom layers. "
                           #"MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           #" kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=2, type=int)
    args.add_argument("--nthreads", help="Optional.Number of threads to execute model", default=4, type=int)
    return parser

#MAIN FUNCTION
def main():

    #SEPARAÇÃO DE ARGUMENTOS
    args = build_argparser().parse_args()

    # CRIAÇÃO DE OBJETO FILE
    nome = "Gender_Recognition" + args.device + ".txt"
    f = open(nome, "a")

    #CARREGAR MODELOS (FICHIEROS XML E BIN)
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    #INSTANCIAR CLASSE CORRESPONDENTE AO INFERENCE ENGINE PARA UM ESPECIFÍCO DISPOSITIVO
    log.info("A criar inference engine")
    ie = IECore()

    #CONFIGURAÇÃO DO OBJETO PARA EXECUTAR TAREFAS EM MULTI-THREADING
    #AVERIGUAR!!
    ie.set_config({'CPU_THREADS_NUM': str(str(args.nthreads))}, 'CPU')

    #CARREGAR MODELO PARA OBJETO INFERENCE ENGINE
    log.info("Carregar ficheiros da rede:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    #PREPARAR DIMENSÕES DO INPUT
    log.info("Preparar dimensões do input")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(args.input)

    #CARREGAR E PRE-PROCESSAR IMAGES DE INPUT

    n, c, h, w = net.inputs[input_blob].shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = cv2.imread(args.input[i])
        if image.shape[:-1] != (h, w):
            log.warning("Imagem {} é ajustada de {} para {}".format(args.input[i], image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image
    log.info("Tamanho do Batch é {}".format(n))

    #PREPARAR MODELO PARA EXECUTAR
    log.info("Preparar modelo para executar")
    exec_net = ie.load_network(network=net, device_name=args.device)

    #COMEÇA INFERÊNCIA EM MODO SÍNCRONO
    log.info("Começa a inferência em modo síncrono")


    #CALCULO DO TEMPO
    #timeInicial = time()
    #timeFinal = time() - timeInicial
    #cont = 0
    #while timeFinal <= 60:
        #timeN = time()
    res = exec_net.infer(inputs={input_blob:images})
        #timeE = time()
        #timeF = ((timeE - timeN) * 1000)
        
        #ESCREVER TEMPO DE INFERÊNCIA PARA FICHEIRO
        #f.write(str(timeF) + "\n")

        #timeFinal = time() - timeInicial
        #cont = cont +1
        #print(timeF)

    #print("Número de iterações em 60s:", cont)
    #f.write(str(cont) + "\n")

    #FECHAR FICHEIRO
    #f.close()
    #PROCESSAR SHAPE DE SAÍDA

    #PREPARA OS RESULTADOS PARA SEREM ANALISADOS
    log.info("Processar shape de saída")
    res = res[out_blob]
    #print("Resultados",res)

    #CRIAÇÃO DOS RESULTADOS ATRAVÉS DAS LABELS E PROBABILIDADES RESULTANTES DA REDE
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
    classid_str = "Classificação"
    probability_str = "Probabilidade"
    contM = 0
    contF = 0
    for i, probs in enumerate(res):
        probs = np.squeeze(probs)

        if probs[0] < 0:
            print("Resultado: Mulher")
            contF = contF+1
        else:
            print("Resultado: Homem")
            contM = contM+1

        #print(probs)
        top_ind = np.argsort(probs)[-args.number_top:][::-1]
        print("Imagem {}\n".format(args.input[i]))
        print(classid_str, probability_str)
        print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
        for id in top_ind:
            det_label = labels_map[id] if labels_map else "{}".format(id)
            label_length = len(det_label)
            space_num_before = (len(classid_str) - label_length) // 2
            space_num_after = len(classid_str) - (space_num_before + label_length) + 2
            space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
            print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
                                          ' ' * space_num_after, ' ' * space_num_before_prob,
                                          probs[id]))
        print("\n")

    #CALCULO DE ACERTO
    contT = contM + contF
    #print(contT)
    #print(n)
    #APRESENTA A TAXA DE ACERTO PARA O BATCH SIZE NA LINHA DE COMANDOS
    print("Taxa de Acerto", (contT/n)*100, "%")

#CHAMA A FUNÇÃO PRINCIPAL
if __name__ == '__main__':
    sys.exit(main() or 0)

