# Projeto
Neural Compute Stick (BenchMarks)

Este projeto está relacionado com a nova tecnologia lançada pela Intel, que tem como objetivo tornar possível a inferência de modelos de deep learning na edge, em dispositivos de pequeno porte e com baixo consumo energético.

A solução proposta pela Movidius, uma das empresa da Intel foi a criação de uma pen denominada de NCS(Neural Compute Stick) que promete cumprir com todos os objetivos enunciados anteriormente.

Neste repositório está presente os scripts que permitiram criar os benchmarks que serão apresentados no relatório, a topologia da rede vgg16, um script que tem como objetivo realizar o treino da rede para o reconhecimento de género através de batch, funções auxiliares para ler os dados através de paths que terão que ser especificados no ficheiro train_and_evaluate.py e por um fim um último script para realizar o grafico ROC e calculo do valor AUC.

Juntamente com todos os comentários existentes nos scripts e com algumas das informações dadas em seguida, irá tornar-se fácil entender toda a implementação do código.

O script principal é o train_and_evaluate.py, este deve ser executado com ajuda do comando python3:

	python3 train_and_evaluate.py

Antes de executar o comando no seu dispositivo, mude as variáveis presentes no script que dizem respeito as diretorias onde guarda o dataset com as imagens das caras das pessoas.
As diretorias devem ter a seginte estrutura, uma directoria para guardar as imagens de treino, outra para guardar as de validação e outra para as de teste. Dentro de cada uma devem estar duas pastas, uma com o nome "male" e outra com "female" e será dentro de cada uma que estarão divididas as imagens por genero. Esta estrutura ajuda na criação das labels e permite que não seja necessário alterar extensivamente o codigo no caso de existencia de mais labels.
No caso de existência de erros com o carregamento de imagens e labels veja o script create_data.py que guarda todas as funções que permitem o procedimento.


Uma otimização realizada no carregamento de imagens é que não são carregadas todas as imagens para uma array, mas sim todos os paths para cada uma das imagens, sendo só depois carregadas imagens para um array conforme o batch size expecificado, para depois ser removido de forma a evitar grandes gastos de memória.
Esta otimização é realizada através da função list_all_data presente no script create_data.py que recebe o path da directoria e retorna um array de tuplos com o path de cada uma das imagens e a label associada.

Na rede são excluídas todas as layers de droupout para evitar erros de compilação quando utilizada a ferramenta da intel (Model Optimizer) para converter o modelo para de forma a executar através da API(Inference Engine)

Existe uma função de evaluate que recebe como input a váriavel com todos os paths das imagens de validação ou de teste retornado pela função list_all_data presente no script create_data.py e o tamanho do batch size pretendido.

Depois de definir todas as funções de treino, de accuracy e loss, é criado um ciclo que treina a rede através de um número fixo de epócas especificado na varíavel "EPOCH" e um cilco dentro deste que calcula o número de batches necessários para percorrer todas as imagens.

O resto de toda a informação está comentada no ficheiro e em qualquer dúvida contacte o criador através do email: Edgar_Daniel_60@hotmail.com




