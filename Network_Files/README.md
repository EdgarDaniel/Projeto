

# PASTA COM IMPLEMENTAÇÕES DE CODIGO DA REDE E METRICAS

O script principal é o train_and_evaluate.py, este deve ser executado com ajuda do comando python3:

	python3 train_and_evaluate.py

Antes de executar o comando no seu dispositivo, mude as variáveis presentes no script que dizem respeito as diretorias onde guarda o dataset com as imagens das caras.
As diretorias devem ter a seguinte estrutura, uma directoria para guardar as imagens de treino, outra para guardar as de validação e outra para as de teste. Dentro de cada uma devem estar duas pastas, uma com o nome "male" e outra com "female" e será dentro de cada uma que estarão divididas as imagens por genero. Esta estrutura ajuda na criação das labels e permite que não seja necessário alterar extensivamente o codigo no caso de existencia de mais labels.
No caso de existência de erros com o carregamento de imagens e labels veja o script create_data.py que guarda todas as funções que permitem o procedimento.


Uma otimização realizada no carregamento de imagens é que não são carregadas todas as imagens para uma array, mas sim todos os paths para cada uma das imagens, sendo só depois carregadas imagens para um array conforme o batch size especificado, para depois ser removido de forma a evitar grandes gastos de memória.
Esta otimização é realizada através da função list_all_data presente no script create_data.py que recebe o path da directoria e retorna um array de tuplos com o path de cada uma das imagens e a label associada.

Existe uma função de evaluate no script train_and_evaluate.py que recebe como input a váriavel com todos os paths das imagens de validação ou de teste retornado pela função list_all_data presente no script create_data.py e o tamanho do batch size pretendido. Esta função calcula a accuracy e a loss como forma de controlo de performance da rede.

Depois de definir todas as funções de treino, de accuracy e loss, é criado um ciclo que treina a rede através de um número fixo de epócas especificado na varíavel "EPOCH" e um cilco dentro deste que calcula o número de batches necessários para percorrer todas as imagens.

Na rede, script vgg.py, são excluídas todas as layers de droupout para evitar erros de compilação quando utilizada a ferramenta da intel (Model Optimizer) para converter o modelo para de forma a executar através da API(Inference Engine)






