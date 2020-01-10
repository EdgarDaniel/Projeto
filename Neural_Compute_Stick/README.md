
# FERRAMENTAS E IMPLEMENTAÇÕES DE CODIGO PARA USAR O MODELO NO NCS

Nesta pasta estão os ficheiro que foram utilizados para converter o modelo para um ficheiro .pb e também os scripts disponbilizados quando a instalação da ferramenta OpenVINO para a conversão do modelo para 3 ficheiros, sendo eles um binário, um xml e mapping que contem a topologia da rede. Por fim está um script com o nome "Programa Final.py" que carrega o ficheiro binário e o xml para que com ajuda da API(Inference Engine), disponibilizada pela Intel, seja possivel carregar a rede para o cpu ou neural compute stick de forma a realizar a inferência.

O primeiro script que tem que ser executado é o "freeze_graph.py" e deve incluir os seguintes parâmetros:

	python3 freeze_graph.py --model_dir '/home/edgardaniel/Desktop/NewModel' --output_node_names model_definition/vgg_16/fc8/BiasAdd

O parâmetro "--model_dir" diz respeito à diretoria onde estão guardados os ficheiros que foram guardados quando terminado o processo de treino.
O parâmetro "--output_node_names" diz respeito à ultima camada da rede que tem a função de devolver as probabilidades associadas às labels, em caso de não saber este parâmetro introduza um nome aleatório que de seguida sera imprimida toda a topologia da rede na linha de comandos de forma a encontrar o nome que pretende utilizar.

O segundo script é o "mo.py" que pode apresentar alguns erros quando executado o seguinte comando:

	python3 mo.py --input_model '/home/edgardaniel/Desktop/NewModel/frozen_model.pb' --input_shape [1,224,224,3]

No caso de apresentar erros sugiro que corra o script "install_prerequisites_tf.sh" de forma a instalar todas as dependências necessárias para as bibliotecas utilizadas. Uma vez que o script utiliza uma biblioteca com nome networkx que ao instalar, recorre à ultima versão existente no repostiorio do pip, pode ser necessário instalar a versão anterior, sendo por isso necessário executar um outro comando:
"pip3 install networkx==2.3.0".
É necessário que ambas as pastas, "mo" e "extensions" estejam na mesma pasta quando executa o script, porque este depende de bibliotecas presentes em ambos.

Em relação aos parâmetros utilizados para a execução deste comando, está o "--input_model" que diz respeito ao caminho onde existe o ficheiro resultante da execução do script "freeze_graph.py", (Em norma está dentro da pasta onde guardou o chekpoint do modelo e os ficheiros que lhe estão relacionados) e o "--input_shape" que especifica as dimensões das imagens, o número de batch size e o tipo de cores utilizado, neste caso RGB o que implica a presença do número 3.

Depois de todos os passos anteriores executados com suceso terá no seu dispositivo 3 ficheiros resultantes do ultimo script executado que diz respeito ao modelo convertido e pronto para executar em cpu ou na neural compute stick.

!! POR COMPLETAR COM OS DADOS DO ULTIMO SCRIPT
