Nesta seção estão disponivies todos os recursos que foram necessários à criação dos resultados de desempenho do equipamento da Intel (NCS).

Primeiro podemos falar sobre o conteúdo da diretoria “Aplicação BenchMark” e qual o papel da mesma no processo de retirada de valores para testar o desempenho. A aplicação “BenchMark_App” foi criada pela Intel e está disponivel no conjunto de ferramentas oferecidas pelo “OpenVino”. Esta foi criada com o intuito de receber como parametro as redes em representação intermédia, ou seja, já depois de terem sido convertidas pelo modelo optimizar da Intel, imagem/imagens para realizar a inferência e mais alguns parametros de controlo que serão referidos posteriormente. O ficheiro “main.cpp” que está dentro da diretoria deve subtituido pelo presente na diretoria: “C:\ProgramFiles(x86)\IntelSWTools\openvino_2019.3.334\deployment_tools\inference_engine\samples\benchmark_app”.

Depois deve compilar todos os exemplos através do srcipt “build_samples.bat”. O resultado desta execução é a criação de uma pasta com todos os exemplos compilados e já prontos a executar na diretoria: "C:\Users\EdgarDaniel\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release”.

Uma vez que para a correta execução do aplicativo é necessário ter presente bibliotecas dependentes, está presente nesta seção também uma diretoria chamada “Bibiliotecas Importantes”. O aplicativo em vez de ser executado dentro da pasta onde foi criado, deve ser copiado para dentro desta pasta e nesse caso já terá todas as dependências necessárias.

Como para executar o aplicativo é necessário uma rede, imagens e conhecimentos acerca de alguns parametros para inserir no comando do terminal, a pasta “Modelos Selecioandos” tem todos os modelos compativies com a unidade de processamento e com a (NCS). Estes modelos já estão convertidos e já têm as imagens utilizadas na realização dos testes de forma a proporcionar os resultados o mais semelhantes possiveis.

Uma vez que para uma correta execução o utilizador tem que conhecimento dos parametros que controlam a execução, tais como número de batch size, execução assincrona ou sincrona, entre outros, está também presente um ficheiro txt que especifica o comando mais complexo usado, ou seja, o comando que permitiu executar um modelo na MYRIAD com um batch size de 30 imagens em modo sincrono. 

Existe a opção de realizar o download de mais modelos através do srcipt oferecido no aplicativo “OpenVino”, dentro da pasta das ferramentas da diretoria “Deployment Tools”, no caso de querer utilizar outros modelos para realizar testes, deve consultar também a documentação da Intel para saber quais os dispositivos compativeis com tais modelos.

No caso da existência de dúvidas deve ler o relatório na seção 4, que aborda todos os testes e implementações.
