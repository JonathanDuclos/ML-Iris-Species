# MLRN-Iris-Species
######  **Utilizacao de Machine Learning e Rede Neural para classificacao de flores segundo as suas caracteristicas.**
###### O projeta apresenta da forma mais rapida e simples o passo-a-passo para que o algoritmo Machine Learning e Rede Neural em Python possa classificacar as flores segundas caracteristicas de suas petalas.

Este algoritmo e dividido em passos, vamos passar pela coleta de dados brutos, preparacao dos dados, criacao da rede neural para iniciar o aprendizado, e por fim o teste para validar a precisao do modelo criado e perda.

- Coleta de dados brutos:

Os dados foram coletados a partir do site da Kraggle (https://www.kaggle.com/) no endereco completo que se refere aos dados sobre as Iris Species em CSV (https://www.kaggle.com/uciml/iris), a planilha vem bruta com dados das petalas. Os dados das sépala incluem (excluindo a identificacao do registro em 'Id') Comprimento, Largura, e da petala incluem tambem comprimento e largura, dados que para determinados valores para esses parametros pode-se saber sobre qual flor se trata ('Species',coluna como consta no CSV), que no caso e nosso conjunto de saida ou resultado que podem ser 'Iris-Setosa', 'Iris-versicolor' ou 'Iris-Virginica'.

- Preparacao dos Dados:

Para a preparacao dos dados, foi usado principalmente a biblioteca Pandas
