# MLRN-Iris-Species
######  **Utilizacao de Machine Learning e Rede Neural para classificacao de flores segundo as suas caracteristicas.**
###### O projeta apresenta da forma mais rapida e simples o passo-a-passo para que o algoritmo Machine Learning e Rede Neural em Python possa classificacar as flores segundas caracteristicas de suas petalas.

Este algoritmo e dividido em passos, vamos passar pela coleta de dados brutos, preparacao dos dados, criacao da rede neural para iniciar o aprendizado, e por fim o teste para validar a precisao do modelo criado e perda.

- Coleta de dados brutos:

Os dados foram coletados a partir do site da Kraggle (https://www.kaggle.com/) no endereco completo que se refere aos dados sobre as Iris Species em CSV (https://www.kaggle.com/uciml/iris), a planilha vem bruta com dados das petalas. Os dados das sépala incluem (excluindo a identificacao do registro em 'Id') Comprimento, Largura, e da petala incluem tambem comprimento e largura, dados que para determinados valores para esses parametros pode-se saber sobre qual flor se trata ('Species',coluna como consta no CSV), que no caso e nosso conjunto de saida ou resultado que podem ser 'Iris-Setosa', 'Iris-versicolor' ou 'Iris-Virginica'.

- Preparacao dos Dados:

Para a preparacao dos dados, foi usado principalmente a biblioteca Pandas. O primeiro passo da preparacao, foi definir uma parte dos dados para treinamento do ML e outra para efetuar o teste validador, checando sua precisao e perda (dando mais enfase para a precisao), de todos os dados, houve uma regra 70/30 (70% para treinar, 30% para teste). Depois de definirmor os dados para treino e para teste, o proximo passo e retirar a parte que nao e utilizada para o treinamento, no caso as colunas de identificador ('Id') e de resultado (nossa coluna 'Species'), o resultado sera utilizado para servir de resposta esperada do treinamento (dados -> treinamento -> resposta correta).

- Criacao das camadas da rede no Keras:

Antes de comecarmos a criacao das camadas, preisamos pensar em algumas coisas: 
	- Primeira delas e como esperamos nossa saida de dados, nosa ja sabemos que temos nosso conjunto de resultado com o nome das especies; 
	- Que parametros vamos usar no treinamento para que possamos, segundo a saida esperada, extrair o maximo de precisao e o minimo de perda do treinamento.

Com estes em mente vamos definir como sera a saida. Nosso resultado envolve 3 possibilidade de resultado ('Iris-Setosa', 'Iris-versicolor' ou 'Iris-Virginica'), o mais adequado seria fazer com que nosso algoritmo treine o modelo a afim de indicar qual destes e a resposta mais adequada, contudo, se o fizermos de forma booleana, a tendencia a erros pode subir, dependendo dos parametros utilizados que para este caso booleano que estamos propondo seria um algoritmo de sigmoid aplicado a cada um dos possiveis resultados, ou seja, com este, deveriamos fazer um teste para cada resultado tentando identificar qual deles atinge o max. da nossa funcao de ativacao que e a sigmoid; entretanto, podemos fazer de uma forma ainda mais adequada na qual podemos verificar as chances que o "modelo ML acredita que seja" a resposta correta para cada um dos resultados. Logo partimos de uma ativacao booleana para ambos que pode gerar maiores erros (dado o caso de que a funcao sigmoid nao seja ativada na saida ainda mais para valores > -5 e < 5 - vide doc. Keras API), para uma questao de probabilidade que para todos os resultados.
Agora que definimos a saida, podemos a partir dela definir os melhores parametros para nosso modelo de Rede Neural que vai treinar o modelo. Como ja definimos, o modelo sera treinado para calcular as probabilidades de ser a resposta que o modelo "acredite que seja", visualmente acompanhe:

Indice:		[0,		1,			2] <br>
Resposta:	['Iris-Setosa',	'Iris-versicolor',	'Iris-Virginica'] <br>
Probabilidade:	['%',		'%',			'%']<br>

Lembrando que em python, os indices do nosso conjunto resposta comeca em 0.

Em codigo, ja sabemos que nosso modelo tera uma saida de 3 neuronios (nos), para que suporte a saida como esperado, e que nossa(s) camada(s) intermediarias anterior a ela tera uma funcao de ativacao softmax, que e mais recomendada para a conversao de vetores para um vetor de probabilidade categorica (como mencionada na documentacao da API Keras). Visualmente temos nosso modelo Sequecial quase inteiro, basta apenas adicionar a camada de entrada dos dados ('Entrada'):

modelo tf.keras.Sequencial([
	tf.keras.layer.Dense(10, activation='relu', name='Entrada'),
	tf.keras.layer.Dense(10, activation='softmax', name='Intermediaria'),
	tf.keras.layer.Dense(3, activation='softmax', name='Saida')
])

