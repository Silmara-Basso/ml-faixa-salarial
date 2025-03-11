# Pipeline de Pré-Processamento, Validação Cruzada e Otimização em Machine Learning

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("Projeto4").getOrCreate()

spark.sparkContext.setLogLevel('ERROR')

df_sil = spark.read.csv("hdfs:///opt/spark/data/dataset.csv", inferSchema=True, header=False)

column_names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                "hours_per_week", "native_country", "income"]

df_sil = df_sil.toDF(*column_names)

df_sil = df_sil.na.fill({"workclass": "Unknown", "occupation": "Unknown", "native_country": "Unknown"})

df_sil = df_sil.dropna()

# Lista com nome das colunas categóricas
categorical_columns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]

# Cria indexers para converter colunas de strings em colunas de índices numéricos
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in categorical_columns]

# Cria encoders para converter índices categóricos em vetores One-Hot Encoding
encoders = [OneHotEncoder(inputCol=column+"_index", outputCol=column+"_vec") for column in categorical_columns]

# Cria um assembler para combinar todas as colunas de recursos em um único vetor de características
assembler = VectorAssembler(inputCols=[c + "_vec" for c in categorical_columns] + ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"], outputCol="features")

# Cria um indexer para a coluna target, transformando-a em rótulos numéricos (ela esta com o simbolo de k no dataset)
labelIndexer = StringIndexer(inputCol="income", outputCol="label")
pipeline = Pipeline(stages=indexers + encoders + [assembler, labelIndexer])
dados_treino, dados_teste = df_sil.randomSplit([0.7, 0.3], seed=42)
pipelineModel = pipeline.fit(dados_treino)

dados_treino_transformado = pipelineModel.transform(dados_treino)
dados_teste_transformado = pipelineModel.transform(dados_teste)

# Cria uma instância do algoritmo de regressão logística
modelo_sil = LogisticRegression(featuresCol="features", labelCol="label")

# Constrói uma grade de parâmetros para otimização, especificando valores para o parâmetro de regularização da regressão logística
modelo_sil_paramGrid = ParamGridBuilder().addGrid(modelo_sil.regParam, [0.1, 0.01]).build()

# Cria um avaliador para modelos de classificação binária, configurado para usar a métrica AUC (Área Sob a Curva ROC)
sil_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

# Configura o validador cruzado com o estimador (regressão logística), a grade de parâmetros, 
# o avaliador e o número de folds
sil_cv = CrossValidator(estimator = modelo_sil, 
                        estimatorParamMaps = modelo_sil_paramGrid, 
                        evaluator = sil_evaluator, 
                        numFolds = 5,
                        parallelism = 3)

# Treina o modelo de regressão logística usando validação cruzada para encontrar os melhores hiperparâmetros
modelo_sil_cv = sil_cv.fit(dados_treino_transformado)

# Aplica o modelo treinado ao conjunto de teste para gerar predições
modelo_sil_cv_predictions = modelo_sil_cv.transform(dados_teste_transformado)

# Avalia o desempenho do modelo no conjunto de teste usando a métrica AUC
auc = sil_evaluator.evaluate(modelo_sil_cv_predictions)

# Criando um DataFrame com a métrica AUC
from pyspark.sql import Row
modelo_sil_cv_auc_df = spark.createDataFrame([Row(auc=auc)])
modelo_sil_cv.write().overwrite().save("hdfs:///opt/spark/data/modelo")
modelo_sil_cv_auc_df.write.csv("hdfs:///opt/spark/data/auc", mode="overwrite")




