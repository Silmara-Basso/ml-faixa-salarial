# ml-faixa-salarial
Pipeline de Pré-Processamento, Validação Cruzada e Otimização em Machine Learning para previsão da faixa salarial

## Para iniciar o cluster

`docker compose -f docker-compose.yml up -d --scale spark-worker-yarn=3`

`docker compose logs`

# Treinar o modelo e salvar a métrica AUC

`docker exec sil-spark-master-yarn spark-submit --master yarn --deploy-mode cluster ./apps/faixa-salarial.py`

# Obter os dados do Cluster 

`docker exec sil-spark-master-yarn hdfs dfs -ls /opt/spark/data`

`docker exec sil-spark-master-yarn hdfs dfs -ls /opt/spark/data/auc`

`docker exec sil-spark-master-yarn hdfs dfs -cat nome-do-arquivo`