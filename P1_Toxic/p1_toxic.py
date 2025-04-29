import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def main(input_path, output_model_path):
    spark = SparkSession.builder.appName("ToxicCommentClassification").getOrCreate()
    # Load data
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    # Assume `comment_text` and `toxic` columns
    df = df.withColumn("label", col("toxic").cast("double")).select("comment_text", "label").na.drop()
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # Build pipeline
    tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=20)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])

    # Train & evaluate
    model = pipeline.fit(train)
    preds = model.transform(test)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator.evaluate(preds)
    print(f"Test AUC = {auc:.4f}")

    # Save model
    model.write().overwrite().save(output_model_path)
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="hdfs path to toxic.csv")
    parser.add_argument("--output", required=True, help="hdfs path to save the trained model")
    args = parser.parse_args()
    main(args.input, args.output)
