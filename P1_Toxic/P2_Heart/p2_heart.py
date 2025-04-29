import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def main(input_path):
    spark = SparkSession.builder.appName("HeartDiseaseLR").getOrCreate()
    # Load and preprocess
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    df = df.drop("education").na.drop()
    df = df.withColumnRenamed("TenYearCHD", "label")

    # Assemble features
    feature_cols = [c for c in df.columns if c != "label"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=20)
    pipeline = Pipeline(stages=[assembler, lr])

    # Train & evaluate
    model = pipeline.fit(df)
    preds = model.transform(df)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator.evaluate(preds)
    print(f"Heart model AUC = {auc:.4f}")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="hdfs path to Framingham CSV")
    args = parser.parse_args()
    main(args.input)
