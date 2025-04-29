import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def main(train_path, test_path, algo):
    spark = SparkSession.builder.appName("CensusTrees").getOrCreate()
    # Load data
    df_train = spark.read.csv(train_path, header=True, inferSchema=True)
    df_test = spark.read.csv(test_path, header=True, inferSchema=True)
    for c in df_train.columns:
        df_train = df_train.filter(df_train[c] != "?")
        df_test = df_test.filter(df_test[c] != "?")
    df_train = df_train.withColumnRenamed("income", "label")
    df_test = df_test.withColumnRenamed("income", "label")

    # Feature engineering (same as P3)
    cat_cols = [f.name for f in df_train.schema.fields if isinstance(f.dataType, StringType) and f.name != "label"]
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="skip") for c in cat_cols]
    encoder = OneHotEncoder(inputCols=[c+"_idx" for c in cat_cols], outputCols=[c+"_vec" for c in cat_cols])
    num_cols = [f.name for f in df_train.schema.fields if isinstance(f.dataType, (IntegerType, DoubleType))]
    assembler = VectorAssembler(inputCols=[c+"_vec" for c in cat_cols] + num_cols, outputCol="features")

    # Choose classifier
    if algo == "rf":
        clf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20)
    else:
        clf = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    pipeline = Pipeline(stages=indexers + [encoder, assembler, clf])

    # Train & evaluate
    model = pipeline.fit(df_train)
    preds = model.transform(df_test)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    acc = evaluator.evaluate(preds)
    print(f"{algo.upper()} test accuracy = {acc:.4f}")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--algo", required=True, choices=["rf","dt"], help="rf or dt")
    args = parser.parse_args()
    main(args.train, args.test, args.algo)
