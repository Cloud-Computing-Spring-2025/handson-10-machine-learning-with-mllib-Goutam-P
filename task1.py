from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Step 2: Load CSV
df = spark.read.csv("customer_churn.csv", header=True, inferSchema=True)

# Step 3: Clean column names
df = df.toDF(*[c.strip() for c in df.columns])

# Step 4: Preprocessing function
def preprocess_data(df):
    # Fill missing TotalCharges
    df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges")))

    # Categorical columns that actually exist
    categorical_cols = ["gender", "SeniorCitizen", "PhoneService", "InternetService"]

    # Index + OneHotEncode
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_Index", handleInvalid="keep") for c in categorical_cols]
    encoders = [OneHotEncoder(inputCol=c + "_Index", outputCol=c + "_Vec") for c in categorical_cols]

    # Numeric columns
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Label column
    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")

    # Assemble features
    assembler_inputs = [c + "_Vec" for c in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    # Pipeline
    pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler])
    model = pipeline.fit(df)
    final_df = model.transform(df).select("features", "label")

    return final_df

# Step 5: Run preprocessing
preprocessed_df = preprocess_data(df)

# Step 6: Show output
preprocessed_df.show(truncate=False)

# Stop Spark
spark.stop()