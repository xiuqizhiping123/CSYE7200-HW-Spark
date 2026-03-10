import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{Imputer, SQLTransformer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession}

object Prediction {
  def main(args: Array[String]): Unit = {
    val resource = getClass.getResource("/titanic/train.csv")
    if (resource == null) throw new RuntimeException("Oops")
    val spark = SparkSession.builder
      .appName("Simple Application")
      .master("local[*]")
      .getOrCreate()
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/titanic/train.csv")
    val test = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/titanic/test.csv")
    eda(df)
    train(df, test).select("features", "probability", "prediction").limit(10).show()
  }

  private def eda(df: DataFrame): Unit = {
    // How many possible Jacks survived?
    df.where((col("age") === 19.0 || col("age") === 20.0) &&
        col("sex") === "male" &&
        col("pclass") === 3 &&
        col("parch") === 0)
      .groupBy("survived")
      .count()
      .show()
  }

  def train(trainDf: DataFrame, testDf: DataFrame): DataFrame = {
    val pipeline = buildPipeline()
    val pipelineModel = pipeline.fit(trainDf)
    pipelineModel.transform(testDf)
  }

  private def buildPipeline(): Pipeline = {
    val sqlTransformer = new SQLTransformer()
      .setStatement("SELECT *, CASE WHEN (SibSp + Parch + 1) = 1 THEN 1 ELSE 0 END AS IsAlone FROM __THIS__")
    val imputer = new Imputer()
      .setInputCol("Age")
      .setOutputCol("Age_Median")
      .setStrategy("median")
    val indexer = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("Sex_Idx")
    val assembler = new VectorAssembler()
      .setInputCols(Array("Age_Median", "Sex_Idx", "Pclass", "IsAlone"))
      .setOutputCol("features")
    val trainer = new LogisticRegression()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
    new Pipeline().setStages(Array(sqlTransformer, imputer, indexer, assembler, trainer))
  }
}