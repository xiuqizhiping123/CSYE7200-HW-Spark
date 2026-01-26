import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object SimpleApp {
  def main(args: Array[String]): Unit = {
    val resource = getClass.getResource("/titanic/train.csv")
    if (resource == null) throw new RuntimeException("Oops")
    val spark = SparkSession.builder
      .appName("Simple Application")
      .master("local[*]")
      .getOrCreate()
    val df = spark.read
      .option("header", "true")
      .option("`inferSchema`", "true")
      .csv("src/main/resources/titanic/train.csv")
    df.groupBy("pclass")
      .agg(avg("fare").as("avg_fare"))
      .orderBy("pclass")
      .show()
    df.groupBy("pclass")
      .agg((sum("survived") * 100.0 / count("*")).as("survival_rate"))
      .orderBy(desc("survival_rate"))
      .show()
    val possibleRoseCount = df
      .where(col("age") === 17.0)
      .where(col("sex") === "female")
      .where(col("pclass") === 1)
      .where(col("parch") === 1)
      .count()
    println(possibleRoseCount)
    val possibleJackCount = df
      .where(col("age") === 19.0 || col("age") === 20.0)
      .where(col("sex") === "male")
      .where(col("pclass") === 3)
      .where(col("parch") === 0)
      .count()
    println(possibleJackCount)
    val resultDf = df.where(col("age").isNotNull)
      .withColumn(
        "age_start",
        when(col("age") < 1.0, 0)
          .otherwise((floor((col("age").cast("double") - 1) / 10) * 10 + 1).cast("int"))
      ).groupBy("age_start")
      .agg(
        count("*").as("count"),
        avg("fare").as("avg_fare"),
        (sum("survived") * 100.0 / count("*")).as("survival_rate")
      )
      .orderBy("age_start")
    val finalResult = resultDf.withColumn(
        "age_group",
        when(col("age_start") === 0, "0-1")
          .otherwise(concat(col("age_start"), lit("-"), col("age_start") + 9))
      )
      .select("age_group", "count", "avg_fare", "survival_rate")
    finalResult.show()
  }
}