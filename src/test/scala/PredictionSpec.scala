import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PredictionSpec extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .appName("TitanicTest")
      .master("local[*]")
      .getOrCreate()
  }

  override def afterAll(): Unit = {
    if (spark != null) spark.stop()
  }

  it should "hello" in {
    val dataPath = getClass.getResource("/titanic/train.csv").getPath
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(dataPath)
    val Array(trainDf, testDf) = df.randomSplit(Array(0.8, 0.2), 42)
    val accuracy = evaluate(Prediction.train(trainDf, testDf))
    println(s"Accuracy: $accuracy")
    accuracy should be > 0.70
  }

  private def evaluate(predictions: DataFrame): Double = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    evaluator.evaluate(predictions)
  }
}