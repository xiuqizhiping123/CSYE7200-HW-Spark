import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object MovieAnalyzer extends App {
  implicit val spark: SparkSession = SparkSession
    .builder()
    .appName("MovieAnalyzer")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  private val parser: TableDataFrameParser = new TableDataFrameParser {}
  private val mdy: DataFrame = parser.parseResource("src/main/resources/movie_metadata.csv")
  private val resultDF = processMovieData(mdy)
  resultDF.show()

  private def processMovieData(df: DataFrame): DataFrame = {
    val extractedColumn = regexp_extract($"content_rating", """(\d{2})$""", 1)
    val cleanedDF = df.withColumn("rating_age",
      when(extractedColumn === "", lit(null))
        .otherwise(extractedColumn.cast("int"))
    )
    val validAgeDF = cleanedDF.filter($"rating_age".isNotNull)
    validAgeDF.select(
        avg($"rating_age").as("mean_rating_age"),
        stddev($"rating_age").as("stddev_rating_age")
      )
  }
}
