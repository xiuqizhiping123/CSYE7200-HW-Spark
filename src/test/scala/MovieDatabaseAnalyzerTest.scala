import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MovieDatabaseAnalyzerTest extends AnyFlatSpec with Matchers {

  implicit val spark: SparkSession = SparkSession
    .builder()
    .appName("MovieDatabaseAnalyzer")
    .master("local[*]")
    .getOrCreate()

  behavior of "parseResource"
  it should "get movie_metadata.csv" in {
    val parser: TableDataFrameParser = new TableDataFrameParser {}
    val mdy: DataFrame = parser.parseResource("src/main/resources/movie_metadata.csv")
    println(mdy.count())
    mdy.count() shouldBe 1609
  }

}