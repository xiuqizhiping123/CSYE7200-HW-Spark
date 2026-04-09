import org.apache.spark.sql.{DataFrame, SparkSession}

trait TableDataFrameParser {

  def parseResource(resource: String)(implicit spark: SparkSession): DataFrame = {
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(resource)
  }
}