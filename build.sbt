ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.18"

lazy val root = (project in file("."))
  .settings(
    name := "untitled",
    libraryDependencies += "org.apache.spark" %% "spark-sql" % "4.1.1"
  )
