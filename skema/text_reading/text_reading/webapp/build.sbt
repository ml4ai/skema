name := "webapp"

// Coordinate this with the grounding subproject.
dependencyOverrides += "xml-apis" % "xml-apis" % "1.4.01"

libraryDependencies ++= Seq(
  "org.scalatestplus.play" %% "scalatestplus-play" % "3.1.2" % Test,
  guice
)