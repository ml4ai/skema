name := "webapp"

// Coordinate this with the grounding subproject.
dependencyOverrides += "xml-apis" % "xml-apis" % "1.4.01"
dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-databind" % "2.12.2"
dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-core" % "2.12.2"

libraryDependencies ++= Seq(
  "org.scalatestplus.play" %% "scalatestplus-play" % "3.1.2" % Test,
  "org.clulab"                 %% "model-streamed-trigram-ser"   % "1.0.0",
  "org.clulab"                 %% "pdf2txt"                             % "1.2.0-RC1",
  "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.12.2",
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.12.0",
  guice
)

// Enable longer timeouts on the webservice controllers (40 min)
PlayKeys.devSettings += "play.server.http.idleTimeout" -> "2400 seconds"