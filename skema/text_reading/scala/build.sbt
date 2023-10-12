name := "skema_text_reading"
organization := "org.clulab"

scalaVersion := "2.12.17"

resolvers ++= Seq(
  "clulab" at "https://artifactory.clulab.org/artifactory/sbt-release"
)

libraryDependencies ++= {
  val breezeVer = "1.2"
  val procVer = "9.0.0-RC1"
  val uJsonVer = "2.0.0"

  Seq(
    "org.scalanlp"               %% "breeze"                              % breezeVer,
    "org.scalanlp"               %% "breeze-natives"                      % breezeVer,
    "org.scalanlp"               %% "breeze-viz"                          % breezeVer,
    "ai.lum"                     %% "common"                              % "0.0.10",
    "org.clulab"                  % "climatechange-model-unigram-ser"     % "1.0.0",
    "org.clulab"                 %% "epidemiology-embeddings-model-ser"   % "1.0.0",
//    "org.clulab"                  % "spaceweather-model-unigram-ser"      % "1.0.0",
    "org.clulab"                  % "glove-840b-300d"                     % "0.1.0" % Test,
    "org.clulab"                 %% "pdf2txt"                             % "1.2.0-RC1",
    "com.typesafe.play"          %% "play-json"                           % "2.9.3",
    "org.clulab"                 %% "processors-main"                     % procVer,
    "org.clulab"                 %% "processors-corenlp"                  % procVer,
    "com.lihaoyi"                %% "requests"                            % "0.7.1",
    "org.scala-lang.modules"     %% "scala-xml"                           % "1.0.6",
    "org.scalatest"              %% "scalatest"                           % "3.0.9" % Test,
    "com.lihaoyi"                %% "ujson"                               % uJsonVer,
    "com.lihaoyi"                %% "upickle"                             % uJsonVer,
    "com.lihaoyi"                %% "ujson-json4s"                        % uJsonVer,
    "com.lihaoyi"                %% "ujson-play"                          % uJsonVer,
    "xml-apis"                    % "xml-apis"                            % "1.4.01",
    "com.github.scopt"           %% "scopt"                               % "4.1.0",
    "com.lihaoyi"                %% "requests"                            % "0.1.8",
    "com.typesafe.scala-logging" %% "scala-logging" % "3.9.3",
    "ch.qos.logback" % "logback-classic" % "1.2.3",
    "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.12.2",
    "com.fasterxml.jackson.core" % "jackson-databind" % "2.12.0"
  )
}

lazy val root = project in file(".")

lazy val webapp = project
    .enablePlugins(PlayScala)
    .aggregate(root)
    .dependsOn(root)

//EclipseKeys.withSource := true

ThisBuild / Test / fork := true // also forces sequential operation
ThisBuild / Test / parallelExecution := false // keeps groups in their order

addCommandAlias("dockerizeWebapp", "webapp/docker:publishLocal")

