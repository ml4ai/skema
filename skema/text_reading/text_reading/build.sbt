name := "skema_text_reading"
organization := "org.clulab"

scalaVersion := "2.12.17"

resolvers ++= Seq(
  "clulab" at "https://artifactory.clulab.org/artifactory/sbt-release"
)

libraryDependencies ++= {
  val procVer = "8.5.3"
  val uJsonVer = "2.0.0"

  Seq(
    "org.clulab"                 %% "pdf2txt"            % "1.1.3",
    "org.clulab"                 %% "processors-main"    % procVer,
    "org.clulab"                 %% "processors-corenlp" % procVer,
    "xml-apis"                    % "xml-apis"           % "1.4.01",
    "ai.lum"                     %% "common"             % "0.0.10",
    "com.lihaoyi"                %% "ujson"              % uJsonVer,
    "com.lihaoyi"                %% "upickle"            % uJsonVer,
    "com.lihaoyi"                %% "ujson-json4s"       % uJsonVer,
    "com.lihaoyi"                %% "ujson-play"         % uJsonVer,
    "com.lihaoyi"                %% "requests"           % "0.7.1",
    "com.typesafe.play"          %% "play-json"          % "2.9.3",
    "org.scala-lang.modules"     %% "scala-xml"          % "1.0.6", // 2.1.0",
    "org.clulab"                  % "glove-840b-300d"    % "0.1.0" % Test,
    "org.scalatest"              %% "scalatest"          % "3.0.9" % Test,
    "org.clulab"                 %  "glove-840b-300d"    % "0.1.0" % Test,
    "org.scalatest"              %% "scalatest"          % "3.0.9" % Test,
    "org.scalanlp" %% "breeze" % "1.1",
    "org.scalanlp" %% "breeze-natives" % "1.1",
    "org.scalanlp" %% "breeze-viz" % "1.1",
  )
}

lazy val root = (project in file("."))
  .aggregate(grounding)
  .dependsOn(grounding % "compile -> compile; test -> test")

lazy val grounding = project.in(file("grounding"))

 lazy val webapp = project
   .enablePlugins(PlayScala)
   .aggregate(root)
   .dependsOn(root)

//EclipseKeys.withSource := true

ThisBuild / Test / fork := true // also forces sequential operation
ThisBuild / Test / parallelExecution := false // keeps groups in their order
