
name := "skema_text_reading"
organization := "org.clulab"

scalaVersion := "2.12.17"

resolvers ++= Seq(
  "jitpack" at "https://jitpack.io", // This provides access to regextools straight from github.
  ("Artifactory" at "http://artifactory.cs.arizona.edu:8081/artifactory/sbt-release").withAllowInsecureProtocol(true)
)

libraryDependencies ++= {
  val procVer = "8.5.2"
  val uJsonVer = "2.0.0"

  Seq(
    "org.clulab"                 %% "processors-main"    % procVer,
    "org.clulab"                 %% "processors-corenlp" % procVer,
    "ai.lum"                     %% "common"             % "0.0.10",
    "com.github.lum-ai"           % "regextools"         % "ee64b773a6", // github version, master commit
    "com.lihaoyi"                %% "ujson"              % uJsonVer,
    "com.lihaoyi"                %% "upickle"            % uJsonVer,
    "com.lihaoyi"                %% "ujson-json4s"       % uJsonVer,
    "com.lihaoyi"                %% "ujson-play"         % uJsonVer,
    "com.lihaoyi"                %% "requests"           % "0.7.1",
    "com.typesafe.play"          %% "play-json"          % "2.9.3",
    "org.scala-lang.modules"     %% "scala-xml"          % "1.0.6", // 2.1.0",
    "org.scalatest"              %% "scalatest"          % "3.0.9" % "test"
  )
}

lazy val root = (project in file("."))
  .aggregate(grounding)
  .dependsOn(grounding)

lazy val grounding = project.in(file("grounding"))

 lazy val webapp = project
   .enablePlugins(PlayScala)
   .aggregate(root)
   .dependsOn(root)

//EclipseKeys.withSource := true
