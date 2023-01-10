
name := "grounding"
description := "The grounding project implements the org.ml4ai.grounding package including the GroundingApp class."

resolvers ++= Seq(
  ("Artifactory" at "http://artifactory.cs.arizona.edu:8081/artifactory/sbt-release").withAllowInsecureProtocol(true)
)

libraryDependencies ++= {
  val procVer = "8.5.2"
  val uJsonVer = "2.0.0"

  Seq(
    // local logging
    "com.typesafe.scala-logging" %% "scala-logging"              % "3.9.4",
    "ch.qos.logback"              % "logback-classic"            % "1.2.8",
    // scala
    "org.clulab"                 %% "processors-main"            % procVer exclude("xml-apis", "xml-apis"),
    "xml-apis"                    % "xml-apis"                   % "1.4.01",
    "org.clulab"                 %% "model-streamed-trigram-ser" % "1.0.0",
    "com.lihaoyi"                %% "ujson"                      % uJsonVer,
    "org.scalatest"              %% "scalatest"                  % "3.0.9" % Test ,
    "org.scalanlp"               %% "breeze"                     % "1.1",
    "org.scalanlp"               %% "breeze-natives"             % "1.1",
    "org.scalanlp"               %% "breeze-viz"                 % "1.1",
  )
}
