import org.clulab.sbt.BuildUtils
import org.clulab.sbt.Resolvers

name := "grounding"
description := "The grounding project implements the org.ml4ai.grounding package including the GroundingApp class."

// Last checked 2021-12-31
val scala11 = "2.11.12" // up to 2.11.12
val scala12 = "2.12.15" // up to 2.12.14
val scala13 = "2.13.7"  // up to 2.13.7
val scala3  = "3.1.0"   // up to 3.1.0

ThisBuild / crossScalaVersions := Seq(scala12, scala11, scala13, scala3)
ThisBuild / scalaVersion := crossScalaVersions.value.head

resolvers ++= Seq(
  ("Artifactory" at "http://artifactory.cs.arizona.edu:8081/artifactory/sbt-release").withAllowInsecureProtocol(true)
//  Resolvers.localResolver,  // Reserve for Two Six.
//  Resolvers.clulabResolver, // glove
//  Resolvers.jitpackResolver // Ontologies
)

libraryDependencies ++= {
  val parallelLibraries = {
    CrossVersion.partialVersion(scalaVersion.value) match {
      case Some((2, major)) if major <= 12 => Seq()
      case _ => Seq("org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4") // up to 1.0.4
    }
  }

  Seq(
    // local logging
    "ch.qos.logback"              % "logback-classic"         % "1.2.10",       // as of 2021-12-31 up to 1.2.10
    "com.typesafe.scala-logging" %% "scala-logging"           % "3.9.4",        // as of 2021-12-31 up to 3.9.4
    // config
    "com.typesafe"                % "config"                  % "1.4.1",        // as of 2021-12-31 up to 1.4.1
    // scala
    "org.scala-lang.modules"     %% "scala-collection-compat" % "2.6.0",        // as of 2021-12-31 up to 2.6.0
    "org.scalatest"              %% "scalatest"               % "3.2.10" % Test, // as of 2021-12-31 up to 3.2.10
    "org.clulab"                 %% "processors-main"         % "8.5.2",
    "org.clulab"                 %% "model-streamed-trigram-ser" % "1.0.0",
    "com.lihaoyi"                %% "ujson"                   % "2.0.0"
  ) ++ parallelLibraries
}

lazy val core = (project in file("."))
  .enablePlugins(BuildInfoPlugin)
  .enablePlugins(DockerPlugin)
  .enablePlugins(JavaAppPackaging)
  .disablePlugins(PlayScala) // , SbtNativePackager)
  .settings(
    assembly / mainClass := Some("org.ml4ai.grounding.apps.HelloWorldApp")
  )

addCommandAlias("dockerize", ";docker:publishLocal")
