name := "webapp"

libraryDependencies ++= Seq(
  "org.scalatestplus.play" %% "scalatestplus-play" % "3.1.2" % Test,
  guice
)

//libraryDependencies ++= Seq(
//  "org.scalatestplus.play" %% "scalatestplus-play" % "5.1" % "test"
//)
// Adds additional packages into Twirl
TwirlKeys.templateImports += "org.clulab.controllers._"

// Adds additional packages into conf/routes
play.sbt.routes.RoutesKeys.routesImport += "org.clulab.binders._"
