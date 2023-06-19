addSbtPlugin("com.typesafe.play"       % "sbt-plugin"           % "2.8.18")
//addSbtPlugin("org.scoverage" % "sbt-scoverage" % "2.0.5")
addSbtPlugin("com.eed3si9n"            % "sbt-assembly"         % "2.1.1")
addSbtPlugin("com.eed3si9n"            % "sbt-buildinfo"        % "0.11.0")
ThisBuild / libraryDependencySchemes ++= Seq(
  "org.scala-lang.modules" %% "scala-xml" % VersionScheme.Always
)