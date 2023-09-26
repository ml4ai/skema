import NativePackagerHelper._
import com.typesafe.sbt.packager.MappingsHelper.directory
import com.typesafe.sbt.packager.docker.{Cmd, CmdLike, DockerChmodType, DockerPermissionStrategy}

val topDir = "/skema/text_reading/webapp"
val appDir = topDir + "/app"
val binDir = appDir + "/bin/" // The second half is determined by the plug-in.  Don't change.
val app = binDir + "webapp"
val port = 9000
val tag = "1.0.0"


Docker / defaultLinuxInstallLocation := appDir
// TODO: can we run this with 11 (i.e., eclipse-temurin:11-jre-focal)?
Docker / dockerBaseImage := "eclipse-temurin:8-jre-focal"
Docker / daemonUser := "nobody"
Docker / dockerExposedPorts := List(port)
Docker / maintainer := "Keith Alcock <docker@keithalcock.com>"
Docker / mappings := (Docker / mappings).value.filter { case (_, string) =>
  // Only allow the app into the /app/bin directory.  Other apps that
  // might be automatically discovered are to be excluded.
  !string.startsWith(binDir) || string == app
}
Docker / packageName := "skema-webapp"
Docker / version := tag
// set our version based on our env variable
dockerEnvVars ++= Map(
  "APP_VERSION" -> scala.util.Properties.envOrElse("APP_VERSION", "???"),
  "APPLICATION_SECRET" -> "this-is-not-a-secure-key-please-change-me",
  // NOTE: the expected min. RAM requirements
  // FIXME: reduce RAM consumption $\le 8GB$
  "_JAVA_OPTIONS" -> "-Xmx10g -Xms10g -Dfile.encoding=UTF-8"
)
dockerAdditionalPermissions += (DockerChmodType.UserGroupPlusExecute, app)
dockerChmodType := DockerChmodType.UserGroupWriteExecute
// dockerCmd := Seq(s"-Dhttp.port=$port")
dockerEntrypoint := Seq(app)
// This is now delegated to skema-webapp.env.
// dockerEnvVars := Map(
//   "_JAVA_OPTIONS" -> "-Xmx16g -Xms16g -Dfile.encoding=UTF-8"
// )
dockerPermissionStrategy := DockerPermissionStrategy.MultiStage
dockerUpdateLatest := true

// Run "show dockerCommands" and use this to edit as appropriate.
dockerCommands := dockerCommands.value.flatMap { dockerCommand: CmdLike =>
  dockerCommand match {
    // Make sure that the appDir can be written for file locking.
    // case Cmd("USER", oldArgs @ _*) if (oldArgs.length == 1 && oldArgs.head == "1001:0") =>
    case Cmd("USER", "1001:0") =>
      Seq(
        Cmd("RUN", "chmod", "775", appDir),
        dockerCommand
      )
    case _ =>
      Seq(dockerCommand)
  }
}

def moveDir(dirname: String): Seq[(File, String)] = {
  val dir = file(dirname)
  val result = dir
    .**(AllPassFilter)
    .pair(relativeTo(dir.getParentFile))
    .map { case (file, _) => (file, file.getPath) }

//  result.foreach { case (file, string) =>
//    println(s"$file -> $string")
//  }
  result
}

// Universal / mappings ++= moveDir("./cache/geonames")

Global / excludeLintKeys += Docker / dockerBaseImage
