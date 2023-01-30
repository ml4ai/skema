assembly / aggregate := false
assembly / assemblyMergeStrategy := {
  case PathList("CHANGES.md")                              => MergeStrategy.discard
  case PathList("com", "sun", "istack", _*)                => MergeStrategy.first
  case PathList("com", "sun", "xml", _*)                   => MergeStrategy.first
  case PathList("javax", "activation", _*)                 => MergeStrategy.first
  case PathList("javax", "xml", "bind", _*)                => MergeStrategy.first
  case PathList("logback.xml")                             => MergeStrategy.last
  case PathList("META-INF", "versions", _*)                => MergeStrategy.first
  case PathList("module-info.class")                       => MergeStrategy.discard 
  case PathList("org", "apache", "commons", "logging", _*) => MergeStrategy.first
  case PathList("org", "bouncycastle", _*)                 => MergeStrategy.first
  // Otherwise just keep one copy if the contents are the same and complain if not.
  case other => // MergeStrategy.deduplicate
    val oldStrategy = (assembly / assemblyMergeStrategy).value
    oldStrategy(other)
}
// This prevents testing in root, then non-aggregation prevents it in other subprojects.
assembly / mainClass := None
assembly / test := {}
