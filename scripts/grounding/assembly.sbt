assembly / aggregate := false
assembly / assemblyMergeStrategy := {

  case x =>
    val oldStrategy = (assembly / assemblyMergeStrategy).value
    oldStrategy(x)
}
// This prevents testing in core, then non-aggregation prevents it in other subprojects.
assembly / test := {}
