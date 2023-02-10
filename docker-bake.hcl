variable "DOCKER_REGISTRY" {
  default = "ghcr.io"
}
variable "DOCKER_ORG" {
  default = "darpa-askem"
}
variable "VERSION" {
  default = "local"
}

# ----------------------------------------------------------------------------------------------------------------------

function "tag" {
  params = [image_name, prefix, suffix]
  result = [ "${DOCKER_REGISTRY}/${DOCKER_ORG}/${image_name}:${check_prefix(prefix)}${VERSION}${check_suffix(suffix)}" ]
}

function "check_prefix" {
  params = [tag]
  result = notequal("",tag) ? "${tag}-": ""
}

function "check_suffix" {
  params = [tag]
  result = notequal("",tag) ? "-${tag}": ""
}

# ----------------------------------------------------------------------------------------------------------------------

group "prod" {
  targets = ["skema-py", "skema-rs"]
}

group "default" {
  targets = ["skema-py-base", "skema-rs-base"]
}

# ----------------------------------------------------------------------------------------------------------------------

target "_platforms" {
  platforms = ["linux/amd64", "linux/arm64"]
}

target "skema-py-base" {
  context = "."
  tags = tag("skema-py", "", "")
  dockerfile = "Dockerfile.skema-py"
}

target "skema-rs-base" {
  context = "."
  tags = tag("skema-rs", "", "")
  dockerfile = "Dockerfile.skema-rs"
}

target "skema-py" {
  inherits = ["_platforms", "skema-py-base"]
}

target "skema-rs" {
  inherits = ["_platforms", "skema-rs-base"]
}

target "data-service-storage" {
  inherits = ["_platforms", "data-service-storage-base"]
}
