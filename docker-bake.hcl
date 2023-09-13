variable "DOCKER_REGISTRY" {
  default = "ghcr.io"
}
variable "DOCKER_ORG" {
  default = "darpa-askem"
}
variable "VERSION" {
  default = "local"
}
variable "SKEMA_TEXT_READING_DOCKERFILE_PATH" {
  default = "docker-tmp"
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
  targets = ["skema-py", "skema-rs", "skema-text-reading"]
}

group "default" {
  targets = ["skema-py-base", "skema-rs-base", "skema-text-reading-base"]
}

# ----------------------------------------------------------------------------------------------------------------------

target "_platforms" {
  # Currently skema-rs and skema-text-reading fails to compile on arm64 so we build only for amd at the moment
  # platforms = ["linux/amd64", "linux/arm64"]
  platforms = ["linux/amd64"]
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

target "skema-text-reading-base" {
  context = "${SKEMA_TEXT_READING_DOCKERFILE_PATH}"
  tags = tag("skema-text-reading", "", "")
  dockerfile = "Dockerfile"
}

target "skema-py" {
  inherits = ["_platforms", "skema-py-base"]
}

target "skema-rs" {
  inherits = ["_platforms", "skema-rs-base"]
}

target "skema-text-reading" {
  inherits = ["_platforms", "skema-text-reading-base"]
}
