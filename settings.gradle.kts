rootProject.name = "numerical-core-root"

include("core", "openblas")

project(":core").name = "numerical-core"
project(":openblas").name = "numerical-core-openblas"
