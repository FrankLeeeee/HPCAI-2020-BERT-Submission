local home = os.getenv("HOME")
local root = pathJoin(home, "apps/software")
local name = myModuleName()
local ver = myModuleVersion()


local basedir      = pathJoin(pathJoin(root, name), ver)
local bin          = pathJoin(basedir, "bin")
local sbin          = pathJoin(basedir, "sbin")

prepend_path("PATH", bin)
prepend_path("PATH", sbin)
