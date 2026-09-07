set working-directory := ".."

default: shear shellcheck

shear:
    cargo shear

# The shell scripts under scripts/, with `-x` so sourced files are followed.
shellcheck:
    shellcheck -x scripts/*.sh
