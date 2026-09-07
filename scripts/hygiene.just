set working-directory := ".."

default: shear zizmor typos

shear:
    cargo shear

# Static analysis of the workflows under .github.
zizmor:
    zizmor .github

# Spelling, over every tracked file. typos.toml lists the words the format uses.
typos:
    typos
