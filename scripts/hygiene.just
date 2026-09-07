set working-directory := ".."

default: shear zizmor

shear:
    cargo shear

# Static analysis of the workflows under .github.
zizmor:
    zizmor .github
