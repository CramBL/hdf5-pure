COMMON_USER_FACING_FEATURES := "ndarray,provenance,serde,zfp,num-complex,checksum"
USER_FACING_FEATURES := "fast-deflate," + COMMON_USER_FACING_FEATURES
# uses deflate in place of fast-deflate because fast-deflate requires libz-ng-sys
USER_FACING_FEATURES_PORTABILITY := "deflate," + COMMON_USER_FACING_FEATURES
