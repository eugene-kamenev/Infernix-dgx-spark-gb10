add_library(infernix_media_decode STATIC
  decode/decode.cpp
  decode/jpeg_decode.cpp
  decode/resize.cpp
)
infernix_internal_includes(infernix_media_decode)
target_link_libraries(infernix_media_decode PRIVATE ${INFERNIX_FFMPEG_TARGET} JPEG::JPEG)
