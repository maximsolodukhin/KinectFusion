target_sources(kinectfusion PRIVATE read_png.cpp write_png.cpp)

# libspng is an implementation detail of the png reader
target_link_libraries(kinectfusion PRIVATE spng_static)
