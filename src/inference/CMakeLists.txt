set(PUBLIC_HEADERS)

set(SOURCES
    src/inference.cpp)

add_executable(inference ${SOURCES})
target_link_libraries(inference PRIVATE cache model operations util)
