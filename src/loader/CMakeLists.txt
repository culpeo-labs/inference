add_library(loader src/loader.cpp)
target_include_directories(loader PUBLIC include)
target_link_libraries(loader PRIVATE glaze::glaze)
target_compile_options(loader PRIVATE -Wall -Wextra)

add_executable(tester tester.cpp)
target_link_libraries(tester loader)
