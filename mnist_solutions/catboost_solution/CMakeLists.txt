set(TARGET "catboost_solution")

add_executable(${TARGET}
    catboost_solution.cpp
    ${SRCS}
    ${HDRS}
)

target_link_directories(${TARGET} PUBLIC
    ${CMAKE_SOURCE_DIR}/contrib/catboost
)

target_link_libraries(${TARGET}
    CONAN_PKG::eigen
    catboostmodel
    dl
    pthread
)


target_include_directories(${TARGET} PUBLIC
    ../include
    ../demo
)

