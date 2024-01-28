set(TARGET "logistic_regression_solution")

add_executable(${TARGET}
    logistic_regression_solution.cpp
    ${SRCS}
    ${HDRS}
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


