set(TARGET "mlp_solution")

add_executable(${TARGET}
    mlp_solution.cpp
    ../src/mnist/mlp_classifier.cpp
    ${SRCS}
    ${HDRS}
)

target_link_libraries(${TARGET} PUBLIC
    CONAN_PKG::eigen
    catboostmodel
    tensorflow
    dl
    pthread
)

target_include_directories(${TARGET} PUBLIC
    ../include
    ../demo
)


