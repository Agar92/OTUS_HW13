set(TARGET "tensorflow_solution")

add_executable(${TARGET}
    tensorflow_solution.cpp
    ../src/mnist/tf_classifier.cpp
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


