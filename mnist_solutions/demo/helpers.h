#pragma once

#include <istream>
#include <vector>

#include <classifier.h>

#include <Eigen/Dense>

#include <mnist/classifier.h>

bool read_features(std::istream& stream, BinaryClassifier::features_t& features);
bool read_features_csv(std::istream& stream, BinaryClassifier::features_t& features);

std::vector<float> read_vector(std::istream&);


namespace mnist{

Eigen::MatrixXf read_mat_from_stream(size_t rows, size_t cols, std::istream& );

Eigen::MatrixXf read_mat_from_file(size_t rows, size_t cols, const std::string&);

std::vector<float> read_vector(std::istream&);

}