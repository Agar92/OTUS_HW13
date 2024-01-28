#include "helpers.h"

#include <iostream>
#include <string>
#include <sstream>
#include <iterator>
#include <fstream>

bool read_features(std::istream& stream, BinaryClassifier::features_t& features) {
    std::string line;
    std::getline(stream, line);

    features.clear();
    std::istringstream linestream{line};
    double value;
    while (linestream, linestream >> value) {
        features.push_back(value);
    }
    return stream.good();
}

bool read_features_csv(std::istream& stream, BinaryClassifier::features_t& features) {
    std::string line;
    std::getline(stream, line);
    features.clear();
    std::stringstream linestream{line};
    double value;
    std::string s="";
    while(getline(linestream, s, ',')){
        if(!s.empty())
        {
            std::string::size_type sz;
            features.push_back( std::stof(s, &sz) );
        }
    }
    return stream.good();
}

std::vector<float> read_vector(std::istream& stream) {
    std::vector<float> result;

    std::copy(std::istream_iterator<float>(stream),
              std::istream_iterator<float>(),
              std::back_inserter(result));
    return result;
}



namespace mnist {

Eigen::MatrixXf read_mat_from_stream(size_t rows, size_t cols, std::istream& stream) {
    Eigen::MatrixXf res(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float val;
            stream >> val;
            res(i, j) = val;
        }
    }
    return res;
}

Eigen::MatrixXf read_mat_from_file(size_t rows, size_t cols, const std::string& filepath) {
    std::ifstream stream{filepath};
    return read_mat_from_stream(rows, cols, stream);
}

std::vector<float> read_vector(std::istream& stream) {
    std::vector<float> result;

    std::copy(std::istream_iterator<float>(stream),
              std::istream_iterator<float>(),
              std::back_inserter(result));
    return result;
}

}