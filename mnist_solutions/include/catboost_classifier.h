#pragma once

#include <string>
#include <memory>
#include <vector>
#include "classifier.h"

#include <catboost/c_api.h>


class CatboostClassifier: public BinaryClassifier {
public:
    CatboostClassifier(const std::string& modepath);

    ~CatboostClassifier() override = default;
    
    float predict_proba(const features_t&) const override;
    std::vector<float> predict_proba_vector(const features_t&) const;

private:
    std::unique_ptr<ModelCalcerHandle, decltype(&ModelCalcerDelete)> model_; 
};
