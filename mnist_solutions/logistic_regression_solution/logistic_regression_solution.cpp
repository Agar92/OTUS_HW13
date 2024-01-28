#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <algorithm>
#include <iterator>

#include <logreg_classifier.h>
#include <helpers.h>

using std::clog;
using namespace std;

int main(int argc, char* argv[])
{
    if(argc != 3){
        cout<<"Invalid number of arguments!"<<endl;
        cout<<"The input should be:\n";
        std::string str(argv[0]);
        auto found=str.find_last_of("/\\");
        std::string result="";
        if(found == std::string::npos) result=str;
        else                           result=str.substr(found+1);
        cout<<result<<"  "<<"path/to/test.csv  path/to/logreg_coef.txt"<<endl;
        cout<<"EXAMPLE:"<<endl;
        cout<<"logistic_regression_solution ../test.csv ../logreg_coef.txt"<<endl;
        std::terminate();
    }
    const std::string logreg_coef_file(argv[2]);
    const std::string test_file(argv[1]);

    std::ifstream istream{logreg_coef_file};
    //if(!istream.is_open()) std::cout<<"FILE logreg_coef.txt NOT OPENED!"<<std::endl;
    //else                   std::cout<<"FILE logreg_coef.txt OPENED!"<<std::endl;
    std::string s="";
    std::vector<std::vector<float>> coefs(10);
    int i=0;
    while( std::getline(istream, s) )
    {
        std::stringstream ss(s);
        auto coef = read_vector(ss);
        //cout<<"i="<<i<<" coef.size()="<<coef.size()<<endl;
        coefs[i]=coef;
        i++;
    }
    istream.close();

    vector<LogregClassifier> predictor;
    for(int i=0; i<10; ++i) predictor.push_back(LogregClassifier{coefs[i]});

    auto features = LogregClassifier::features_t{};

    double y_pred_expected = 0.0;

    std::ifstream test_data{test_file};//test_data_logreg.txt"};
    //if(!test_data.is_open()) std::cout<<"FILE test_data NOT OPENED!"<<std::endl;
    //else                     std::cout<<"FILE test_data OPENED!"<<std::endl;
    int EQUAL=0, NOT_EQUAL=0, TOTAL=0;
    for (;;) {
        test_data >> y_pred_expected;
        if (!read_features_csv(test_data, features)) {
            break;
        }
        vector<float> probabilities(10);
        for(int i=0; i<10; ++i) 
        {
            auto y_pred = predictor[i].predict_proba(features);
            probabilities[i]=y_pred;
        }
        auto iter=std::max_element(probabilities.begin(),probabilities.end());
        const int y_pred=std::distance(probabilities.begin(),iter);
        if(y_pred_expected == y_pred) EQUAL++;
        else                          NOT_EQUAL++;
        TOTAL++;
    }
    cout<<"accuracy="<<((float)EQUAL)/TOTAL<<endl;
}
