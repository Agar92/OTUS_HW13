#include <fstream>
#include <iostream>


#include <mnist/tf_classifier.h>

#include <helpers.h>

using namespace mnist;
using namespace std;

const size_t width = 28;
const size_t height = 28;
const size_t output_dim = 10;



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
        cout<<result<<"  "<<"path/to/test.csv  path/to/saved_model"<<endl;
        cout<<"EXAMPLE:"<<endl;
        cout<<"tensorflow_solution ../test.csv ../saved_model"<<endl;
        std::terminate();
    }
    const std::string saved_model_file(argv[2]);
    const std::string test_file(argv[1]);
    auto clf = TfClassifier{saved_model_file, width, height};//{"../saved_model", width, height};
    auto features = TfClassifier::features_t{};
    std::ifstream test_data{test_file};//{"../test.csv"};
    //if( !test_data.is_open() ) cout<<"FILE test_data NOT OPENED!"<<endl;
    //else                       cout<<"FILE test_data OPENED!"<<endl;
    int EQUAL=0, NOT_EQUAL=0, TOTAL=0;
    for (;;) {
        size_t y_pred_expected;
        test_data >> y_pred_expected;
        if (!read_features_csv(test_data, features)) {
            break;
        }
        auto y_pred = clf.predict(features);
        //cout<<"y_pred_expected="<<y_pred_expected<<" y_pred="<<y_pred<<endl;
        if(y_pred_expected == y_pred) EQUAL++;
        else                          NOT_EQUAL++;
        TOTAL++;
    }
    cout<<"accuracy="<<((float)EQUAL)/TOTAL<<endl;
}
