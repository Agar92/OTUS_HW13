#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <algorithm>
#include <iterator>

#include <mnist/mlp_classifier.h>
#include <helpers.h>

using std::clog;
using namespace std;
using namespace mnist;

const size_t input_dim = 784;
const size_t hidden_dim = 128;
const size_t output_dim = 10;

int main(int argc, char* argv[])
{
    if(argc != 4){
        cout<<"Invalid number of arguments!"<<endl;
        cout<<"The input should be:\n";
        std::string str(argv[0]);
        auto found=str.find_last_of("/\\");
        std::string result="";
        if(found == std::string::npos) result=str;
        else                           result=str.substr(found+1);
        cout<<result<<"  "<<"path/to/test.csv  path/to/w1.txt path/to/w2.txt"<<endl;
        cout<<"EXAMPLE:"<<endl;
        cout<<"mlp_solution ../test.csv ../w1.txt ../w2.txt"<<endl;
        std::terminate();
    }
    const std::string test_file(argv[1]);
    const std::string w1_file(argv[2]);
    const std::string w2_file(argv[3]);
    //
    auto w1 = read_mat_from_file(input_dim, hidden_dim, w1_file);//"../w1.txt");
    auto w2 = read_mat_from_file(hidden_dim, output_dim, w2_file);//"../w2.txt");
    auto clf = MlpClassifier{w1.transpose(), w2.transpose()};
    auto features = MlpClassifier::features_t{};
    std::ifstream test_data{"../test.csv"};///{"../test_data_mlp.txt"};
    //if( !test_data.is_open() ) cout<<"FILE test_data NOT OPENED!"<<endl;
    //else                       cout<<"FILE test_data OPENED!"<<endl;
    int EQUAL=0, NOT_EQUAL=0, TOTAL=0;
    for (;;) {
        size_t y_pred_expected;
        test_data >> y_pred_expected;
        if (!read_features_csv(test_data, features)) {
            break;
        }
        /*
        cout<<"features:"<<endl;
        int j=1;
        for(auto i : features) cout<<i<<"|"<<j<<" ", j++;
        cout<<endl;
        */
        auto y_pred = clf.predict(features);
        //cout<<"y_pred_expected="<<y_pred_expected<<" y_pred="<<y_pred<<endl;
        if(y_pred_expected == y_pred) EQUAL++;
        else                          NOT_EQUAL++;
        TOTAL++;
    }
    cout<<"accuracy="<<((float)EQUAL)/TOTAL<<endl;
}
