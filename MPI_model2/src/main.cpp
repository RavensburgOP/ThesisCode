#include "diffusion.h"

arma::mat randomParam(int size);
arma::mat safeSoftMax(arma::mat logs, double etrunc = 30);
double logLikWnOuPairs(arma::mat x, arma::vec t, arma::vec alpha, arma::vec mu, arma::vec sigma, int maxK = 2, double etrunc = 30);

int main(int argc, char* argv[]){
    MPI_Init(NULL, NULL);

    int burn_in = 5;

    arma::mat data;
    data.load("../data/FullHomstradSample.csv", arma::csv_ascii);
    arma::ivec sizes;
    sizes.load("../data/FullHomstradArray.csv");
    arma::mat datat = trans(data);
    arma::mat param;
    param ={{13,   11.5, 0, -2, 0, 1, 1},
                   {11.5, 11.5, 0.5, 2, 0, 1, 1}};//,
    //                {9.5, 9.5, 0, 2, -1, 1, 1}};//,
    //                {2.5, 0.5, 0, 2, M_PI/2, 1, 1},
    //                {1.8, 0.5, 0, 2, M_PI/2, 1, 1},
    //                {2.3, 0.5, 0, 2, M_PI/2, 1, 1}};

    // param = randomParam(2);


    // arma::vec v = trans(param.row(0));

    // std::cout << logLikWnOuPairs(data.cols(0,3), data.col(4), v.subvec(0,2), v.subvec(3,4), v.subvec(5,6), 1) << std::endl;

    Diffusion test(1, 5, datat, sizes, param);
    // test.saveDataWithState("../data/test/output.csv");

    for (int i=0; i<burn_in; i++){
        test.sample_discrete();
    }
    for (int i=0; i<10; i++){
        test.sample_discrete();
        test.optimise_parameters();
        test.saveStates("../data/test/states.csv");
    }

    test.saveDataWithState("../data/test/output.csv");
    test.saveParameters("../data/params.csv");
    test.saveTransition("../data/transmat.csv");
    // // test.Do_unit_test();
    MPI_Finalize();
    return 0;
}

arma::mat randomParam(int size){
    // arma::arma_rng::set_seed_random();
    std::default_random_engine generator(time(0));
    arma::mat temp = arma::ones<arma::mat>(size, 7);
    for (int i=0; i<size;i++){
        std::uniform_real_distribution<double> alpha12(1e-2,10.0);
        temp(i, 0) = alpha12(generator);
        temp(i, 1) = alpha12(generator);
        double tempVal = temp(i,0)*temp(i,1)/10;
        std::uniform_real_distribution<double> alpha3(-tempVal,tempVal);
        temp(i,2) = alpha3(generator);

        // Check that alpha3 is within limits
        double testalpha = temp(i,0) * temp(i,1) - temp(i,2) * temp(i,2);
        if(testalpha <= 0) temp(i,2) = std::signbit(temp(i,2)) * sqrt(temp(i,0) * temp(i,1)) * 0.9999;

        std::uniform_real_distribution<double> mu12(-M_PI,M_PI);
        temp(i,3) = mu12(generator);
        temp(i,4) = mu12(generator);
    }
    return temp;
}
