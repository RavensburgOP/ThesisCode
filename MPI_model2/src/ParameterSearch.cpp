#include "diffusion.h"

arma::mat randomParam(int size);

int main(int argc, char* argv[]){
    MPI_Init(NULL, NULL);

    int WR, WS;

    MPI_Comm_rank(MPI_COMM_WORLD, &WR);
    MPI_Comm_size(MPI_COMM_WORLD, &WS);

    unsigned int start = static_cast<unsigned int>(atoi(argv[1]));
    unsigned int stop = static_cast<unsigned int>(atoi(argv[2]));
    unsigned int step = static_cast<unsigned int>(atoi(argv[3]));

    // Coarse grained search
    int num_parameter_sets = 100;
    int start_coarse_search = start;
    int end_coarse_search = stop;
    int step_size_coarse = step;

    int num_grid_steps = (end_coarse_search-start_coarse_search)/step_size_coarse;

    // General model parameters
    int train_iterations = 100;

    arma::mat data;
    data.load("../data/FullHomstradSample.dat", arma::csv_ascii);
    data = trans(data);
    arma::ivec sizes;
    sizes.load("../data/FullHomstradArray.dat");

    arma::mat loglikMat(num_parameter_sets, 50);
    // loglikMat.load("../data/loglikMat.csv");

    arma::mat old_param, new_param, prev_old_param;

    for(int i=start_coarse_search; i<end_coarse_search+1; i+=step_size_coarse){
        int i_idx = (i-start_coarse_search)/step_size_coarse;
        double best_loglik = -HUGE_VAL;

        for (int k=0; k<num_parameter_sets; k++){
            std::string save_path = "../data/trained_models/";
            if (i<10){
                save_path += "0";
            }
            save_path += std::to_string(i) + "_" + std::to_string(k);

            new_param = randomParam(i);
            // Set parameter to something determined before hand
            // if (i>start_coarse_search){
            //     new_param.rows(0,prev_old_param.n_cols-1) = trans(prev_old_param);
            // }

            Diffusion model_trainer(2, 30, data, sizes, new_param);

            double loglik = 0;
            double prev_loglik = 1000;
            int counter = 0;
            model_trainer.sample_discrete();
            while(std::abs(loglik - prev_loglik)>10){
                prev_loglik = loglik;
                if(counter > train_iterations){
                    break;
                }
                model_trainer.sample_discrete();
                model_trainer.optimise_parameters();
                if(WR==0){
                    loglik = model_trainer.get_loglik();
                }
                MPI_Bcast(&loglik, 1, MPI::DOUBLE,0,MPI_COMM_WORLD);
                counter++;
            }

            if (WR==0){
                loglikMat(k, i) = loglik;
                std::string saveLogLik = "../data/loglikMat_";
                if (i<10){
                    saveLogLik += "0";
                }
                saveLogLik += std::to_string(i) + ".csv";
                loglikMat.save(saveLogLik, arma::csv_ascii);
            }

            std::string save_states = save_path + "states.csv";
            std::string save_params = save_path + "params.csv";
            std::string save_trans = save_path + "trans.csv";

            model_trainer.saveStates(save_states);
            model_trainer.saveParameters(save_params);
            model_trainer.saveTransition(save_trans);

            if (loglik>best_loglik) old_param = model_trainer.get_parameters();

        }
        prev_old_param = old_param;
    }

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
