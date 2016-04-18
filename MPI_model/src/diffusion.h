#include <armadillo>
#include <nlopt.hpp>
#include <mpi.h>
#include <cstdlib>
#include <fstream>
#include <cassert>

class Diffusion{
public:
    // Constructors
    Diffusion(arma::mat &new_data);
    Diffusion(int new_maxK, int new_etrunc, arma::mat &new_data, arma::ivec new_size, arma::mat initial_parameters);

    // E-step
    void sample_discrete();

    // M-step
    void optimise_parameters();

    // Public declaration for friend function used by nlopt
    friend double object_wrapper(const std::vector<double> & x, std::vector<double> & grad, void* dataPtr);

    // Save functions
    void saveDataWithState(std::string name);
    void saveParameters(std::string name);
    void saveTransition(std::string name);
    void saveStates(std::string name);

    // Getters
    double get_loglik();
    arma::mat get_parameters();

    // Testing
    void Do_unit_test();

private:
    // Mpi information
    int WR, WS, datasize, localcount, job, jobCount;
    MPI_Datatype ArmadilloRow, ParameterCol, ParameterSet, ArmaInt;
    int *sendcounts, *displ;
    std::vector<int> jobQ, jobList;
    bool need_data, firstIter;
    int count;

    // Mpi functions
    void distributeSendCounts();

    int JobCheck();
    void subsetData();

    // Data size constants
    int elem_pr_row;
    arma::ivec pairIdx, protId_cs;

    // Model constants
    int lk, etrunc;
    arma::vec twokpi;

    // Data manipulation
    void sortData();
    arma::uvec sort_indexes(const std::vector<int> &v);
    double wrapMax(double x, double max);
    double wrapMinMax(double x);

    // Loglikelihood calculation
    arma::mat parameters;
    void markov_blanket();
    arma::vec logLikWnOuPairsTime(arma::mat data, int hidden_state);
    arma::mat safeSoftMax(arma::mat logs);
    arma::mat transMat, transCount, transCountTemp;
    arma::vec init_state, init_state_count;

    // Nlopt
    double DistributedEstimation(bool manager);
    void SetupNlopt(uint p);
    double objective_function(const std::vector<double> & x, std::vector<double> & grad, void* ptr);


    // Data
    bool e_step_done, m_step_done;
    const arma::mat &data;
    std::vector<int> tsc, state_count;

    // recv_data is resized multiple times and is therefor not called in constructor
    double loc_loglik, loglik;
    arma::mat recv_data, send_data, grad_send, grad_recv, sorted_data;
    arma::vec logLik_send, logLik_recv;
    arma::ivec state, sorted_state;
    arma::ivec send_state, recv_state;
};
