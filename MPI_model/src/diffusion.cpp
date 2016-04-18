#include "diffusion.h"

// MAJOR TODO: MAKE SURE ALL ARMADILLO ITERATIONS ARE COLUMN MAJOR

// Forward declaration
double object_wrapper(const std::vector<double> & x, std::vector<double> & grad, void* dataPtr);

Diffusion::Diffusion(arma::mat &new_data) : data(new_data){
    MPI_Comm_rank(MPI_COMM_WORLD, &WR);
    MPI_Comm_size(MPI_COMM_WORLD, &WS);

    // Number of elements in the armadillo row
    elem_pr_row = 5;

    // Create new armadillo row datatype for mpi scatterv
    MPI_Type_contiguous(elem_pr_row, MPI::DOUBLE, &ArmadilloRow);
    MPI_Type_commit(&ArmadilloRow);

    // Create parameter datatype for Bcast
    MPI_Type_contiguous(parameters.n_cols, MPI::DOUBLE, &ParameterCol);
    MPI_Type_commit(&ParameterCol);

    MPI_Type_contiguous(parameters.n_rows, ParameterCol, &ParameterSet);
    MPI_Type_commit(&ParameterSet);

    // If you're looking at this because you're getting overflow errors when selecting states, it's because armadillo integers are only 8 bytes. You'll have to change to unsigned integers (uword) to solve it (or stop using armadillo vectors for storing them, but that will bring you a whole other range of issues)
    MPI_Type_contiguous(8, MPI::BYTE, &ArmaInt);
    MPI_Type_commit(&ArmaInt);

    // arma::arma_rng::set_seed_random();

    e_step_done = false;
    m_step_done = false;
}

Diffusion::Diffusion(int new_maxK, int new_etrunc, arma::mat &new_data, arma::ivec new_sizes, arma::mat initial_parameters): Diffusion(new_data){
    // Sequence of winding numbers
    lk = 2 * new_maxK + 1;
    twokpi = arma::linspace<arma::vec>(-new_maxK * 2 * M_PI, new_maxK * 2 * M_PI, lk);

    etrunc = new_etrunc;

    parameters = trans(initial_parameters);

    state_count.resize(parameters.n_cols);
    tsc.resize(parameters.n_cols);

    transMat.resize(parameters.n_cols, parameters.n_cols);
    if (WR==0){
        std::cout << "Initial parameters" << std::endl;
        std::cout << trans(parameters) << std::endl;
        state = arma::randi(data.n_cols, arma::distr_param(0, parameters.n_cols-1));

        // Initialise transition matrix
        transMat.fill(arma::fill::randu);
        transMat += 0.5;
        transMat.each_col() /= sum(transMat, 1);
    }

    MPI_Bcast(parameters.memptr(), parameters.size(), MPI::DOUBLE, 0, MPI_COMM_WORLD);


    transCount.resize(parameters.n_cols, parameters.n_cols);
    transCountTemp.resize(parameters.n_cols, parameters.n_cols);

    arma::vec emit_prop(parameters.n_cols);

    logLik_send.set_size(parameters.n_cols);
    logLik_send.fill(arma::fill::zeros);
    logLik_recv = logLik_send;

    grad_send.set_size(parameters.n_rows, parameters.n_cols);
    grad_send.fill(arma::fill::zeros);
    grad_recv = grad_send;

    pairIdx = new_sizes;
}

void Diffusion::distributeSendCounts(){
    if(WR==0){
        datasize = send_data.n_cols;

        // Set Mpi information
        displ = new int[WS];
        sendcounts = new int[WS];
        int sum = 0;
        for (int i=0; i<WS;i++){
            sendcounts[i] = (datasize / WS) + ((datasize % WS) > i);
            displ[i] = sum;
            sum += sendcounts[i];
        }
    }
    MPI_Bcast(&datasize, 1, MPI::INT, 0, MPI_COMM_WORLD);
    localcount = (datasize / WS) + ((datasize % WS) > WR);
    recv_data.resize(elem_pr_row, localcount);
}

void Diffusion::sample_discrete(){
    // printf("[%i] Sample disc\n", WR);
    // Move this to sample_discrete

    loc_loglik = 0;

    // It should be possible to delete this without messing anything up
    if (WR==0){
        send_data = data;
    }
    MPI_Bcast(transMat.memptr(), transMat.size(), MPI::DOUBLE, 0, MPI_COMM_WORLD);

    transCount.fill(arma::fill::zeros);

    if (WR==0){
        // Make a cumulative sum of the indices
        protId_cs.resize(pairIdx.size()+1);
        protId_cs(0) = 0;
        arma::ivec temp = cumsum(pairIdx);
        protId_cs.subvec(1, protId_cs.size()-1) = temp.subvec(0, temp.size()-1);
    }

    std::fill(state_count.begin(), state_count.end(), 0);
    // Only send out proteins if there is more than one process
    if (WS>1){
        if (WR==0){
            int proteinId;
            MPI_Status status;
            for (int i=0; i<pairIdx.size();i++){
                // (I)D (A)nd (S)ize
                int ias[2];
                ias[0] = i;
                ias[1] = pairIdx(i);
                MPI_Recv(&proteinId, 1, MPI::INT,MPI::ANY_SOURCE, 0,MPI_COMM_WORLD, &status);

                // If worker has a protein, receive sampled states
                if(proteinId > -1){
                    int size = pairIdx(proteinId);
                    recv_state.resize(size);
                    MPI_Recv(recv_state.memptr(), size, ArmaInt, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    state.subvec(protId_cs(proteinId), protId_cs(proteinId+1)-1) = recv_state;
                }

                // Send protein id and size
                MPI_Send(&ias, 2, MPI::INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);

                // Send a protein out for sampling
                send_data = data.cols(protId_cs(i), protId_cs(i+1)-1);
                send_state = state.subvec(protId_cs(i), protId_cs(i+1)-1);
                MPI_Send(send_data.memptr(), ias[1], ArmadilloRow, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                MPI_Send(send_state.memptr(), ias[1], ArmaInt, status.MPI_SOURCE,0, MPI_COMM_WORLD);
            }

            // Tell all processes, that they are done
            for(int i=1; i<WS; i++){
                MPI_Recv(&proteinId, 1, MPI::INT,MPI::ANY_SOURCE, 0,MPI_COMM_WORLD, &status);
                if(proteinId > -1){

                    int size = pairIdx(proteinId);
                    recv_state.resize(size);
                    MPI_Recv(recv_state.memptr(), size, ArmaInt, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    state.subvec(protId_cs(proteinId), protId_cs(proteinId+1)-1) = recv_state;

                }
                // (I)D (A)nd (S)ize
                int ias[2];
                ias[0] = -1;
                ias[1] = 0;
                // Send protein size
                MPI_Send(&ias, 2, MPI::INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            }
        }
        else{
            // (I)D (A)nd (S)ize
            int ias[2];
            ias[0] = -1;
            ias[1] = 0;
            while(true){
                // Ask for protein
                MPI_Send(&ias, 1, MPI::INT, 0, 0, MPI_COMM_WORLD);

                // Receive protein size
                if(ias[0]>-1){
                    MPI_Send(send_state.memptr(), ias[1], ArmaInt, 0, 0, MPI_COMM_WORLD);
                }

                MPI_Recv(&ias, 2, MPI::INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // If there are no more proteins stop loop
                if (ias[1]==0) {
                    break;
                }

                // Initialise correct sized matrix
                recv_data.resize(elem_pr_row, ias[1]);
                recv_state.resize(ias[1]);

                // Receive Protein
                MPI_Recv(recv_data.memptr(), ias[1], ArmadilloRow, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_state.memptr(), ias[1], ArmaInt, 0,0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Do analysis
                markov_blanket();
            }
        }
    }
    else if(WR==0){
        for (int i=0; i<pairIdx.size();i++){
            recv_data = data.cols(protId_cs(i), protId_cs(i+1)-1);
            recv_state = state.subvec(protId_cs(i), protId_cs(i+1)-1);

            markov_blanket();
            state.subvec(protId_cs(i), protId_cs(i+1)-1) = send_state;
        }
    }

    MPI_Reduce(&loc_loglik, &loglik, 1, MPI::DOUBLE, MPI::SUM, 0, MPI_COMM_WORLD);

    e_step_done = true;
    m_step_done = false;
}

void Diffusion::optimise_parameters(){
    // Reduce transition matrix
    MPI_Reduce(transCount.memptr(), transCountTemp.memptr(), parameters.n_cols*parameters.n_cols, MPI::DOUBLE, MPI::SUM, 0, MPI::COMM_WORLD);

    if(WR==0){
        // Pseudo count
        transCountTemp += 1;
        transMat = arma::conv_to<arma::mat>::from(transCountTemp);
        transMat.each_col() /= sum(transMat, 1);
    }

    // Reduce state count
    MPI_Allreduce(&state_count.front(), &tsc.front(), parameters.n_cols, MPI::INT, MPI::SUM,MPI::COMM_WORLD);

    int count;
    jobList.resize(WS);

    // 1) Sort data
    if(WR==0){
        // Needs to sort data and calculate which states have assigned points
        sortData();

        // Count the number of nonzero parameters, then create a pool with idx for unoptimised states

        int num = count_if(tsc.begin(), tsc.end(), []( int x ) { return x > 0; } );
        count = 0;

        // Create jobQ
        jobQ.assign(std::max(WS, num),-1);
        for (int i=0; i<tsc.size(); i++){
            if(tsc[i]>0){
                jobQ[count] = i;
                // subProcMat(count, 1) = -1;
                count++;
            }
        }
        for (int i=0; i<WS; i++){
            jobList[i] = jobQ[i];
        }
        jobCount = num-WS;
        subsetData();
    }

    // Distribute jobs
    MPI_Scatter(&jobList.front(), 1, MPI::INT, &job, 1, MPI::INT, 0, MPI_COMM_WORLD);

    need_data = true;
    int done = 1;
    JobCheck();
    while(done!=3){
        if(job > -1){
            SetupNlopt(job);
            job=-2;
            // printf("[%i] Jobcheck after nlopt\n", WR);
            done = JobCheck();
        }
        else if (job==-1){
            // printf("[%i] is now worker\n", WR);
            DistributedEstimation(false);
            done = JobCheck();
        }
    }

    if (WR==0) std::cout << "Final parameters for m-step" << std::endl;
    if (WR==0) std::cout << trans(parameters) << std::endl;


    e_step_done = false;
    m_step_done = true;
}

void Diffusion::sortData(){
    // Find the sorted index
    arma::uvec idx = sort_index(state);
    // Copy the sorted order to new array
    sorted_state = state.rows(idx);

    arma::uvec cols = arma::linspace<arma::uvec>(0,data.n_rows-1,data.n_rows);

    sorted_data = data.submat(cols, idx);

  // if(WR==0){
  //       arma::mat outMatrix(sorted_data.n_rows+1, sorted_data.n_cols);
  //       outMatrix.rows(1, 5) = sorted_data;
  //       std::vector<double> stateVec(sorted_state.begin(), sorted_state.end());
  //       arma::vec stateVecArm(stateVec);
  //       outMatrix.row(0) = trans(stateVecArm);
  //       outMatrix = trans(outMatrix);

  //       std::cout << "sorted_data " << outMatrix << std::endl;
  //   }
}

int Diffusion::JobCheck(){
    // printf("[%i] Job check %i\n", WR, job);
    // 1 if all have jobs, but we're not done. 2 if someone needs a job and 3 if all are finished.
    std::vector<int> joblist(WS);
    // printf("[%i] has job %i\n", WR, job);
    MPI_Gather(&job, 1, MPI::INT, &joblist.front(), 1, MPI::INT, 0, MPI_COMM_WORLD);
    int ret=1;

    if(WR==0){
        bool allWO = true;
        bool allHave = true;
        for (int i=0; i<WS;i++){
            if (joblist[i]==-2){
                // A process has finished and need a job
                allHave=false;
            }
            else if (joblist[i] > -1){
                // The process is part of the workpool
                allWO=false;
            }
        }
        if(allWO && (jobCount < 1)){
            // m-step is done
            ret=3;
        }
        else if(!allHave){
            // A process needs a new assignment
            ret=2;
        }
    }

    MPI_Bcast(&ret, 1, MPI::INT, 0, MPI_COMM_WORLD);
    if(ret==2){
        if(WR==0){
            for (int i=0; i<WS;i++){
                if (joblist[i]==-2){
                    // Send a nlopt job if any available
                    if (jobCount > 0){
                        int newjob = jobQ[jobQ.size()-jobCount];
                        // Just copy to self if out of a job
                        if (i==0){
                            job=newjob;
                        }
                        else{
                            MPI_Send(&newjob, 1, MPI::INT, i, 0, MPI_COMM_WORLD);
                        }
                        jobList[i] = newjob;
                        jobCount--;
                    }
                    else{
                        int newjob = -1;
                        // Just copy to self if out of a job
                        if (i==0){
                            job=newjob;
                        }
                        else{
                            MPI_Send(&newjob, 1, MPI::INT, i, 0, MPI_COMM_WORLD);
                        }
                        jobList[i] = newjob;
                    }
                }
            }
        }
        if(job==-2 && WR>0){
            MPI_Recv(&job, 1, MPI::INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        need_data = true;
    }

    if(need_data){
        // Send out new data
        MPI_Bcast(&jobList.front(), WS, MPI::INT, 0, MPI_COMM_WORLD);
        if(WR==0){
            subsetData();

        }
        distributeSendCounts();
        MPI_Scatterv(send_data.memptr(), sendcounts, displ, ArmadilloRow, recv_data.memptr(), localcount, ArmadilloRow, 0, MPI_COMM_WORLD);

        need_data = false;
    }

    return ret;
}

void Diffusion::subsetData(){
    // This function is very likely to produce the desired result. Past Christian is not a complete moron. Now go bughunt somewhere else.
    int send_size = 0;
    for(int i=0; i<WS; i++){
        if (jobList[i]>-1){
            // Add the number of datapoints that each process needs
            send_size += tsc[jobList[i]];
        }
    }
    if(send_size == data.n_cols){
        if (e_step_done){
            send_data = sorted_data;
        }
        else{
            send_data = data;
        }
    }

    else{
        send_data.resize(5, send_size);
        int idxSS = 0;
        int idxSE = -1;
        // The cumulative sum is first index of each state in the dataset
        std::vector<int> cs(tsc.size()+1);
        int sum = 0;
        for(int i=0;i<tsc.size();i++){
            // if(jobList[i]>-1)
            sum += tsc[i];
            cs[i+1] = sum;
        }
        std::vector<int> sortJobList = jobList;
        sort(begin(sortJobList), end(sortJobList));

        // YES IT IS SUPPOSED TO LOOK LIKE THAT!
        for(int i=0; i<WS;i++){
            int p = sortJobList[i];
            if (p>-1){
                idxSE += tsc[p];
                // Cut out datapoints from each state from the sorted array
                arma::mat insert = sorted_data.cols(cs[p], (cs[p+1]-1));
                send_data.cols(idxSS, idxSE) = insert;
                idxSS += tsc[p];
            }
        }
    }
}

double Diffusion::DistributedEstimation(bool manager){
    // The bulk of calculations for nlopt

    logLik_send.fill(arma::fill::zeros);

    // Receive parameter rows from workers and distribute full parameter set
    if(WR==0){
        // Receive parameters from all worker processes doing setupNlopt
        for(int i=0; i<jobList.size();i++){
            int p = jobList[i];
            if (p > -1 && i!=0){
                MPI_Recv(parameters.colptr(p), 7, MPI::DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    // // Distribute new parameter sets
    MPI_Bcast(parameters.memptr(), parameters.size(), MPI::DOUBLE, 0, MPI_COMM_WORLD);

    const double h = 1e-3;

    // Calculate P(Data | parameters)
    std::vector<int> tsccs(tsc.size()+1);
    std::vector<int> state_count(tsc.size());

    // Find out which states our datapoints are assigned to
    std::vector<int> sortJobList = jobList;
    sort(begin(sortJobList), end(sortJobList));

    // Cumulative sum of counts in the subset for indexing
    int sum = 0;
    for(int i=0; i<jobList.size();i++){
        int p = jobList[i];
        if (p<0) continue;
        for (int j=p; j<tsc.size();j++){
            tsccs[j+1] += tsc[p];
        }
    }

    int mystart = (datasize / WS) * WR + ((datasize % WS) < WR ? (datasize % WS) : WR);
    int myend = mystart + (datasize / WS) + ((datasize % WS) > WR)-1;

    for(int i=0; i<sortJobList.size();i++){
        int p0 = sortJobList[i];
        if (p0<0) continue;
        int fir = std::min(myend, tsccs[p0+1]-1);
        int sec = std::max(mystart, tsccs[p0]);
        state_count[p0] = std::max(0, fir - sec);
    }

    mystart = 0;
    myend = 0;

    grad_send.fill(arma::fill::zeros);


    arma::mat gradVec(state_count[1]+1, grad_send.n_rows);
    for(uint p = 0; p<state_count.size(); p++){
        // Check if there is data for that hidden state
        if(state_count[p] == 0) continue;;
        myend = mystart+state_count[p];
        // if(p==1) printf("[%i] From %i to %i on state %i\n",WR, mystart, myend, p);
        arma::mat DataSubset = recv_data.cols(mystart, myend);

        // for(int i=0; i<WS; i++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     if (WR==i){
        //         std::cout << trans(DataSubset);
        //     }
        // }

        // Likelihood of data
        // printf("[%i] From %i to %i on state %i\n",WR, mystart, myend, p);
        arma::vec temp = logLikWnOuPairsTime(DataSubset, p);
        logLik_send(p) = arma::sum(temp);

        // Gradient
        // printf("[%i] Localcount = %i. Subset from %i to %i on state %i\n",WR, localcount, mystart, myend, p);
        for (int i=0; i<grad_send.n_rows;i++){
            parameters(i,p) += h;
            temp = logLikWnOuPairsTime(DataSubset, p);
            if(p == 1) gradVec.col(i) = temp;
            grad_send(i,p) = arma::sum(temp);
            parameters(i,p) -= h;
        }
        count += p;

        // printf("[%i] From %i to %i on state %i\n",WR, mystart, myend, p);
         mystart = myend+1;
    }

    grad_recv.fill(arma::fill::zeros);
    logLik_recv.fill(arma::fill::zeros);
    MPI_Allreduce(grad_send.memptr(), grad_recv.memptr(), grad_send.size(), MPI::DOUBLE, MPI::SUM, MPI_COMM_WORLD);
    MPI_Allreduce(logLik_send.memptr(), logLik_recv.memptr(), parameters.n_cols, MPI::DOUBLE, MPI::SUM, MPI_COMM_WORLD);

    for (int i=0; i<logLik_recv.size();i++){
        grad_recv.col(i) -= logLik_recv(i);
    }
    grad_recv /= h;

}

void Diffusion::SetupNlopt(uint p){
    // printf("[%i] SetupNlopt\n", WR);
    // finds most likely parameter values based on assigned data
    double opt_f;

    // Creates nlopt maxlik object and set search algorithm
    nlopt::opt lOpt(nlopt::LD_LBFGS, 7);
    nlopt::opt opt(nlopt::AUGLAG, 7);
    opt.set_local_optimizer(lOpt);

    // Set boundary constraints for diffusion
    double lower[7] = {1e-2, 1e-2, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, 0.99, 0.99};
    double upper[7] = {25, 25, HUGE_VAL, HUGE_VAL, HUGE_VAL, 1.01, 1.01};
    std::vector<double> vLower(lower, lower + sizeof(lower) / sizeof(double));
    std::vector<double> vUpper(upper, upper + sizeof(upper) / sizeof(double));
    opt.set_lower_bounds(vLower);
    opt.set_upper_bounds(vUpper);

    // Add constraint
    // opt.add_inequality_constraint(AlphaConstraint, NULL, 1e-8);

    // Stopping criterion
    opt.set_xtol_abs(0.001);
    opt.set_maxeval(200);

    // Set the pointer of objective_function
    opt.set_max_objective(object_wrapper, this);

    std::vector<double> x = arma::conv_to< std::vector<double> >::from(parameters.col(p));

    firstIter = true;
    // Run optimisation

    nlopt::result res = opt.optimize(x, opt_f);

    printf("[%i] Done optimising %i\n", WR, p);
}

double Diffusion::objective_function(const std::vector<double> & x, std::vector<double> & grad, void* dataPtr){
    // printf("[%i] objective_function\n", WR);
    // printf("[%i] Jobcheck in obj func\n", WR);
    if(!firstIter){
        int notused = JobCheck();
        if (notused==3){
            printf("[%i] Told job is done, but still in function", WR);
        }
    }
    firstIter=false;

    // send x to root who then broadcasts a new parameter matrix
    if (WR>0){

        MPI_Send(&x.front(), 7, MPI::DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else{
        arma::vec ins(x);
        parameters.col(job) = ins;
    }

    DistributedEstimation(true);
    if (job == 1) count++;

    if (!grad.empty()) {
        grad = arma::conv_to< std::vector<double> >::from(grad_recv.col(job));
    }

    // if(job==1){

    //     std::cout << WR << " ";
    //     for(int i=0; i<grad.size(); i++){
    //         std::cout << grad[i] << " ";
    //     }
    //     for(int i=0; i<x.size(); i++){
    //         std::cout << x[i] << " ";
    //     }
    //     std::cout << job << " LL= " << logLik_recv(job) << std::endl;
    // }
    // if(count>5){
    //     exit(0);
    // }
    return logLik_recv(job);
}

double object_wrapper(const std::vector<double> & x, std::vector<double> & grad, void* dataPtr){
    // grad.resize(7);
    Diffusion *obj = static_cast<Diffusion *>(dataPtr);
    return obj->objective_function(x, grad, dataPtr);
}

arma::vec Diffusion::logLikWnOuPairsTime(arma::mat in_data, int hidden_state) {
    // printf("[%i] loglik\n", WR);
  /*
   * Create basic objects
   */

  // Number of pairs
  int N = in_data.n_cols;
  in_data = trans(in_data);

  // Needs to create alpha, mu and sigma from hidden_state and t from x
  arma::mat x = in_data.cols(0,3);
  arma::vec t = in_data.col(4);

  arma::vec params = parameters.col(hidden_state);
  arma::vec alpha = params.subvec(0, 2);
  arma::vec mu = params.subvec(3, 4);
  for (int i=0; i<mu.size(); i++){
    mu(i) = wrapMinMax(mu(i));
  }

  arma::vec sigma = params.subvec(5, 6);

  // Create log-likelihoods
  arma::vec loglikinitial;
  arma::vec logliktpd;
  arma::vec loglik;

  // Create and initialize A
  double quo = sigma(0) / sigma(1);
  arma::mat A(2, 2);
  A(0, 0) = alpha(0);
  A(1, 1) = alpha(1);
  A(0, 1) = alpha(2) * quo;
  A(1, 0) = alpha(2) / quo;

  // Create and initialize Sigma
  arma::mat Sigma = diagmat(square(sigma));

  // Bivariate vector (2 * K1 * PI, 2 * K2 * PI) for weighting
  arma::vec twokepivec(2);

  // Bivariate vector (2 * K1 * PI, 2 * K2 * PI) for wrapping
  arma::vec twokapivec(2);

  /*
   * Check for symmetry and positive definiteness of A^(-1) * Sigma
   */

  // Add a penalty to the loglikelihood in case any assumption is violated
  double penalty = 0;

  // Only positive definiteness can be violated with the parametrization of A
  double testalpha = alpha(0) * alpha(1) - alpha(2) * alpha(2);

  // Check positive definiteness
  if(testalpha <= 0) {
    // std::cout << "PENALTY!" << std::endl;
    // Add a penalty
    penalty = (-testalpha * 10000 + 10);
    if(e_step_done) penalty /= tsc[hidden_state];
    else penalty /= localcount;

    // Update alpha(2) such that testalpha > 0
    alpha(2) = std::signbit(alpha(2)) * sqrt(alpha(0) * alpha(1)) * 0.9999;

    // Reset A to a matrix with positive determinant
    A(0, 1) = alpha(2) * quo;
    A(1, 0) = alpha(2) / quo;

  }

  // Determinant of A
  double detA = alpha(0) * alpha(1) - alpha(2) * alpha(2);

  // A^(-1) * Sigma
  arma::mat AInvSigma(2, 2);
  AInvSigma(0, 0) = alpha(1) * Sigma(0, 0);
  AInvSigma(0, 1) = -alpha(2) * sigma(0) * sigma(1);
  AInvSigma(1, 0) = AInvSigma(0, 1);
  AInvSigma(1, 1) = alpha(0) * Sigma(1, 1);
  AInvSigma = AInvSigma / detA;

  // Inverse of (1/2 * A^(-1) * Sigma): 2 * Sigma^(-1) * A
  arma::mat invSigmaA(2, 2);
  invSigmaA(0, 0) = 2 * alpha(0) / Sigma(0, 0);
  invSigmaA(0, 1) = 2 * alpha(2) / (sigma(0) * sigma(1));
  invSigmaA(1, 0) = invSigmaA(0, 1);
  invSigmaA(1, 1) = 2 * alpha(1) / Sigma(1, 1);

  // Determinant of invSigmaA
  double detInvSigmaA = 4 / (Sigma(0, 0) * Sigma(1, 1)) * detA;

  // For normalizing constants
  double l2pi = log(2 * M_PI);

  // Log-normalizing constant for the Gaussian with covariance SigmaA
  double lognormconstSigmaA = -l2pi + log(detInvSigmaA) / 2;

  /*
   * Computation of Gammat and exp(-t * A) analytically
   */

  // Quantities for computing exp(-t * A)
  double s = (alpha(0) + alpha(1)) / 2;
  double q = sqrt(fabs((alpha(0) - s) * (alpha(1) - s) - alpha(2) * alpha(2)));

  // Avoid indetermination in sinh(q * t) / q when q == 0
  if(q == 0){

    q = 1e-6;

  }

  // s1(-t) and s2(-t)
  arma::vec est = exp(-s * t);
  arma::vec eqt = exp(q * t);
  arma::vec eqtinv = 1 / eqt;
  arma::vec cqt = (eqt + eqtinv) / 2;
  arma::vec sqt = (eqt - eqtinv) / (2 * q);
  arma::vec s1t = est % (cqt + s * sqt);
  arma::vec s2t = -est % sqt;

  // s1(-2t) and s2(-2t)
  est = est % est;
  eqt = eqt % eqt;
  eqtinv = eqtinv % eqtinv;
  cqt = (eqt + eqtinv) / 2;
  sqt = (eqt - eqtinv) / (2 * q);
  arma::vec s12t = est % (cqt + s * sqt);
  arma::vec s22t = -est % sqt;

  /*
   * Weights of the winding numbers for each data point
   */

  // We store the weights in a matrix to skip the null later in the computation of the tpd
  arma::mat weightswindsinitial(N, lk * lk);
  weightswindsinitial.fill(lognormconstSigmaA);



  // Loop in the data
  for(int i = 0; i < N; i++){

    // Compute the factors in the exponent that do not depend on the windings
    arma::vec xmu = x.submat(i, 0, i, 1).t() - mu;
    arma::vec xmuinvSigmaA = invSigmaA * xmu;
    double xmuinvSigmaAxmudivtwo = -dot(xmuinvSigmaA, xmu) / 2;

    // Loop in the winding weight K1
    for(int wek1 = 0; wek1 < lk; wek1++){

      // 2 * K1 * PI
      twokepivec(0) = twokpi(wek1);

      // Compute once the index
      int wekl1 = wek1 * lk;

      // Loop in the winding weight K2
      for(int wek2 = 0; wek2 < lk; wek2++){

        // 2 * K2 * PI
        twokepivec(1) = twokpi(wek2);

        // Negative exponent
        weightswindsinitial(i, wekl1 + wek2) += xmuinvSigmaAxmudivtwo - dot(xmuinvSigmaA, twokepivec) - dot(invSigmaA * twokepivec, twokepivec) / 2;

      }

    }

  }

  // The unstandardized weights of the tpd give the required wrappings for the initial loglikelihood
  loglikinitial = log(sum(exp(weightswindsinitial), 1));

//   if(hidden_state == 1) {
//     // for (int i=0; i<state_count.size(); i++){
//     //     std::cout << state_count[i] << " ";
//     // }
//     // std::cout << std::endl;
//     std::cout << "Hidden state parameters" << std::endl;
//     std::cout << localcount << std::endl;
//     std::cout << N << std::endl;
//     std::cout << loglikinitial << std::endl;
// };

  // Standardize weights for the tpd
  weightswindsinitial = safeSoftMax(weightswindsinitial);

  /*
   * Computation of the tpd: wrapping + weighting
   */

  // The evaluations of the tpd are stored in a vector, no need to keep track of wrappings
  arma::vec tpdfinal(N); tpdfinal.zeros();

  // Loop in the data
  for(int i = 0; i < N; i++){

    // Initial point x0 varying with i
    arma::vec x0 = x.submat(i, 0, i, 1).t();

    // Exp(-ti * A)
    arma::mat ExptiA = s2t(i) * A;
    ExptiA.diag() += s1t(i);

    // Gammati
    arma::mat Gammati = ((1 - s12t(i)) * AInvSigma - s22t(i) * Sigma) / 2;

    // Inverse and log-normalizing constant for the Gammat
    arma::mat invGammati = inv_sympd(Gammati);
    double lognormconstGammati = -l2pi + log(det(invGammati)) / 2;

    // Loop on the winding weight K1
    for(int wek1 = 0; wek1 < lk; wek1++){

      // 2 * K1 * PI
      twokepivec(0) = twokpi(wek1);

      // Compute once the index
      int wekl1 = wek1 * lk;

      // Loop on the winding weight K2
      for(int wek2 = 0; wek2 < lk; wek2++){

        // Skip zero weights
        if(weightswindsinitial(i, wekl1 + wek2) > 0){

          // 2 * K1 * PI
          twokepivec(1) = twokpi(wek2);

          // muti
          arma::vec muti = mu + ExptiA * (x0 + twokepivec - mu);

          // Compute the factors in the exponent that do not depend on the windings
          arma::vec xmuti = x.submat(i, 2, i, 3).t() - muti;
          arma::vec xmutiInvGammati = invGammati * xmuti;
          double xmutiInvGammatixmutidiv2 = -dot(xmutiInvGammati, xmuti) / 2;

          // Loop in the winding wrapping K1
          for(int wak1 = 0; wak1 < lk; wak1++){

            // 2 * K1 * PI
            twokapivec(0) = twokpi(wak1);

            // Loop in the winding wrapping K2
            for(int wak2 = 0; wak2 < lk; wak2++){

              // 2 * K2 * PI
              twokapivec(1) = twokpi(wak2);

              // Decomposition of the negative exponent
              double exponent = xmutiInvGammatixmutidiv2 - dot(xmutiInvGammati, twokapivec) - dot(invGammati * twokapivec, twokapivec) / 2 + lognormconstGammati;

              // Tpd
              tpdfinal(i) += exp(exponent) * weightswindsinitial(i, wekl1 + wek2);

            }

          }

        }

      }

    }

  }

  // Logarithm of tpd
  // tpdfinal = log(tpdfinal);

  // Set log(0) to -trunc, as this is the truncation of the negative exponentials
  tpdfinal.elem(find_nonfinite(tpdfinal)).fill(-etrunc);

  // // Log-likelihood from tpd
  logliktpd = log(tpdfinal);

  // // Final loglikelihood
  loglik = loglikinitial + logliktpd;

  // // Check if it is finite
  loglik.elem(find_nonfinite(loglik)).fill(-100);

  return loglik - penalty;
  // return tpdfinal;

}

double Diffusion::get_loglik(){
    return loglik;
}

arma::mat Diffusion::get_parameters(){
    return parameters;
}

arma::mat Diffusion::safeSoftMax(arma::mat logs) {

  // Maximum of logs by rows to avoid overflows
  arma::vec m = max(logs, 1);

  // Recenter by columns
  logs.each_col() -= m;

  // Ratios by columns
  logs.each_col() -= log(sum(exp(logs), 1));

  // Truncate exponential by using a lambda function - requires C++ 11
  logs.transform([this](double val) { return (val < -etrunc) ? double(0) : double(exp(val)); });

  return logs;
}

void Diffusion::saveDataWithState(std::string name){
    if(WR==0){
        arma::mat outMatrix(data.n_rows+1, data.n_cols);
        outMatrix.rows(1, 5) = data;
        std::vector<double> stateVec(state.begin(), state.end());
        arma::vec stateVecArm(stateVec);
        outMatrix.row(0) = trans(stateVecArm);
        outMatrix = trans(outMatrix);
        outMatrix.save(name, arma::csv_ascii);
        // std::cout << outMatrix << std::endl;
    }
}

void Diffusion::saveStates(std::string name){
    if(WR==0){
        state.save(name, arma::csv_ascii);
    }
}

void Diffusion::saveParameters(std::string name){
    if(WR==0){
        parameters.save(name, arma::csv_ascii);
    }
}

void Diffusion::saveTransition(std::string name){
    if(WR==0){
        transMat.save(name, arma::csv_ascii);
    }
}

arma::uvec Diffusion::sort_indexes(const std::vector<int> &v) {

  // initialize original index locations
  arma::uvec idx = arma::linspace<arma::uvec>(0,v.size()-1,v.size());

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2];});

  return idx;
}

/* wrap x -> [0,max) */
double Diffusion::wrapMax(double x, double max){
    /* integer math: `(max + x % max) % max` */
    return fmod(max + fmod(x, max), max);
}
/* wrap x -> [-pi, pi) */
double Diffusion::wrapMinMax(double x){
    return -M_PI + wrapMax(x - -M_PI, M_PI - -M_PI);
}

void Diffusion::markov_blanket(){
    // Calculates likelihood of hidden state given child, previous hidden state and child of previous hidden state
    double ppll, pmll;

    send_state.resize(recv_state.n_rows);

    // shuffle positions for random sampling
    arma::ivec seq = arma::linspace<arma::ivec>(0, recv_state.size()-1, recv_state.size());
    arma::ivec seq_shuf = arma::shuffle(seq);
    send_state.resize(recv_state.size());
    arma::vec sampleVec(seq_shuf.size(), arma::fill::randu);

    // Check entire code below for cache misses
    arma::mat logLikMat(recv_state.size(), parameters.n_cols);

    // Calculate likelihoods for each datapoint
    for(int p=0; p<parameters.n_cols; p++){
        arma::vec temp = logLikWnOuPairsTime(recv_data, p);
        logLikMat.col(p) = temp;
    }

    logLikMat = trans(logLikMat);
    for (int p=0; p<seq_shuf.size();p++){
        // Hidden states
        int i = seq_shuf(p);
        // Previous hidden state and child
        arma::vec c = logLikMat.col(i);
        c = exp(c);
        // If neighbour to the right, find likelihood from transition
        if(i<seq_shuf.size()-2){
            int pp1 = recv_state(i+1);
            for(int ppt=0; ppt<c.size(); ppt++){
                ppll = transMat(ppt,pp1);
                c(ppt) *= ppll;
            }
        }
        // If neighbour to the left, find likelihood from transition
        if (i>0){
            int pm1 = recv_state(i-1);
            for(int ppt=0; ppt<c.size(); ppt++){
                pmll = transMat(pm1,ppt);
                c(ppt) *= pmll;
            }
        }
        arma::vec c_temp = c;
        // Sample new hidden state
        c /= accu(c);
        c = cumsum(c);
        for(int j=0; j<c.n_rows; j++){
            // if (c(j) > sampleVec(i)){
            if (c(j) > 0.5){
                send_state(i) = j;
                recv_state(i) = j;
                loc_loglik += log(c_temp(j));
                break;
            }
        }
    }

    // Count up transitions and states
    arma::vec fwrAlg;
    for (int i=0; i<seq_shuf.size();i++){
        int p = send_state(i);
        state_count[p]++;
        if(i<seq_shuf.size()-2){
            int pp = send_state(i+1);
            transCount(p, pp)++;
        }
    }
}
