// Single Layer Restricted Boltzmann Machines
//
// Author: Eric Yuan
// Blog: http://eric-yuan.me
// You are FREE to use the following code for ANY purpose.
//
// To run this code, you should have Armadillo in your computer.
// Have fun with it :)

#include <armadillo>
#include <math.h>
#include <fstream>
#include <iostream>
#include <random>  

using namespace arma;
using namespace std;

#define elif else if
#define HIDDEN_SIZE 200
#define BATCH_SIZE 2000

mat 
concatenateMat(vector<mat> &vec){

    int height = vec[0].n_rows;
    int width = vec[0].n_cols;
    mat res = zeros<mat>(height * width, vec.size());
    for(int i=0; i<vec.size(); i++){
        mat img = vec[i];
        img.reshape(height * width, 1);
        res.col(i) = img.col(0);
    }
    res = res / 255.0;
    return res;
}

int 
ReverseInt (int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void 
read_Mnist(string filename, vector<mat> &vec){
    ifstream file (filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i){
            mat tp(n_rows, n_cols);
            for(int r = 0; r < n_rows; ++r){
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp(r, c) = (double) temp;
                }
            }
            vec.push_back(tp);
        }
    }
}

void
readData(mat &x, string xpath){
    //read MNIST iamge into Arma Mat vector
    vector<mat> vec;
    read_Mnist(xpath, vec);
    random_shuffle(vec.begin(), vec.end());
    x = concatenateMat(vec);
}

mat 
sigmoid(mat M){
    return 1.0 / (exp(-M) + 1.0);
}

void
matRandomInit(mat &m, int rows, int cols, double scaler){
    m = randn<mat>(rows, cols);
    m = m * scaler;
}

mat
getBernoulliMatrix(mat &prob){
    // randu builds a Uniformly distributed matrix
    mat ran = randu<mat>(prob.n_rows, prob.n_cols);
    mat res = zeros<mat>(prob.n_rows, prob.n_cols);
    res.elem(find(prob > ran)).ones();
    return res;
}

void
save2txt(mat &data, string str, int step){
    string s = std::to_string(step);
    str += s;
    str += ".txt";
    FILE *pOut = fopen(str.c_str(), "w");
    for(int i=0; i<data.n_rows; i++){
        for(int j=0; j<data.n_cols; j++){
            fprintf(pOut, "%lf", data(i, j));
            if(j == data.n_cols - 1) fprintf(pOut, "\n");
            else fprintf(pOut, " ");
        }
    }
    fclose(pOut);
}

mat
RBM_training(mat x, int hidSize, int batchSize, int cd_k){

    int nfeatures = x.n_rows;
    int nsamples = x.n_cols;
    // b is hidden layer;
    // c is visible layer
    mat w, b, c;
    matRandomInit(w, nfeatures, hidSize, 0.12);
    matRandomInit(b, hidSize, 1, 0);
    matRandomInit(c, nfeatures, 1, 0);
    int counter = 0;
    double lrateW = 0.01; //Learning rate for weights 
    double lrateC = 0.01; //Learning rate for biases of visible units 
    double lrateB = 0.01; //Learning rate for biases of hidden units 
    double weightcost = 0.0002;   
    double initialmomentum = 0.5;
    double finalmomentum = 0.9;
    double errsum = 0.0;
    double momentum;
    mat incW = zeros(w.n_rows, w.n_cols);
    mat incB = zeros(b.n_rows, b.n_cols);
    mat incC = zeros(c.n_rows, c.n_cols);

    while(1){
        // start positive phase
        int randomNum = ((long)rand() + (long)rand()) % (nsamples - batchSize);
        mat data = x.cols(randomNum, randomNum + batchSize - 1);
        data = getBernoulliMatrix(data);
        mat poshidprobs = sigmoid(w.t() * data + repmat(b, 1, batchSize));
        poshidprobs = normalise(poshidprobs, 1, 0);
        mat posprods = data * poshidprobs.t() / batchSize;
        mat poshidact = sum(poshidprobs, 1) / batchSize;
        mat posvisact = sum(data, 1) / batchSize;

        // end of positive phase
        mat poshidprobs_temp = poshidprobs;
        mat poshidstates, negdata;
        // start negative phase
        // CD-K alg
        for(int i = 0; i < cd_k; i++){

            poshidstates = getBernoulliMatrix(poshidprobs_temp);
            negdata = sigmoid(w * poshidstates + repmat(c, 1, batchSize));
            negdata = getBernoulliMatrix(negdata);
            poshidprobs_temp = sigmoid(w.t() * negdata + repmat(b, 1, batchSize));
            poshidprobs_temp = normalise(poshidprobs_temp, 1, 0);
        }
        mat neghidprobs = poshidprobs_temp;
        mat negprods = negdata * neghidprobs.t() / batchSize;
        mat neghidact = sum(neghidprobs, 1) / batchSize;
        mat negvisact = sum(negdata, 1) / batchSize;
        
        //end of negative phase
        double err = accu(pow(mean(data - negdata, 1), 2.0));
        //errsum = err + errsum;
        if(counter > 10) momentum = finalmomentum;
        else momentum = initialmomentum;

        // update weights and biases
        incW = momentum * incW + lrateW * ((posprods - negprods) - weightcost * w);
        incC = momentum * incC + lrateC * (posvisact - negvisact);
        incB = momentum * incB + lrateB * (poshidact - neghidact);
        w += incW;
        c += incC;
        b += incB;
        cout<<"counter = "<<counter<<", error = "<<err<<endl;
        if(counter % 100 == 0){
            save2txt(w, "w/w_", counter / 100);
            save2txt(b, "b/b_", counter / 100);
            save2txt(c, "c/c_", counter / 100);
        }
        if(counter >= 10000) break;
        ++ counter;
    }
    return w;
}

int 
main(int argc, char** argv){

    long start, end;
    start = clock();
    mat trainX;
    readData(trainX, "mnist/train-images-idx3-ubyte");
    cout<<"Read trainX successfully, including "<<trainX.n_rows<<" features and "<<trainX.n_cols<<" samples."<<endl;
    // Finished reading data
    mat w = RBM_training(trainX, HIDDEN_SIZE, BATCH_SIZE, 1);
    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
    return 0;
}
