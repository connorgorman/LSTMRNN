#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <Eigen/Dense>
#include "fileReader.h"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_real.hpp>
#include "Eigen_Util.h"
#include "LSTM.h"
#include "softmax.h"
#include <fstream>
#include "config.h"
#include <chrono>
#include <ctime>

class LSTM;

class Layer
{
	
public:

	Layer(Config c);

	template<typename Derived>
	void initMatrix(const Eigen::MatrixBase<Derived> &input_const, type range);

	void initLSTMVector(FileReader &fr);

	#ifdef PEEPHOLE
	template<typename Derived>
	void initDiagonal(const Eigen::DiagonalBase<Derived> &input_const, type range);
	#endif


	template<typename Derived>
	void forwardPropagation(const Eigen::MatrixBase<Derived> &inputMiniBatch);

	template<typename Derived>
	void backwardPropagation(const Eigen::MatrixBase<Derived> &outputMiniBatch);

	void clearRunningErrors();

	type getError();

	void applyGradients();

	void clearGradients();

	template<typename Derived>
	void checkGradient(const Eigen::MatrixBase<Derived> &mat_const, const Eigen::MatrixBase<Derived> &dErr_mat_const, std::string mat_name, type ep);

	void checkGradients(type ep);

	void dumpMatrices();

	void loadWeights(std::string filename);

	template<typename Derived>
	void clipGradients(const Eigen::MatrixBase<Derived> &gradient_const, type gradThres);

	void printMatrices();


	/*** Public Members ***/
	std::vector<LSTM> LSTMs;

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Whi; 
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Whf; 
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Whc; 
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Who;

	#ifdef PEEPHOLE
	
	// Eigen::DiagonalMatrix<type, Eigen::Dynamic, Eigen::Dynamic> Wci;
	// Eigen::DiagonalMatrix<type, Eigen::Dynamic, Eigen::Dynamic> Wcf;
	// Eigen::DiagonalMatrix<type, Eigen::Dynamic, Eigen::Dynamic> Wco;

	// Eigen::DiagonalMatrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_Wci;
	// Eigen::DiagonalMatrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_Wcf;
	// Eigen::DiagonalMatrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_Wco;


	// Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Wxi; 
	// Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Wxf; 
	// Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Wxc; 
	// Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Wxo; 

	// Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_Wxi; 
	// Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic>  dErr_Wxf; 
	// Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic>  dErr_Wxc; 
	// Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic>  dErr_Wxo; 
	
	#else
	
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Mi;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Mf;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Mo;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Mc;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> W;

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_Mi;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_Mf;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_Mo;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_Mc;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_W;

	#endif
	
	Eigen::Matrix<type, Eigen::Dynamic, 1> Bi;
	Eigen::Matrix<type, Eigen::Dynamic, 1> Bf;
	Eigen::Matrix<type, Eigen::Dynamic, 1> Bc;
	Eigen::Matrix<type, Eigen::Dynamic, 1> Bo;

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> D;

	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> inputMiniBatch;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> outputMiniBatch;



	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> first_hidden;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> first_cell;

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> runningErrH;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> runningErrC;

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic>  dErr_Whi; 
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic>  dErr_Whf; 
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic>  dErr_Whc; 
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic>  dErr_Who;
	
	
	Eigen::Matrix<type, Eigen::Dynamic, 1> dErr_Bi;
	Eigen::Matrix<type, Eigen::Dynamic, 1> dErr_Bf;
	Eigen::Matrix<type, Eigen::Dynamic, 1> dErr_Bc;
	Eigen::Matrix<type, Eigen::Dynamic, 1> dErr_Bo;

	boost::random::mt19937 gen;

	Softmax * softmax;

	Config config;

};

#endif
