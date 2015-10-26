#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <Eigen/Dense>
#include "LSTM.h"
#include "Eigen_Util.h"
#include "layer.h"
#include "config.h"
#include <fstream>
#include <chrono>
#include <ctime>

class Layer;

class Softmax
{

public:

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> D;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_D;
	Eigen::Matrix<type, 1, Eigen::Dynamic> softmax_denominator;
	Eigen::Matrix<type, Eigen::Dynamic, 1> Bk;
	Eigen::Matrix<type, Eigen::Dynamic, 1> dErr_Bk;

	Eigen::Matrix<type, Eigen::Dynamic, 1> sums;


	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> prob;
	type perplexity = 0.0;


	int batchSize;
	int hiddenSize;

	Layer * layer;

	boost::random::mt19937 gen;

	Config config;


	void initSoftmax(int outputVocabSize, int hiddenSize, int batchSize, Layer * layer);

	template<typename Derived, typename Derived2, typename Derived3> 
	void computeGradient(const Eigen::MatrixBase<Derived> &h_t, const Eigen::MatrixBase<Derived2> &input, const Eigen::MatrixBase<Derived3> &dErr_h_t_const);

	template<typename Derived, typename Derived2> 
	type computeError(const Eigen::MatrixBase<Derived> &h_t, const Eigen::MatrixBase<Derived2> &input);

	void resetDenominator();

	template<typename Derived>
	void computeProbability(const Eigen::MatrixBase<Derived> &h_t);

	void applyGradients();

	void checkGradients(type ep);

	template<typename Derived>
	void checkGradient(const Eigen::MatrixBase<Derived> &mat, const Eigen::MatrixBase<Derived> &dErr_mat, std::string mat_name, type ep);

	template<typename Derived>
	void initMatrix(const Eigen::MatrixBase<Derived> &input_const, type range);

	template<typename Derived>
	void clipGradients(const Eigen::MatrixBase<Derived> &input_const, type gradThres);

	void clearGradients();

	void dumpMatrices();

	void loadWeights(std::string filename);

	void printMatrices();


};


#endif
