#ifndef LSTM_H
#define LSTM_H

#include <Eigen/Dense>
#include "Eigen_Util.h"
#include "layer.h"
#include <chrono>
#include <ctime>
#include <fstream>

// #include "layer.hpp"

class Layer;

class LSTM
{
	
	public:

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> i_t; //input gate
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> f_t; // forget gate
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> c_t; // current cell value
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> o_t; // output gate
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> h_t; // hidden value
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> g_t; // tanh value

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_i_t; // gradient wrt i_t
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_f_t; // gradient wrt f_t
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_c_t; // gradient wrt c_t
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_o_t; // gradient wrt o_t
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_h_t; // gradient wrt h_t
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_g_t; // gradient wrt g_t
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_c_t_1; // gradient wrt c_t - 1
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> dErr_h_t_1; // gradient wrt h_t - 1


	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> runningErrH_t2n;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> runningErrC_t2n; // gradient wrt c_t from t to n

	// Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic>  

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> prev_h;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> prev_c;

	Eigen::Matrix<int,Eigen::Dynamic, 1> input;
	Eigen::Matrix<int,Eigen::Dynamic, 1> output;

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> intermediate;

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> tanh_prime;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> one_minus_tanh_sq_prime;

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> tanh_ct;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> one_minus_tanh_sq_ct;

	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> it_one_minus_it;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> ft_one_minus_ft;
	Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> ot_one_minus_ot;

	#ifndef PEEPHOLE
		Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Zi;
		Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Zf;
		Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Zc;
		Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic> Zo;
	#endif

	template<typename Derived>
	void compute_intermediate(const Eigen::MatrixBase<Derived> &w_mat);

	void compute_i_t();

	void compute_f_t();

	void compute_c_t();

	void compute_o_t();

	void compute_h_t();

	void compute_g_t();

	template<typename Derived, typename Derived2>
	void setupForwardProp(const Eigen::MatrixBase<Derived> &prev_h, 
							const Eigen::MatrixBase<Derived> &prev_c, 
							const Eigen::MatrixBase<Derived2> &inputMiniBatch, 
							int index);

	template<typename Derived>
	void setupBackwardProp(const Eigen::MatrixBase<Derived> &output, int index);

	template<typename Derived>
	void print_matrices(const Eigen::MatrixBase<Derived> &mat, std::string name);
	
	void forwardPropagation();

	template<typename Derived, typename Derived2>
	void backwardPropagation(const Eigen::MatrixBase<Derived> &runningErrH_const, 
						const Eigen::MatrixBase<Derived2> &runningErrC_const);

	LSTM(int cell_size, int batchsize, Layer * layer);

	Layer * layer;

};

#endif
