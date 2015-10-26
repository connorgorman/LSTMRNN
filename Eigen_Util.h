/****************************************

	Written by Barret Zoph 2015

****************************************/

#ifndef EIGEN_UTIL_H
#define EIGEN_UTIL_H

#include <Eigen/Dense>

// Functions that take non-const matrices as arguments
// are supposed to declare them const and then use this
// to cast away constness.
#define UNCONST(t,c,uc) Eigen::MatrixBase<t> &uc = const_cast<Eigen::MatrixBase<t>&>(c);

#define UNCONST_DIAG(t,c,uc) Eigen::DiagonalBase<t> &uc = const_cast<Eigen::DiagonalBase<t>&>(c);

struct sigmoid_functor {
  type operator() (type x) const { return 1.0/(1.0+std::exp(-x)); }
};

struct tanh_functor {
  type operator() (type x) const { return std::tanh(x); }
};

struct tanh_sq_functor {
  type operator() (type x) const { return std::tanh(x)*std::tanh(x); }
};


struct exp_functor {
  type operator() (type x) const { return std::exp(x); }
};

struct log_functor {
	type operator() (type x) const { return std::log(x); }
};

//Does elementwise sigmoid operation on matrix
template <typename DerivedIn, typename DerivedOut>
void elemwise_sigmoid(const Eigen::MatrixBase<DerivedIn> &input_const, const Eigen::MatrixBase<DerivedOut> &output_const) {
	UNCONST(DerivedOut, output_const, output);
	output = input_const.array().unaryExpr(sigmoid_functor());
	output.matrix();
}

//Does elementsize tanh operation on matrix
template <typename DerivedIn, typename DerivedOut>
void elemwise_tanh(const Eigen::MatrixBase<DerivedIn> &input_const, const Eigen::MatrixBase<DerivedOut> &output_const) {
	UNCONST(DerivedOut, output_const, output);
	output = input_const.array().unaryExpr(tanh_functor());
	output.matrix();
}

//Calculate the hidden state for LSTM with this
template <typename Derived>
void hidden_calc(const Eigen::MatrixBase<Derived> &c_t,const Eigen::MatrixBase<Derived> &h_t_const,const Eigen::MatrixBase<Derived> &o_t) {
	UNCONST(Derived, h_t_const, h_t);
	h_t = o_t.array()*(c_t.array().unaryExpr(tanh_functor()));
	h_t.matrix();
}

#endif