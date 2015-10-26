// #include "layer.h"

template<typename Derived>
void LSTM::compute_intermediate(const Eigen::MatrixBase<Derived> &w_mat)
{

	for(int i = 0; i < intermediate.cols(); i++)
	{
		if( input(i) != -1 )
			intermediate.col(i) = w_mat.col( input(i) );
		else
			intermediate.col(i) = w_mat.col(0);
	}
}

void LSTM::compute_i_t()
{
	#ifdef PEEPHOLE
		elemwise_sigmoid( (intermediate + layer->Whi*prev_h + layer->Wci * prev_c).colwise() + layer->Bi , i_t  );
	#else
	
		elemwise_sigmoid( (layer->Mi * intermediate + layer->Whi*prev_h).colwise() + layer->Bi, i_t );

		// Eigen::MatrixXd temp = (intermediate + layer->Whi*prev_h) + layer->Bi; 
	
	#endif

	// elemwise_sigmoid( temp , i_t  );
}

void LSTM::compute_f_t()
{
	#ifdef PEEPHOLE

		elemwise_sigmoid( (intermediate + layer->Whf*prev_h + layer->Wcf * prev_c).colwise() + layer->Bf, f_t );

		// Eigen::MatrixXd temp = (intermediate + layer->Whf*prev_h + layer->Wcf * prev_c).colwise() + layer->Bf;
	#else

		elemwise_sigmoid( ( layer->Mf * intermediate + layer->Whf*prev_h).colwise() + layer->Bf, f_t );
		// Eigen::MatrixXd temp = (intermediate + layer->Whf*prev_h).colwise() + layer->Bf;
	#endif

	// elemwise_sigmoid( temp , f_t );
}

void LSTM::compute_g_t()
{
	// (Mc * intermediate + layer->Whc*prev_h).colwise() + layer->Bc;
	elemwise_tanh( (layer->Mc * intermediate + layer->Whc*prev_h).colwise() + layer->Bc, g_t );
}

void LSTM::compute_c_t()
{
	c_t = (f_t.array() * prev_c.array() + i_t.array() * g_t.array()).matrix(); 
}

void LSTM::compute_o_t()
{

	#ifdef PEEPHOLE
		// Eigen::MatrixXd temp = (intermediate + layer->Who * prev_h + layer->Wco*c_t).colwise() + layer->Bo;

		elemwise_sigmoid( (intermediate + layer->Who * prev_h + layer->Wco*c_t).colwise() + layer->Bo, o_t );

	#else
		// Eigen::MatrixXd temp = (intermediate + layer->Who * prev_h).colwise() + layer->Bo;

		elemwise_sigmoid( (layer->Mo * intermediate + layer->Who * prev_h).colwise() + layer->Bo, o_t);
	
	#endif

	// elemwise_sigmoid(temp, o_t);

}

void LSTM::compute_h_t()
{
	h_t = o_t.array() * c_t.array().unaryExpr(tanh_functor());
}

template<typename Derived, typename Derived2>
void LSTM::setupForwardProp(const Eigen::MatrixBase<Derived> &prev_h, 
						const Eigen::MatrixBase<Derived> &prev_c, 
						const Eigen::MatrixBase<Derived2> &inputMiniBatch, 
						int index)
{


	this->prev_h.noalias() = prev_h;
	this->prev_c.noalias() = prev_c;
	this->input = inputMiniBatch.col(index);
	intermediate.setZero(intermediate.rows(), intermediate.cols());

}

template<typename Derived>
void LSTM::setupBackwardProp(const Eigen::MatrixBase<Derived> &output, int index)
{
	this->output = output.col(index);
}

template<typename Derived>
void LSTM::print_matrices(const Eigen::MatrixBase<Derived> &mat, std::string name)
{
	std::cout << "\n\n" << name << "\n" << mat << std::endl;
}

template<typename Derived, typename Derived2>
void LSTM::backwardPropagation(const Eigen::MatrixBase<Derived> &runningErrH_const, 
						const Eigen::MatrixBase<Derived2> &runningErrC_const)
{
	UNCONST(Derived, runningErrH_const, runningErrH_t2n);
	UNCONST(Derived2, runningErrC_const, runningErrC_t2n);
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;

	// start = std::chrono::system_clock::now();

	//elemwise_tanh( g_t, tanh_prime);
	tanh_prime = g_t;
	elemwise_tanh( c_t, tanh_ct);

	one_minus_tanh_sq_prime = (1 - g_t.array().square() ).matrix();
	one_minus_tanh_sq_ct = (1 - c_t.array().unaryExpr(tanh_sq_functor()) ).matrix();

	it_one_minus_it = (1 - i_t.array() ) * i_t.array();
	ft_one_minus_ft = (1 - f_t.array() ) * f_t.array();
	ot_one_minus_ot = (1 - o_t.array() ) * o_t.array();

	// end = std::chrono::system_clock::now();
	// elapsed_seconds = end-start;
	// printf("First - first in seconds: %f \n", elapsed_seconds.count());
	// start = std::chrono::system_clock::now();

	//H TO N
	layer->softmax->computeGradient(h_t, output, dErr_h_t);
	runningErrH_t2n += dErr_h_t ;

	// end = std::chrono::system_clock::now();
	// elapsed_seconds = end-start;
	// printf("First - second in seconds: %f \n", elapsed_seconds.count());
	// start = std::chrono::system_clock::now();

	/*** dErr_c_t_to N ***/
	dErr_c_t = ( runningErrH_t2n.array() * (o_t.array() * one_minus_tanh_sq_ct.array()).transpose()).matrix();
	runningErrC_t2n += dErr_c_t;

	/*** dErr_o_t ***/
	dErr_o_t = (runningErrH_t2n.array() * tanh_ct.transpose().array()).matrix();

	/*** dErr_f_t ***/
	dErr_f_t = ( runningErrC_t2n.array() * prev_c.transpose().array() ).matrix();

	/*** dErr_g_t or dErr_tanh(c') ***/
	dErr_g_t = ( runningErrC_t2n.array() * i_t.transpose().array() ).matrix();

	/*** dErr_i_t ***/
	dErr_i_t = ( runningErrC_t2n.array() * tanh_prime.transpose().array() ).matrix();

	/*** dErr_h_t_1 ***/
	dErr_h_t_1.noalias()  =  (layer->Who.transpose() * (dErr_o_t.transpose().array() * ot_one_minus_ot.array() ).matrix()).transpose() \
	+   (layer->Whf.transpose() * ( dErr_f_t.transpose().array() * ft_one_minus_ft.array() ).matrix()).transpose() \
	+ (layer->Whi.transpose() * ( dErr_i_t.transpose().array() * it_one_minus_it.array() ).matrix()).transpose() \
	+ (layer->Whc.transpose() * ( dErr_g_t.transpose().array() * one_minus_tanh_sq_prime.array() ).matrix()).transpose();

	/*** dErr_c_t_1 ***/
	dErr_c_t_1 = runningErrC_t2n.array() * f_t.transpose().array();

	layer->dErr_Whi.noalias()  += ( prev_h * ( dErr_i_t.array() * (it_one_minus_it).transpose().array() ).matrix() ).transpose();
	layer->dErr_Whf.noalias()  += ( prev_h * ( dErr_f_t.array() * (ft_one_minus_ft).transpose().array() ).matrix() ).transpose();
	layer->dErr_Whc.noalias()  += ( prev_h * ( runningErrC_t2n.array() * ( i_t.array() * one_minus_tanh_sq_prime.array() ).transpose().array() ).matrix() ).transpose();
	layer->dErr_Who.noalias()  += ( prev_h * ( dErr_o_t.array() * (ot_one_minus_ot).transpose().array() ).matrix() ).transpose();

	compute_intermediate(layer->W);

	// end = std::chrono::system_clock::now();
	// elapsed_seconds = end-start;
	// printf("Second in seconds: %f \n", elapsed_seconds.count());
	// start = std::chrono::system_clock::now();

	layer->dErr_Mi.noalias()  += (dErr_i_t.transpose().array() * it_one_minus_it.array()).matrix() * intermediate.transpose();
	layer->dErr_Mf.noalias()  += (dErr_f_t.transpose().array() * ft_one_minus_ft.array()).matrix() * intermediate.transpose();
	layer->dErr_Mo.noalias()  += (dErr_o_t.transpose().array() * ot_one_minus_ot.array()).matrix() * intermediate.transpose();
	layer->dErr_Mc.noalias()  += (dErr_g_t.transpose().array() * one_minus_tanh_sq_prime.array()).matrix() * intermediate.transpose();

	layer->dErr_Bi.noalias()  += (( dErr_i_t.array() * it_one_minus_it.transpose().array() ).matrix().colwise().sum()).transpose();
	layer->dErr_Bf.noalias()  += (( dErr_f_t.array() * ft_one_minus_ft.transpose().array() ).matrix().colwise().sum()).transpose();
	layer->dErr_Bc.noalias()  += (( dErr_g_t.array() * one_minus_tanh_sq_prime.transpose().array() ).matrix().colwise().sum()).transpose();
	layer->dErr_Bo.noalias()  += (( dErr_o_t.array() * ot_one_minus_ot.transpose().array() ).matrix().colwise().sum()).transpose();


	Zi = dErr_i_t.array()*(it_one_minus_it).matrix().transpose().array();
	Zf = dErr_f_t.array()*(ft_one_minus_ft).matrix().transpose().array();
	Zo = dErr_o_t.array()*(ot_one_minus_ot).matrix().transpose().array();
	Zc = dErr_g_t.array()*(one_minus_tanh_sq_prime).matrix().transpose().array();

	for(int i=0; i<input.rows(); i++) {
		if(input(i)!=-1) {
			for(int j=0; j<layer->dErr_W.rows(); j++) {
				float sumtemp = Zi.row(i) * layer->Mi.col(j);
				sumtemp += Zf.row(i) * layer->Mf.col(j);
				sumtemp += Zo.row(i) * layer->Mo.col(j);
				sumtemp += Zc.row(i) * layer->Mc.col(j);
				layer->dErr_W(j, input(i)) += sumtemp;
			}
		}
	}


	// end = std::chrono::system_clock::now();
	// elapsed_seconds = end-start;
	// printf("Third in seconds: %f \n", elapsed_seconds.count());

}

void LSTM::forwardPropagation()
{

	#ifdef PEEPHOLE
		compute_intermediate( layer->Wxi );
	#else
		compute_intermediate( layer->W);
	#endif

	compute_i_t();

	#ifdef PEEPHOLE
		compute_intermediate( layer->Wxf );
	#endif


	compute_f_t();
	
	#ifdef PEEPHOLE
		compute_intermediate( layer->Wxc );
	#endif

	compute_g_t();
	compute_c_t();


	#ifdef PEEPHOLE
		compute_intermediate( layer->Wxo );
	#endif

	compute_o_t();
	compute_h_t();


	for(int i = 0; i < input.rows(); i++)
	{
		if(input(i) == -1)
		{
			h_t.col(i).setZero();
			c_t.col(i).setZero();
		}
	}	
}

LSTM::LSTM(int cell_size, int batchsize, Layer * layer)
{
	i_t.resize(cell_size, batchsize);
	f_t.resize(cell_size, batchsize);
	c_t.resize(cell_size, batchsize);
	o_t.resize(cell_size, batchsize);
	h_t.resize(cell_size, batchsize);
	g_t.resize(cell_size, batchsize);

	dErr_i_t.resize(batchsize, cell_size); // gradient wrt i_t
	dErr_f_t.resize(batchsize, cell_size); // gradient wrt f_t
	dErr_c_t.resize(batchsize, cell_size);// gradient wrt c_t
	dErr_o_t.resize(batchsize, cell_size);// gradient wrt o_t
	dErr_h_t.resize(batchsize, cell_size);// gradient wrt h_t
	dErr_g_t.resize(batchsize, cell_size); // gradient wrt g_t
	dErr_c_t_1.resize(batchsize, cell_size); // gradient wrt c_t - 1
	dErr_h_t_1.resize(batchsize, cell_size); // gradient wrt h_t - 1


	runningErrH_t2n.resize(batchsize, cell_size); // gradient wrt h_t from t to n
	runningErrC_t2n.resize(batchsize, cell_size); // gradient wrt c_t from t to n

	intermediate.resize(cell_size, batchsize);

	tanh_prime.resize(cell_size, batchsize);
	one_minus_tanh_sq_prime.resize(cell_size, batchsize);

	tanh_ct.resize(cell_size, batchsize);
	one_minus_tanh_sq_ct.resize(cell_size, batchsize);

	it_one_minus_it.resize(cell_size, batchsize);
	ft_one_minus_ft.resize(cell_size, batchsize);
	ot_one_minus_ot.resize(cell_size, batchsize);

	Zi.resize(batchsize, cell_size);
	Zf.resize(batchsize, cell_size);
	Zc.resize(batchsize, cell_size);
	Zo.resize(batchsize, cell_size);

	prev_h.resize(cell_size, batchsize);
	prev_c.resize(cell_size, batchsize);


	this->layer = layer;
}
