// #include "config.h"
// #include "Eigen_Util.h"
// #include <boost/random/uniform_real.hpp>
// #include <iostream>
// #include "layer.h"

Layer::Layer(Config c)
{
	this->config = c;
	type uniform_rng = config.uniform_rng;
	int inputVocabSize = config.inputVocabSize;
	int outputVocabSize = config.outputVocabSize;
	int hiddenSize = config.hiddenSize;
	int batchSize = config.batchSize;

	#ifdef PEEPHOLE
	
	#else

	Mi.resize(hiddenSize, hiddenSize);
	initMatrix(Mi, uniform_rng);

	Mf.resize(hiddenSize, hiddenSize);
	initMatrix(Mf, uniform_rng);

	Mc.resize(hiddenSize, hiddenSize);
	initMatrix(Mc, uniform_rng);

	Mo.resize(hiddenSize, hiddenSize);
	initMatrix(Mo, uniform_rng);

	W.resize(hiddenSize, inputVocabSize);
	initMatrix(W, uniform_rng);

	#endif

	Whi.resize(hiddenSize, hiddenSize);
	initMatrix(Whi, uniform_rng);

	Bi.resize(hiddenSize, 1);
	initMatrix(Bi, uniform_rng);


	Whf.resize(hiddenSize, hiddenSize); 
	initMatrix(Whf, uniform_rng);

	Bf.resize(hiddenSize, 1);
	initMatrix(Bf, uniform_rng);


	Whc.resize(hiddenSize, hiddenSize);
	initMatrix(Whc, uniform_rng);

	Bc.resize(hiddenSize, 1);
	initMatrix(Bc, uniform_rng);


	Who.resize(hiddenSize, hiddenSize);
	initMatrix(Who, uniform_rng);

	Bo.resize(hiddenSize, 1);
	initMatrix(Bo, uniform_rng);


	first_hidden.resize(hiddenSize, batchSize);
	first_hidden.setZero(hiddenSize, batchSize);

	first_cell.resize(hiddenSize, batchSize);
	first_cell.setZero(hiddenSize, batchSize);

	runningErrH.resize(batchSize, hiddenSize);
	runningErrH.setZero(batchSize, hiddenSize);

	runningErrC.resize(batchSize, hiddenSize);
	runningErrC.setZero(batchSize, hiddenSize);


	#ifdef PEEPHOLE

	#else

	dErr_Mi.resize(hiddenSize, hiddenSize);
	dErr_Mi.setZero(hiddenSize, hiddenSize);

	dErr_Mf.resize(hiddenSize, hiddenSize);
	dErr_Mf.setZero(hiddenSize, hiddenSize);

	dErr_Mc.resize(hiddenSize, hiddenSize);
	dErr_Mc.setZero(hiddenSize, hiddenSize);

	dErr_Mo.resize(hiddenSize, hiddenSize);
	dErr_Mo.setZero(hiddenSize, hiddenSize);

	dErr_W.resize(hiddenSize, inputVocabSize);
	dErr_W.setZero(hiddenSize, inputVocabSize);


	#endif


	dErr_Whi.resize(hiddenSize, hiddenSize);
	dErr_Whi.setZero(hiddenSize, hiddenSize);

	dErr_Bi.resize(hiddenSize, 1);
	dErr_Bi.setZero(hiddenSize, 1);

	dErr_Whf.resize(hiddenSize, hiddenSize); 
	dErr_Whf.setZero(hiddenSize, hiddenSize); 

	dErr_Bf.resize(hiddenSize, 1);
	dErr_Bf.setZero(hiddenSize, 1);

	dErr_Whc.resize(hiddenSize, hiddenSize);
	dErr_Whc.setZero(hiddenSize, hiddenSize);

	dErr_Bc.resize(hiddenSize, 1);
	dErr_Bc.setZero(hiddenSize, 1);


	dErr_Who.resize(hiddenSize, hiddenSize);
	dErr_Who.setZero(hiddenSize, hiddenSize);

	dErr_Bo.resize(hiddenSize, 1);
	dErr_Bo.setZero(hiddenSize, 1);

}

template<typename Derived>
void Layer::backwardPropagation(const Eigen::MatrixBase<Derived> &outputMiniBatch)
{
	clearRunningErrors();

	LSTMs[LSTMs.size()-1].setupBackwardProp(outputMiniBatch, LSTMs.size() - 1);
	LSTMs[LSTMs.size()-1].backwardPropagation(runningErrH, runningErrC);

	for(int i = LSTMs.size() - 2; i > -1; i--)
	{
		LSTMs[i].setupBackwardProp(outputMiniBatch, i);
		LSTMs[i].backwardPropagation( LSTMs[i+1].dErr_h_t_1, LSTMs[i+1].dErr_c_t_1);
	}
	
}

void Layer::clearRunningErrors()
{
	runningErrH.setZero();
	runningErrC.setZero();
}

template<typename Derived>
void Layer::forwardPropagation(const Eigen::MatrixBase<Derived> &inputMiniBatch)
{

	LSTMs[0].setupForwardProp(first_hidden, first_cell, inputMiniBatch, 0);
	LSTMs[0].forwardPropagation();

	for(int i = 1; i < inputMiniBatch.cols(); i++)
	{
		LSTMs[i].setupForwardProp( LSTMs[i-1].h_t, LSTMs[i-1].c_t, inputMiniBatch, i);
		LSTMs[i].forwardPropagation();
	}
}

void Layer::initLSTMVector(FileReader &fr)
{
	LSTMs.clear();

	this->inputMiniBatch = fr.inputMiniBatch;
	this->outputMiniBatch = fr.outputMiniBatch;

	for(int i = 0; i < fr.max_length; i++)
	{
		LSTM lstm = LSTM( config.hiddenSize, config.batchSize, this );
		lstm.input = fr.inputMiniBatch.col(i);
		LSTMs.push_back(lstm);
	}
}

template<typename Derived>
void Layer::initMatrix(const Eigen::MatrixBase<Derived> &input_const, type range) {
	UNCONST(Derived, input_const, input);
	boost::uniform_real<> ud(-range, range);

	for(int i = 0; i < input.cols(); i++)
	{
		for(int j = 0; j < input.rows(); j++)
		{
			input(j, i) = ud(gen);
		}
	}
}

template<typename Derived>
void Layer::checkGradient(const Eigen::MatrixBase<Derived> &mat_const, const Eigen::MatrixBase<Derived> &dErr_mat_const, std::string mat_name, type ep)
{
	UNCONST(Derived, mat_const, mat);
	UNCONST(Derived, dErr_mat_const, dErr_mat);

	for(int i = 0; i < mat.rows() ; i++)
	{
		for(int j = 0; j < mat.cols(); j++)
		{
			type orig = mat(i, j);

			mat(i,j) = orig + ep;
			type top_list = getError();
			mat(i,j) = orig - ep;
			type low_list = getError();

			type calcGrad = (top_list - low_list) / ( 2 * ep);
			type actualGrad = dErr_mat(i, j);

			type diff = std::abs( calcGrad - actualGrad );

			if( diff > .0001 )
				std::cout << mat_name << "(" << i << ", " << j << "): " << std::abs( calcGrad - actualGrad ) << std::endl;

			mat(i, j) = orig;
		}
	}
}

void Layer::checkGradients(type ep)
{

	checkGradient(Whi, dErr_Whi, "Whi", ep);
	checkGradient(Whf, dErr_Whf, "Whf", ep);
	checkGradient(Whc, dErr_Whc, "Whc", ep);
	checkGradient(Who, dErr_Who, "Who", ep);


	checkGradient(Mi, dErr_Mi, "Mi", ep);
	checkGradient(Mf, dErr_Mf, "Mf", ep);
	checkGradient(Mo, dErr_Mo, "Mo", ep);
	checkGradient(Mc, dErr_Mc, "Mc", ep);

	checkGradient(W, dErr_W, "W", ep);

	checkGradient(Bi, dErr_Bi, "Bi", ep);
	checkGradient(Bf, dErr_Bf, "Bf", ep);
	checkGradient(Bc, dErr_Bc, "Bc", ep);
	checkGradient(Bo, dErr_Bo, "Bo", ep);

}

type Layer::getError()
{
	/*** Run forward prop ***/
	type error = 0.0;

	LSTMs[0].setupForwardProp(first_hidden, first_cell, inputMiniBatch, 0);
	LSTMs[0].forwardPropagation();

	error += softmax->computeError(LSTMs[0].h_t, outputMiniBatch.col(0));

	for(int i = 1; i < inputMiniBatch.cols(); i++)
	{
		LSTMs[i].setupForwardProp( LSTMs[i-1].h_t, LSTMs[i-1].c_t, inputMiniBatch, i);
		LSTMs[i].forwardPropagation();

		error += softmax->computeError(LSTMs[i].h_t, outputMiniBatch.col(i));
	}

	return error;
}

void Layer::applyGradients()
{

	type gt = config.gradThres;
	type alpha = config.alpha;
	int batchSize = config.batchSize;

	dErr_Mi = dErr_Mi / batchSize;
	dErr_Mf = dErr_Mf / batchSize;
	dErr_Mc = dErr_Mc / batchSize;
	dErr_Mo = dErr_Mo / batchSize;

	dErr_W = dErr_W / batchSize;

	dErr_Whi = dErr_Whi / batchSize;
	dErr_Bi = dErr_Bi / batchSize;

	dErr_Whf = dErr_Whf / batchSize;
	dErr_Bf = dErr_Bf / batchSize;

	dErr_Whc = dErr_Whc / batchSize;
	dErr_Bc = dErr_Bc / batchSize;

	dErr_Who = dErr_Who / batchSize;
	dErr_Bo = dErr_Bo / batchSize;


	clipGradients(dErr_Mi, gt);
	clipGradients(dErr_Mf, gt);
	clipGradients(dErr_Mc, gt);
	clipGradients(dErr_Mo, gt);

	clipGradients(dErr_W, gt);

	clipGradients(dErr_Whi, gt);
	clipGradients(dErr_Bi, gt);

	clipGradients(dErr_Whf, gt);
	clipGradients(dErr_Bf, gt);

	clipGradients(dErr_Whc, gt);
	clipGradients(dErr_Bc, gt);

	clipGradients(dErr_Who, gt);
	clipGradients(dErr_Bo, gt);


	Mi += alpha * dErr_Mi;
	Mf += alpha * dErr_Mf;
	Mc += alpha * dErr_Mc;
	Mo += alpha * dErr_Mo;

	W += alpha * dErr_W;

	Whi += alpha * dErr_Whi;
	Bi += alpha * dErr_Bi;

	Whf += alpha * dErr_Whf;
	Bf += alpha * dErr_Bf;

	Whc += alpha * dErr_Whc;
	Bc += alpha * dErr_Bc;

	Who += alpha * dErr_Who;
	Bo += alpha * dErr_Bo;

}

template<typename Derived>
void Layer::clipGradients(const Eigen::MatrixBase<Derived> &gradient_const, type gradThres)
{
	UNCONST(Derived, gradient_const, gradient);

	type matrixNorm = std::sqrt( (gradient.array() * gradient.array()).sum() );

	if( matrixNorm > gradThres )
	{
		gradient = gradient * gradThres / matrixNorm;
	}
}

void Layer::clearGradients()
{

	dErr_Mi.setZero(dErr_Mi.rows(), dErr_Mi.cols());
	dErr_Mf.setZero(dErr_Mf.rows(), dErr_Mf.cols());
	dErr_Mo.setZero(dErr_Mo.rows(), dErr_Mo.cols());
	dErr_Mc.setZero(dErr_Mc.rows(), dErr_Mc.cols());

	dErr_W.setZero(dErr_W.rows(), dErr_W.cols());

	dErr_Whi.setZero(dErr_Whi.rows(), dErr_Whi.cols());
	dErr_Bi.setZero(dErr_Bi.rows(), dErr_Bi.cols());

	dErr_Whf.setZero(dErr_Whf.rows(), dErr_Whf.cols());
	dErr_Bf.setZero(dErr_Bf.rows(), dErr_Bf.cols());

	dErr_Whc.setZero(dErr_Whc.rows(), dErr_Whc.cols());
	dErr_Bc.setZero(dErr_Bc.rows(), dErr_Bc.cols());

	dErr_Who.setZero(dErr_Who.rows(), dErr_Who.cols());
	dErr_Bo.setZero(dErr_Bo.rows(), dErr_Bo.cols());
}

void Layer::dumpMatrices()
{

	std::ofstream ofile("Layer_Weights.txt");
	for(int i = 0; i < Mi.rows(); i++ )
	{
		for(int j = 0; j < Mi.cols(); j++){
			ofile << Mi(i,j) << " ";
		}
	}

	ofile << "\n";

	for(int i = 0; i < Mf.rows(); i++ )
	{
		for(int j = 0; j < Mf.cols(); j++){
			ofile << Mf(i,j) << " ";
		}
	}

	ofile << "\n";

	for(int i = 0; i < Mo.rows(); i++ )
	{
		for(int j = 0; j < Mo.cols(); j++){
			ofile << Mo(i,j) << " ";
		}
	}
	
	ofile << "\n";


	for(int i = 0; i < Mc.rows(); i++ )
	{
		for(int j = 0; j < Mc.cols(); j++){
			ofile << Mc(i,j) << " ";
		}
	}

	ofile << "\n";


	for(int i = 0; i < W.rows(); i++ )
	{
		for(int j = 0; j < W.cols(); j++){
			ofile << W(i,j) << " ";
		}
	}

	ofile << "\n";
	


	for(int i = 0; i < Whi.rows(); i++ )
	{
		for(int j = 0; j < Whi.cols(); j++){
			ofile << Whi(i,j) << " ";
		}
	}

	ofile << "\n";

	for(int i = 0; i < Bi.rows(); i++ )
	{
		for(int j = 0; j < Bi.cols(); j++){
			ofile << Bi(i,j) << " ";
		}
	}

	ofile << "\n";
	
	

	for(int i = 0; i < Whf.rows(); i++ )
	{
		for(int j = 0; j < Whf.cols(); j++){
			ofile << Whf(i,j) << " ";
		}
	}

	ofile << "\n";

	for(int i = 0; i < Bf.rows(); i++ )
	{
		for(int j = 0; j < Bf.cols(); j++){
			ofile << Bf(i,j) << " ";
		}
	}

	ofile << "\n";
	
	

	for(int i = 0; i < Whc.rows(); i++ )
	{
		for(int j = 0; j < Whc.cols(); j++){
			ofile << Whc(i,j) << " ";
		}
	}

	ofile << "\n";

	for(int i = 0; i < Bc.rows(); i++ )
	{
		for(int j = 0; j < Bc.cols(); j++){
			ofile << Bc(i,j) << " ";
		}
	}

	ofile << "\n";

	

	for(int i = 0; i < Who.rows(); i++ )
	{
		for(int j = 0; j < Who.cols(); j++){
			ofile << Who(i,j) << " ";
		}
	}

	ofile << "\n";

	for(int i = 0; i < Bo.rows(); i++ )
	{
		for(int j = 0; j < Bo.cols(); j++){
			ofile << Bo(i,j) << " ";
		}
	}

	ofile.close();

}


void Layer::loadWeights(std::string filename)
{

	std::ifstream ifile(filename);

	std::string line;
	getline(ifile, line);
	std::stringstream ss(line);

	for(int i = 0; i < Mi.rows(); i++ )
	{
		for(int j = 0; j < Mi.cols(); j++){
			ss >> Mi(i,j);
		}
	}

	ss.clear();
	getline(ifile, line);
	ss.str(line);

	for(int i = 0; i < Mf.rows(); i++ )
	{
		for(int j = 0; j < Mf.cols(); j++){
			ss >> Mf(i,j);
		}
	}

	ss.clear();
	getline(ifile, line);
	ss.str(line);

	for(int i = 0; i < Mo.rows(); i++ )
	{
		for(int j = 0; j < Mo.cols(); j++){
			ss >> Mo(i,j);
		}
	}
	
		ss.clear();
		getline(ifile, line);
		ss.str(line);

	for(int i = 0; i < Mc.rows(); i++ )
	{
		for(int j = 0; j < Mc.cols(); j++){
			ss >> Mc(i,j);
		}
	}

		ss.clear();
		getline(ifile, line);
		ss.str(line);

	for(int i = 0; i < W.rows(); i++ )
	{
		for(int j = 0; j < W.cols(); j++){
			ss >> W(i,j);
		}
	}

		ss.clear();
		getline(ifile, line);
		ss.str(line);	


	for(int i = 0; i < Whi.rows(); i++ )
	{
		for(int j = 0; j < Whi.cols(); j++){
			ss >> Whi(i,j);
		}
	}

		ss.clear();
		getline(ifile, line);
		ss.str(line);

	for(int i = 0; i < Bi.rows(); i++ )
	{
		for(int j = 0; j < Bi.cols(); j++){
			ss >> Bi(i,j);
		}
	}

		ss.clear();
		getline(ifile, line);
		ss.str(line);	
	

	for(int i = 0; i < Whf.rows(); i++ )
	{
		for(int j = 0; j < Whf.cols(); j++){
			ss >> Whf(i,j);
		}
	}

		ss.clear();
		getline(ifile, line);
		ss.str(line);

	for(int i = 0; i < Bf.rows(); i++ )
	{
		for(int j = 0; j < Bf.cols(); j++){
			ss >> Bf(i,j);
		}
	}

		ss.clear();
		getline(ifile, line);
		ss.str(line);	
	

	for(int i = 0; i < Whc.rows(); i++ )
	{
		for(int j = 0; j < Whc.cols(); j++){
			ss >> Whc(i,j);
		}
	}

		ss.clear();
		getline(ifile, line);
		ss.str(line);

	for(int i = 0; i < Bc.rows(); i++ )
	{
		for(int j = 0; j < Bc.cols(); j++){
			ss >> Bc(i,j);
		}
	}

		ss.clear();
		getline(ifile, line);
		ss.str(line);
	

	for(int i = 0; i < Who.rows(); i++ )
	{
		for(int j = 0; j < Who.cols(); j++){
			ss >> Who(i,j);
		}
	}

		ss.clear();
		getline(ifile, line);
		ss.str(line);

	for(int i = 0; i < Bo.rows(); i++ )
	{
		for(int j = 0; j < Bo.cols(); j++){
			ss >> Bo(i,j);
		}
	}

	ifile.close();

}

void Layer::printMatrices()
{

	std::cout << "\n\n MI \n\n";

	for(int i = 0; i < Mi.rows(); i++ )
	{
		for(int j = 0; j < Mi.cols(); j++){
			std::cout << Mi(i,j) << " ";
		}
		std::cout << "\n";

	}

	std::cout << "\n\n MF \n\n";

	for(int i = 0; i < Mf.rows(); i++ )
	{
		for(int j = 0; j < Mf.cols(); j++){
			std::cout << Mf(i,j) << " ";
		}
	std::cout << "\n";

	}

	std::cout << "\n\n MO \n\n";


	for(int i = 0; i < Mo.rows(); i++ )
	{
		for(int j = 0; j < Mo.cols(); j++){
			std::cout << Mo(i,j) << " ";
		}
	std::cout << "\n";

	}
	
	std::cout << "\n\n MC \n\n";

	for(int i = 0; i < Mc.rows(); i++ )
	{
		for(int j = 0; j < Mc.cols(); j++){
			std::cout << Mc(i,j) << " ";
		}
	std::cout << "\n";

	}

	std::cout << "\n\n W \n\n";


	for(int i = 0; i < W.rows(); i++ )
	{
		for(int j = 0; j < W.cols(); j++){
			std::cout << W(i,j) << " ";
		}
	std::cout << "\n";

	}

	std::cout << "\n\n WHI \n\n";
	


	for(int i = 0; i < Whi.rows(); i++ )
	{
		for(int j = 0; j < Whi.cols(); j++){
			std::cout << Whi(i,j) << " ";
		}
	std::cout << "\n";

	}

	std::cout << "\n\n BI \n\n";

	for(int i = 0; i < Bi.rows(); i++ )
	{
		for(int j = 0; j < Bi.cols(); j++){
			std::cout << Bi(i,j) << " ";
		}
	std::cout << "\n";

	}

	std::cout << "\n\n WHF \n\n";
	

	for(int i = 0; i < Whf.rows(); i++ )
	{
		for(int j = 0; j < Whf.cols(); j++){
			std::cout << Whf(i,j) << " ";
		}
	std::cout << "\n";

	}

	std::cout << "\n\n BF \n\n";

	for(int i = 0; i < Bf.rows(); i++ )
	{
		for(int j = 0; j < Bf.cols(); j++){
			std::cout << Bf(i,j) << " ";
		}
	std::cout << "\n";

	}

	std::cout << "\n\n WHC \n\n";
	
	

	for(int i = 0; i < Whc.rows(); i++ )
	{
		for(int j = 0; j < Whc.cols(); j++){
			std::cout << Whc(i,j) << " ";
		}
	std::cout << "\n";

	}

	std::cout << "\n\n BC \n\n";

	for(int i = 0; i < Bc.rows(); i++ )
	{
		for(int j = 0; j < Bc.cols(); j++){
			std::cout << Bc(i,j) << " ";
		}
	std::cout << "\n";

	}

	std::cout << "\n\n WHO \n\n";

	

	for(int i = 0; i < Who.rows(); i++ )
	{
		for(int j = 0; j < Who.cols(); j++){
			std::cout << Who(i,j) << " ";
		}
	std::cout << "\n";

	}

	std::cout << "\n\n BO \n\n";

	for(int i = 0; i < Bo.rows(); i++ )
	{
		for(int j = 0; j < Bo.cols(); j++){
			std::cout << Bo(i,j) << " ";
		}
	std::cout << "\n";

	}

}


#ifdef PEEPHOLE
template<typename Derived>
void Layer::initDiagonal(const Eigen::DiagonalBase<Derived> &input_const, type range) {
	UNCONST_DIAG(Derived,input_const,input);
	boost::uniform_real<> ud(-range, range);
	for(int i=0; i<input.diagonal().size(); i++) {
		input.diagonal()(i) = ud(gen);
	}
}
#endif

