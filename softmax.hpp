	

	void Softmax::initSoftmax(int outputVocabSize, int hiddenSize, int batchSize, Layer * layer)
	{

		this->config = layer->config;

		D.resize(outputVocabSize, hiddenSize);
		initMatrix(D, 1.0);

		dErr_D.resize(outputVocabSize, hiddenSize);
		dErr_D.setZero(outputVocabSize, hiddenSize);

		Bk.resize(outputVocabSize, 1);
		initMatrix(Bk, 1.0);

		dErr_Bk.resize(outputVocabSize, 1);
		dErr_Bk.setZero(outputVocabSize, 1);

		softmax_denominator.resize(1, batchSize);
		softmax_denominator.setZero(1, batchSize);

		prob.resize(outputVocabSize, batchSize);
		prob.setZero(outputVocabSize, batchSize);

		this->hiddenSize = hiddenSize;
		this->batchSize = batchSize;

		this->layer = layer;
	}

	template<typename Derived>
	void Softmax::initMatrix(const Eigen::MatrixBase<Derived> &input_const, type range) {
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

	template<typename Derived, typename Derived2, typename Derived3> 
	void Softmax::computeGradient(const Eigen::MatrixBase<Derived> &h_t, const Eigen::MatrixBase<Derived2> &output, const Eigen::MatrixBase<Derived3> &dErr_h_t_const)
	{
		UNCONST(Derived3, dErr_h_t_const, dErr_h_t);
		dErr_h_t.setZero();

		// std::chrono::time_point<std::chrono::system_clock> start, end;
		// std::chrono::duration<type> elapsed_seconds;
		// start = std::chrono::system_clock::now();

		computeProbability(h_t);

		// start = std::chrono::system_clock::now();


		/*** H_T GRADIENT ***/
		for(int j=0; j<dErr_h_t.rows(); j++) {
			if(output(j)!=-1) {
				for(int i=0; i<D.rows(); i++) {
					dErr_h_t.row(j) -= (prob(i,j))*(D.row(i));
				}
			}
		}

		// end = std::chrono::system_clock::now();
		// elapsed_seconds = end-start;
		// printf("softmax: 2nd in seconds: %f \n", elapsed_seconds.count());
		// start = std::chrono::system_clock::now();

		//Set intitial vector, for error (Dk transpose in gradient sheet)
		for(int i=0; i<dErr_h_t.rows(); i++) {
			if(output(i)!=-1) {
				dErr_h_t.row(i) += D.row(output(i));
			}
		}


		// end = std::chrono::system_clock::now();
		// elapsed_seconds = end-start;
		// printf("softmax: 3rd in seconds: %f \n", elapsed_seconds.count());
		// start = std::chrono::system_clock::now();

		/*** D GRADIENT ***/
		dErr_D += -1 * prob * h_t.transpose();

		for(int i = 0; i < h_t.cols(); i++)
		{
			if(output(i) != -1)
				dErr_D.row(output(i)) += h_t.col(i).transpose(); 
		}


		// end = std::chrono::system_clock::now();
		// elapsed_seconds = end-start;
		// printf("softmax: fourth in seconds: %f \n", elapsed_seconds.count());
		// start = std::chrono::system_clock::now();

		/*** Bk GRADIENT ***/
		for(int i = 0; i < output.rows(); i++)
		{
			if( output(i) != -1){
				dErr_Bk(output(i)) += 1;
				dErr_Bk += -1 * prob.col(i);

			}
		}


		// end = std::chrono::system_clock::now();
		// elapsed_seconds = end-start;
		// printf("softmax: fifth in seconds: %f \n", elapsed_seconds.count());

	}


	template<typename Derived, typename Derived2> 
	type Softmax::computeError(const Eigen::MatrixBase<Derived> &h_t, const Eigen::MatrixBase<Derived2> &output){
		
		computeProbability(h_t);

		type error = 0.0;

		for(int i = 0; i < output.rows(); i++)
		{
			if(output(i) != -1)
				error += std::log( prob(output(i), i) );
		}

		return error;
	}


	void Softmax::resetDenominator()
	{
		softmax_denominator.setZero(1, batchSize);
	}


	template<typename Derived>
	void Softmax::computeProbability(const Eigen::MatrixBase<Derived> &h_t)
	{
		prob = ( (D*h_t).colwise() + Bk).unaryExpr(exp_functor()).matrix();
		
		softmax_denominator = prob.colwise().sum();

		for(int i = 0; i < prob.rows(); i++)
			prob.row(i) = prob.row(i).array() / softmax_denominator.array();
	}

	void Softmax::applyGradients()
	{

		type gt = config.gradThres;
		type alpha = config.alpha;

		dErr_D = dErr_D / config.batchSize;
		dErr_Bk = dErr_Bk / config.batchSize;

		clipGradients(dErr_D, gt);
		clipGradients(dErr_Bk, gt);

		D += alpha * dErr_D;
		Bk += alpha * dErr_Bk;
	}

	template<typename Derived>
	void Softmax::checkGradient(const Eigen::MatrixBase<Derived> &mat_const, const Eigen::MatrixBase<Derived> &dErr_mat_const, std::string mat_name, type ep)
	{
		UNCONST(Derived, mat_const, mat);
		UNCONST(Derived, dErr_mat_const, dErr_mat);

		for(int i = 0; i < mat.rows() ; i++)
		{
			for(int j = 0; j < mat.cols(); j++)
			{
				type orig = mat(i, j);

				mat(i,j) = orig + ep;
				type top_list = layer->getError();
				mat(i,j) = orig - ep;
				
				type low_list = layer->getError();
				type calcGrad = (top_list - low_list) / ( 2 * ep);
				type actualGrad = dErr_mat(i, j);

				type diff = std::abs( calcGrad - actualGrad );

				if( diff > .00001 )
					std::cout << mat_name << "(" << i << ", " << j << "): " << std::abs( calcGrad - actualGrad ) << std::endl;

				mat(i, j) = orig;
			}
		}
	}

	template<typename Derived>
	void Softmax::clipGradients(const Eigen::MatrixBase<Derived> &gradient_const, type gradThres)
	{
		UNCONST(Derived, gradient_const, gradient);

		type matrixNorm = std::sqrt( (gradient.array() * gradient.array()).sum() );

		if( matrixNorm > gradThres )
		{
			gradient = gradient * gradThres / matrixNorm;
		}
	}

	void Softmax::checkGradients(type ep)
	{

		checkGradient(D, dErr_D, "D", ep);
		checkGradient(Bk, dErr_Bk, "Bk", ep);

	}

	void Softmax::clearGradients()
	{
		dErr_D.setZero(dErr_D.rows(), dErr_D.cols());
		dErr_Bk.setZero(dErr_Bk.rows(), dErr_Bk.cols());
	}

	void Softmax::dumpMatrices()
	{
		std::ofstream ofile("Softmax_Weights.txt");

		for(int i = 0; i < D.rows(); i++)
		{
			for(int j = 0; j < D.cols(); j++)
			{
				ofile << D(i,j)  << " ";
			}
		}

		ofile << "\n";

		for(int i = 0; i < Bk.rows(); i++)
		{
			for(int j = 0; j < Bk.cols(); j++)
			{
				ofile << Bk(i,j)  << " ";
			}
		}

		ofile.close();

	}

	void Softmax::loadWeights(std::string filename)
	{

		std::ifstream ifile(filename);

		std::string line;
		getline(ifile, line);
		std::stringstream ss(line);

		for(int i = 0; i < D.rows(); i++)
		{
			for(int j = 0; j < D.cols(); j++)
			{
				ss >> D(i,j);
			}
		}

		ss.clear();
		getline(ifile, line);
		ss.str(line);

		for(int i = 0; i < Bk.rows(); i++)
		{
			for(int j = 0; j < Bk.cols(); j++)
			{
				ss >> Bk(i,j);
			}
		}

		ifile.close();

	}

	void Softmax::printMatrices()
	{
		std::cout << "\n\n D Matrix \n\n";
		for(int i = 0; i < D.rows(); i++)
		{
			for(int j = 0; j < D.cols(); j++)
			{
				std::cout << D(i,j)  << " ";
			}
			std::cout << "\n";
		}
		
		std::cout << "\n\n Bk Matrix \n\n";

		for(int i = 0; i < Bk.rows(); i++)
		{
			for(int j = 0; j < Bk.cols(); j++)
			{
				std::cout << Bk(i,j)  << " ";
			}
			std::cout << "\n";
		}

	}


