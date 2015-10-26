#ifndef FILE_READER_H
#define FILE_READER_H

#include <fstream>
#include <sstream>
#include <iostream>
#include "Eigen/Dense"

class FileReader
{

public:
	FileReader(std::string filename, int batchSize)
	{
		this->filename = filename;
		this->batchSize = batchSize;

		ifile.open(filename);

	}

	~FileReader()
	{
		ifile.close();
	}

	void resetFile()
	{
		ifile.close();
		ifile.open(filename);
	}

	void readNextBatch()
	{

		numWordsInBatch = 0;

		std::vector<std::vector<int>> inp_vectors;
		std::vector<std::vector<int>> out_vectors;

		int max_len = 0;

		for(int i = 0; i < batchSize; i++)
		{

			// std::cout << "here";

			std::vector<int> inputs;
			std::vector<int> outputs;

			std::string input_line;
			std::string output_line;

			std::getline( ifile, input_line );
			std::getline( ifile, output_line );

			// std::cout << input_line << std::endl;
			// std::cout << output_line << std::endl;

			std::stringstream si(input_line);
			std::stringstream so(output_line);

			while( si.good() )
			{
				int val;
				si >> val;
				inputs.push_back(val);
			}

			while( so.good() )
			{
				int val;
				so >> val;
				outputs.push_back(val);
			}

			inp_vectors.push_back(inputs);
			out_vectors.push_back(outputs);

			if(inputs.size() > max_len)
			{
				max_len = inputs.size();
			}
		}

		// std::cout << inp_vectors.size() << "\n";
		// std::cout << inp_vectors[0].size() << "\n";
		// std::cout << inp_vectors[1].size() << "\n";

		// std::cout << out_vectors.size() << "\n";
		// std::cout << out_vectors[0].size() << "\n";
		// std::cout << out_vectors[1].size() << "\n";

		inputMiniBatch.resize(batchSize, max_len);
		outputMiniBatch.resize(batchSize, max_len);

		for(int i = 0; i < batchSize; i++)
		{
			int input_size = inp_vectors[i].size();
			int output_size = out_vectors[i].size();

			for(int j = 0; j < max_len; j++)
			{
				if( j < input_size){
					inputMiniBatch(i,j) = inp_vectors[i][j];
					outputMiniBatch(i,j) = out_vectors[i][j];
					numWordsInBatch++;
				}
				else{
					inputMiniBatch(i,j) = -1;
					outputMiniBatch(i,j) = -1;
				}
			}
		}

		// std::cout << "**** Inputs ****" << std::endl;
		// std::cout << inputMiniBatch << std::endl;

		// std::cout << std::endl << "**** Outputs ****" << std::endl;
		// std::cout << outputMiniBatch << "\n\n";


		max_length = max_len;
	}

	int max_length;

	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> inputMiniBatch;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> outputMiniBatch;

	int numWordsInBatch;


private:
	std::string filename;
	int batchSize;
	std::ifstream ifile;
};

#endif

