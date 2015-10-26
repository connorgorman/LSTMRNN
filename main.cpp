typedef float type;

#include "fileReader.h"
#include "config.h"

#include "layer.h"
#include "LSTM.h"
#include "softmax.h"

#include "layer.hpp"
#include "LSTM.hpp"
#include "softmax.hpp"

#include <math.h>
#include <chrono>
#include <ctime>

type computePerplexity()
{

	type perplexity = 0.0;
	int numWords = 0;
	
	Config p_config;
	Layer p_layer(p_config);

	FileReader p_fr(p_config.test_file, p_config.batchSize);

	Softmax p_softmax;

	p_layer.softmax = &p_softmax;
	p_softmax.initSoftmax(p_config.outputVocabSize, p_config.hiddenSize, p_config.batchSize, &p_layer);

	p_layer.loadWeights("Layer_Weights.txt");
	p_softmax.loadWeights("Softmax_Weights.txt");

	for(int i = 0;  i < p_config.numTestBatches; i++)
	{
		p_fr.readNextBatch();
		p_layer.initLSTMVector(p_fr);
		
		perplexity += p_layer.getError();
		numWords += p_fr.numWordsInBatch;

		p_layer.clearRunningErrors();
		p_layer.clearGradients();
		p_softmax.clearGradients();
	}

	perplexity = perplexity/std::log(2.0); //Change to base 2 log
	perplexity = std::pow(2,-1*perplexity/numWords);

	std::cout << "PERPLEXITY: " << perplexity << std::endl;
	return perplexity;
}

int main(int argc, char* argv[])
{

	Config config;
	Layer l(config);

	FileReader fr(config.train_file, config.batchSize);

	Softmax softmax;
	softmax.initSoftmax(config.outputVocabSize, config.hiddenSize, config.batchSize, &l);

	l.softmax = &softmax;

	type prevPerplexity = 10000000;

	if(config.loadWeights)
	{
		l.loadWeights("Layer_Weights.txt");
		softmax.loadWeights("Softmax_Weights.txt");

		if(config.verbose)
		{
			softmax.printMatrices();
			l.printMatrices();
		}
	}

	if(config.trainModel)
	{

		for(int i = 0; i < config.numPasses; i++)
		{
			std::chrono::time_point<std::chrono::system_clock> start, end, batchStart, batchEnd;
		    start = std::chrono::system_clock::now();
			std::cout << "PASS NUMBER: " << i << std::endl;
			fr.resetFile();


			for(int j = 0; j < config.numTrainBatches; j++)
			{
				if( j % 10 == 0){
					batchEnd = std::chrono::system_clock::now();
					std::chrono::duration<type> elapsed_seconds = batchEnd-batchStart;
					printf("Time Taken %f \n", elapsed_seconds.count());
					std::cout << "Batch number: " << j << std::endl;
					batchStart = std::chrono::system_clock::now();
				}

				fr.readNextBatch();
				l.initLSTMVector(fr);
				
				l.forwardPropagation(fr.inputMiniBatch);
				l.backwardPropagation(fr.outputMiniBatch);

				if(config.checkGradients){
					l.checkGradients(config.ep);
					softmax.checkGradients(config.ep);
				}

				softmax.applyGradients();
				l.applyGradients();

				l.clearRunningErrors();
				l.clearGradients();
				softmax.clearGradients();

				if( j % 100 == 0 && j != 0){
					if( config.getPerplexity)
					{
						type perplexity = 0.0;
						int numWords = 0;

						FileReader p_fr(config.test_file, config.batchSize);
						for (int i = 0; i < config.numTestBatches; i++){
							p_fr.readNextBatch();
							l.initLSTMVector(p_fr);
							perplexity += l.getError();
							l.clearRunningErrors();
							l.clearGradients();
							softmax.clearGradients();
							numWords +=p_fr.numWordsInBatch;
						}

						perplexity = perplexity/std::log(2.0); //Change to base 2 log
						perplexity = std::pow(2,-1*perplexity/numWords);
						std::cout << "PERPLEXITY: " << perplexity << std::endl;

					}
				}

			}

			end = std::chrono::system_clock::now();
	 
		    std::chrono::duration<type> elapsed_seconds = end-start;
		    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
		 
		   std::cout << "Finished Computing 1 Pass in " << elapsed_seconds.count() << "s\n";

		    l.dumpMatrices();

			softmax.dumpMatrices();

		}

		if(config.verbose)
		{
			l.printMatrices();
			softmax.printMatrices();
		}

	}

	return 0;
}