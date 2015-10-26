#ifndef CONFIG_H
#define CONFIG_H


class Config
{
public:
	static constexpr type uniform_rng = 0.01;
	int inputVocabSize = 10001;
	int outputVocabSize = 10001;

	int numPasses = 20;

	int hiddenSize = 200;

	type ep = .00001;

	type gradThres = 1;

	type alpha = .7;

	bool verbose = false;

	bool loadWeights = false;
	bool getPerplexity = true;
	bool trainModel = true;

	int num_train_sentences = 84120 / 2;
	int num_test_sentences = 7520 / 2;

	int batchSize = 20;
	bool checkGradients = false;

	// std::string train_file = "ptb.train.txt.sorted.integerized";
	std::string train_file = "ptb.train.txt.sorted.integerized";
	int numTrainBatches = num_train_sentences / batchSize;

	// std::string test_file = "ptb.test.txt.sorted.integerized";
	std::string test_file = "ptb.test.txt.sorted.integerized";
	int numTestBatches = num_test_sentences / batchSize;


};

#endif

