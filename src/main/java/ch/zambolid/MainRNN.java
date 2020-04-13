package ch.zambolid;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Zurich University of Applied Sciences (ZHAW), Institute for Business
 * Information Systems (IWI), Center for Information Systems and Technologies
 * (CIS)
 * 
 * @author Alexandre de Spindler (desa@zhaw.ch)
 * 
 */
public class MainRNN {

	private static final Logger log = LoggerFactory.getLogger(MainRNN.class);

	public static void main(String[] args) throws IOException, InterruptedException {

		log.info("> Hello RNN :-)");

		int vectorSize = 300;
		int truncateTextToLength = 256; // Truncate reviews with length (# words) greater than this
		int numberOfClasses = 2; // The number of possible outcomes

		int nEpochs = 1; // Number of training epochs
		int seed = 123;

		Nd4j.getMemoryManager().setAutoGcWindow(10000); // https://deeplearning4j.org/workspaces

		log.info("> Preparing Data ...");

		int batchSize = 32;

		WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(Paths.WORD_VECTORS_PATH));
		DataSetIterator trainData = getDataSetIterator(true, wordVectors, batchSize, truncateTextToLength);
		DataSetIterator testData = getDataSetIterator(false, wordVectors, batchSize, truncateTextToLength);

		log.info("> Building Model ...");

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.updater(new Adam(5e-3))
				.l2(1e-5)
				.weightInit(WeightInit.XAVIER)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(1.0)
				.list()
				.layer(new LSTM.Builder().nIn(vectorSize).nOut(256).activation(Activation.TANH).build())
				.layer(new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
						.lossFunction(LossFunctions.LossFunction.MCXENT)
						.nIn(256)
						.nOut(numberOfClasses)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		log.info("> Training & Testing Model ...");
		model.setListeners(new ScoreIterationListener(100),
				new EvaluativeListener(testData, 1, InvocationType.EPOCH_END));
		model.fit(trainData, nEpochs);

		// log.info("> Applying Model ...");
		// TODO if needed

		log.info("> Good Bye ;-(");

	}

	private static DataSetIterator getDataSetIterator(boolean isTraining, WordVectors wordVectors, int minibatchSize,
			int maxSentenceLength) throws IOException, InterruptedException {

		if (isTraining) {
			return new ClassifiedTextIterator4RNN(
					new String[] { "classifiedtextdata/lines-comedy_training.csv",
							"classifiedtextdata/lines-thriller_training.csv" },
					69908, new String[] { "comedy", "thriller" }, wordVectors, minibatchSize, maxSentenceLength);
		} else {
			return new ClassifiedTextIterator4RNN(
					new String[] { "classifiedtextdata/lines-comedy_testing.csv",
							"classifiedtextdata/lines-thriller_testing.csv" },
					69908, new String[] { "comedy", "thriller" }, wordVectors, minibatchSize, maxSentenceLength);
		}
	}
}
