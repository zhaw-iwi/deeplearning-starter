package ch.zhaw.deeplearning4j;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
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
public class MainCNN {

	private static final Logger log = LoggerFactory.getLogger(MainCNN.class);

	public static void main(String[] args) throws IOException, InterruptedException {

		log.info("> Hello CNN :-)");

		int vectorSize = 300;
		int truncateTextToLength = 256; // Truncate reviews with length (# words) greater than this
		int cnnLayerFeatureMaps = 100; // Number of feature maps / channels / depth for each CNN layer
		int numberOfClasses = 2; // The number of possible outcomes
		PoolingType globalPoolingType = PoolingType.MAX;

		int nEpochs = 1; // Number of training epochs

		Nd4j.getMemoryManager().setAutoGcWindow(10000); // https://deeplearning4j.org/workspaces

		log.info("> Preparing Data ...");
		
		int batchSize = 32;

		// Load word vectors and get the DataSetIterators for training and testing
		WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(Paths.WORD_VECTORS_PATH));
		DataSetIterator trainDataIterator = getDataSetIterator(true, wordVectors, batchSize, truncateTextToLength);
		DataSetIterator testDataIterator = getDataSetIterator(false, wordVectors, batchSize, truncateTextToLength);

		log.info("> Building Model ...");

		ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder().weightInit(WeightInit.RELU)
				.activation(Activation.LEAKYRELU)
				.updater(new Adam(0.01))
				.convolutionMode(ConvolutionMode.Same) // This
														// is
														// important
														// so we
														// can
														// 'stack'
														// the
														// results
														// later
				.l2(0.0001)
				.graphBuilder()
				.addInputs("input")
				.addLayer("cnn3",
						new ConvolutionLayer.Builder().kernelSize(3, vectorSize)
								.stride(1, vectorSize)
								.nOut(cnnLayerFeatureMaps)
								.build(),
						"input")
				.addLayer("cnn4",
						new ConvolutionLayer.Builder().kernelSize(4, vectorSize)
								.stride(1, vectorSize)
								.nOut(cnnLayerFeatureMaps)
								.build(),
						"input")
				.addLayer("cnn5",
						new ConvolutionLayer.Builder().kernelSize(5, vectorSize)
								.stride(1, vectorSize)
								.nOut(cnnLayerFeatureMaps)
								.build(),
						"input")
				// MergeVertex performs depth concatenation on activations:
				// 3x[minibatch,100,length,300] to 1x[minibatch,300,length,300]
				.addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
				// Global pooling: pool over x/y locations (dimensions 2 and 3): Activations
				// [minibatch,300,length,300] to [minibatch, 300]
				.addLayer("globalPool",
						new GlobalPoolingLayer.Builder().poolingType(globalPoolingType).dropOut(0.5).build(), "merge")
				.addLayer("out",
						new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
								.activation(Activation.SOFTMAX)
								.nOut(numberOfClasses)
								.build(),
						"globalPool")
				.setOutputs("out")
				// Input has shape [minibatch, channels=1, length=1 to 256, 300]
				.setInputTypes(InputType.convolutional(truncateTextToLength, vectorSize, 1))
				.build();

		ComputationGraph model = new ComputationGraph(config);
		model.init();

		log.info("> Training & Testing Model ...");
		model.setListeners(new ScoreIterationListener(100),
				new EvaluativeListener(testDataIterator, 1, InvocationType.EPOCH_END));
		model.fit(trainDataIterator, nEpochs);

		log.info("> Testing Model ...");
		Evaluation eval = model.evaluate(testDataIterator);
		log.info(eval.stats());

		log.info("> Good Bye ;-(");

	}

	private static DataSetIterator getDataSetIterator(boolean isTraining, WordVectors wordVectors, int minibatchSize,
			int maxSentenceLength) throws IOException, InterruptedException {

		if (isTraining) {
			return new ClassifiedTextIterator4CNN.Builder(new String[] { "classifiedtextdata/lines-comedy_training.csv",
					"classifiedtextdata/lines-thriller_training.csv" }, new String[] { "comedy", "thriller" })
							.wordVectors(wordVectors)
							.minibatchSize(minibatchSize)
							.maxSentenceLength(maxSentenceLength)
							.build();
		} else {
			return new ClassifiedTextIterator4CNN.Builder(new String[] { "classifiedtextdata/lines-comedy_testing.csv",
					"classifiedtextdata/lines-thriller_testing.csv" }, new String[] { "comedy", "thriller" })
							.wordVectors(wordVectors)
							.minibatchSize(minibatchSize)
							.maxSentenceLength(maxSentenceLength)
							.build();
		}
	}
}
