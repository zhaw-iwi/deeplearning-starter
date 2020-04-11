package ch.zambolid;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
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
public class Main {

	private static final Logger log = LoggerFactory.getLogger(Main.class);

	public static void main(String[] args) throws IOException, InterruptedException {

		log.info("> Hello Deeplearning4j :-)");

		int vectorSize = 5;
		int numberOfClasses = 5; // The number of possible outcomes
		int indexOfLabelColumn = vectorSize; // if vector has n features, then label is at column n

		int neuronsPerHidenLayer = 10; // Number of feature maps / channels / depth for each CNN layer
		int nEpochs = 1000; // Number of training epochs
		int seed = 123;

		log.info("> Preparing Data ...");

		DataSet trainData = Main.getDataSetFromCSV("randomdataTraining.csv", 0, 50000, indexOfLabelColumn,
				numberOfClasses);
		DataSet testData = Main.getDataSetFromCSV("randomdataTesting.csv", 0, 5000, indexOfLabelColumn,
				numberOfClasses);

		// We need to normalize our data. We'll use NormalizeStandardize (which gives us
		// mean 0, unit variance):
		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(trainData); // Collect the statistics (mean/stdev) from the training data. This does not
									// modify the input data
		normalizer.transform(trainData); // Apply normalization to the training data
		normalizer.transform(testData); // Apply normalization to the test data. This is using statistics
										// calculated from the *training* set

		log.info("> Building Model ...");

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.activation(Activation.SOFTMAX)
				.weightInit(WeightInit.XAVIER)
				.updater(new Adam(0.01))
				.l2(0.0005)
				.list()
				.layer(new DenseLayer.Builder().nIn(vectorSize).nOut(neuronsPerHidenLayer).build())
				.layer(new DenseLayer.Builder().nIn(neuronsPerHidenLayer).nOut(neuronsPerHidenLayer).build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX)
						.nIn(neuronsPerHidenLayer)
						.nOut(numberOfClasses)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		log.info("> Training Model ...");
		model.setListeners(new ScoreIterationListener(10)); // Print score every 10 iterations
		for (int i = 0; i < nEpochs; i++) {
			model.fit(trainData);
			// log.info("*** Completed epoch {} ***", i);
		}

		log.info("> Testing Model ...");

		Evaluation eval = new Evaluation(numberOfClasses);
		INDArray output = model.output(testData.getFeatures());

		eval.eval(testData.getLabels(), output);
		log.info(eval.stats());

		log.info("> Good Bye ;-(");

	}

	public static DataSet getDataSetFromCSV(String pathToFile, int rowOffset, int batchSize, int indexLabel,
			int numOfClasses) throws IOException, InterruptedException {
		RecordReader rr = new CSVRecordReader(rowOffset);
		rr.initialize(new FileSplit(new File(pathToFile)));
		RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, indexLabel, numOfClasses);
		return iterator.next();
	}
}
