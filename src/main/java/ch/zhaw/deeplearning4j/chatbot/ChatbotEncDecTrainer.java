package ch.zhaw.deeplearning4j.chatbot;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
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
public class ChatbotEncDecTrainer {

	private static final Logger log = LoggerFactory.getLogger(ChatbotEncDecTrainer.class);

	public static final String TRAINING_DATA_FILENAME = "classifieddialoguepairs/dialoguepairs-comedy.csv";
	public static final String MODEL_FILENAME = "classifieddialoguepairs/chatbotencdec.zip";

	private final File trainingDataFile;
	private final File modelFile;

	private final WordVectors wordVectors;
	private final TokenizerFactory tokenizerFactory;

	// Model Parameters
	private final int embeddingWidth; // this is defined by the WordVectors instance
	private final int hiddenLayerWidth;
	private final double learningRate;
	private final int tbttSize;

	// Training Parameters
	private final int truncateTextToLength;
	private final int batchSize;
	private final int nEpochs;

	private ComputationGraph model;

	public ChatbotEncDecTrainer(File trainingDataFile, File modelFile, Builder builder) {

		log.info("> Hello ChatbotEncDecTrainer :)");

		this.trainingDataFile = trainingDataFile;
		this.modelFile = modelFile;

		this.wordVectors = builder.wordVectors;
		this.tokenizerFactory = builder.tokenizerFactory;

		this.embeddingWidth = builder.wordVectors.getWordVector(builder.wordVectors.vocab().wordAtIndex(0)).length;
		this.hiddenLayerWidth = builder.hiddenLayerWidth;
		this.learningRate = builder.learningRate;
		this.tbttSize = builder.tbttSize;

		this.truncateTextToLength = builder.maxSentenceLength;
		this.batchSize = builder.minibatchSize;
		this.nEpochs = builder.epochs;
	}

	public void go() throws IOException {
		Nd4j.getMemoryManager().setAutoGcWindow(2000);
		this.buildModel();
		this.trainModel(this.trainingDataFile);
		this.saveModel(this.modelFile);
	}

	public static void main(String[] args) throws IOException {

		File trainingDataFile = new File(ChatbotEncDecTrainer.TRAINING_DATA_FILENAME);
		File modelFile = new File(ChatbotEncDecTrainer.MODEL_FILENAME);

		File wordVectorsFile = new File(ChatbotEncDecWordVectors.WORDVECTORS_FILENAME);
		WordVectors wordVectors = ChatbotEncDecWordVectors.load(wordVectorsFile);

		ChatbotEncDecTrainer trainer = new ChatbotEncDecTrainer.Builder(trainingDataFile, modelFile)
				.wordVectors(wordVectors)
				.build();

		trainer.go();
	}

	private void buildModel() {

		log.info("> Building Model ...");

		final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.updater(new RmsProp(learningRate))
				.weightInit(WeightInit.XAVIER)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

		final GraphBuilder graphBuilder = builder.graphBuilder()
				.backpropType(BackpropType.Standard)
				.tBPTTBackwardLength(tbttSize)
				.tBPTTForwardLength(tbttSize)
				.addInputs("encoderInput", "decoderInput")
				// TODO .setInputTypes(InputType.recurrent(dict.size()),
				// InputType.recurrent(dict.size()))
				.addLayer("encoder",
						new LSTM.Builder().nIn(embeddingWidth)
								.nOut(hiddenLayerWidth)
								.activation(Activation.TANH)
								.build(),
						"encoderInput")
				.addVertex("thoughtVector", new LastTimeStepVertex("encoderInput"), "encoder")
				.addVertex("dup", new DuplicateToTimeSeriesVertex("decoderInput"), "thoughtVector")
				.addVertex("merge", new MergeVertex(), "decoderInput", "dup")
				.addLayer("decoder",
						new LSTM.Builder().nIn(embeddingWidth + hiddenLayerWidth)
								.nOut(hiddenLayerWidth)
								.activation(Activation.TANH)
								.build(),
						"merge")
				.addLayer("output",
						new RnnOutputLayer.Builder().nIn(hiddenLayerWidth)
								.nOut(embeddingWidth)
								.activation(Activation.SOFTMAX)
								.lossFunction(LossFunctions.LossFunction.MCXENT)
								.build(),
						"decoder")
				.setOutputs("output");

		this.model = new ComputationGraph(graphBuilder.build());
		this.model.init();

		log.info("> Building Model DONE");
	}

	private void trainModel(File trainingDataFile) throws IOException {

		log.info("> Training Model ...");

		MultiDataSetIterator trainingData = new QAIterator4EncDecLSTM.Builder(trainingDataFile)
				.wordVectors(this.wordVectors)
				.tokenizerFactory(this.tokenizerFactory)
				.minibatchSize(this.batchSize)
				.maxSentenceLength(this.truncateTextToLength)
				.build();

		this.model.addListeners(new ScoreIterationListener(100));
		this.model.fit(trainingData, this.nEpochs);

		log.info("> Training Model DONE");
	}

	private void saveModel(File modelFile) throws IOException {

		log.info("> Saving Model ... to " + modelFile.getAbsolutePath());

		ModelSerializer.writeModel(this.model, modelFile, true);

		log.info("> Saving Model DONE");

	}

	public static class Builder {

		private WordVectors wordVectors;
		private TokenizerFactory tokenizerFactory;
		private int maxSentenceLength = 256;
		private int minibatchSize = 32;
		private int epochs = 1;

		private int hiddenLayerWidth = 512;
		private double learningRate = 1e-1;
		private int tbttSize = 25;

		private File trainingDataFile;
		private File modelFile;

		public Builder(File trainingDataFile, File modelFile) {
			this.trainingDataFile = trainingDataFile;
			this.modelFile = modelFile;
			this.tokenizerFactory = new DefaultTokenizerFactory();
			this.tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
		}

		/**
		 * Provide the WordVectors instance that should be used for training
		 */
		public Builder wordVectors(WordVectors wordVectors) {
			this.wordVectors = wordVectors;
			return this;
		}

		/**
		 * The {@link TokenizerFactory} that should be used. Defaults to
		 * {@link DefaultTokenizerFactory}
		 */
		public Builder tokenizerFactory(TokenizerFactory tokenizerFactory) {
			this.tokenizerFactory = tokenizerFactory;
			return this;
		}

		/**
		 * Maximum sentence/document length. If sentences exceed this, they will be
		 * truncated to this length by taking the first 'maxSentenceLength' known words.
		 */
		public Builder maxSentenceLength(int maxSentenceLength) {
			this.maxSentenceLength = maxSentenceLength;
			return this;
		}

		/**
		 * Minibatch size to use for the bot
		 */
		public Builder minibatchSize(int minibatchSize) {
			this.minibatchSize = minibatchSize;
			return this;
		}

		/**
		 * Number of epochs the bot will be trained. Defaults to 1.
		 */
		public Builder epochs(int epochs) {
			this.epochs = epochs;
			return this;
		}

		public Builder hiddenLayerWidth(int hiddenLayerWidth) {
			this.hiddenLayerWidth = hiddenLayerWidth;
			return this;
		}

		public Builder learningRate(double learningRate) {
			this.learningRate = learningRate;
			return this;
		}

		public Builder tbttSize(int tbttSize) {
			this.tbttSize = tbttSize;
			return this;
		}

		public ChatbotEncDecTrainer build() {
			if (wordVectors == null) {
				throw new IllegalStateException("Cannot build ChatbotEncDecTrainer without a WordVectors instance");
			}

			return new ChatbotEncDecTrainer(this.trainingDataFile, this.modelFile, this);
		}

	}
}
