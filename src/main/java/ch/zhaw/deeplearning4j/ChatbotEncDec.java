package ch.zhaw.deeplearning4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Layer;
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
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ChatbotEncDec {

	private static final Logger log = LoggerFactory.getLogger(ChatbotEncDec.class);
	private static final String MODEL_FILENAME = "chatbotencdec.zip"; // filename of the model
	private static final String BACKUP_MODEL_FILENAME = "chatbotencdec.bak.zip";

	private final WordVectors wordVectors;
	private final int vectorSize;

	// Model Parameters
	private double learningRate = 1e-1;
	private int tbttSize = 25;
	private int embeddingWidth = 300;
	private int hiddenLayerWidth = 512;

	// Training Parameters
	int truncateTextToLength = 256;
	int nEpochs = 1;
	int batchSize = 32;

	// Using Parameters
	private TokenizerFactory tokenizerFactory;

	private ComputationGraph model;

	public static void main(String[] args) throws IOException {
		ChatbotEncDec bot = new ChatbotEncDec(true);
		System.out.println(bot.getAforQ("how are you feeling?", 3));
	}

	public ChatbotEncDec(boolean createNew) throws IOException {

		log.info("> Hello ChatbotEncDec :)");

		Nd4j.getMemoryManager().setAutoGcWindow(2000);

		log.info("> Loading Word Vectors ...");
		this.wordVectors = WordVectorSerializer.loadStaticModel(new File(Paths.WORD_VECTORS_PATH));
		this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
		log.info("> Loading Word Vectors DONE");
		this.tokenizerFactory = new DefaultTokenizerFactory();
		this.tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

		// starting either creating and training or using the model

		File networkFile = new File(toTempPath(MODEL_FILENAME));

		if (createNew) {
			this.model = this.buildModel();
			this.trainModel(networkFile);
			this.saveModel(networkFile);
		} else {
			this.model = this.loadModel(networkFile);
		}
	}

	public List<Collection<String>> getAforQ(String q, int top) {

		// 1. Tokenize
		// q
		List<String> qTokenList = this.tokenizeSentence(q);
		if (qTokenList.isEmpty()) {
			qTokenList.add(QAIterator4EncDecLSTM.EMPTY_LINE_REPLACEMENT);
		}
		// q is reversed
		Collections.reverse(qTokenList);

		// decode only contains LINE_START words
		List<String> decodeTokenList = this.tokenizeSentence(QAIterator4EncDecLSTM.LINE_START);

		// TODO maybe truncate Q?

		// 2. Map Tokens to Word Vectors
		INDArray qVectors, decodeVectors;
		qVectors = this.wordVectors.getWordVectors(qTokenList).transpose();
		decodeVectors = this.wordVectors.getWordVectors(decodeTokenList).transpose();

		// 3. Put everything into NDArrays
		INDArray input = Nd4j.create(new int[] { 1, this.vectorSize, qTokenList.size() }, 'f');
		input.put(new INDArrayIndex[] { NDArrayIndex.point(1), NDArrayIndex.all(),
				NDArrayIndex.interval(0, qTokenList.size()) }, qVectors);

		INDArray decode = Nd4j.create(new int[] { 1, this.vectorSize, 1 }, 'f');
		decode.put(new INDArrayIndex[] { NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.point(1) },
				decodeVectors);

		// feed forward
		this.model.rnnClearPreviousState();
		this.model.feedForward(new INDArray[] { input, decode }, false, false);

		// get some of the layers/vertices
		org.deeplearning4j.nn.layers.recurrent.LSTM decoder = (org.deeplearning4j.nn.layers.recurrent.LSTM) this.model
				.getLayer("decoder");
		Layer output = this.model.getLayer("output");
		GraphVertex mergeVertex = this.model.getVertex("merge");
		INDArray thoughtVector = mergeVertex.getInputs()[1];
		LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();

		List<INDArray> aVectors = new ArrayList<INDArray>();
		// TODO change this "40" (max length of answer?)
		for (int nOfWords = 0; nOfWords < 40; nOfWords++) {
			mergeVertex.setInputs(decode, thoughtVector);
			INDArray merged = mergeVertex.doForward(false, mgr);
			INDArray activateDec = decoder.rnnTimeStep(merged, mgr);
			INDArray out = output.activate(activateDec, false, mgr);

			INDArray wordVector = out.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.point(1));
			aVectors.add(wordVector);

			if (this.wordVectors.wordsNearest(wordVector, 1)
					.iterator()
					.next()
					.equalsIgnoreCase(QAIterator4EncDecLSTM.LINE_END)) {
				break;
			}

			// set next decode
			decode.put(new INDArrayIndex[] { NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.point(1) },
					wordVector);

		}

		return this.getMostProbableText(aVectors, top);
	}

	private List<Collection<String>> getMostProbableText(List<INDArray> wordVectors, int top) {
		List<Collection<String>> result = new ArrayList<Collection<String>>(wordVectors.size());
		for (INDArray current : wordVectors) {
			result.add(this.wordVectors.wordsNearest(current, top));
		}
		return result;
	}

	private List<String> tokenizeSentence(String sentence) {
		Tokenizer t = this.tokenizerFactory.create(sentence);

		List<String> result = new ArrayList<>();
		while (t.hasMoreTokens()) {
			String token = t.nextToken();
			if (!this.wordVectors.outOfVocabularySupported() && !this.wordVectors.hasWord(token)) {
				// TODO how to handle unknown words?
				// result.add(UNKNOWN_WORD_SENTINEL);
			} else {
				result.add(token);
			}
		}
		return result;
	}

	private ComputationGraph buildModel() {

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

		ComputationGraph model = new ComputationGraph(graphBuilder.build());
		model.init();

		log.info("> Building Model DONE");

		return model;
	}

	private void trainModel(File networkFile) throws IOException {

		log.info("> Training Model ...");

		log.info("> Preparing Data ...");

		MultiDataSetIterator trainingData = new QAIterator4EncDecLSTM.Builder(
				"classifieddialoguepairs/dialoguepairs-comedy.csv").wordVectors(this.wordVectors)
						.minibatchSize(this.batchSize)
						.maxSentenceLength(this.truncateTextToLength)
						.build();

		log.info("> Training Model START");

		this.model.addListeners(new ScoreIterationListener(100));
		this.model.fit(trainingData, this.nEpochs);

		log.info("> Training Model DONE");
	}

	private void saveModel(File networkFile) throws IOException {

		log.info("> Saving Model ... to " + networkFile.getAbsolutePath());

		File backup = new File(toTempPath(BACKUP_MODEL_FILENAME));
		if (networkFile.exists()) {
			if (backup.exists()) {
				backup.delete();
			}
			networkFile.renameTo(backup);
		}
		ModelSerializer.writeModel(this.model, networkFile, true);

		log.info("> Saving Model DONE");

	}

	private ComputationGraph loadModel(File networkFile) throws IOException {

		log.info("> Loading Model ...");

		ComputationGraph result = ComputationGraph.load(networkFile, true);

		log.info("> Loading Model DONE");

		return result;
	}

	private String toTempPath(String path) {
		return System.getProperty("java.io.tmpdir") + "/" + path;
	}

}
