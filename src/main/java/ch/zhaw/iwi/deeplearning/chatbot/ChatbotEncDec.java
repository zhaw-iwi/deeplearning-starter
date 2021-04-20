package ch.zhaw.iwi.deeplearning.chatbot;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ChatbotEncDec {

	private static final Logger log = LoggerFactory.getLogger(ChatbotEncDec.class);

	private final WordVectors wordVectors;
	private final TokenizerFactory tokenizerFactory;
	private final int embeddingWidth;

	private ComputationGraph model;

	public ChatbotEncDec(File modelFile, Builder builder) throws IOException {

		log.info("> Hello ChatbotEncDec :)");

		this.wordVectors = builder.wordVectors;
		this.embeddingWidth = builder.wordVectors.getWordVector(builder.wordVectors.vocab().wordAtIndex(0)).length;
		this.tokenizerFactory = builder.tokenizerFactory;

		this.loadModel(modelFile);
	}

	public static void main(String[] args) throws IOException {
		File modelFile = new File(ChatbotEncDecTrainer.MODEL_FILENAME);
		
		File wordVectorsFile = new File(ChatbotEncDecWordVectors.WORDVECTORS_FILENAME);
		WordVectors wordVectors = ChatbotEncDecWordVectors.load(wordVectorsFile);

		ChatbotEncDec bot = new ChatbotEncDec.Builder(modelFile).wordVectors(wordVectors).build();

		Scanner scanner = new Scanner(System.in);
		System.out.println("Q > ");
		String q = scanner.nextLine();
		List<Collection<String>> a;
		while (!q.equalsIgnoreCase("exit")) {

			a = bot.getAforQ(q, 3);
			System.out.println("A > " + a);

			System.out.println("Q > ");
			q = scanner.nextLine();
		}
	}

	private void loadModel(File networkFile) throws IOException {

		log.info("> Loading Model ...");
		this.model = ComputationGraph.load(networkFile, true);
		log.info("> Loading Model DONE");
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
		INDArray input = Nd4j.create(new int[] { 1, this.embeddingWidth, qTokenList.size() }, 'f');
		input.put(new INDArrayIndex[] { NDArrayIndex.point(1), NDArrayIndex.all(),
				NDArrayIndex.interval(0, qTokenList.size()) }, qVectors);

		INDArray decode = Nd4j.create(new int[] { 1, this.embeddingWidth, 1 }, 'f');
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

	public static class Builder {

		private WordVectors wordVectors;
		private TokenizerFactory tokenizerFactory;

		private File modelFile;

		public Builder(File modelFile) {
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

		public ChatbotEncDec build() throws IOException {
			if (wordVectors == null) {
				throw new IllegalStateException("Cannot build ChatbotEncDecTrainer without a WordVectors instance");
			}

			return new ChatbotEncDec(this.modelFile, this);
		}

	}

}
