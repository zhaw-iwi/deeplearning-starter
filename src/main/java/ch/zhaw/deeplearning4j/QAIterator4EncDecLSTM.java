package ch.zhaw.deeplearning4j;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
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
public class QAIterator4EncDecLSTM implements MultiDataSetIterator {

	private static final Logger log = LoggerFactory.getLogger(QAIterator4EncDecLSTM.class);

	// private static final String UNKNOWN_WORD_SENTINEL = "UNKNOWN_WORD";
	public static final String EMPTY_LINE_REPLACEMENT = "well";
	public static final String LINE_START = "say";
	public static final String LINE_END = "right";

	private final String pathToCSVFile;
	private final WordVectors wordVectors;
	private final int vectorSize;
	private final TokenizerFactory tokenizerFactory;

	private final int minibatchSize;
	private final int maxSentenceLength;

	private int cursor;
	private boolean done;
	private List<String> knownWords;
	private List<String> unknownWords;

	public QAIterator4EncDecLSTM(String pathToCSVFile, Builder builder) {

		this.pathToCSVFile = builder.pathToCSVFile;
		this.wordVectors = builder.wordVectors;
		this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
		this.tokenizerFactory = builder.tokenizerFactory;
		this.tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

		this.minibatchSize = builder.minibatchSize;
		this.maxSentenceLength = builder.maxSentenceLength;

		this.reset();
	}

	public static void main(String[] args) {

		Nd4j.getMemoryManager().setAutoGcWindow(5000);

		WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(Paths.WORD_VECTORS_PATH));

		MultiDataSetIterator it;
		MultiDataSet current;
		int count = 0;

		it = new QAIterator4EncDecLSTM.Builder("classifieddialoguepairs/dialoguepairs-comedy.csv")
				.wordVectors(wordVectors)
				.minibatchSize(32)
				.maxSentenceLength(200)
				.build();

		count = 0;
		while (it.hasNext()) {
			current = it.next();
			count++;
			System.out.println("MultiDataSet[" + count + "] " + current.getMemoryFootprint());
		}
	}

	private MultiDataSet nextDataSet(int numberOfExamples) throws Exception {

		CSVRecordReader reader = new CSVRecordReader();
		reader.initialize(new FileSplit(new File(pathToCSVFile)));

		// 1. Skip lines from previous batches
		int numberOfLinesReadPreviousBatches = 0;
		while (reader.hasNext() && numberOfLinesReadPreviousBatches < this.cursor) {
			reader.next();
			numberOfLinesReadPreviousBatches++;
		}
		// Reached end of file while skipping?
		if (!reader.hasNext()) {
			QAIterator4EncDecLSTM.log.warn(
					"QAIterator4EncDecLSTM.nextDataSet(int) reached the end of a file while TRYING TO SKIP LINES from previous batches");
			this.done = true;
		}

		// 2. Read lines for current batch and tokenise
		List<List<String>> qTokens = new ArrayList<List<String>>(numberOfExamples);
		List<List<String>> aTokens = new ArrayList<List<String>>(numberOfExamples);

		List<String> currentQTokens;
		List<String> currentATokens;
		int maxLength = 0;
		int numberOfLinesRead = 0;
		while (reader.hasNext() && numberOfLinesRead < numberOfExamples) {
			List<Writable> line = reader.next();

			currentQTokens = this.tokenizeSentence(line.get(0).toString());
			currentATokens = this.tokenizeSentence(line.get(1).toString());

			// TODO check for empty lines and handle these!!!
			if (currentQTokens.isEmpty()) {
				currentQTokens.add(EMPTY_LINE_REPLACEMENT);
			}
			if (currentATokens.isEmpty()) {
				currentATokens.add(EMPTY_LINE_REPLACEMENT);
			}

			// TODO remove this
			// System.out.println(currentQTokens + "\n\t" + currentATokens);
			maxLength = Math.max(maxLength, Math.max(currentQTokens.size(), currentATokens.size()));

			qTokens.add(numberOfLinesRead, currentQTokens);
			aTokens.add(numberOfLinesRead, currentATokens);

			numberOfLinesRead++;
		}
		// Reached end of file while reading batch?
		// Else: reached end of file when done reading batch?
		if (!reader.hasNext() && numberOfLinesRead < numberOfExamples) {
			QAIterator4EncDecLSTM.log.warn(
					"QAIterator4EncDecLSTM.nextDataSet(int) reached the end of a file while TRYING TO READ LINES from current batch");
			numberOfExamples = numberOfLinesRead;
			this.done = true;
		} else if (!reader.hasNext()) {
			QAIterator4EncDecLSTM.log.warn(
					"QAIterator4EncDecLSTM.nextDataSet(int) reached the end of a file and was able to read full batch of lines");
			this.done = true;
		}
		this.cursor += numberOfExamples;
		reader.close();

		// TODO possibly unnecessary but for now ...
		if (qTokens.size() != aTokens.size()) {
			throw new Exception("QAIterator4EncDecLSTM has got qTokens.size() !- aTokens.size() :-(");
		}

		// for truncation ...
		if (maxLength > this.maxSentenceLength) {
			maxLength = this.maxSentenceLength;
		}

		// 3. Map Tokens to Word Vectors, the put Vectors for input, prediction and
		// decode into NDArrays

		INDArray input = Nd4j.create(new int[] { numberOfExamples, this.vectorSize, maxLength }, 'f');
		INDArray prediction = Nd4j.create(new int[] { numberOfExamples, this.vectorSize, maxLength }, 'f');
		INDArray decode = Nd4j.create(new int[] { numberOfExamples, this.vectorSize, maxLength }, 'f');
		INDArray inputMask = Nd4j.zeros(numberOfExamples, maxLength);
		// this mask is also used for the decoder input, the length is the same
		INDArray predictionMask = Nd4j.zeros(numberOfExamples, maxLength);

		List<String> currentQTokenList, currentATokenList, currentDecoderTokenList;
		int sequenceLengthQ, sequenceLengthA;
		INDArray currentQVectors, currentAVectors, currentDecodeVectors;

		for (int i = 0; i < qTokens.size(); i++) {
			// ... for truncation
			currentQTokenList = qTokens.get(i);
			sequenceLengthQ = Math.min(maxLength, currentQTokenList.size());
			currentATokenList = aTokens.get(i);
			// put LINE_END at the end of A
			currentATokenList.add(LINE_END);
			sequenceLengthA = Math.min(maxLength, currentATokenList.size());

			// Decoder is A offset by 1, with "go" in the beginning and NOT the "eos" at the
			// end of prediction
			currentDecoderTokenList = new ArrayList<String>(sequenceLengthA);
			// put LINE_START at the beginning of decoder
			currentDecoderTokenList.add(0, LINE_START);
			for (int j = 0; j < sequenceLengthA; j++) {
				if (j < sequenceLengthA - 1) {
					currentDecoderTokenList.add(currentATokenList.get(j));
				}
			}

			// word vectors for Q and A ad Decoder

			// Q is reversed (see paper)
			List<String> reversedQTokenList = currentQTokenList.subList(0, sequenceLengthQ);
			Collections.reverse(reversedQTokenList);
			try {
				currentQVectors = this.wordVectors.getWordVectors(reversedQTokenList).transpose();
				currentAVectors = this.wordVectors.getWordVectors(currentATokenList.subList(0, sequenceLengthA))
						.transpose();
				currentDecodeVectors = this.wordVectors.getWordVectors(currentDecoderTokenList).transpose();
			} catch (IllegalStateException e) {
				// TODO this is for debugging purposes only
				// TODO HANDLE empty list in tokenization: replace unknown words with something
				// known
				throw e;
			}

			try {

				// Q goes into input
				input.put(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all(),
						NDArrayIndex.interval(0, sequenceLengthQ) }, currentQVectors);
				inputMask.put(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.interval(0, sequenceLengthQ) },
						Nd4j.ones(sequenceLengthQ));

				// A goes into prediction
				prediction.put(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all(),
						NDArrayIndex.interval(0, sequenceLengthA) }, currentAVectors);
				predictionMask.put(
						new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.interval(0, sequenceLengthA) },
						Nd4j.ones(sequenceLengthA));

				// Aaaaand the decoder
				decode.put(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.all(),
						NDArrayIndex.interval(0, sequenceLengthA) }, currentDecodeVectors);

			} catch (IllegalStateException e) {
				// TODO this is for debugging purposes only
				System.out.println(">>> " + e.getMessage() + "\n>>> " + currentQTokenList + "\n>>> " + currentATokenList
						+ "\n>>> " + currentDecoderTokenList);
				throw e;
			}
		}

		return new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] { input, decode }, new INDArray[] { prediction },
				new INDArray[] { inputMask, predictionMask }, new INDArray[] { predictionMask });
	}

	private List<String> tokenizeSentence(String sentence) {
		Tokenizer t = this.tokenizerFactory.create(sentence);

		List<String> result = new ArrayList<>();
		while (t.hasMoreTokens()) {
			String token = t.nextToken();
			if (!this.wordVectors.outOfVocabularySupported() && !this.wordVectors.hasWord(token)) {
				// TODO how to handle unknown words?
				// result.add(UNKNOWN_WORD_SENTINEL);
				this.unknownWords.add(token);
			} else {
				this.knownWords.add(token);
				result.add(token);
			}
		}
		return result;
	}

	@Override
	public boolean hasNext() {
		return !this.done;
	}

	@Override
	public MultiDataSet next() {
		return this.next(this.minibatchSize);
	}

	@Override
	public MultiDataSet next(int num) {
		if (!this.hasNext()) {
			throw new NoSuchElementException(
					"ClassifiedTextIterator.next(int) cannot return any more data given the number of lines available per class");
		}
		try {
			return this.nextDataSet(num);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public MultiDataSetPreProcessor getPreProcessor() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		return true;
	}

	@Override
	public void reset() {
		this.cursor = 0;
		this.done = false;
		this.knownWords = new ArrayList<String>();
		this.unknownWords = new ArrayList<String>();
	}

	public static class Builder {

		private WordVectors wordVectors;
		private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
		private int maxSentenceLength = -1;
		private int minibatchSize = 32;

		private String pathToCSVFile;

		public Builder(String pathToCSVFile) {
			this.pathToCSVFile = pathToCSVFile;
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
		 * Minibatch size to use for the DataSetIterator
		 */
		public Builder minibatchSize(int minibatchSize) {
			this.minibatchSize = minibatchSize;
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

		public QAIterator4EncDecLSTM build() {
			if (wordVectors == null) {
				throw new IllegalStateException(
						"Cannot build ClassifiedTextIterator4Rnn without a WordVectors instance");
			}

			return new QAIterator4EncDecLSTM(this.pathToCSVFile, this);
		}

	}

}
