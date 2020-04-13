package ch.zambolid;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
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
public class ClassifiedTextIterator4RNN implements DataSetIterator {
	public enum UnknownWordHandling {
		RemoveWord, UseUnknownVector
	}

	private static final Logger log = LoggerFactory.getLogger(ClassifiedTextIterator4RNN.class);

	private final String[] pathsToCSVFilePerClass;
	private final int smallestNumberOfLines;
	private final String[] labels;
	private final int numberOfClasses;

	private final WordVectors wordVectors;
	private final int vectorSize;

	private final int minibatchSize;
	private final int maxSentenceLength;

	private final TokenizerFactory tokenizerFactory;

	private int cursor;

	/**
	 * @param pathsToCSVFilePerClass the CSV file for each class containing one line
	 *                               per text. The length of this list is the number
	 *                               of classes. The first will be class = 0, the
	 *                               second class = 1 and the last one class =
	 *                               (number of classes -1)
	 * @param smallestNumberOfLines  the number of lines of the file with the least
	 *                               number of lines
	 * @param labels                 will be matched with pathsToCSVFilesPerClass
	 *                               along position
	 * 
	 * @throws InterruptedException
	 * @throws IOException
	 */
	public ClassifiedTextIterator4RNN(String[] pathsToCSVFilePerClass, int smallestNumberOfLines, String[] labels,
			Builder builder) throws IOException, InterruptedException {

		this.wordVectors = builder.wordVectors;
		this.minibatchSize = builder.minibatchSize;
		this.maxSentenceLength = builder.maxSentenceLength;

		this.pathsToCSVFilePerClass = pathsToCSVFilePerClass;
		this.smallestNumberOfLines = smallestNumberOfLines;
		this.labels = labels;
		this.numberOfClasses = this.pathsToCSVFilePerClass.length;

		this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

		this.tokenizerFactory = builder.tokenizerFactory;
		this.tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor()); // TODO StemmingPreprocessor?

		this.reset();
	}

	/**
	 * This is for testing purposes only! There is no other use in running this from
	 * here...
	 * 
	 * @param args
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public static void main(String[] args) throws IOException, InterruptedException {

		Nd4j.getMemoryManager().setAutoGcWindow(5000);

		WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(Paths.WORD_VECTORS_PATH));
		DataSetIterator it = new ClassifiedTextIterator4RNN.Builder(
				new String[] { "classifiedtextdata/lines-comedy_training.csv",
						"classifiedtextdata/lines-thriller_training.csv" },
				69908, new String[] { "comedy", "thriller" }).wordVectors(wordVectors)
						.minibatchSize(32)
						.maxSentenceLength(200)
						.build();

		DataSet dataSet = it.next();
		System.out.println(dataSet.getFeatures());
	}

	public int totalExamples() {
		return this.smallestNumberOfLines * this.numberOfClasses;
	}

	/**
	 * @param numberOfExamples make sure numberOfExamples % numberOfClasses == 0 so
	 *                         we can return the same number of examples per class
	 * @return
	 */
	private DataSet nextDataSet(int numberOfExamples) throws Exception {

		if (numberOfExamples % this.numberOfClasses != 0) {
			throw new IllegalArgumentException(
					"ClassifiedTextIterator.nextDataSet(int) cannot read an equal amount of examples per class with the given numberOfExamples and numberOfClasses");
		}
		int numPerClass = numberOfExamples / this.numberOfClasses;
		int cursorPerClass = this.cursor / this.numberOfClasses;

		// 1 Read the (numberOfExamples / numberOfClasses) lines per class
		List<List<String>> nLinesPerClass = new ArrayList<List<String>>(this.numberOfClasses);

		SentenceIterator sentenceIt;
		for (String currentCSV : this.pathsToCSVFilePerClass) {

			sentenceIt = new LineSentenceIterator(new File(currentCSV));

			// skip lines < this.cursor
			int linesSkipped = 0;
			while (linesSkipped < cursorPerClass && sentenceIt.hasNext()) {
				sentenceIt.nextSentence();
				linesSkipped++;
			}
			if (!sentenceIt.hasNext()) {
				throw new Exception(
						"ClassifiedTextIterator.nextDataSet(int) is trying skip more lines than available in file (next() despite hasNext() == null?)");
			}

			List<String> nLines = new ArrayList<String>(numPerClass);
			int linesRead = 0;
			while (linesRead < numPerClass && sentenceIt.hasNext()) {
				nLines.add(sentenceIt.nextSentence());
				linesRead++;
			}
			nLinesPerClass.add(nLines);
			if (linesRead < numPerClass && !sentenceIt.hasNext()) {
				// throw new Exception(
				// "ClassifiedTextIterator.nextDataSet() was unable to read (numberOfExamples /
				// numberOfClasses) of lines because less lines are left in file");
				ClassifiedTextIterator4RNN.log.error(
						"ClassifiedTextIterator.nextDataSet(int) was unable to read (numberOfExamples / numberOfClasses) of lines because less lines are left in file (probably in last batch?)");
				numPerClass = linesRead;
			}
		}

		// 2 Tokenize all lines per class
		List<List<List<String>>> allTokensPerClass = new ArrayList<List<List<String>>>(this.numberOfClasses);
		List<List<String>> allTokensForOneClass;

		List<String> currentTokens;
		int maxLength = 0;

		for (List<String> linesForOneClass : nLinesPerClass) {
			allTokensForOneClass = new ArrayList<List<String>>(numPerClass);
			for (String currentLine : linesForOneClass) {
				currentTokens = this.tokenizeSentence(currentLine);

				if (currentTokens.isEmpty()) {
					// TODO handle this!
					ClassifiedTextIterator4RNN.log
							.error("Line \"" + currentLine + "\" is left empty after WordVectors.hasWord()");
					ClassifiedTextIterator4RNN.log
							.warn("Line \"" + currentLine + "\" is replaced with word \"this be the\"");
					currentTokens.add("this");
					currentTokens.add("be");
					currentTokens.add("the");
				}

				allTokensForOneClass.add(currentTokens);
				maxLength = Math.max(maxLength, currentTokens.size());
			}
			allTokensPerClass.add(allTokensForOneClass);
		}

		if (maxLength > maxSentenceLength) {
			maxLength = maxSentenceLength;
		}

		// 3 Map Tokens to Word Vectors, then put Word Vectors and Labels into NDArrays
		INDArray features = Nd4j.create(new int[] { numberOfExamples, vectorSize, maxLength }, 'f');
		INDArray labels = Nd4j.create(new int[] { numberOfExamples, numberOfClasses, maxLength }, 'f');

		INDArray featuresMask = Nd4j.zeros(numberOfExamples, maxLength);
		INDArray labelsMask = Nd4j.zeros(numberOfExamples, maxLength);

		List<String> currentTokenList;
		int codeForCurrentClass;
		int point;

		int currentSequenceLength;
		INDArray currentVectors;
		int currentIndexForLabel;
		int currentLastIndex;
		for (int i = 0; i < numPerClass; i++) {
			codeForCurrentClass = 0;
			for (List<List<String>> tokensForOneClass : allTokensPerClass) {
				currentTokenList = tokensForOneClass.get(i);
				currentSequenceLength = Math.min(currentTokenList.size(), maxLength);

				try {
					currentVectors = this.wordVectors.getWordVectors(currentTokenList.subList(0, currentSequenceLength))
							.transpose();
				} catch (IllegalStateException e) {
					throw e;
				}

				point = i * this.numberOfClasses + codeForCurrentClass;

				features.put(new INDArrayIndex[] { NDArrayIndex.point(point), NDArrayIndex.all(),
						NDArrayIndex.interval(0, currentSequenceLength) }, currentVectors);
				featuresMask.get(new INDArrayIndex[] { NDArrayIndex.point(point),
						NDArrayIndex.interval(0, currentSequenceLength) }).assign(1);

				currentIndexForLabel = codeForCurrentClass;
				currentLastIndex = Math.min(currentTokenList.size(), maxLength);
				labels.putScalar(new int[] { point, currentIndexForLabel, currentLastIndex - 1 }, 1.0);
				labelsMask.putScalar(new int[] { point, currentLastIndex - 1 }, 1.0);

				codeForCurrentClass++;
			}
		}

		this.cursor += numberOfExamples;
		return new DataSet(features, labels, featuresMask, labelsMask);
	}

	private List<String> tokenizeSentence(String sentence) {
		Tokenizer t = this.tokenizerFactory.create(sentence);

		List<String> result = new ArrayList<>();
		while (t.hasMoreTokens()) {
			String token = t.nextToken();
			if (!this.wordVectors.outOfVocabularySupported() && !this.wordVectors.hasWord(token)) {
				continue;
			}
			result.add(token);
		}
		return result;
	}

	@Override
	public boolean hasNext() {
		return this.cursor < this.totalExamples();
	}

	@Override
	public DataSet next() {
		return this.next(this.minibatchSize);
	}

	@Override
	public DataSet next(int num) {
		if (this.cursor >= this.smallestNumberOfLines * this.numberOfClasses) {
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
	public int inputColumns() {
		return this.vectorSize;
	}

	@Override
	public int totalOutcomes() {
		return this.numberOfClasses;
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
	}

	@Override
	public int batch() {
		return this.minibatchSize;
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public List<String> getLabels() {
		return Arrays.asList(this.labels);
	}

	/**
	 * Used post training to convert a String to a features INDArray that can be
	 * passed to the network output method
	 * 
	 * @param text      text to vectorise
	 * @param maxLength maxLength Maximum length (if text is longer than this:
	 *                  truncate to maxLength). Use Integer.MAX_VALUE to not
	 *                  truncate
	 * @return Features array for the given input String
	 */
	public INDArray loadFeaturesFromString(String text, int maxLength) {
		List<String> tokens = tokenizerFactory.create(text).getTokens();
		List<String> tokensFiltered = new ArrayList<>();
		for (String t : tokens) {
			if (wordVectors.hasWord(t))
				tokensFiltered.add(t);
		}
		int outputLength = Math.min(maxLength, tokensFiltered.size());

		INDArray features = Nd4j.create(1, vectorSize, outputLength);

		int count = 0;
		for (int j = 0; j < tokensFiltered.size() && count < maxLength; j++) {
			String token = tokensFiltered.get(j);
			INDArray vector = wordVectors.getWordVectorMatrix(token);
			if (vector == null) {
				continue; // Word not in word vectors
			}
			features.put(new INDArrayIndex[] { NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j) },
					vector);
			count++;
		}

		return features;
	}

	public static class Builder {

		private WordVectors wordVectors;
		private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
		private int maxSentenceLength = -1;
		private int minibatchSize = 32;

		private String[] pathsToCSVFilePerClass;
		private int smallestNumberOfLines;
		private String[] labels;

		public Builder(String[] pathsToCSVFilePerClass, int smallestNumberOfLines, String[] labels) {
			this.pathsToCSVFilePerClass = pathsToCSVFilePerClass;
			this.smallestNumberOfLines = smallestNumberOfLines;
			this.labels = labels;
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

		public ClassifiedTextIterator4RNN build() throws IOException, InterruptedException {
			if (wordVectors == null) {
				throw new IllegalStateException(
						"Cannot build ClassifiedTextIterator4Rnn without a WordVectors instance");
			}

			return new ClassifiedTextIterator4RNN(this.pathsToCSVFilePerClass, this.smallestNumberOfLines, this.labels,
					this);
		}

	}
}
