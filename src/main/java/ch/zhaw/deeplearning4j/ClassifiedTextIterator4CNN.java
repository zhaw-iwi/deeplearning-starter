package ch.zhaw.deeplearning4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.Format;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Zurich University of Applied Sciences (ZHAW), Institute for Business
 * Information Systems (IWI), Center for Information Systems and Technologies
 * (CIS)
 * 
 * @author Alexandre de Spindler (desa@zhaw.ch)
 * 
 */
public class ClassifiedTextIterator4CNN implements DataSetIterator {

	private final CnnSentenceDataSetIterator it;

	/**
	 * @param pathsToCSVFilePerClass the CSV file for each class containing one line
	 *                               per text. The length of this list is the number
	 *                               of classes. The first will be class = 0, the
	 *                               second class = 1 and the last one class =
	 *                               (number of classes -1)
	 * @param labels                 will be matched with pathsToCSVFilesPerClass
	 *                               along position
	 */
	public ClassifiedTextIterator4CNN(String[] pathsToCSVFilePerClass, String[] labels, Builder builder) {

		LabeledSentenceProvider sentenceProvider = ClassifiedTextIterator4CNN
				.createSentenceProvider(pathsToCSVFilePerClass, labels);

		this.it = new CnnSentenceDataSetIterator.Builder(Format.CNN2D).sentenceProvider(sentenceProvider)
				.wordVectors(builder.wordVectors)
				.tokenizerFactory(builder.tokenizerFactory)
				.minibatchSize(builder.minibatchSize)
				.maxSentenceLength(builder.maxSentenceLength)
				.useNormalizedWordVectors(false)
				.build();

	}

	private static LabeledSentenceProvider createSentenceProvider(String[] pathsToCSVFilePerClass, String[] labels) {
		List<String> texts = new ArrayList<String>();
		List<String> textsLabels = new ArrayList<String>();

		SentenceIterator[] sentenceIterators = new SentenceIterator[pathsToCSVFilePerClass.length];
		for (int i = 0; i < pathsToCSVFilePerClass.length; i++) {
			sentenceIterators[i] = new LineSentenceIterator(new File(pathsToCSVFilePerClass[i]));
		}

		while (ClassifiedTextIterator4CNN.allSentenceIteratorsHaveNext(sentenceIterators)) {
			for (int i = 0; i < pathsToCSVFilePerClass.length; i++) {
				texts.add(sentenceIterators[i].nextSentence());
				textsLabels.add(labels[i]);
			}
		}

		return new CollectionLabeledSentenceProvider(texts, textsLabels);
	}

	private static boolean allSentenceIteratorsHaveNext(SentenceIterator[] sentenceIterators) {
		for (SentenceIterator current : sentenceIterators) {
			if (!current.hasNext()) {
				return false;
			}
		}
		return true;
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
		DataSetIterator it = new ClassifiedTextIterator4CNN.Builder(new String[] {
				"classifiedtextdata/lines-comedy_training.csv", "classifiedtextdata/lines-thriller_training.csv" },
				new String[] { "comedy", "thriller" }).wordVectors(wordVectors)
						.minibatchSize(32)
						.maxSentenceLength(200)
						.build();

		DataSet current;
		int count = 0;
		while (it.hasNext()) {
			current = it.next();
			count++;
			System.out.println("DataSet[" + count + "] " + current.numExamples());
		}
	}

	@Override
	public boolean hasNext() {
		return this.it.hasNext();
	}

	@Override
	public DataSet next() {
		return this.it.next();
	}

	@Override
	public DataSet next(int num) {
		return this.it.next(num);
	}

	@Override
	public int inputColumns() {
		return this.it.inputColumns();
	}

	@Override
	public int totalOutcomes() {
		return this.it.totalOutcomes();
	}

	@Override
	public boolean resetSupported() {
		return this.it.resetSupported();
	}

	@Override
	public boolean asyncSupported() {
		return this.it.asyncSupported();
	}

	@Override
	public void reset() {
		this.it.reset();
	}

	@Override
	public int batch() {
		return this.it.batch();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		this.it.setPreProcessor(preProcessor);
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		return this.it.getPreProcessor();
	}

	@Override
	public List<String> getLabels() {
		return this.it.getLabels();
	}

	public static class Builder {

		private WordVectors wordVectors;
		private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
		private int maxSentenceLength = -1;
		private int minibatchSize = 32;

		private String[] pathsToCSVFilePerClass;
		private String[] labels;

		public Builder(String[] pathsToCSVFilePerClass, String[] labels) {
			this.pathsToCSVFilePerClass = pathsToCSVFilePerClass;
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

		public ClassifiedTextIterator4CNN build() throws IOException, InterruptedException {
			if (wordVectors == null) {
				throw new IllegalStateException(
						"Cannot build ClassifiedTextIterator4Rnn without a WordVectors instance");
			}

			return new ClassifiedTextIterator4CNN(this.pathsToCSVFilePerClass, this.labels, this);
		}

	}

}
