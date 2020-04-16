package ch.zhaw.deeplearning4j;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.NoSuchElementException;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class QAIterator4EncDecLSTM implements MultiDataSetIterator {

	private static final Logger log = LoggerFactory.getLogger(QAIterator4EncDecLSTM.class);

	private final String pathToCSVFile;
	private final int minibatchSize;

	private int cursor;
	private boolean done;

	public QAIterator4EncDecLSTM(String pathToCSVFile, Builder builder) {

		this.pathToCSVFile = builder.pathToCSVFile;
		this.minibatchSize = builder.minibatchSize;

		this.reset();

	}

	public static void main(String[] args) {

		// Nd4j.getMemoryManager().setAutoGcWindow(5000);

		WordVectors wordVectors = null; // WordVectorSerializer.loadStaticModel(new File(Paths.WORD_VECTORS_PATH));

		MultiDataSetIterator it;
		MultiDataSet current;
		int count = 0;

		it = new QAIterator4EncDecLSTM.Builder("classifieddialoguepairs/dialoguepairs-comedy.csv")
				.wordVectors(wordVectors).minibatchSize(32).maxSentenceLength(200).build();

		count = 0;
		while (it.hasNext()) {
			current = it.next();
			count++;
			System.out.println("MultiDataSet[" + count + "] "); // + current.getMemoryFootprint()
		}
	}

	private MultiDataSet nextDataSet(int numberOfExamples) throws IOException, InterruptedException {

		CSVRecordReader reader = new CSVRecordReader();
		reader.initialize(new FileSplit(new File(pathToCSVFile)));

		while (reader.hasNext()) {
			List<Writable> line = reader.next();
			System.out.println(line.get(0) + "\n\t" + line.get(1));
		}

		this.done = true;

		reader.close();

		return null;
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
		// TODO Auto-generated method stub

	}

	@Override
	public MultiDataSetPreProcessor getPreProcessor() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean resetSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean asyncSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void reset() {
		this.cursor = 0;
		this.done = false;
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
				// throw new IllegalStateException(
				// "Cannot build ClassifiedTextIterator4Rnn without a WordVectors instance");
			}

			return new QAIterator4EncDecLSTM(this.pathToCSVFile, this);
		}

	}

}
