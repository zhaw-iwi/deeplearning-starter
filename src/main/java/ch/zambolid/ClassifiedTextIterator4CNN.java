package ch.zambolid;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.Format;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;

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
	 * @param smallestNumberOfLines  the number of lines of the file with the least
	 *                               number of lines
	 * @param labels                 will be matched with pathsToCSVFilesPerClass
	 *                               along position
	 * @param wordVectors            WordVectors object
	 * @param batchSize              Size of each minibatch for training
	 * @param truncateLength         If text exceeds this length, it will be
	 *                               truncated
	 */
	public ClassifiedTextIterator4CNN(String[] pathsToCSVFilePerClass, int smallestNumberOfLines, String[] labels,
			WordVectors wordVectors, int batchSize, int truncateLength) {

		List<String> texts = new ArrayList<String>();
		List<String> textsLabels = new ArrayList<String>();

		SentenceIterator sentenceIt;
		for (int i = 0; i < pathsToCSVFilePerClass.length; i++) {

			sentenceIt = new LineSentenceIterator(new File(pathsToCSVFilePerClass[i]));
			while (sentenceIt.hasNext()) {
				texts.add(sentenceIt.nextSentence());
				textsLabels.add(labels[i]);
			}
		}

		LabeledSentenceProvider sentenceProvider = new CollectionLabeledSentenceProvider(texts, textsLabels);

		this.it = new CnnSentenceDataSetIterator.Builder(Format.CNN2D).sentenceProvider(sentenceProvider)
				.wordVectors(wordVectors).minibatchSize(batchSize).maxSentenceLength(truncateLength)
				.useNormalizedWordVectors(false).build();

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

		WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(
				"D:\\Java\\EclipseWorkspace\\word2vec-GoogleNews-vectors\\GoogleNews-vectors-negative300.bin.gz"));
		DataSetIterator it = new ClassifiedTextIterator4CNN(
				new String[] { "classifiedtextdata/lines-comedy_training.csv",
						"classifiedtextdata/lines-thriller_training.csv" },
				69908, new String[] { "comedy", "thriller" }, wordVectors, 100, 200);
		DataSet dataSet = it.next();
		System.out.println(dataSet.getFeatures());
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

}
