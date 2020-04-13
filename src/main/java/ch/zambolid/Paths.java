package ch.zambolid;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

public class Paths {

	// https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
	public static final String WORD_VECTORS_PATH = "C:\\Git\\GoogleNews-vectors-negative300.bin.gz";

	public static LabeledSentenceProvider createSentenceProvider(String[] pathsToCSVFilePerClass, String[] labels) {
		List<String> texts = new ArrayList<String>();
		List<String> textsLabels = new ArrayList<String>();

		SentenceIterator[] sentenceIterators = new SentenceIterator[pathsToCSVFilePerClass.length];
		for (int i = 0; i < pathsToCSVFilePerClass.length; i++) {
			sentenceIterators[i] = new LineSentenceIterator(new File(pathsToCSVFilePerClass[i]));
		}

		while (Paths.allSentenceIteratorsHaveNext(sentenceIterators)) {
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
}
