package ch.zhaw.iwi.deeplearning;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MainWordEmbedder {

	private static final Logger log = LoggerFactory.getLogger(MainWordEmbedder.class);

	public static void main(String[] args) throws IOException {

		MainWordEmbedder.log.info("Accessing file line by line");
		String filePath = new File("mtsamples/mtsamples-transcription.txt").getAbsolutePath();
		SentenceIterator iterator = new BasicLineIterator(filePath);
		iterator.setPreProcessor(new SentencePreProcessor() {
			@Override
			public String preProcess(String sentence) {
				return sentence.toLowerCase();
			}
		});

		MainWordEmbedder.log.info("Tokenising");
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		MainWordEmbedder.log.info("Building model");
		Word2Vec vec = new Word2Vec.Builder().minWordFrequency(3)
				.layerSize(100)
				.seed(42)
				.windowSize(5)
				.iterate(iterator)
				.tokenizerFactory(t)
				.build();

		MainWordEmbedder.log.info("Training model");
		vec.fit();

		// TODO SAVE
		// WordVectorSerializer.writeWordVectors(vec, "PATHTOFILE.txt");

		MainWordEmbedder.log.info("Testing");

		Collection<String> neighbours = vec.wordsNearest("heart", 7);
		System.out.println(neighbours);

		Collection<String> neighboursSum = vec.wordsNearestSum("heart", 7);
		System.out.println(neighboursSum);
	}

}
