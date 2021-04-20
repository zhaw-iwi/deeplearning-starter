package ch.zhaw.iwi.deeplearning.chatbot;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.factory.Nd4j;
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
public class ChatbotEncDecWordVectors {

	private static final Logger log = LoggerFactory.getLogger(ChatbotEncDecWordVectors.class);

	public static final String WORDVECTORS_FILENAME = "classifieddialoguepairs/ChatbotEncDecWordVector.txt";
	public static final int WORDVECTORS_SIZE_DEFAULT = 300;

	public static void main(String[] args) throws IOException {

		log.info("> Hello ChatbotEncDecWordVectors :-)");

		Nd4j.getMemoryManager().setAutoGcWindow(2000);

		File trainingDataFile = new File(ChatbotEncDecTrainer.TRAINING_DATA_FILENAME);
		File wordVectorsFile = new File(ChatbotEncDecWordVectors.WORDVECTORS_FILENAME);

		TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

		ChatbotEncDecWordVectors.embed(trainingDataFile, wordVectorsFile, tokenizerFactory,
				ChatbotEncDecWordVectors.WORDVECTORS_SIZE_DEFAULT);
	}

	public static void embed(File trainingDataFile, File wordVectorsFile, TokenizerFactory t, int embeddingWidth)
			throws IOException {

		SentenceIterator lines = new BasicLineIterator(trainingDataFile);

		log.info("> Building Model ...");
		Word2Vec model = new Word2Vec.Builder().minWordFrequency(2)
				.iterations(5)
				.layerSize(embeddingWidth)
				.seed(42)
				.windowSize(20)
				.iterate(lines)
				.tokenizerFactory(t)
				.build();
		log.info("> Building Model DONE");

		log.info("> Training Model ...");
		model.fit();
		log.info("> Training Model DONE");

		log.info("> Saving Model ... to " + wordVectorsFile.getAbsolutePath());
		WordVectorSerializer.writeWordVectors(model.lookupTable(), wordVectorsFile);
		log.info("> Saving Model DONE");
	}

	public static WordVectors load(File wordVectorsFile) {

		log.info("> Loading Model ... from " + wordVectorsFile.getAbsolutePath());
		WordVectors result = WordVectorSerializer.readWord2VecModel(wordVectorsFile);
		log.info("> Loading Model DONE");

		return result;
	}

}
