package ch.zhaw.deeplearning4j;

import java.io.File;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MainEncDec {

	private static final Logger log = LoggerFactory.getLogger(MainEncDec.class);

	public static void main(String[] args) {

		log.info("> Hello EncDec :-)");

		double learningRate = 1e-1;
		int tbttSize = 25;
		int embeddingWidth = 300;
		int hiddenLayerWidth = 512;

		Nd4j.getMemoryManager().setAutoGcWindow(2000);

		log.info("> Preparing Data ...");

		// WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new
		// File(Paths.WORD_VECTORS_PATH));

		int batchSize = 32;

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
				// TODO .setInputTypes(InputType.recurrent(dict.size()), InputType.recurrent(dict.size()))
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

		log.info("> Training Model ...");

		log.info("> Testing Model ...");

		log.info("> Good Bye ;-(");
	}

}
