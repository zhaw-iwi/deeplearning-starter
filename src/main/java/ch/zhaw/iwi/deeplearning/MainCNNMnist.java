/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package ch.zhaw.iwi.deeplearning;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Zurich University of Applied Sciences (ZHAW), Institute for Business
 * Information Systems (IWI), Center for Information Systems and Technologies
 * (CIS)
 * 
 * http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
 * 
 * @author Alexandre de Spindler (desa@zhaw.ch)
 * 
 */
public class MainCNNMnist {
	private static final Logger log = LoggerFactory.getLogger(MainCNNMnist.class);
	private static final String BASE_PATH = "mnist";

	public static void main(String[] args) throws Exception {

		log.info("> Hello CNN :-)");

		int height = 28; // height of the picture in px
		int width = 28; // width of the picture in px
		int channels = 1; // single channel for grayscale images
		int outputNum = 10; // 10 digits classification
		int batchSize = 54; // number of samples that will be propagated through the network in each
							// iteration
		int nEpochs = 1; // number of training epochs

		int seed = 1234; // number used to initialize a pseudorandom number generator.

		log.info("> Preparing Data ...");

		DataSetIterator trainIter = MainCNNMnist.getDataSetFromFolders(BASE_PATH + "/mnist_png/training", height, width,
				channels, batchSize, outputNum, seed);
		DataSetIterator testIter = MainCNNMnist.getDataSetFromFolders(BASE_PATH + "/mnist_png/testing", height, width,
				channels, batchSize, outputNum, seed);

		// pixel values from 0-255 to 0-1 (min-max scaling)
		DataNormalization imageScaler = new ImagePreProcessingScaler();
		imageScaler.fit(trainIter);
		trainIter.setPreProcessor(imageScaler);
		testIter.setPreProcessor(imageScaler);

		log.info("> Building Model ...");

		// reduce the learning rate as the number of training epochs increases
		// iteration #, learning rate
		Map<Integer, Double> learningRateSchedule = new HashMap<>();
		learningRateSchedule.put(0, 0.06);
		learningRateSchedule.put(200, 0.05);
		learningRateSchedule.put(600, 0.028);
		learningRateSchedule.put(800, 0.0060);
		learningRateSchedule.put(1000, 0.001);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.l2(0.0005) // ridge regression value
				.updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(5, 5).nIn(channels)
						.stride(1, 1)
						.nOut(20)
						.activation(Activation.IDENTITY)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(5, 5).stride(1, 1) // nIn need not specified in later layers
						.nOut(50)
						.activation(Activation.IDENTITY)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for
																					// normal image
				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		log.info("> Training & Testing Model ...");

		net.setListeners(new ScoreIterationListener(10));
		log.info("Total num of params: {}", net.numParams());

		// evaluation while training (the score should go down)
		for (int i = 0; i < nEpochs; i++) {
			net.fit(trainIter);
			log.info("Completed epoch {}", i);
			Evaluation eval = net.evaluate(testIter);
			log.info(eval.stats());

			trainIter.reset();
			testIter.reset();
		}

		log.info("> Good Bye ;-(");
	}

	private static DataSetIterator getDataSetFromFolders(String pathToRootFolder, int height, int width, int channels,
			int batchSize, int outputNum, int seed) throws IOException {

		Random randNumGen = new Random(seed);

		File data = new File(pathToRootFolder);
		FileSplit dataSplit = new FileSplit(data, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // use parent directory name as the image
																				// label
		ImageRecordReader dataRR = new ImageRecordReader(height, width, channels, labelMaker);
		dataRR.initialize(dataSplit);
		DataSetIterator result = new RecordReaderDataSetIterator(dataRR, batchSize, 1, outputNum);

		return result;
	}
}
