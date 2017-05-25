package Clustering;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.conversion.InputDriver;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class FuzzyKmeans {

    private static final Logger log = LoggerFactory.getLogger(FuzzyKmeans.class);

    private String dataPath = "testdata";

    private static final String M_OPTION = FuzzyKMeansDriver.M_OPTION;

    private final String BASE_PATH = "ClusteringDir/FuzzyKmeansDir/";

    private final String POINTS_PATH = BASE_PATH + "points";

    private final String OUTPUT_PATH = BASE_PATH + "output";

    private final String CLUSTERS_PATH = BASE_PATH + "clusters";

    private Path input; // the input directory path

    private Path output; // the output directory path

    private String measureClass; // the DistanceMeasure to use

    private double convergenceDelta; // the double convergence criteria for iterations

    int maxIterations; // the int maximum number of iterations

    float fuzziness;

    private static final String DIRECTORY_CONTAINING_CONVERTED_INPUT = "data";

    public static void main(String[] args) throws Exception {
        final FuzzyKmeans application = new FuzzyKmeans();
        args = new String[] {"-i","ClusteringDir/FuzzyKmeansDir/breastCancer.csv","-cd","5","-x","10","-ow","-k","2","-m","2.0"};
        try {
            application.runFuzzyKmeans(args);
        }
        catch (final Exception e) {
            log.error("Clustering.FuzzyKmeans failed", e);
        }
    }

    private void runFuzzyKmeans(String[] args) throws Exception {
        final Configuration configuration = new Configuration();

        DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
        GroupBuilder gbuilder = new GroupBuilder();

        Option inputOpt = DefaultOptionCreator.inputOption().create();
        Option distanceMeasureOpt = DefaultOptionCreator.distanceMeasureOption().create();
        Option t1Opt = DefaultOptionCreator.t1Option().create();
        Option t2Opt = DefaultOptionCreator.t2Option().create();
        Option convergenceOpt = DefaultOptionCreator.convergenceOption().create();
        Option maxIterationsOpt = DefaultOptionCreator.maxIterationsOption().create();
        Option overwriteOpt = DefaultOptionCreator.overwriteOption().create();
        Option mOpt = obuilder.withLongName(M_OPTION).withShortName(M_OPTION).withRequired(true)
                .withArgument(new ArgumentBuilder().withName(M_OPTION).withMinimum(1).withMaximum(1).create())
                .withDescription("coefficient normalization factor, must be greater than 1").create();
        Option helpOpt = DefaultOptionCreator.helpOption();

        Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(mOpt)
                .withOption(distanceMeasureOpt).withOption(convergenceOpt).withOption(maxIterationsOpt)
                .withOption(t1Opt).withOption(t2Opt)
                .withOption(overwriteOpt).withOption(helpOpt).create();

        Parser parser = new Parser();
        parser.setGroup(group);
        CommandLine cmdLine = parser.parse(args);
        if (cmdLine.hasOption("help")) {
            CommandLineUtil.printHelp(group);
            return;
        }
        input = new Path(cmdLine.getValue(inputOpt).toString());
        output = new Path(OUTPUT_PATH);
        measureClass = cmdLine.getValue(distanceMeasureOpt).toString();
        if (measureClass == null) {
            measureClass = SquaredEuclideanDistanceMeasure.class.getName();
        }
        convergenceDelta = Double.parseDouble(cmdLine.getValue(convergenceOpt).toString());
        maxIterations = Integer.parseInt(cmdLine.getValue(maxIterationsOpt).toString());
        fuzziness = Float.parseFloat(cmdLine.getValue(mOpt).toString());

        if (cmdLine.hasOption(overwriteOpt)) {
            HadoopUtil.delete(configuration, output);
        }

        DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
        double t1 = Double.parseDouble(cmdLine.getValue(t1Opt).toString());
        double t2 = Double.parseDouble(cmdLine.getValue(t2Opt).toString());

        Path directoryContainingConvertedInput = new Path(output, DIRECTORY_CONTAINING_CONVERTED_INPUT);
        InputDriver.runJob(input, directoryContainingConvertedInput, "org.apache.mahout.math.RandomAccessSparseVector");
        Path canopyOutput = new Path(output, "canopies");
        CanopyDriver.run(configuration, directoryContainingConvertedInput, canopyOutput, measure, t1, t2, false, 0.0,
                false);


        FuzzyKMeansDriver.run(directoryContainingConvertedInput, new Path(canopyOutput, "clusters-0-final"), output,
                convergenceDelta, maxIterations, fuzziness, true, true, 0.0, false);
        readAndPrintOutputValues(configuration, output.toString());
    }

    private void readAndPrintOutputValues(Configuration conf, String outputPath)
            throws IOException {
        final Path input = new Path(outputPath + "/clusteredPoints/part-m-00000");
        final SequenceFile.Reader reader = new SequenceFile.Reader(conf, SequenceFile.Reader.file(input));
        final IntWritable key = new IntWritable();
        final WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();

        File file = new File(BASE_PATH+"result.txt");
        if (!file.exists()) {
            file.createNewFile();
        }
        FileWriter fileWriter = new FileWriter(file.getAbsoluteFile());
        BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);

        Integer vecNum = 0;
        while (reader.next(key, value)) {
            String[] values = value.toString().split("vec");
            String content = values[0]+"vec: "+vecNum+values[1].substring(1)+" belongs to cluster "+key.toString()+"\n";
            bufferedWriter.write(content);
            vecNum++;
        }
        reader.close();
        bufferedWriter.close();

    }

}