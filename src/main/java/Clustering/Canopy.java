package Clustering;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.conversion.InputDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Canopy {

    private static final Logger log = LoggerFactory.getLogger(Canopy.class);

    private final String BASE_PATH = "ClusteringDir/CanopyDir/";
    private final String OUTPUT_PATH = BASE_PATH + "output";

    private Path input; // the input directory path

    private Path output; // the output directory path

    private String measureClass; // the DistanceMeasure to use

    private double convergenceDelta; // the double convergence criteria for iterations

    int maxIterations; // the int maximum number of iterations

    private static final String DIRECTORY_CONTAINING_CONVERTED_INPUT = "data";

    public static void main(String[] args) throws Exception {
        final Canopy application = new Canopy();
        args = new String[] {"-i","ClusteringDir/CanopyDir/testdata","-cd","0.001","-x","10","-ow","-t1","1.0","-t2","0.95"};
        try {
            application.runCanopy(args);
        }
        catch (final Exception e) {
            log.error("Clustering.Canopy failed", e);
        }
    }

    private void runCanopy(String[] args) throws Exception {
        final Configuration configuration = new Configuration();

        GroupBuilder gbuilder = new GroupBuilder();

        Option inputOpt = DefaultOptionCreator.inputOption().create();
        Option distanceMeasureOpt = DefaultOptionCreator.distanceMeasureOption().create();
        Option t1Opt = DefaultOptionCreator.t1Option().create();
        Option t2Opt = DefaultOptionCreator.t2Option().create();
        Option convergenceOpt = DefaultOptionCreator.convergenceOption().create();
        Option maxIterationsOpt = DefaultOptionCreator.maxIterationsOption().create();
        Option overwriteOpt = DefaultOptionCreator.overwriteOption().create();
        Option helpOpt = DefaultOptionCreator.helpOption();

        Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(distanceMeasureOpt)
                .withOption(convergenceOpt).withOption(maxIterationsOpt).withOption(t1Opt).withOption(t2Opt)
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
        if (cmdLine.hasOption(overwriteOpt)) {
            HadoopUtil.delete(configuration, output);
        }

        DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
        double t1 = Double.parseDouble(cmdLine.getValue(t1Opt).toString());
        double t2 = Double.parseDouble(cmdLine.getValue(t2Opt).toString());

        Path directoryContainingConvertedInput = new Path(output, DIRECTORY_CONTAINING_CONVERTED_INPUT);
        log.info("Preparing Input");
        InputDriver.runJob(input, directoryContainingConvertedInput, "org.apache.mahout.math.RandomAccessSparseVector");
        log.info("Running Clustering.Canopy to get initial clusters");
        Path canopyOutput = new Path(output, "canopies");
        CanopyDriver.run(configuration, directoryContainingConvertedInput, canopyOutput, measure, t1, t2, false, 0.0,
                false);
        log.info("Running KMeans");
        KMeansDriver.run(configuration, directoryContainingConvertedInput, new Path(canopyOutput, Cluster.INITIAL_CLUSTERS_DIR
                + "-final"), output, convergenceDelta, maxIterations, true, 0.0, false);
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