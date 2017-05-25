package Clustering;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
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

public class Kmeans {
    private static final Logger log = LoggerFactory.getLogger(Kmeans.class);

    private String dataPath = "testdata";
    private final String BASE_PATH = "ClusteringDir/KmeansDir/";
    private final String POINTS_PATH = BASE_PATH + "points";
    private final String CLUSTERS_PATH = BASE_PATH + "clusters";
    private final String OUTPUT_PATH = BASE_PATH + "output";
    private Path inputPath;
    private Path outputPath;
    private int numberOfCluster;
    private String measureClass; // the DistanceMeasure to use
    private double convergenceDelta = 0.5; // the double convergence criteria for iterations
    int maxIterations = 10; // the int maximum number of iterations

    public static void main(String[] args) {
        final Kmeans application = new Kmeans();
        args = new String[] {"-i","ClusteringDir/KmeansDir/breastCancer.csv","-k","2","-cd","0.001","-x","10"};
        try {
            application.runKmeans(args);
        }
        catch (final Exception e) {
            log.error("Clustering.Kmeans failed", e);
        }
    }

    private void runKmeans(String[] args) throws Exception {

        final Configuration configuration = new Configuration();

        GroupBuilder gbuilder = new GroupBuilder();

        Option inputOpt = DefaultOptionCreator.inputOption().create();
        Option distanceMeasureOpt = DefaultOptionCreator.distanceMeasureOption().create();
        Option numClustersOpt = DefaultOptionCreator.numClustersOption().create();
        Option convergenceOpt = DefaultOptionCreator.convergenceOption().create();
        Option maxIterationsOpt = DefaultOptionCreator.maxIterationsOption().create();
        Option helpOpt = DefaultOptionCreator.helpOption();

        Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(distanceMeasureOpt)
                .withOption(numClustersOpt).withOption(convergenceOpt)
                .withOption(maxIterationsOpt).withOption(helpOpt).create();

        Parser parser = new Parser();
        parser.setGroup(group);
        CommandLine cmdLine = parser.parse(args);
        if (cmdLine.hasOption("help")) {
            CommandLineUtil.printHelp(group);
            return;
        }
        numberOfCluster = Integer.parseInt(cmdLine.getValue(numClustersOpt).toString());
        // Create input directories for data
        final File pointsDir = new File(POINTS_PATH);
        if (!pointsDir.exists()) {
            pointsDir.mkdir();
        }
        // read the point values and generate vectors from input data

        dataPath = cmdLine.getValue(inputOpt).toString();
        // Write data to sequence hadoop sequence files
        List<DenseVector> vectors = toDenseVector(configuration);

        if (measureClass == null) {
            measureClass = SquaredEuclideanDistanceMeasure.class.getCanonicalName();
//            measureClass = EuclideanDistanceMeasure.class.getName();
        }
        // Write initial centers for clusters
        writeClusterInitialCenters(configuration, measureClass, vectors);

        // Run K-means algorithm
        inputPath = new Path(POINTS_PATH);
        final Path clustersPath = new Path(CLUSTERS_PATH);
        outputPath = new Path(OUTPUT_PATH);
        convergenceDelta = Double.parseDouble(cmdLine.getValue(convergenceOpt).toString());
        maxIterations = Integer.parseInt(cmdLine.getValue(maxIterationsOpt).toString());
        HadoopUtil.delete(configuration, outputPath);

        KMeansDriver.run(configuration, inputPath, clustersPath, outputPath, convergenceDelta, maxIterations, true, 0, false);

        // Read and print output values
        readAndPrintOutputValues(configuration);
    }

    private void writeClusterInitialCenters(final Configuration conf, String measureClass, List<DenseVector> points)
            throws IOException {
        final Path writerPath = new Path(CLUSTERS_PATH + "/part-00000");

        final SequenceFile.Writer writer =
                SequenceFile.createWriter(conf, SequenceFile.Writer.file(writerPath),
                        SequenceFile.Writer.keyClass(Text.class),
                        SequenceFile.Writer.valueClass(Kluster.class));

        Random rand = new Random();
        for (int i = 0; i < numberOfCluster; i++) {
            int randomNum = rand.nextInt((points.size() - 0) + 1);
            final Vector vec = points.get(randomNum);

            // write the initial centers
            DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
            final Kluster cluster = new Kluster(vec, i, measure);
            writer.append(new Text(cluster.getIdentifier()), cluster);
        }
        writer.close();
    }

    private void readAndPrintOutputValues(final Configuration configuration)
            throws IOException {
        final Path input = new Path(OUTPUT_PATH + "/clusteredPoints/part-m-00000");
        //final Path input = new Path(OUTPUT_PATH + "/" + Cluster.FINAL_ITERATION_SUFFIX + "/part-r-00000");

        final SequenceFile.Reader reader = new SequenceFile.Reader(configuration, SequenceFile.Reader.file(input));
        final IntWritable key = new IntWritable();
        final WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();

        File file = new File(BASE_PATH+"result.txt");
        if (!file.exists()) {
            file.createNewFile();
        }
        FileWriter fileWriter = new FileWriter(file.getAbsoluteFile());
        BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
        Integer vecNum = 0;
        int correctCount = 0;
        List<String> raw = FileUtils.readLines(new File(BASE_PATH+"breastCancerWithClass.csv"));
        int[] targets = new int[raw.size()];
        int i = 0;
        for (String line:raw) {
            targets[i++] = Integer.parseInt(line.split(",")[10]) == 4? 1:0;
        }
        while (reader.next(key, value)) {
            String[] values = value.toString().split("vec");
            String content = values[0]+"vec: "+vecNum+values[1].substring(1)+" belongs to cluster "+key.toString()+"\n";
            bufferedWriter.write(content);
            if (key.get() == targets[vecNum]) {
                correctCount++;
            }
            vecNum++;
        }
        reader.close();
        float correctRatio = (float)correctCount/(float)targets.length;
        int inCorrectCount = targets.length - correctCount;
        float inCorrectRatio = 1 - correctRatio;
        bufferedWriter.write("Correctly Clustered Instances: " + correctCount + " " + correctRatio + "\n");
        bufferedWriter.write("Incorrectly Clustered Instances: " + inCorrectCount + " " + inCorrectRatio+ "\n");
        bufferedWriter.close();

    }


    private List<DenseVector> toDenseVector(Configuration conf) throws FileNotFoundException, IOException{
        List<DenseVector> positions = new ArrayList<DenseVector>();
        DenseVector position;
        BufferedReader br;
        br = new BufferedReader(new FileReader(this.dataPath));

        String sCurrentLine;
        while ((sCurrentLine = br.readLine()) != null) {
            double[] features = new double[9];
            String[] values = sCurrentLine.split(",");
            for(int indx=0; indx<features.length;indx++){
                features[indx] = Float.parseFloat(values[indx]);

            }
            position = new DenseVector(features);
            positions.add(position);
        }

        final Path path = new Path(POINTS_PATH + "/pointsFile");
        FileSystem fs = FileSystem.get(conf);
        SequenceFile.Writer writer = new SequenceFile.Writer(fs,  conf, path, Text.class, VectorWritable.class);

        VectorWritable vec = new VectorWritable();
        Integer count = 0;

        for(DenseVector vector : positions){
            vec.set(vector);
            writer.append(new Text(count.toString()), vec);
            count++;
        }
        writer.close();
        return positions;
    }

}

