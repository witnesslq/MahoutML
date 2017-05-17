package Classification;


import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.RegressionResultAnalyzer;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.builder.DecisionTreeBuilder;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.DescriptorException;
import org.apache.mahout.classifier.df.mapreduce.Builder;
import org.apache.mahout.classifier.df.mapreduce.Classifier;
import org.apache.mahout.classifier.df.mapreduce.inmem.InMemBuilder;
import org.apache.mahout.classifier.df.mapreduce.partial.PartialBuilder;
import org.apache.mahout.classifier.df.tools.Describe;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;

public class RadomForest {
    private static final Logger log = LoggerFactory.getLogger(RadomForest.class);

    private final String BASE_PATH = "ClassificationDir/RandomForestDir/";

    private Path dataPath;

    private FileSystem dataFS;

    private Path datasetPath;

    private Path outputPath;

    private Integer m; // Number of variables to select at each tree-node

    private boolean complemented; // tree is complemented

    private Integer minSplitNum; // minimum number for split

    private Double minVarianceProportion; // minimum proportion of the total variance for split

    private int nbTrees; // Number of trees to grow

    private Long seed; // Random seed

    private boolean isPartial; // use partial data implementation

    private boolean analyze; // analyze the classification results ?

    private boolean useMapreduce; // use the mapreduce classifier ?

    private Path modelPath; // path where the forest is stored

    private FileSystem outFS;



    public static void main(String[] args) throws Exception {
        String[] describeArgs = new String[] {"-p","ClassificationDir/RandomForestDir/input/iris.data",
                "-f", "ClassificationDir/RandomForestDir/input/iris.info", "-d", "4","N","L"};

        String[] buildArgs = new String[] {"-d","ClassificationDir/RandomForestDir/input/iris.data",
                "-ds", "ClassificationDir/RandomForestDir/input/iris.info",
                "-o", "ClassificationDir/RandomForestDir/model", "-t", "30"};

        String[] testArgs = new String[] {"-i","ClassificationDir/RandomForestDir/input/iris.data",
                "-ds", "ClassificationDir/RandomForestDir/input/iris.info",
                "-m", "ClassificationDir/RandomForestDir/model/forest.seq",
                "-o", "ClassificationDir/RandomForestDir/prediction"};
        final RadomForest application = new RadomForest();

        try {
            application.describeForest(describeArgs);
            application.buildForest(buildArgs);
            application.testForest(testArgs);
        }
        catch (final Exception e) {
            log.error("Classification.RandomForest failed", e);
        }
    }

    private void describeForest(String[] args) throws IOException, DescriptorException {
        HadoopUtil.delete(new Configuration(), new Path(args[Arrays.asList(args).indexOf("-f")+1]));
        try {
            Describe.main(args);
        } catch (Exception e) {
            log.error("Exception", e);
        }
    }

    private void buildForest(String[] args) throws Exception {
        final Configuration configuration = new Configuration();

        DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
        ArgumentBuilder abuilder = new ArgumentBuilder();
        GroupBuilder gbuilder = new GroupBuilder();

        Option dataOpt = obuilder.withLongName("data").withShortName("d").withRequired(true)
                .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
                .withDescription("Data path").create();

        Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true)
                .withArgument(abuilder.withName("dataset").withMinimum(1).withMaximum(1).create())
                .withDescription("Dataset path").create();

        Option selectionOpt = obuilder.withLongName("selection").withShortName("sl").withRequired(false)
                .withArgument(abuilder.withName("m").withMinimum(1).withMaximum(1).create())
                .withDescription("Optional, Number of variables to select randomly at each tree-node.\n"
                        + "For classification problem, the default is square root of the number of explanatory variables.\n"
                        + "For regression problem, the default is 1/3 of the number of explanatory variables.").create();

        Option noCompleteOpt = obuilder.withLongName("no-complete").withShortName("nc").withRequired(false)
                .withDescription("Optional, The tree is not complemented").create();

        Option minSplitOpt = obuilder.withLongName("minsplit").withShortName("ms").withRequired(false)
                .withArgument(abuilder.withName("minsplit").withMinimum(1).withMaximum(1).create())
                .withDescription("Optional, The tree-node is not divided, if the branching data size is "
                        + "smaller than this value.\nThe default is 2.").create();

        Option minPropOpt = obuilder.withLongName("minprop").withShortName("mp").withRequired(false)
                .withArgument(abuilder.withName("minprop").withMinimum(1).withMaximum(1).create())
                .withDescription("Optional, The tree-node is not divided, if the proportion of the "
                        + "variance of branching data is smaller than this value.\n"
                        + "In the case of a regression problem, this value is used. "
                        + "The default is 1/1000(0.001).").create();

        Option seedOpt = obuilder.withLongName("seed").withShortName("sd").withRequired(false)
                .withArgument(abuilder.withName("seed").withMinimum(1).withMaximum(1).create())
                .withDescription("Optional, seed value used to initialise the Random number generator").create();

        Option partialOpt = obuilder.withLongName("partial").withShortName("p").withRequired(false)
                .withDescription("Optional, use the Partial Data implementation").create();

        Option nbtreesOpt = obuilder.withLongName("nbtrees").withShortName("t").withRequired(true)
                .withArgument(abuilder.withName("nbtrees").withMinimum(1).withMaximum(1).create())
                .withDescription("Number of trees to grow").create();

        Option outputOpt = obuilder.withLongName("output").withShortName("o").withRequired(true)
                .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
                .withDescription("Output path, will contain the Decision Forest").create();

        Option helpOpt = obuilder.withLongName("help").withShortName("h")
                .withDescription("Print out help").create();

        Group group = gbuilder.withName("Options").withOption(dataOpt).withOption(datasetOpt)
                .withOption(selectionOpt).withOption(noCompleteOpt).withOption(minSplitOpt)
                .withOption(minPropOpt).withOption(seedOpt).withOption(partialOpt).withOption(nbtreesOpt)
                .withOption(outputOpt).withOption(helpOpt).create();

        Parser parser = new Parser();
        parser.setGroup(group);
        CommandLine cmdLine = parser.parse(args);
        if (cmdLine.hasOption("help")) {
            CommandLineUtil.printHelp(group);
            return;
        }

        isPartial = cmdLine.hasOption(partialOpt);
        String dataName = cmdLine.getValue(dataOpt).toString();
        String datasetName = cmdLine.getValue(datasetOpt).toString();
        String outputName = cmdLine.getValue(outputOpt).toString();
        nbTrees = Integer.parseInt(cmdLine.getValue(nbtreesOpt).toString());

        if (cmdLine.hasOption(selectionOpt)) {
            m = Integer.parseInt(cmdLine.getValue(selectionOpt).toString());
        }
        complemented = !cmdLine.hasOption(noCompleteOpt);
        if (cmdLine.hasOption(minSplitOpt)) {
            minSplitNum = Integer.parseInt(cmdLine.getValue(minSplitOpt).toString());
        }
        if (cmdLine.hasOption(minPropOpt)) {
            minVarianceProportion = Double.parseDouble(cmdLine.getValue(minPropOpt).toString());
        }
        if (cmdLine.hasOption(seedOpt)) {
            seed = Long.valueOf(cmdLine.getValue(seedOpt).toString());
        }

        dataPath = new Path(dataName);
        datasetPath = new Path(datasetName);
        outputPath = new Path(outputName);

        FileSystem ofs = outputPath.getFileSystem(configuration);
        if (ofs.exists(outputPath)) {
            HadoopUtil.delete(configuration, outputPath);
        }
        DecisionTreeBuilder treeBuilder = new DecisionTreeBuilder();
        if (m != null) {
            treeBuilder.setM(m);
        }

        treeBuilder.setComplemented(complemented);
        if (minSplitNum != null) {
            treeBuilder.setMinSplitNum(minSplitNum);
        }
        if (minVarianceProportion != null) {
            treeBuilder.setMinVarianceProportion(minVarianceProportion);
        }
        Builder forestBuilder;
        if (isPartial) {
            log.info("Partial Mapred implementation");
            forestBuilder = new PartialBuilder(treeBuilder, dataPath, datasetPath, seed, configuration);
        } else {
            log.info("InMem Mapred implementation");
            forestBuilder = new InMemBuilder(treeBuilder, dataPath, datasetPath, seed, configuration);
        }

        forestBuilder.setOutputDirName(outputPath.getName());

        log.info("Building the forest...");
        long time = System.currentTimeMillis();

        DecisionForest forest = forestBuilder.build(nbTrees);

        if (forest == null) {
            return;
        }

        time = System.currentTimeMillis() - time;
        log.info("Build Time: {}", DFUtils.elapsedTime(time));
        log.info("Forest num Nodes: {}", forest.nbNodes());
        log.info("Forest mean num Nodes: {}", forest.meanNbNodes());
        log.info("Forest mean max Depth: {}", forest.meanMaxDepth());

        // store the decision forest in the output path
        Path forestPath = new Path(outputPath, "forest.seq");
        log.info("Storing the forest in: {}", forestPath);
        DFUtils.storeWritable(configuration, forestPath, forest);

    }

    private  void testForest(String[] args) throws Exception {
        final Configuration configuration = new Configuration();

        DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
        ArgumentBuilder abuilder = new ArgumentBuilder();
        GroupBuilder gbuilder = new GroupBuilder();

        Option inputOpt = DefaultOptionCreator.inputOption().create();

        Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true).withArgument(
                abuilder.withName("dataset").withMinimum(1).withMaximum(1).create()).withDescription("Dataset path")
                .create();

        Option modelOpt = obuilder.withLongName("model").withShortName("m").withRequired(true).withArgument(
                abuilder.withName("path").withMinimum(1).withMaximum(1).create()).
                withDescription("Path to the Decision Forest").create();

        Option outputOpt = DefaultOptionCreator.outputOption().create();

        Option analyzeOpt = obuilder.withLongName("analyze").withShortName("a").withRequired(false).create();

        Option mrOpt = obuilder.withLongName("mapreduce").withShortName("mr").withRequired(false).create();

        Option helpOpt = DefaultOptionCreator.helpOption();

        Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(datasetOpt).withOption(modelOpt)
                .withOption(outputOpt).withOption(analyzeOpt).withOption(mrOpt).withOption(helpOpt).create();

        Parser parser = new Parser();
        parser.setGroup(group);
        CommandLine cmdLine = parser.parse(args);

        if (cmdLine.hasOption("help")) {
            CommandLineUtil.printHelp(group);
            return;
        }

        String dataName = cmdLine.getValue(inputOpt).toString();
        String datasetName = cmdLine.getValue(datasetOpt).toString();
        String modelName = cmdLine.getValue(modelOpt).toString();
        String outputName = cmdLine.hasOption(outputOpt) ? cmdLine.getValue(outputOpt).toString() : null;
        analyze = cmdLine.hasOption(analyzeOpt);
        useMapreduce = true;

        dataPath = new Path(dataName);
        datasetPath = new Path(datasetName);
        modelPath = new Path(modelName);
        if (outputName != null) {
            outputPath = new Path(outputName);
        }

        // make sure the output file does not exist
        if (outputPath != null) {
            outFS = outputPath.getFileSystem(configuration);
            if (outFS.exists(outputPath)) {
                throw new IllegalArgumentException("Output path already exists");
            }
        }

        // make sure the decision forest exists
        FileSystem mfs = modelPath.getFileSystem(configuration);
        if (!mfs.exists(modelPath)) {
            throw new IllegalArgumentException("The forest path does not exist");
        }

        // make sure the test data exists
        dataFS = dataPath.getFileSystem(configuration);
        if (!dataFS.exists(dataPath)) {
            throw new IllegalArgumentException("The Test data path does not exist");
        }
        if (useMapreduce) {
            mapreduce(configuration);
        }
    }

    private void mapreduce(Configuration conf) throws ClassNotFoundException, IOException, InterruptedException {
        if (outputPath == null) {
            throw new IllegalArgumentException("You must specify the ouputPath when using the mapreduce implementation");
        }

        Classifier classifier = new Classifier(modelPath, dataPath, datasetPath, outputPath, conf);

        classifier.run();

        if (analyze) {
            double[][] results = classifier.getResults();
            if (results != null) {
                Dataset dataset = Dataset.load(conf, datasetPath);
                if (dataset.isNumerical(dataset.getLabelId())) {
                    RegressionResultAnalyzer regressionAnalyzer = new RegressionResultAnalyzer();
                    regressionAnalyzer.setInstances(results);
                    log.info("{}", regressionAnalyzer);
                } else {
                    ResultAnalyzer analyzer = new ResultAnalyzer(Arrays.asList(dataset.labels()), "unknown");
                    for (double[] res : results) {
                        analyzer.addInstance(dataset.getLabelString(res[0]),
                                new ClassifierResult(dataset.getLabelString(res[1]), 1.0));
                    }
                    log.info("{}", analyzer);
                }
            }
        }
    }

}
