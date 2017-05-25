package Classification;

import java.io.*;
import java.util.*;

import com.google.common.collect.Lists;
import com.google.common.io.Resources;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.classifier.evaluation.Auc;
import org.apache.mahout.classifier.sgd.CsvRecordFactory;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogisticRegression {
    private static final Logger log = LoggerFactory.getLogger(LogisticRegression.class);

    private static LogisticModelParameters lmp;
    private String inputFile; // the input directory path
    private String outputFile;
    private int passes;
    private boolean scores;

    public static void main(String[] args) throws Exception {
        String[] trainArg = new String[] {"-i","ClassificationDir/LogisticRegressionDir/input/breastCancer.csv",
                "-o", "ClassificationDir/LogisticRegressionDir/model/modelOutput",
                "--predictors", "Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion"
                ,"Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses",
        "--types","numeric","--target","Class","--categories","2","--passes","500","--features","100","--rate","1"};

        String[] testArg = new String[] {"--input","ClassificationDir/LogisticRegressionDir/input/breastCancerTest.csv",
                "--model", "ClassificationDir/LogisticRegressionDir/model/modelOutput",
                "--scores", "--auc","--confusion"};

        final LogisticRegression application = new LogisticRegression();

        // Train a model
        application.train(trainArg);

        // Test the model
        application.testModel(testArg);
    }

    public void train(String[] arg) throws IOException {
        DefaultOptionBuilder obuilder = new DefaultOptionBuilder();


        Option scoresOpt = obuilder.withLongName("scores").withDescription("output score diagnostics during training").create();

        ArgumentBuilder argumentBuilder = new ArgumentBuilder();

        Option inputFileOpt = DefaultOptionCreator.inputOption().create();

        Option outputFileOpt = DefaultOptionCreator.outputOption().create();

        Option predictorsOpt = obuilder.withLongName("predictors")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("p").create())
                .withDescription("a list of predictor variables")
                .create();

        Option typesOpt = obuilder.withLongName("types").withRequired(true)
                .withArgument(argumentBuilder.withName("t").create())
                .withDescription("a list of predictor variable types (numeric, word, or text)")
                .create();

        Option targetOpt = obuilder.withLongName("target").withRequired(true)
                .withArgument(argumentBuilder.withName("target").withMaximum(1).create())
                .withDescription("the name of the target variable")
                .create();

        Option featuresOpt = obuilder.withLongName("features").withArgument(argumentBuilder.withName("numFeatures").withDefault("1000")
                .withMaximum(1).create()).withDescription("the number of internal hashed features to use").create();

        Option passesOpt = obuilder.withLongName("passes").withArgument(argumentBuilder.withName("passes").withDefault("2")
                .withMaximum(1).create()).withDescription("the number of times to pass over the input data").create();

        Option lambdaOpt = obuilder.withLongName("lambda").withArgument(argumentBuilder.withName("lambda").withDefault("1e-4").withMaximum(1).create())
                .withDescription("the amount of coefficient decay to use").create();

        Option rateOpt = obuilder.withLongName("rate").withArgument(argumentBuilder.withName("learningRate").withDefault("1e-3").withMaximum(1).create())
                .withDescription("the learning rate").create();

        Option targetCategoriesOpt = obuilder.withLongName("categories").withRequired(true)
                .withArgument(argumentBuilder.withName("number").withMaximum(1).create())
                .withDescription("the number of target categories to be considered").create();

        Option noBiasOpt = obuilder.withLongName("noBias")
                .withDescription("don't include a bias term")
                .create();

        Option helpOpt = obuilder.withLongName("help").withShortName("h")
                .withDescription("Print out help").create();

        Group group = new GroupBuilder().withOption(helpOpt).withOption(inputFileOpt).withOption(outputFileOpt).withOption(noBiasOpt)
                .withOption(targetOpt).withOption(targetCategoriesOpt).withOption(predictorsOpt).withOption(typesOpt)
                .withOption(passesOpt).withOption(lambdaOpt).withOption(rateOpt).withOption(featuresOpt).create();

        Parser parser = new Parser();
        parser.setGroup(group);
        CommandLine cmdLine = parser.parseAndHelp(arg);
        if (cmdLine.hasOption("help")) {
            CommandLineUtil.printHelp(group);
        }
        List<String> typeList = Lists.newArrayList();
        for (Object x : cmdLine.getValues(typesOpt)) {
            typeList.add(x.toString());
        }

        List<String> predictorList = Lists.newArrayList();
        for (Object x : cmdLine.getValues(predictorsOpt)) {
            predictorList.add(x.toString());
        }
        lmp = new LogisticModelParameters();
        lmp.setTypeMap(predictorList, typeList);
        lmp.setTargetVariable(cmdLine.getValue(targetOpt).toString());
        lmp.setMaxTargetCategories(Integer.parseInt(cmdLine.getValue(targetCategoriesOpt).toString()));
        lmp.setNumFeatures(Integer.parseInt(cmdLine.getValue(featuresOpt).toString()));
        lmp.setLambda(Double.parseDouble(cmdLine.getValue(lambdaOpt).toString()));
        lmp.setLearningRate(Double.parseDouble(cmdLine.getValue(rateOpt).toString()));
        lmp.setUseBias(cmdLine.hasOption(noBiasOpt));

        scores = cmdLine.hasOption(scoresOpt.toString());
        passes = Integer.parseInt(cmdLine.getValue(passesOpt).toString());
        inputFile = cmdLine.getValue(inputFileOpt).toString();
        outputFile = cmdLine.getValue(outputFileOpt).toString();

        List<String> raw = FileUtils.readLines(new File(inputFile));
        String header = raw.get(0);
        List<String> content = raw.subList(1, raw.size());
        // parse data
        CsvRecordFactory csv = lmp.getCsvRecordFactory();
        OnlineLogisticRegression olr = lmp.createRegression();
        csv.firstLine(header);

        for(int pass = 0; pass < passes; pass++) {
            for (String line : content) {
                Vector input = new RandomAccessSparseVector(lmp.getNumFeatures());
                int targetValue = csv.processLine(line, input);
                olr.train(targetValue, input);
            }
        }
        try {
            OutputStream modelOutput = new FileOutputStream(outputFile);
            lmp.saveTo(modelOutput);
        } catch (Exception e){
            log.error("Save to file fail...");
        }
    }

    void testModel(String[] arg) throws Exception {
        DefaultOptionBuilder builder = new DefaultOptionBuilder();

        Option help = builder.withLongName("help").withDescription("print this list").create();

        Option quiet = builder.withLongName("quiet").withDescription("be extra quiet").create();

        Option auc = builder.withLongName("auc").withDescription("print AUC").create();
        Option confusion = builder.withLongName("confusion").withDescription("print confusion matrix").create();

        Option scores = builder.withLongName("scores").withDescription("print scores").create();

        ArgumentBuilder argumentBuilder = new ArgumentBuilder();
        Option inputFileOption = builder.withLongName("input")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
                .withDescription("where to get training data")
                .create();

        Option modelFileOption = builder.withLongName("model")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("model").withMaximum(1).create())
                .withDescription("where to get a model")
                .create();

        Group group = new GroupBuilder().withOption(help).withOption(quiet)
                .withOption(auc).withOption(scores).withOption(confusion)
                .withOption(inputFileOption).withOption(modelFileOption)
                .create();

        Parser parser = new Parser();
        parser.setGroup(group);
        CommandLine cmdLine = parser.parseAndHelp(arg);
        if (cmdLine.hasOption("help")) {
            CommandLineUtil.printHelp(group);
            return;
        }
        String inputFile = cmdLine.getValue(inputFileOption).toString();
        String modelFile = cmdLine.getValue(modelFileOption).toString();
        boolean showAuc = cmdLine.hasOption(auc);
        boolean showScores = cmdLine.hasOption(scores);
        boolean showConfusion = cmdLine.hasOption(confusion);

        if (!showAuc && !showConfusion && !showScores) {
            showAuc = true;
            showConfusion = true;
        }

        Auc collector = new Auc();
        LogisticModelParameters lmp = LogisticModelParameters.loadFrom(new File(modelFile));

        CsvRecordFactory csv = lmp.getCsvRecordFactory();
        OnlineLogisticRegression lr = lmp.createRegression();
        BufferedReader in = open(inputFile);
        String line = in.readLine();
        csv.firstLine(line);
        line = in.readLine();
        File file = new File("ClassificationDir/LogisticRegressionDir/result.txt");
        if (!file.exists()) {
            file.createNewFile();
        }
        PrintStream out = new PrintStream(new BufferedOutputStream(new FileOutputStream("ClassificationDir/LogisticRegressionDir/result.txt")));
        System.setOut(out);
        if (showScores) {
            System.out.println("\"target\",\"model-output\",\"log-likelihood\"");
        }
        while (line != null) {
            Vector v = new SequentialAccessSparseVector(lmp.getNumFeatures());
            int target = csv.processLine(line, v);

            double score = lr.classifyScalar(v);
            if (showScores) {
                System.out.printf(Locale.ENGLISH, "%d,%.3f,%.6f%n", target, score, lr.logLikelihood(target, v));
            }
            collector.add(target, score);
            line = in.readLine();
        }

        if (showAuc) {
            System.out.printf(Locale.ENGLISH, "AUC = %.3f%n", collector.auc());
        }
        if (showConfusion) {
            Matrix m = collector.confusion();
            System.out.printf(Locale.ENGLISH, "confusion: [[%.1f, %.1f], [%.1f, %.1f]]%n",
                    m.get(0, 0), m.get(1, 0), m.get(0, 1), m.get(1, 1));
            m = collector.entropy();
            System.out.printf(Locale.ENGLISH, "entropy: [[%.1f, %.1f], [%.1f, %.1f]]%n",
                    m.get(0, 0), m.get(1, 0), m.get(0, 1), m.get(1, 1));
        }
        out.close();
        System.setOut(System.out);
    }

    static BufferedReader open(String inputFile) throws IOException {
        InputStream in;
        try {
            in = Resources.getResource(inputFile).openStream();
        } catch (IllegalArgumentException e) {
            in = new FileInputStream(new File(inputFile));
        }
        return new BufferedReader(new InputStreamReader(in, Charsets.UTF_8));
    }
}

